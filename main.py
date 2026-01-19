#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
小说数据合成脚本（生成 OpenAI messages 格式 jsonl）

功能：
- 递归遍历小说目录，识别文本文件（默认不打印路径/文件名/内容，只显示hash id）
- 健壮解码：BOM -> charset_normalizer(可选) -> 多候选打分
- 清洗：控制字符、换行、明显乱码替换符重复等
- 调用 https://chat.dphn.ai/api/chat （SSE 流式响应）获取 XML 格式输出
- 产出训练数据：{"messages":[{"role":"user","content": user_prompt}, {"role":"assistant","content": formatted}]}
- 断点续跑：state 文件记录 done/failed（按文件相对路径hash）
- 失败重试：指数退避
- 并发：--workers N
- 终端打印模型原始输出（默认开启）用于视觉反馈
- 重点增强（按用户要求）：
  1) 分片处理：仅失败分片重试（不整本打回）
  2) final 汇总步骤单独重试
  3) XML 不全：先做“修复追问”，再重试
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import threading
import unicodedata
import urllib.request
import urllib.error
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple


API_URL = "https://chat.dphn.ai/api/chat"

PRINT_LOCK = threading.Lock()     # 终端输出锁（防并发串台）
IO_LOCK = threading.Lock()        # 写jsonl/state锁（防并发写错乱）
PRINT_MODEL_OUTPUT = True         # 默认打印模型输出，可用 --no-print-model-output 关闭
XML_REPAIR = True                 # 默认启用XML修复追问，可用 --no-xml-repair 关闭


# ---------------------------
# 文件ID与状态
# ---------------------------

def file_id(rel_path: str) -> str:
    """不在state中保存真实路径，避免泄露：用hash作为key。"""
    return hashlib.sha1(rel_path.encode("utf-8", errors="ignore")).hexdigest()


def load_state(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {"done": {}, "failed": {}}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    obj.setdefault("done", {})
    obj.setdefault("failed", {})
    return obj


def save_state(path: str, state: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: str, record: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------
# 文本识别/解码/清洗（健壮）
# ---------------------------

def is_probably_text(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return False

    sample = data[:8192]
    total = len(sample)
    if total == 0:
        return False

    bad = 0
    for b in sample:
        if b in (9, 10, 13):  # \t \n \r
            continue
        if 32 <= b <= 126:
            continue
        if b >= 0x80:  # 可能是中文等
            continue
        bad += 1

    return (bad / total) < 0.06


def _bom_encoding(data: bytes) -> Optional[str]:
    if data.startswith(b"\xEF\xBB\xBF"):
        return "utf-8-sig"
    if data.startswith(b"\xFF\xFE\x00\x00") or data.startswith(b"\x00\x00\xFE\xFF"):
        return "utf-32"
    if data.startswith(b"\xFF\xFE") or data.startswith(b"\xFE\xFF"):
        return "utf-16"
    return None


def _decode_score(text: str) -> Tuple[int, int, int]:
    repl = text.count("�")
    ctrl = 0
    odd = 0
    for ch in text:
        o = ord(ch)
        if ch in ("\n", "\r", "\t"):
            continue
        if (0 <= o <= 0x1F) or (0x7F <= o <= 0x9F):
            ctrl += 1
        cat = unicodedata.category(ch)
        if cat in ("Co", "Cn"):
            odd += 1
    return (repl, ctrl, odd)


def try_decode(data: bytes) -> Tuple[str, str]:
    """返回 (text, encoding_used)"""
    bom = _bom_encoding(data)
    if bom:
        try:
            return data.decode(bom, errors="replace"), bom
        except Exception:
            pass

    # 可选依赖：charset_normalizer
    try:
        from charset_normalizer import from_bytes  # type: ignore
        best = from_bytes(data).best()
        if best and best.encoding:
            enc = str(best.encoding)
            try:
                return data.decode(enc, errors="replace"), enc
            except Exception:
                pass
    except Exception:
        pass

    candidates = [
        "utf-8",
        "utf-8-sig",
        "gb18030",
        "gbk",
        "big5",
        "utf-16",
        "utf-16le",
        "utf-16be",
    ]

    best_text = None
    best_enc = None
    best_score = None
    for enc in candidates:
        try:
            t = data.decode(enc, errors="replace")
        except Exception:
            continue
        score = _decode_score(t)
        if best_score is None or score < best_score:
            best_score = score
            best_text = t
            best_enc = enc

    if best_text is not None:
        return best_text, (best_enc or "unknown")

    return data.decode("utf-8", errors="replace"), "utf-8(replace)"


_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")  # 保留\t \n \r


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"(�){4,}", "�", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{6,}", "\n\n\n", text)
    return text.strip()


def read_text_file(path: str) -> Tuple[bool, str, str]:
    with open(path, "rb") as f:
        data = f.read()
    if not is_probably_text(data[:8192]):
        return False, "", ""
    text, enc = try_decode(data)
    return True, clean_text(text), enc


# ---------------------------
# 分块（处理超长文本）
# ---------------------------

def chunk_text(text: str, max_chars: int, overlap: int = 200) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    paras = text.split("\n\n")
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        c = "\n\n".join(cur).strip()
        if c:
            chunks.append(c)
        cur = []
        cur_len = 0

    for p in paras:
        p = p.strip()
        if not p:
            continue
        add_len = len(p) + (2 if cur else 0)
        if cur_len + add_len <= max_chars:
            cur.append(p)
            cur_len += add_len
        else:
            if len(p) > max_chars:
                flush()
                start = 0
                while start < len(p):
                    end = min(len(p), start + max_chars)
                    chunks.append(p[start:end].strip())
                    start = end - overlap if end < len(p) else end
            else:
                flush()
                cur.append(p)
                cur_len = len(p)
    flush()

    if overlap > 0 and len(chunks) >= 2:
        out: List[str] = []
        out.append(chunks[0])
        prev_tail = chunks[0][-overlap:]
        for c in chunks[1:]:
            out.append((prev_tail + "\n" + c).strip())
            prev_tail = c[-overlap:]
        return out

    return chunks


# ---------------------------
# SSE 调用与输出打印
# ---------------------------

@dataclass
class ChatConfig:
    model: str = "dolphinserver:24B"
    template: str = "logical"
    timeout: int = 180
    min_delay: float = 0.6


def _print_model_output(label: str, raw: str) -> None:
    with PRINT_LOCK:
        sys.stdout.write(f"\n===== MODEL OUTPUT BEGIN [{label}] =====\n")
        sys.stdout.write(raw)
        if not raw.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.write(f"===== MODEL OUTPUT END   [{label}] =====\n")
        sys.stdout.flush()


def sse_chat_completion(payload: Dict, cfg: ChatConfig) -> str:
    headers = {
        "accept": "text/event-stream",
        "accept-language": "zh-CN,zh;q=0.9",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "content-type": "application/json",
        "origin": "https://chat.dphn.ai",
        "referer": "https://chat.dphn.ai/",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")

    full: List[str] = []
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk:
                    continue
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {})
                content = delta.get("content")
                if content:
                    full.append(content)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read(4096).decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        raise RuntimeError(f"HTTPError {e.code}: {body[:300]}") from e

    return "".join(full)


def build_payload(user_content: str, cfg: ChatConfig) -> Dict:
    return {
        "messages": [{"role": "user", "content": user_content}],
        "model": cfg.model,
        "template": cfg.template,
    }


# ---------------------------
# 提示词与XML解析
# ---------------------------

SYSTEM_PROMPT = """你是一个用于“小说文本整理与数据合成”的助手。你必须严格按用户要求输出XML标签，不要输出任何额外解释。
你只能对格式做优化（标点、换行、段落、标题、少量Markdown排版），不得改写小说表达的意义与具体内容。

硬性要求（必须遵守）：
1) 统一输出为【简体中文】：如果原文包含繁体字，请转换为简体字（不改变含义）。
2) 清理无意义符号与连续标点：将明显无意义的重复标点/分隔符压缩或移除（例如“。。。。。”、“———”、“======”、“＊＊＊＊＊”等）。
3) 若原文用纯符号做分隔（如一行全是“====”/“——”/“***”），请改为Markdown分隔线：`---`（或等价的Markdown水平分割线）。
4) 允许删除明显无意义/广告/重复/乱码碎片；若发现内容位置错乱，可以通过移动其位置来修复，但不得凭空编造。"""

USER_PROMPT_TEMPLATE = """请阅读下方<xiaoshuo>中的小说原文（可能包含乱码、缺标点、断行紊乱等），并完成三件事：

任务A：整理小说格式
- 将繁体字转换为简体字，统一为简体中文输出（不得改变含义）。
- 清理无意义的连续标点/符号分隔（如长串句号/破折号/等号/星号等）；原本纯符号分隔行用Markdown分隔线 `---` 替代。
- 不改变原文意义与细节，不改写情节/人物设定/对话内容。
- 允许：补全/修正标点；调整换行段落；补充合理的标题层级；用少量Markdown提升可读性（如章节标题、分隔线）。
- 允许删除：明显无意义的重复、广告、水印、随机乱码碎片（例如长串无意义符号）。
- 若有段落明显错位，可移动到更合适位置以恢复阅读顺序（不要新增内容）。

任务B：写一段“综述”
- 用较短但精炼的文字概括小说内容，可包含主线、关键人物关系、冲突与结局走向（若原文无结局则说明）。

任务C：基于综述生成“用户提示词”
- 这是一条给大模型的用户指令，目标是让模型写出与该小说一致的故事。
- 表达方式尽量多样（例如“请写一篇小说……/写个故事……/创作一段长篇……”等），但必须忠于综述。
- 用户提示词可以长或短：短则抓住最突出的要点；长则覆盖更多细节。

输出要求（非常重要）：
1) 只输出一个<result>根标签，内部必须包含且仅包含三个标签：<formatted>、<summary>、<user_prompt>
2) 标签必须成对闭合；不要输出任何多余文字，不要用代码块。
3) <formatted>中放整理后的小说全文；<summary>放综述；<user_prompt>放用户提示词。

示例（仅示意格式，不代表内容）：
<result>
<formatted>
...整理后的正文...
</formatted>
<summary>
...综述...
</summary>
<user_prompt>
请写一篇小说，讲述……
</user_prompt>
</result>

下面是小说原文：
<xiaoshuo>
{xiaoshuo}
</xiaoshuo>
"""

CHUNK_PROMPT_TEMPLATE = """你将收到小说的一部分片段（chunk）。请完成两件事：
1) 整理该片段的格式（同样不得改变意义，可修正标点/换行/少量Markdown，可删除明显无意义重复/乱码）。
   - 将繁体字转换为简体字，统一为简体中文输出（不得改变含义）。
   - 清理无意义的连续标点/符号分隔；原本纯符号分隔行用Markdown分隔线 `---` 替代。
2) 为该片段写一个极简“片段摘要”（用于后续合并为总综述）。

输出要求（非常重要）：
- 只输出一个<chunk>根标签，内部必须包含两个标签：<formatted>、<chunk_summary>
- 不要输出任何多余文字。

示例（仅示意格式，不代表内容）：
<chunk>
<formatted>
...整理后的片段...
</formatted>
<chunk_summary>
...片段摘要...
</chunk_summary>
</chunk>

片段如下：
<xiaoshuo>
{xiaoshuo}
</xiaoshuo>
"""

FINAL_FROM_SUMMARIES_PROMPT_TEMPLATE = """你将收到多个片段摘要，请合并为小说的总综述，并基于总综述生成用户提示词。

输出要求（非常重要）：
1) 只输出一个<final>根标签，内部必须包含两个标签：<summary>、<user_prompt>
2) 不要输出任何多余文字。

示例（仅示意格式，不代表内容）：
<final>
<summary>
...总综述...
</summary>
<user_prompt>
...用户提示词...
</user_prompt>
</final>

片段摘要如下：
<summaries>
{summaries}
</summaries>
"""

REPAIR_RESULT_TEMPLATE = """你刚才的输出可能不是严格合规的XML，或缺少必要标签。
请你基于“原始输出文本”尽最大可能恢复为一个严格合规的XML，并且只输出XML本身。

规则：
- 必须只输出一个<result>根标签，且包含且仅包含：<formatted>、<summary>、<user_prompt>
- 所有标签必须成对闭合
- 不要输出任何解释、前后缀、代码块
- 不要编造新内容：尽量从原始输出中提取并整理；如果某个标签确实无法从原始输出恢复，可留空字符串，但标签必须存在

原始输出文本如下（按原样提供）：
<<<RAW_OUTPUT
{raw}
RAW_OUTPUT>>>
"""

REPAIR_CHUNK_TEMPLATE = """你刚才的输出可能不是严格合规的XML，或缺少必要标签。
请你基于“原始输出文本”尽最大可能恢复为一个严格合规的XML，并且只输出XML本身。

规则：
- 必须只输出一个<chunk>根标签，且包含且仅包含：<formatted>、<chunk_summary>
- 所有标签必须成对闭合
- 不要输出任何解释、前后缀、代码块
- 不要编造新内容：尽量从原始输出中提取并整理；如果某个标签确实无法从原始输出恢复，可留空字符串，但标签必须存在

原始输出文本如下（按原样提供）：
<<<RAW_OUTPUT
{raw}
RAW_OUTPUT>>>
"""

REPAIR_FINAL_TEMPLATE = """你刚才的输出可能不是严格合规的XML，或缺少必要标签。
请你基于“原始输出文本”尽最大可能恢复为一个严格合规的XML，并且只输出XML本身。

规则：
- 必须只输出一个<final>根标签，且包含且仅包含：<summary>、<user_prompt>
- 所有标签必须成对闭合
- 不要输出任何解释、前后缀、代码块
- 不要编造新内容：尽量从原始输出中提取并整理；如果某个标签确实无法从原始输出恢复，可留空字符串，但标签必须存在

原始输出文本如下（按原样提供）：
<<<RAW_OUTPUT
{raw}
RAW_OUTPUT>>>
"""


def _extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.S | re.I)
    if not m:
        return None
    return m[-1].strip()


def parse_result_xml(raw: str) -> Tuple[str, str, str]:
    formatted = _extract_tag(raw, "formatted")
    summary = _extract_tag(raw, "summary") or ""
    user_prompt = _extract_tag(raw, "user_prompt")
    if not formatted or not user_prompt:
        raise ValueError("XML不完整：缺少 <formatted> 或 <user_prompt>")
    return formatted, summary, user_prompt


def parse_chunk_xml(raw: str) -> Tuple[str, str]:
    formatted = _extract_tag(raw, "formatted")
    chunk_summary = _extract_tag(raw, "chunk_summary")
    if not formatted or not chunk_summary:
        raise ValueError("XML不完整：缺少 <formatted> 或 <chunk_summary>")
    return formatted, chunk_summary


def parse_final_xml(raw: str) -> Tuple[str, str]:
    summary = _extract_tag(raw, "summary") or ""
    user_prompt = _extract_tag(raw, "user_prompt")
    if not user_prompt:
        raise ValueError("XML不完整：缺少 <user_prompt>")
    return summary, user_prompt


def repair_xml(raw: str, *, kind: str, cfg: ChatConfig, label: str) -> str:
    """
    kind: result / chunk / final
    返回修复后的raw（仍需再次parse验证）
    """
    # 避免标记冲突（极小概率）
    safe_raw = raw.replace("<<<RAW_OUTPUT", "<<<RAW_OUT").replace("RAW_OUTPUT>>>", "RAW_OUT>>>")
    if kind == "result":
        prompt = REPAIR_RESULT_TEMPLATE.format(raw=safe_raw)
    elif kind == "chunk":
        prompt = REPAIR_CHUNK_TEMPLATE.format(raw=safe_raw)
    elif kind == "final":
        prompt = REPAIR_FINAL_TEMPLATE.format(raw=safe_raw)
    else:
        raise ValueError(f"unknown kind: {kind}")

    payload = build_payload(SYSTEM_PROMPT + "\n\n" + prompt, cfg)
    fixed = sse_chat_completion(payload, cfg)
    if PRINT_MODEL_OUTPUT:
        _print_model_output(f"{label}::repair::{kind}", fixed)
    return fixed


# ---------------------------
# 重试
# ---------------------------

def call_with_retries(fn, *, max_retries: int, base_delay: float = 2.0, max_delay: float = 60.0, jitter: float = 0.2):
    last = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt >= max_retries:
                break
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay *= (1.0 + random.uniform(-jitter, jitter))
            time.sleep(max(0.0, delay))
    raise last  # type: ignore


# ---------------------------
# 处理小说文本（单次/分片）
# ---------------------------

def process_text(text: str, *, cfg: ChatConfig, max_input_chars: int, label: str, max_retries: int) -> Tuple[str, str, str]:
    """
    返回 formatted, summary, user_prompt

    关键策略：
    - 单次：请求失败/解析失败 -> 修复追问 -> 仍失败 -> 重试该次请求
    - 分片：每个chunk独立重试/修复，不整本打回；final也独立重试/修复
    """

    def single_attempt() -> Tuple[str, str, str]:
        user_content = SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(xiaoshuo=text)
        payload = build_payload(user_content, cfg)
        raw = sse_chat_completion(payload, cfg)
        if PRINT_MODEL_OUTPUT:
            _print_model_output(label, raw)
        try:
            return parse_result_xml(raw)
        except ValueError:
            if not XML_REPAIR:
                raise
            fixed = repair_xml(raw, kind="result", cfg=cfg, label=label)
            return parse_result_xml(fixed)

    def chunk_attempt(chunk_text_: str, idx: int) -> Tuple[str, str]:
        prompt = CHUNK_PROMPT_TEMPLATE.format(xiaoshuo=chunk_text_)
        payload = build_payload(SYSTEM_PROMPT + "\n\n" + prompt, cfg)
        raw = sse_chat_completion(payload, cfg)
        if PRINT_MODEL_OUTPUT:
            _print_model_output(f"{label}::chunk{idx}", raw)
        try:
            return parse_chunk_xml(raw)
        except ValueError:
            if not XML_REPAIR:
                raise
            fixed = repair_xml(raw, kind="chunk", cfg=cfg, label=f"{label}::chunk{idx}")
            return parse_chunk_xml(fixed)

    def final_attempt(summaries_text: str) -> Tuple[str, str]:
        final_prompt = FINAL_FROM_SUMMARIES_PROMPT_TEMPLATE.format(summaries=summaries_text)
        payload = build_payload(SYSTEM_PROMPT + "\n\n" + final_prompt, cfg)
        raw = sse_chat_completion(payload, cfg)
        if PRINT_MODEL_OUTPUT:
            _print_model_output(f"{label}::final", raw)
        try:
            return parse_final_xml(raw)
        except ValueError:
            if not XML_REPAIR:
                raise
            fixed = repair_xml(raw, kind="final", cfg=cfg, label=f"{label}::final")
            return parse_final_xml(fixed)

    # 单次处理（也对这一次做重试，不依赖整本重试）
    if len(text) <= max_input_chars:
        result = call_with_retries(single_attempt, max_retries=max_retries)
        time.sleep(cfg.min_delay)
        return result

    # 分片处理
    chunks = chunk_text(text, max_chars=max_input_chars, overlap=200)
    formatted_parts: List[str] = []
    summaries: List[str] = []

    for i, ch in enumerate(chunks, start=1):
        fpart, s = call_with_retries(lambda: chunk_attempt(ch, i), max_retries=max_retries)
        formatted_parts.append(fpart.strip())
        summaries.append(f"[chunk {i}] {s.strip()}")
        time.sleep(cfg.min_delay)

    formatted_full = "\n\n".join(formatted_parts).strip()

    # final 单独重试（不重跑chunks）
    summary, user_prompt = call_with_retries(lambda: final_attempt("\n".join(summaries)), max_retries=max_retries)
    time.sleep(cfg.min_delay)

    return formatted_full, summary, user_prompt


def iter_files(root: str) -> Iterable[Tuple[str, str]]:
    root = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            abs_path = os.path.join(dirpath, name)
            rel_path = os.path.relpath(abs_path, root)
            yield abs_path, rel_path


@dataclass
class TaskResult:
    fid: str
    rel_path: str
    status: str  # ok / skipped / failed / dry_run_ok
    record: Optional[Dict] = None
    meta: Optional[Dict] = None
    err: Optional[str] = None


def process_file(abs_path: str, rel_path: str, *, args, cfg: ChatConfig) -> TaskResult:
    fid = file_id(rel_path)

    try:
        is_text, text, enc = read_text_file(abs_path)
    except Exception as e:
        return TaskResult(fid=fid, rel_path=rel_path, status="failed", err=f"read_error: {type(e).__name__}")

    if (not is_text) or (len(text) < args.min_chars):
        return TaskResult(
            fid=fid,
            rel_path=rel_path,
            status="skipped",
            meta={"status": "skipped_nontext_or_too_short", "len": len(text), "encoding": enc},
        )

    if args.dry_run:
        return TaskResult(
            fid=fid,
            rel_path=rel_path,
            status="dry_run_ok",
            meta={"status": "dry_run_ok", "len": len(text), "encoding": enc},
        )

    try:
        formatted, summary, user_prompt = process_text(
            text,
            cfg=cfg,
            max_input_chars=args.max_input_chars,
            label=fid,
            max_retries=args.max_retries,
        )

        record = {
            "messages": [
                {"role": "user", "content": user_prompt.strip()},
                {"role": "assistant", "content": formatted.strip()},
            ]
        }
        meta = {
            "status": "ok",
            "len_in": len(text),
            "encoding": enc,
            "len_user_prompt": len(user_prompt),
            "len_formatted": len(formatted),
            "has_summary": bool(summary.strip()),
        }
        return TaskResult(fid=fid, rel_path=rel_path, status="ok", record=record, meta=meta)

    except Exception as e:
        return TaskResult(fid=fid, rel_path=rel_path, status="failed", err=f"{type(e).__name__}: {str(e)[:220]}")


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="小说->训练数据(jsonl)合成脚本（SSE接口）")
    ap.add_argument("--root", default="小说", help="小说目录（默认: 小说）")
    ap.add_argument("--out", default="dataset.jsonl", help="输出jsonl路径（默认: dataset.jsonl）")
    ap.add_argument("--state", default="dataset.state.json", help="状态文件路径（默认: dataset.state.json）")
    ap.add_argument("--model", default="dolphinserver:24B", help="模型名（默认与curl.http一致）")
    ap.add_argument("--template", default="logical", help="template（默认与curl.http一致）")
    ap.add_argument("--timeout", type=int, default=180, help="单次请求超时秒数")
    ap.add_argument("--min-delay", type=float, default=0.6, help="每个worker内请求间最小延迟秒数")
    ap.add_argument("--max-retries", type=int, default=4, help="单次API调用失败重试次数（指数退避）。分片模式下每个chunk与final都单独使用该值。")
    ap.add_argument("--max-input-chars", type=int, default=12000, help="单次喂给模型的最大字符数，超出则分块")
    ap.add_argument("--min-chars", type=int, default=200, help="过短文本跳过（默认200）")
    ap.add_argument("--max-files", type=int, default=0, help="最多处理多少个文件（0表示不限制）")
    ap.add_argument("--workers", type=int, default=1, help="并发worker数（默认1）")
    ap.add_argument("--show-path", action="store_true", help="打印文件相对路径（默认不打印，避免泄露）")
    ap.add_argument("--dry-run", action="store_true", help="只验证文本识别/解码/清洗，不调用模型、不写jsonl")
    ap.add_argument("--no-print-model-output", dest="no_print_model_output", action="store_true",
                    help="不在终端打印模型原始输出（默认会打印）")
    ap.add_argument("--no-xml-repair", dest="no_xml_repair", action="store_true",
                    help="关闭XML不全时的修复追问（默认开启）")
    args = ap.parse_args()

    global PRINT_MODEL_OUTPUT, XML_REPAIR
    PRINT_MODEL_OUTPUT = (not getattr(args, "no_print_model_output", False))
    XML_REPAIR = (not getattr(args, "no_xml_repair", False))

    cfg = ChatConfig(model=args.model, template=args.template, timeout=args.timeout, min_delay=args.min_delay)

    out_path = os.path.abspath(args.out)
    state_path = os.path.abspath(args.state)
    state = load_state(state_path)

    # 收集pending
    pending: List[Tuple[str, str, str]] = []
    for abs_path, rel_path in iter_files(args.root):
        fid = file_id(rel_path)
        if fid in state["done"]:
            continue
        pending.append((abs_path, rel_path, fid))
        if args.max_files and len(pending) >= args.max_files:
            break

    if not pending:
        print("no pending files")
        print(f"out_jsonl={out_path}")
        print(f"state={state_path}")
        return

    workers = max(1, int(args.workers))
    print(f"pending={len(pending)} workers={workers} print_model_output={PRINT_MODEL_OUTPUT} xml_repair={XML_REPAIR}")

    processed = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_file, abs_path, rel_path, args=args, cfg=cfg) for abs_path, rel_path, _ in pending]

        for fut in as_completed(futs):
            r: TaskResult = fut.result()
            label = r.rel_path if args.show_path else f"id={r.fid[:10]}"

            with IO_LOCK:
                if r.status == "ok":
                    append_jsonl(out_path, r.record)  # type: ignore[arg-type]
                    processed += 1
                    state["done"][r.fid] = r.meta or {"status": "ok"}
                    state["failed"].pop(r.fid, None)
                    save_state(state_path, state)
                    meta = r.meta or {}
                    print(f"[ok]   {label} | in={meta.get('len_in')} enc={meta.get('encoding')} out+=1")
                elif r.status in ("skipped", "dry_run_ok"):
                    skipped += 1
                    state["done"][r.fid] = r.meta or {"status": r.status}
                    save_state(state_path, state)
                    meta = r.meta or {}
                    print(f"[skip] {label} | {meta.get('status')} len={meta.get('len')} enc={meta.get('encoding')}")
                else:
                    failed += 1
                    state["failed"][r.fid] = {"reason": r.err or "unknown"}
                    save_state(state_path, state)
                    print(f"[fail] {label} | {r.err}")

    print("\n=== done ===")
    print(f"processed_ok={processed} skipped={skipped} failed={failed}")
    print(f"out_jsonl={out_path}")
    print(f"state={state_path}")


if __name__ == "__main__":
    main()
