#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
小说数据合成脚本（生成 OpenAI messages 格式 jsonl）

核心目标：
- 读取小说文本
- 清洗（仅做格式修正，不改写内容）
- 生成：用户提示词 / 综述 / 分段综述 / 整理后的小说正文
- 输出为训练数据 jsonl（OpenAI messages）

重要设计说明：
1) 该接口不支持 system role，因此“系统提示词”应当视为普通文本指令并入 user content。
   但不同步骤需要的指令不同：格式清洗≠总结≠生成用户提示词。
   => 本脚本为不同步骤使用不同的提示词（避免无关约束让模型困惑）。

2) template 选择：
   - logical：适合做“格式修正/结构化输出/XML修复”
   - summary ：适合做“综述/摘要合并”
   - creative：适合做“用户提示词（让模型写小说的指令）”

3) 分片策略：
   - chunk 逐片处理：每片单独重试/修复（不整本打回）
   - final 汇总步骤（合并总综述、生成用户提示词）也单独重试/修复
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
# 进一步后处理：中文标点/元信息去除/标题markdown（程序兜底）
# ---------------------------

_CJK_RANGE_RE = re.compile(r"[\u3400-\u9FFF]")  # CJK扩展A+基本区

def _is_cjk(ch: str) -> bool:
    return bool(ch) and bool(_CJK_RANGE_RE.match(ch))


def normalize_punctuation_zh(text: str) -> str:
    """
    将常见英文标点在“中文上下文”中替换为中文标点（程序兜底，不完全依赖模型）：
    , . ? ! : ; () [] -> ， 。 ？ ！ ： ； （） 【】
    """
    if not text:
        return text

    s = text
    out: List[str] = []
    n = len(s)

    def prev_char(i: int) -> str:
        return s[i - 1] if i - 1 >= 0 else ""

    def next_char(i: int) -> str:
        return s[i + 1] if i + 1 < n else ""

    for i, ch in enumerate(s):
        p = prev_char(i)
        nx = next_char(i)

        # 句点：避免小数 3.14
        if ch == ".":
            if (p.isdigit() and nx.isdigit()):
                out.append(ch)
            elif _is_cjk(p) or _is_cjk(nx):
                out.append("。")
            else:
                out.append(ch)
            continue

        mapping = {
            ",": "，",
            "?": "？",
            "!": "！",
            ":": "：",
            ";": "；",
            "(": "（",
            ")": "）",
            "[": "【",
            "]": "】",
        }
        if ch in mapping:
            if _is_cjk(p) or _is_cjk(nx):
                out.append(mapping[ch])
            else:
                out.append(ch)
            continue

        out.append(ch)

    return "".join(out)


_METADATA_LINE_RE = re.compile(
    r"""^\s*(?:
    作者|作\s*者|Author|
    书名|作品名|小说名|
    写作时间|创作时间|更新时间|发布时间|更新于|发表于|
    来源|出处|首发|转载|转自|来自|
    网站|网址|链接|地址|
    QQ群|群号|公众号|微信|微博|
    TXT|TXT下载|电子书|完整版|全文|
    扫描|校对|排版|整理|制作|上传|
    版权|声明|免责声明
    )\s*[:：]?\s*.*$""",
    flags=re.X | re.I,
)

_URL_LINE_RE = re.compile(r"^\s*(https?://\S+|www\.\S+)\s*$", flags=re.I)


def strip_metadata_lines(text: str, *, head: int = 30, tail: int = 30) -> str:
    """
    尝试删除明显的作者/网站/时间/来源等元信息行（只在开头/结尾窗口内删除，降低误伤）。
    """
    if not text:
        return text
    lines = text.splitlines()
    n = len(lines)
    keep: List[str] = []
    for idx, ln in enumerate(lines):
        s = ln.strip()
        in_window = (idx < head) or (idx >= max(0, n - tail))
        if in_window:
            if not s:
                keep.append(ln)
                continue
            if _URL_LINE_RE.match(s):
                continue
            if _METADATA_LINE_RE.match(s):
                continue
            # 常见“本书来自xx/更多精彩请访问xx”
            if ("本书来自" in s) or ("更多精彩" in s and ("访问" in s or "下载" in s)):
                continue
        keep.append(ln)
    return "\n".join(keep).strip()


_CHAPTER_TITLE_RE = re.compile(r"^\s*(第[0-9一二三四五六七八九十百千两]+[章节回卷部集][^\n]{0,30})\s*$")
_SPECIAL_TITLE_RE = re.compile(r"^\s*(序章|序言|楔子|引子|前言|后记|尾声|番外)\s*$")


def add_markdown_headings(text: str) -> str:
    """
    程序兜底：对明显的章节标题行补上 Markdown 标题符号（避免模型漏加）。
    仅对整行匹配的标题操作，避免误伤正文。
    """
    if not text:
        return text
    out: List[str] = []
    for ln in text.splitlines():
        raw = ln.rstrip()
        s = raw.strip()
        if not s:
            out.append(raw)
            continue
        if s.startswith("#"):
            out.append(raw)
            continue
        if _CHAPTER_TITLE_RE.match(s) or _SPECIAL_TITLE_RE.match(s):
            out.append("## " + s)
            continue
        out.append(raw)
    return "\n".join(out).strip()


_MD_HR_RE = re.compile(r"^\s*(?:---|\*\*\*|___)\s*$")
_MD_FENCE_RE = re.compile(r"^\s*```")

def normalize_markdown_spacing(text: str) -> str:
    """
    让正文更符合常用 Markdown 阅读习惯（可读性兜底，不做复杂语义解析）：
    - 对“疑似自然段边界”的单换行，提升为“段落空行”（插入一个空行）
    - 标题/分隔线前后保证空行
    - 连续空行最多保留2个
    - 尽量不干扰引用块/列表/代码块
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    lines = [ln.rstrip() for ln in t.split("\n")]

    def is_blank(x: str) -> bool:
        return not x.strip()

    def is_heading(x: str) -> bool:
        return x.lstrip().startswith("#")

    def is_hr(x: str) -> bool:
        return bool(_MD_HR_RE.match(x.strip()))

    def is_quote(x: str) -> bool:
        return x.lstrip().startswith(">")

    def is_list_item(x: str) -> bool:
        xs = x.lstrip()
        if re.match(r"^(\-|\*|\+)\s+\S+", xs):
            return True
        if re.match(r"^\d+\.\s+\S+", xs):
            return True
        return False

    def ends_paragraph(x: str) -> bool:
        xs = x.strip()
        if not xs:
            return False
        # 常见段落结尾：句末标点/引号/省略号等
        enders = "。！？!?…"
        if xs[-1] in enders:
            return True
        # 对话常见：……”
        if xs.endswith("……") or xs.endswith("...") or xs.endswith("……】"):
            return True
        if xs[-1] in "”』」）】":
            if len(xs) >= 2 and xs[-2] in enders:
                return True
        return False

    out: list[str] = []
    in_code = False
    i = 0
    while i < len(lines):
        cur = lines[i]

        # 代码围栏：尽量保持原样
        if _MD_FENCE_RE.match(cur):
            in_code = not in_code
            out.append(cur)
            i += 1
            continue

        if in_code:
            out.append(cur)
            i += 1
            continue

        # 空行：压缩到最多2个
        if is_blank(cur):
            j = i
            while j < len(lines) and is_blank(lines[j]):
                j += 1
            # 保留最多2个空行
            out.append("")
            if j - i >= 2:
                out.append("")
            i = j
            continue

        out.append(cur)
        j = i + 1
        if j >= len(lines):
            break
        nxt = lines[j]

        # 原本就有空行：交给下一轮
        if is_blank(nxt):
            i += 1
            continue

        # 引用块：不改（我们程序生成的引用块已经很规范）
        if is_quote(cur) or is_quote(nxt):
            i += 1
            continue

        # 列表：不插空行，避免列表断裂
        if is_list_item(cur) and is_list_item(nxt):
            i += 1
            continue

        # 标题/分隔线：前后留空行（这里处理“cur 与 nxt 之间”）
        if is_heading(cur) or is_hr(cur) or is_heading(nxt) or is_hr(nxt):
            out.append("")
            i += 1
            continue

        # 普通文本：仅在“疑似段落结束”时插入空行
        if ends_paragraph(cur):
            out.append("")
        i += 1

    # 去掉末尾多余空行
    while out and is_blank(out[-1]):
        out.pop()
    return "\n".join(out).strip()


def postprocess_formatted_text(text: str) -> str:
    """
    对模型输出的 formatted 做最后兜底：
    - 去元信息
    - 英文标点->中文标点（中文上下文）
    - 补 Markdown 标题
    - Markdown段落空行（提高可读性/渲染一致性）
    """
    t = (text or "").strip()
    t = strip_metadata_lines(t)
    t = normalize_punctuation_zh(t)
    t = add_markdown_headings(t)
    t = normalize_markdown_spacing(t)
    return t.strip()


def postprocess_short_text(text: str) -> str:
    """
    给 summary / chunk_summary / user_prompt 做轻量后处理（主要是中文标点）。
    """
    t = (text or "").strip()
    t = normalize_punctuation_zh(t)
    return t.strip()


# ---------------------------
# 分块（处理超长文本）
# ---------------------------

def chunk_text(text: str, max_chars: int, *, lookback: int = 100) -> List[str]:
    """
    分片优化：
    - 优先在接近 max_chars 处向前找“换行”
    - 其次找句末标点（。！？?!）
    - 都找不到才硬切
    lookback：向前搜索范围（建议 100）
    """
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    i = 0
    n = len(text)

    sentence_delims = ["。", "！", "？", "!", "?"]

    while i < n:
        end = min(n, i + max_chars)
        if end >= n:
            cut = n
        else:
            win_start = max(i, end - max(1, int(lookback)))
            window = text[win_start:end]

            cut = None

            # 1) 换行优先
            nl = window.rfind("\n")
            if nl != -1:
                candidate = win_start + nl + 1
                if candidate > i:
                    cut = candidate

            # 2) 句末标点
            if not cut:
                best = -1
                for d in sentence_delims:
                    pos = window.rfind(d)
                    if pos > best:
                        best = pos
                if best != -1:
                    candidate = win_start + best + 1
                    if candidate > i:
                        cut = candidate

            # 3) 硬切兜底
            if not cut or cut <= i:
                cut = end

        piece = text[i:cut].strip()
        if piece:
            chunks.append(piece)

        # 保障前进
        if cut <= i:
            i = end
        else:
            i = cut

    return chunks


# ---------------------------
# SSE 调用与输出打印
# ---------------------------

@dataclass
class ChatConfig:
    model: str = "dolphinserver:24B"
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


def build_payload(user_content: str, cfg: ChatConfig, *, template: str) -> Dict:
    # 接口不支持 system role，这里只发一个 user message
    return {
        "messages": [{"role": "user", "content": user_content}],
        "model": cfg.model,
        "template": template,
    }


# ---------------------------
# 提示词与XML解析
# ---------------------------

BASE_XML_INSTRUCTIONS = """你是一个严格的文本处理助手。你必须严格按要求输出XML标签，不要输出任何额外解释、前后缀、代码块。"""

FORMAT_RULES = """你只能对文本“格式”做优化（标点、换行、段落、标题、少量Markdown排版），不得改写小说表达的意义与具体内容。

必须遵守：
1) 统一输出为【简体中文】：如果原文包含繁体字，请转换为简体字（不改变含义）。
2) 将英文标点符号替换为中文标点符号（在中文语境下）：, . ? ! : ; ( ) [ ] -> ， 。 ？ ！ ： ； （） 【】。
3) 清理无意义符号与连续标点：将明显无意义的重复标点/分隔符压缩或移除（例如“。。。。。”、“———”、“======”、“＊＊＊＊＊”等）。
4) 若原文用纯符号做分隔（如一行全是“====”/“——”/“***”），请改为Markdown分隔线：`---`。
5) 允许删除明显无意义/广告/重复/乱码碎片；若发现内容位置错乱，可以通过移动其位置来修复，但不得凭空编造。
6) 删除“作者信息/来源/写作时间/网站地址/链接/转载声明/版权声明/QQ群/公众号”等非正文元信息（通常位于开头或结尾）。"""

MARKDOWN_RULES = """Markdown排版增强（能做就做）：
- 自然段落之间必须用“空一行”分隔（Markdown段落需要空行）：每个自然段之间至少保留1个空行。
- 对话建议“一人一句成段”，不同人物说话尽量分成不同段落。
- 章节/卷/回/番外/序章等标题行：尽量使用Markdown标题（优先 `## 标题`；整书大标题可用 `# 标题`）。
- 需要分隔的地方：优先使用 `---`，并保证分隔线前后各有空行。
- 不要滥用列表/表格；保证小说正文可读性。"""

SUMMARY_RULES = """请写一段较短但信息密度高的综述，尽量包含：主线、关键人物关系、核心冲突、阶段性转折、结局走向；如果原文未完结/无结局，请明确说明。"""

USER_PROMPT_RULES = """基于给定的“综述”，生成一条【给大模型的用户指令】（user prompt），目标是让模型创作出与该小说一致的故事（人物、背景、冲突、走向尽量贴合）。
要求：
- 这条指令应当像真实用户在聊天框里发出的请求，不能提“根据综述/根据摘要/如下综述”等字眼。
- 可长可短：短则抓住最突出的要点；长则覆盖更多细节。
- 表达方式尽量多样（例如“请写一篇小说……/写个故事……/创作一段长篇……”等），但必须忠于综述，不要杜撰综述未提到的关键设定。
- 尽量把“你希望模型写出来的东西”说清楚：题材风格（若能从综述推断）、主角/关系、开端处境、关键矛盾、故事节奏与结尾倾向（开放/悲剧/圆满等）。
- 只输出一条用户提示词，不要多条备选。"""

FORMAT_PROMPT_TEMPLATE = f"""{BASE_XML_INSTRUCTIONS}

你将收到小说原文（可能包含乱码、缺标点、断行紊乱等）。请完成：整理小说格式。

{FORMAT_RULES}
{MARKDOWN_RULES}

输出要求（非常重要）：
1) 只输出一个<format>根标签，内部必须包含且仅包含一个标签：<formatted>
2) 标签必须成对闭合；不要输出任何多余文字

示例（仅示意格式）：
<format>
<formatted>
...整理后的正文...
</formatted>
</format>

下面是小说原文：
<xiaoshuo>
{{xiaoshuo}}
</xiaoshuo>
"""

CHUNK_PROMPT_TEMPLATE = f"""{BASE_XML_INSTRUCTIONS}

你将收到小说的一部分片段（chunk）。请完成两件事：
1) 整理该片段的格式（不得改变意义，可修正标点/换行/少量Markdown，可删除明显无意义重复/乱码/元信息）。
2) 为该片段写一个“片段摘要”（用于后续合并为总综述，也会用于训练数据中的“前文提要”）。

{FORMAT_RULES}
{MARKDOWN_RULES}

片段摘要要求：
- 尽量客观、精炼、覆盖本片段新增的关键情节与人物动作
- 不要写评价，不要写推理过程
- 不要提“chunk/片段/摘要”字样，直接描述内容即可
- 尽量使用中文标点（，。！？：；等）

输出要求（非常重要）：
- 只输出一个<chunk>根标签，内部必须包含两个标签：<formatted>、<chunk_summary>
- 不要输出任何多余文字

示例（仅示意格式）：
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
{{xiaoshuo}}
</xiaoshuo>
"""

SUMMARY_FROM_FORMATTED_PROMPT_TEMPLATE = f"""{BASE_XML_INSTRUCTIONS}

你将收到“整理后的小说正文”。请基于该正文写一段总综述。

{SUMMARY_RULES}

输出要求（非常重要）：
1) 只输出一个<summary_result>根标签，内部必须包含且仅包含：<summary>
2) 不要输出任何多余文字

示例：
<summary_result>
<summary>
...综述...
</summary>
</summary_result>

整理后的小说正文如下：
<formatted_text>
{{formatted}}
</formatted_text>
"""

FINAL_FROM_SUMMARIES_PROMPT_TEMPLATE = f"""{BASE_XML_INSTRUCTIONS}

你将收到多个“片段摘要”。请将它们合并为小说的总综述。

{SUMMARY_RULES}

额外要求：
- 不要在综述中出现“第X段/第X片/chunk”等字样
- 尽量去重、合并同义信息，保证连贯

输出要求（非常重要）：
1) 只输出一个<final>根标签，内部必须包含且仅包含：<summary>
2) 不要输出任何多余文字

示例：
<final>
<summary>
...总综述...
</summary>
</final>

片段摘要如下：
<summaries>
{{summaries}}
</summaries>
"""

USER_PROMPT_FROM_SUMMARY_PROMPT_TEMPLATE = f"""{BASE_XML_INSTRUCTIONS}

你将收到一段“综述”。请生成一条用户提示词。

{USER_PROMPT_RULES}

输出要求（非常重要）：
1) 只输出一个<prompt_result>根标签，内部必须包含且仅包含：<user_prompt>
2) 不要输出任何多余文字

示例：
<prompt_result>
<user_prompt>
请写一篇小说，讲述……
</user_prompt>
</prompt_result>

综述如下：
<summary>
{{summary}}
</summary>
"""

# 兜底：一步到位生成 summary + user_prompt
FINAL_BOTH_FROM_SUMMARIES_PROMPT_TEMPLATE = f"""{BASE_XML_INSTRUCTIONS}

你将收到多个片段摘要，请合并为小说的总综述，并基于总综述生成用户提示词。

总综述要求：
{SUMMARY_RULES}

用户提示词要求：
{USER_PROMPT_RULES}

输出要求（非常重要）：
1) 只输出一个<final>根标签，内部必须包含且仅包含两个标签：<summary>、<user_prompt>
2) 不要输出任何多余文字

示例：
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
{{summaries}}
</summaries>
"""


# ---------------------------
# XML 解析
# ---------------------------

def _extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.S | re.I)
    if not m:
        return None
    return m[-1].strip()


def parse_formatted_xml(raw: str) -> str:
    formatted = _extract_tag(raw, "formatted")
    if not formatted:
        raise ValueError("XML不完整：缺少 <formatted>")
    return formatted


def parse_summary_xml(raw: str) -> str:
    summary = _extract_tag(raw, "summary")
    if summary is None:
        raise ValueError("XML不完整：缺少 <summary>")
    return summary.strip()


def parse_user_prompt_xml(raw: str) -> str:
    user_prompt = _extract_tag(raw, "user_prompt")
    if not user_prompt:
        raise ValueError("XML不完整：缺少 <user_prompt>")
    return user_prompt.strip()


def parse_chunk_xml(raw: str) -> Tuple[str, str]:
    formatted = _extract_tag(raw, "formatted")
    chunk_summary = _extract_tag(raw, "chunk_summary")
    if not formatted or not chunk_summary:
        raise ValueError("XML不完整：缺少 <formatted> 或 <chunk_summary>")
    return formatted.strip(), chunk_summary.strip()


def parse_final_both_xml(raw: str) -> Tuple[str, str]:
    summary = _extract_tag(raw, "summary") or ""
    user_prompt = _extract_tag(raw, "user_prompt")
    if not user_prompt:
        raise ValueError("XML不完整：缺少 <user_prompt>")
    return summary.strip(), user_prompt.strip()


# ---------------------------
# XML 修复追问（尽量结构化）
# ---------------------------

REPAIR_FORMAT_TEMPLATE = """你刚才的输出可能不是严格合规的XML，或缺少必要标签。
请你基于“原始输出文本”尽最大可能恢复为一个严格合规的XML，并且只输出XML本身。

规则：
- 必须只输出一个<format>根标签，且包含且仅包含：<formatted>
- 所有标签必须成对闭合
- 不要输出任何解释、前后缀、代码块
- 不要编造新内容：尽量从原始输出中提取并整理；如果无法恢复，可留空字符串，但标签必须存在

原始输出文本如下（按原样提供）：
<<<RAW_OUTPUT
{raw}
RAW_OUTPUT>>>
"""

REPAIR_SUMMARY_TEMPLATE = """你刚才的输出可能不是严格合规的XML，或缺少必要标签。
请你基于“原始输出文本”尽最大可能恢复为一个严格合规的XML，并且只输出XML本身。

规则：
- 必须只输出一个<summary_result>根标签，且包含且仅包含：<summary>
- 所有标签必须成对闭合
- 不要输出任何解释、前后缀、代码块
- 不要编造新内容：尽量从原始输出中提取并整理；如果无法恢复，可留空字符串，但标签必须存在

原始输出文本如下（按原样提供）：
<<<RAW_OUTPUT
{raw}
RAW_OUTPUT>>>
"""

REPAIR_USER_PROMPT_TEMPLATE = """你刚才的输出可能不是严格合规的XML，或缺少必要标签。
请你基于“原始输出文本”尽最大可能恢复为一个严格合规的XML，并且只输出XML本身。

规则：
- 必须只输出一个<prompt_result>根标签，且包含且仅包含：<user_prompt>
- 所有标签必须成对闭合
- 不要输出任何解释、前后缀、代码块
- 不要编造新内容：尽量从原始输出中提取并整理；如果无法恢复，可留空字符串，但标签必须存在

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
- 不要编造新内容：尽量从原始输出中提取并整理；如果无法恢复，可留空字符串，但标签必须存在

原始输出文本如下（按原样提供）：
<<<RAW_OUTPUT
{raw}
RAW_OUTPUT>>>
"""

REPAIR_FINAL_TEMPLATE = """你刚才的输出可能不是严格合规的XML，或缺少必要标签。
请你基于“原始输出文本”尽最大可能恢复为一个严格合规的XML，并且只输出XML本身。

规则：
- 必须只输出一个<final>根标签，且包含且仅包含：<summary>
- 所有标签必须成对闭合
- 不要输出任何解释、前后缀、代码块
- 不要编造新内容：尽量从原始输出中提取并整理；如果无法恢复，可留空字符串，但标签必须存在

原始输出文本如下（按原样提供）：
<<<RAW_OUTPUT
{raw}
RAW_OUTPUT>>>
"""

REPAIR_FINAL_BOTH_TEMPLATE = """你刚才的输出可能不是严格合规的XML，或缺少必要标签。
请你基于“原始输出文本”尽最大可能恢复为一个严格合规的XML，并且只输出XML本身。

规则：
- 必须只输出一个<final>根标签，且包含且仅包含：<summary>、<user_prompt>
- 所有标签必须成对闭合
- 不要输出任何解释、前后缀、代码块
- 不要编造新内容：尽量从原始输出中提取并整理；如果无法恢复，可留空字符串，但标签必须存在

原始输出文本如下（按原样提供）：
<<<RAW_OUTPUT
{raw}
RAW_OUTPUT>>>
"""


def repair_xml(raw: str, *, kind: str, cfg: ChatConfig, label: str, template_repair: str) -> str:
    """
    kind: format / summary / user_prompt / chunk / final / final_both
    返回修复后的raw（仍需再次parse验证）
    """
    safe_raw = raw.replace("<<<RAW_OUTPUT", "<<<RAW_OUT").replace("RAW_OUTPUT>>>", "RAW_OUT>>>")

    if kind == "format":
        prompt = REPAIR_FORMAT_TEMPLATE.format(raw=safe_raw)
    elif kind == "summary":
        prompt = REPAIR_SUMMARY_TEMPLATE.format(raw=safe_raw)
    elif kind == "user_prompt":
        prompt = REPAIR_USER_PROMPT_TEMPLATE.format(raw=safe_raw)
    elif kind == "chunk":
        prompt = REPAIR_CHUNK_TEMPLATE.format(raw=safe_raw)
    elif kind == "final":
        prompt = REPAIR_FINAL_TEMPLATE.format(raw=safe_raw)
    elif kind == "final_both":
        prompt = REPAIR_FINAL_BOTH_TEMPLATE.format(raw=safe_raw)
    else:
        raise ValueError(f"unknown kind: {kind}")

    payload = build_payload(prompt, cfg, template=template_repair)
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
# 输出拼装（把综述/前文提要用引用块插入assistant内容）
# ---------------------------

def _wrap_line_zh(line: str, max_len: int = 60, backtrack: int = 12) -> List[str]:
    """
    把一行超长中文文本按标点/长度进行软换行（用于引用块可读性）。
    """
    s = (line or "").strip()
    if not s:
        return [""]
    res: List[str] = []
    punct = "，。！？；：、"
    while len(s) > max_len:
        cut = max_len
        window = s[: max_len + 1]
        best = -1
        for p in punct:
            pos = window.rfind(p)
            if pos > best:
                best = pos
        if best >= max_len - backtrack and best > 0:
            cut = best + 1
        res.append(s[:cut].rstrip())
        s = s[cut:].lstrip()
    if s:
        res.append(s)
    return res


def wrap_zh_text(text: str, *, max_len: int = 60) -> str:
    """
    保留原始换行；若单行过长则按中文标点/长度拆行。
    """
    if not text:
        return ""
    out: List[str] = []
    for ln in text.splitlines():
        if not ln.strip():
            out.append("")
            continue
        out.extend(_wrap_line_zh(ln, max_len=max_len))
    return "\n".join(out).strip()


def _quote_block(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    body = wrap_zh_text(body, max_len=60)

    lines = body.splitlines()
    out: List[str] = [f"> **{title}**", ">"]
    for ln in lines:
        ln = ln.rstrip()
        if ln:
            out.append(f"> {ln}")
        else:
            out.append(">")
    return "\n".join(out).strip()


def assemble_assistant_content(*, formatted_parts: List[str], summary: str, chunk_summaries: List[str]) -> str:
    parts = [p.strip() for p in formatted_parts if p and p.strip()]
    if not parts:
        return ""

    out: List[str] = []
    sblock = _quote_block("综述", summary)
    if sblock:
        out.append(sblock)

    # 非分片：只加总综述，不加“前文提要”
    if len(parts) == 1:
        out.append(parts[0])
        return "\n\n".join(out).strip()

    # 分片：在每段之间插入“前文提要”（用上一段的chunk_summary）
    out.append(parts[0])
    for i in range(1, len(parts)):
        prev_sum = ""
        if i - 1 < len(chunk_summaries):
            prev_sum = chunk_summaries[i - 1]
        rblock = _quote_block("前文提要", prev_sum)
        if rblock:
            out.append(rblock)
        out.append(parts[i])

    return "\n\n".join([x for x in out if x and x.strip()]).strip()


# ---------------------------
# 处理小说文本（单次/分片）
# ---------------------------

@dataclass
class ProcessTextResult:
    formatted_parts: List[str]
    summary: str
    chunk_summaries: List[str]  # 与 formatted_parts 等长（分片时），非分片为空
    user_prompt: str


def process_text(
    text: str,
    *,
    cfg: ChatConfig,
    template_format: str,
    template_summary: str,
    template_creative: str,
    template_repair: str,
    max_input_chars: int,
    chunk_lookback: int,
    label: str,
    max_retries: int,
) -> ProcessTextResult:
    """
    分步骤策略：
    - format：logical
    - summary：summary
    - user_prompt：creative
    """

    def format_attempt(x: str) -> str:
        prompt = FORMAT_PROMPT_TEMPLATE.format(xiaoshuo=x)
        payload = build_payload(prompt, cfg, template=template_format)
        raw = sse_chat_completion(payload, cfg)
        if PRINT_MODEL_OUTPUT:
            _print_model_output(f"{label}::format", raw)
        try:
            return parse_formatted_xml(raw)
        except ValueError:
            if not XML_REPAIR:
                raise
            fixed = repair_xml(raw, kind="format", cfg=cfg, label=f"{label}::format", template_repair=template_repair)
            return parse_formatted_xml(fixed)

    def summary_from_formatted_attempt(formatted: str) -> str:
        prompt = SUMMARY_FROM_FORMATTED_PROMPT_TEMPLATE.format(formatted=formatted)
        payload = build_payload(prompt, cfg, template=template_summary)
        raw = sse_chat_completion(payload, cfg)
        if PRINT_MODEL_OUTPUT:
            _print_model_output(f"{label}::summary", raw)
        try:
            return parse_summary_xml(raw)
        except ValueError:
            if not XML_REPAIR:
                raise
            fixed = repair_xml(raw, kind="summary", cfg=cfg, label=f"{label}::summary", template_repair=template_repair)
            return parse_summary_xml(fixed)

    def user_prompt_from_summary_attempt(summary: str) -> str:
        prompt = USER_PROMPT_FROM_SUMMARY_PROMPT_TEMPLATE.format(summary=summary)
        payload = build_payload(prompt, cfg, template=template_creative)
        raw = sse_chat_completion(payload, cfg)
        if PRINT_MODEL_OUTPUT:
            _print_model_output(f"{label}::user_prompt", raw)
        try:
            return parse_user_prompt_xml(raw)
        except ValueError:
            if not XML_REPAIR:
                raise
            fixed = repair_xml(raw, kind="user_prompt", cfg=cfg, label=f"{label}::user_prompt", template_repair=template_repair)
            return parse_user_prompt_xml(fixed)

    def chunk_attempt(chunk_text_: str, idx: int) -> Tuple[str, str]:
        prompt = CHUNK_PROMPT_TEMPLATE.format(xiaoshuo=chunk_text_)
        payload = build_payload(prompt, cfg, template=template_format)
        raw = sse_chat_completion(payload, cfg)
        if PRINT_MODEL_OUTPUT:
            _print_model_output(f"{label}::chunk{idx}", raw)
        try:
            return parse_chunk_xml(raw)
        except ValueError:
            if not XML_REPAIR:
                raise
            fixed = repair_xml(raw, kind="chunk", cfg=cfg, label=f"{label}::chunk{idx}", template_repair=template_repair)
            return parse_chunk_xml(fixed)

    def final_summary_attempt(summaries_text: str) -> str:
        prompt = FINAL_FROM_SUMMARIES_PROMPT_TEMPLATE.format(summaries=summaries_text)
        payload = build_payload(prompt, cfg, template=template_summary)
        raw = sse_chat_completion(payload, cfg)
        if PRINT_MODEL_OUTPUT:
            _print_model_output(f"{label}::final_summary", raw)
        try:
            return parse_summary_xml(raw)
        except ValueError:
            if not XML_REPAIR:
                raise
            fixed = repair_xml(raw, kind="final", cfg=cfg, label=f"{label}::final_summary", template_repair=template_repair)
            return parse_summary_xml(fixed)

    def final_both_fallback_attempt(summaries_text: str) -> Tuple[str, str]:
        prompt = FINAL_BOTH_FROM_SUMMARIES_PROMPT_TEMPLATE.format(summaries=summaries_text)
        payload = build_payload(prompt, cfg, template=template_repair)
        raw = sse_chat_completion(payload, cfg)
        if PRINT_MODEL_OUTPUT:
            _print_model_output(f"{label}::final_both_fallback", raw)
        try:
            return parse_final_both_xml(raw)
        except ValueError:
            if not XML_REPAIR:
                raise
            fixed = repair_xml(raw, kind="final_both", cfg=cfg, label=f"{label}::final_both_fallback", template_repair=template_repair)
            return parse_final_both_xml(fixed)

    # 非分片：format -> summary -> user_prompt
    if len(text) <= max_input_chars:
        formatted = call_with_retries(lambda: format_attempt(text), max_retries=max_retries)
        formatted = postprocess_formatted_text(formatted)
        time.sleep(cfg.min_delay)

        summary = call_with_retries(lambda: summary_from_formatted_attempt(formatted), max_retries=max_retries)
        summary = postprocess_short_text(summary)
        time.sleep(cfg.min_delay)

        user_prompt = call_with_retries(lambda: user_prompt_from_summary_attempt(summary), max_retries=max_retries)
        user_prompt = postprocess_short_text(user_prompt)
        time.sleep(cfg.min_delay)

        return ProcessTextResult(formatted_parts=[formatted], summary=summary, chunk_summaries=[], user_prompt=user_prompt)

    # 分片：chunk(format+chunk_summary) -> final_summary -> user_prompt
    chunks = chunk_text(text, max_chars=max_input_chars, lookback=chunk_lookback)
    formatted_parts: List[str] = []
    chunk_summaries: List[str] = []

    for i, ch in enumerate(chunks, start=1):
        fpart, s = call_with_retries(lambda: chunk_attempt(ch, i), max_retries=max_retries)
        formatted_parts.append(postprocess_formatted_text(fpart))
        chunk_summaries.append(postprocess_short_text(s))
        time.sleep(cfg.min_delay)

    summaries_text = "\n".join([f"[段落{i}] {s}" for i, s in enumerate(chunk_summaries, start=1)])

    try:
        summary = call_with_retries(lambda: final_summary_attempt(summaries_text), max_retries=max_retries)
        summary = postprocess_short_text(summary)
        time.sleep(cfg.min_delay)

        user_prompt = call_with_retries(lambda: user_prompt_from_summary_attempt(summary), max_retries=max_retries)
        user_prompt = postprocess_short_text(user_prompt)
        time.sleep(cfg.min_delay)

        return ProcessTextResult(formatted_parts=formatted_parts, summary=summary, chunk_summaries=chunk_summaries, user_prompt=user_prompt)
    except Exception:
        summary2, up2 = call_with_retries(lambda: final_both_fallback_attempt(summaries_text), max_retries=max_retries)
        summary2 = postprocess_short_text(summary2)
        up2 = postprocess_short_text(up2)
        time.sleep(cfg.min_delay)
        return ProcessTextResult(formatted_parts=formatted_parts, summary=summary2, chunk_summaries=chunk_summaries, user_prompt=up2)


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
        r = process_text(
            text,
            cfg=cfg,
            template_format=args.template_format,
            template_summary=args.template_summary,
            template_creative=args.template_creative,
            template_repair=args.template_repair,
            max_input_chars=args.max_input_chars,
            chunk_lookback=args.chunk_lookback,
            label=fid,
            max_retries=args.max_retries,
        )

        assistant_content = assemble_assistant_content(
            formatted_parts=r.formatted_parts,
            summary=r.summary,
            chunk_summaries=r.chunk_summaries,
        )

        record = {
            "messages": [
                {"role": "user", "content": r.user_prompt.strip()},
                {"role": "assistant", "content": assistant_content.strip()},
            ],
            "summary": (r.summary or "").strip(),
            "chunk_summaries": [s.strip() for s in (r.chunk_summaries or [])],
            "source_text": text,
        }

        meta = {
            "status": "ok",
            "len_in": len(text),
            "encoding": enc,
            "num_chunks": len(r.formatted_parts),
            "len_user_prompt": len(r.user_prompt or ""),
            "len_summary": len(r.summary or ""),
            "len_assistant": len(assistant_content or ""),
            "has_summary": bool((r.summary or "").strip()),
            "has_chunk_summaries": bool(r.chunk_summaries),
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

    # template：拆分为不同步骤使用（避免总是logical）
    ap.add_argument("--template", default="logical", help="兼容参数：等同于 --template-format（默认: logical）")
    ap.add_argument("--template-format", default=None, help="格式清洗用template（默认: logical 或由 --template 指定）")
    ap.add_argument("--template-summary", default="summary", help="总结/合并综述用template（默认: summary）")
    ap.add_argument("--template-creative", default="creative", help="生成用户提示词用template（默认: creative）")
    ap.add_argument("--template-repair", default="logical", help="XML修复追问用template（默认: logical）")

    ap.add_argument("--timeout", type=int, default=180, help="单次请求超时秒数")
    ap.add_argument("--min-delay", type=float, default=0.6, help="每个worker内请求间最小延迟秒数")
    ap.add_argument("--max-retries", type=int, default=4, help="单次API调用失败重试次数（指数退避）。分片模式下每个chunk与final都单独使用该值。")
    ap.add_argument("--max-input-chars", type=int, default=12000, help="单次喂给模型的最大字符数，超出则分块")

    ap.add_argument("--chunk-lookback", type=int, default=100, help="分片时在 max-input-chars 附近向前寻找自然切分点的范围（优先换行，其次句号等；默认100）")

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

    # 兼容：--template 作为 format template
    if args.template_format is None:
        args.template_format = args.template

    global PRINT_MODEL_OUTPUT, XML_REPAIR
    PRINT_MODEL_OUTPUT = (not getattr(args, "no_print_model_output", False))
    XML_REPAIR = (not getattr(args, "no_xml_repair", False))

    cfg = ChatConfig(model=args.model, timeout=args.timeout, min_delay=args.min_delay)

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
    print(
        f"pending={len(pending)} workers={workers} "
        f"templates=format:{args.template_format} summary:{args.template_summary} creative:{args.template_creative} repair:{args.template_repair} "
        f"chunk_lookback={args.chunk_lookback} "
        f"print_model_output={PRINT_MODEL_OUTPUT} xml_repair={XML_REPAIR}"
    )

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
                    print(f"[ok]   {label} | in={meta.get('len_in')} enc={meta.get('encoding')} chunks={meta.get('num_chunks')} out+=1")
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
