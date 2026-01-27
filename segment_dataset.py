# segment_dataset.py
# 用法示例：
#   python segment_dataset.py --input dataset.jsonl --output dataset.segmented.jsonl --max-context-chars 1000 --end-marker "（完）"
#
# 说明：
# - 默认不复制原始 record 里除 messages 以外的字段（避免把 source_text 等巨大字段重复 N 次）。
# - 如果你确实需要保留原字段用于追踪，可加 --keep-extra-fields

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def wrap_zh_text(text: str, width: int = 60) -> str:
    """
    简单中文友好换行：尽量不拆 ASCII 连续串（单词/数字/下划线），其余按字符宽度计数。
    """
    if not text:
        return ""

    token_re = re.compile(r"[A-Za-z0-9_]+|.", re.S)
    tokens = token_re.findall(text)

    lines: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            lines.append("".join(cur).rstrip())
            cur = []
            cur_len = 0

    for tk in tokens:
        if tk == "\n":
            flush()
            continue

        tk_len = len(tk)
        if cur_len + tk_len > width and cur:
            flush()

        cur.append(tk)
        cur_len += tk_len

    flush()
    return "\n".join(lines).strip()


def quote_block(title: str, body: str, width: int = 60) -> str:
    """
    生成：
    > **title**
    >
    > wrapped...
    """
    body = (body or "").strip()
    if not body:
        return f"> **{title}**"

    # 保留段落（空行）结构
    paras = re.split(r"\n\s*\n", body)
    out_lines = [f"> **{title}**", ">",]
    for pi, p in enumerate(paras):
        p = p.strip()
        if not p:
            continue
        wrapped = wrap_zh_text(p, width=width)
        for ln in wrapped.splitlines():
            out_lines.append(f"> {ln}".rstrip())
        if pi != len(paras) - 1:
            out_lines.append(">")
    return "\n".join(out_lines).strip()


def extract_quote_block_body(text: str, title: str) -> Optional[str]:
    """
    从文本中提取第一个以 '> **title**' 开头的引用块内容（去掉 '> ' 前缀），返回 body。
    """
    lines = text.splitlines()
    pat = re.compile(rf"^\s*>\s*\*\*{re.escape(title)}\*\*\s*$")
    start = None
    for i, ln in enumerate(lines):
        if pat.match(ln):
            start = i
            break
    if start is None:
        return None

    body_lines: List[str] = []
    i = start + 1
    while i < len(lines) and lines[i].lstrip().startswith(">"):
        ln = lines[i].lstrip()
        # ln startswith '>'
        content = ln[1:].lstrip()
        # 兼容只有一个 '>' 的空行
        body_lines.append(content)
        i += 1

    body = "\n".join(body_lines).strip()
    # 清理开头可能的空行
    body = re.sub(r"^\s*\n+", "", body)
    return body.strip()


def remove_first_quote_block(text: str, title: str) -> Tuple[str, str]:
    """
    移除第一段指定 title 的引用块，返回 (body, rest_text)。
    若不存在则返回 ("", text)。
    """
    lines = text.splitlines()
    pat = re.compile(rf"^\s*>\s*\*\*{re.escape(title)}\*\*\s*$")
    start = None
    for i, ln in enumerate(lines):
        if pat.match(ln):
            start = i
            break
    if start is None:
        return "", text.strip()

    body_lines: List[str] = []
    i = start + 1
    while i < len(lines) and lines[i].lstrip().startswith(">"):
        ln = lines[i].lstrip()
        content = ln[1:].lstrip()
        body_lines.append(content)
        i += 1

    body = "\n".join(body_lines).strip()
    rest = "\n".join(lines[i:]).lstrip("\n").strip()
    body = re.sub(r"^\s*\n+", "", body).strip()
    return body, rest


def parse_segment_bodies_from_assistant(assistant: str) -> List[str]:
    """
    从你合成脚本的长文 assistant_content 中解析每段“正文”。
    合成脚本长文格式大致为：
      > **综述** ...
      ## 第1段：...
      <正文1>
      > **前文提要** (第1段提要)
      ## 第2段：...
      <正文2>
      ...
    这里会：
    - 去掉每段开头的 '## 第X段...' 行
    - 去掉每段末尾紧跟的 '> **前文提要** ...' 块
    """
    summary_body, rest = remove_first_quote_block(assistant, "综述")

    # 没有 “## 第X段” 就当作短文：rest 即正文
    heading_re = re.compile(r"(?m)^##\s*第(\d+)段.*$")
    matches = list(heading_re.finditer(rest))
    if not matches:
        body = rest.strip()
        return [body] if body else []

    bodies: List[str] = []
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(rest)
        chunk = rest[start:end].strip()

        # 去掉 chunk 内部（通常在末尾）的“前文提要”引用块
        pre_m = re.search(r"(?m)^\s*>\s*\*\*前文提要\*\*\s*$", chunk)
        if pre_m:
            chunk = chunk[:pre_m.start()].rstrip()

        bodies.append(chunk.strip())

    return bodies


_CN_NUM = "零一二三四五六七八九十"


def to_cn_ordinal(n: int) -> str:
    """
    返回“第X”，用于提示词里的段编号。
    - 1~99：尽量输出中文（可选）
    - 100+：回退到阿拉伯数字，避免越界
    """
    if n <= 0:
        return f"第{n}"

    # 100+ 直接用数字，稳定且更短
    if n >= 100:
        return f"第{n}"

    # 1~99 中文（简单实现）
    cn = "零一二三四五六七八九"
    if n < 10:
        return f"第{cn[n]}"
    if n == 10:
        return "第十"
    if n < 20:
        return f"第十{cn[n-10]}"
    tens = n // 10
    ones = n % 10
    if ones == 0:
        return f"第{cn[tens]}十"
    return f"第{cn[tens]}十{cn[ones]}"


def build_initial_user_prompt(orig_user_prompt: str, end_marker: str) -> str:
    rule = (
        f"\n\n写作规则：先给出一段 Markdown 引用块 `> **综述**`；"
        f"若内容较长可分段输出，每次只写一段正文并在文末给出 `> **前文提要**`（概括这一段）。"
        f"只有当全文结束时，最后另起一行写“{end_marker}”。"
    )
    return (orig_user_prompt or "").strip() + rule


def build_continue_user_prompt(
    summary: str,
    prev_summaries: List[Tuple[int, str]],
    seg_idx: int,
    total: int,
    *,
    end_marker: str,
    max_context_chars: int,
    wrap_width: int = 60,
) -> str:
    # 先应用“>1000只保留综述+最后一段提要”的规则（仅对 summary+提要本体计数）
    plain = (summary or "") + "\n".join([s for _, s in prev_summaries])
    if len(plain) > max_context_chars and prev_summaries:
        prev_summaries = [prev_summaries[-1]]

    header = (
        f"继续写第{seg_idx}段（共{total}段）。"
        f"只写这一段正文，并在文末给出 `> **前文提要**`。"
        f"只有当全文结束时，最后另起一行写“{end_marker}”。"
        f"\n\n"
    )

    blocks = [quote_block("综述", summary, width=wrap_width)]
    for idx, s in prev_summaries:
        blocks.append(quote_block(f"{to_cn_ordinal(idx)}段", s, width=wrap_width))

    return header + "\n\n".join(blocks)


def build_assistant_short(summary: str, body: str, end_marker: str, wrap_width: int = 60) -> str:
    parts = [
        quote_block("综述", summary, width=wrap_width),
        (body or "").strip(),
        end_marker.strip(),
    ]
    return "\n\n".join([p for p in parts if p])


def build_assistant_long_seg1(summary: str, body: str, seg_summary: str, wrap_width: int = 60) -> str:
    parts = [
        quote_block("综述", summary, width=wrap_width),
        (body or "").strip(),
        quote_block("前文提要", seg_summary, width=wrap_width),
    ]
    return "\n\n".join([p for p in parts if p])


def build_assistant_long_segN(body: str, seg_summary: str, end_marker: Optional[str], wrap_width: int = 60) -> str:
    parts = [
        (body or "").strip(),
        quote_block("前文提要", seg_summary, width=wrap_width),
    ]
    if end_marker:
        parts.append(end_marker.strip())
    return "\n\n".join([p for p in parts if p])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="原始 dataset.jsonl")
    ap.add_argument("--output", required=True, help="输出分段后的 jsonl")
    ap.add_argument("--max-context-chars", type=int, default=1000, help="综述+提要超过则只保留综述+最后一段提要")
    ap.add_argument("--end-marker", default="（完）", help="结束标记（建议自然一点）")
    ap.add_argument("--wrap-width", type=int, default=60, help="引用块内换行宽度")
    ap.add_argument("--keep-extra-fields", action="store_true", help="是否复制 messages 之外的原字段（会膨胀文件体积）")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output

    total_in = 0
    total_out = 0
    skipped = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            total_in += 1
            try:
                rec = json.loads(line)
            except Exception:
                skipped += 1
                continue

            msgs = rec.get("messages") or []
            if len(msgs) < 2:
                skipped += 1
                continue

            orig_user = (msgs[0].get("content") or "").strip()
            assistant = (msgs[1].get("content") or "").strip()

            # summary：优先从 assistant 的引用块抽取；不行再用 rec["summary"]
            summary = extract_quote_block_body(assistant, "综述")
            if summary is None:
                summary = (rec.get("summary") or "").strip()

            # 长文判断：以 chunk_summaries 为准（你合成脚本会写入这个字段）:contentReference[oaicite:2]{index=2}
            chunk_summaries = rec.get("chunk_summaries") or []
            chunk_summaries = [str(s).strip() for s in chunk_summaries if str(s).strip()]

            if not chunk_summaries:
                # 短文：去掉综述块后 remainder 作为正文
                _, body = remove_first_quote_block(assistant, "综述")
                new_user = build_initial_user_prompt(orig_user, args.end_marker)
                new_asst = build_assistant_short(summary or "", body or "", args.end_marker, wrap_width=args.wrap_width)

                out_rec: Dict[str, Any] = {"messages": [{"role": "user", "content": new_user},
                                                       {"role": "assistant", "content": new_asst}]}
                out_rec["meta"] = {"orig_line": line_no, "segment_index": 1, "segment_total": 1, "mode": "short"}
                if args.keep_extra_fields:
                    for k, v in rec.items():
                        if k != "messages":
                            out_rec.setdefault(k, v)

                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                total_out += 1
                continue

            # 长文：解析每段正文
            bodies = parse_segment_bodies_from_assistant(assistant)
            total = len(chunk_summaries)

            if len(bodies) != total:
                # 尽量不硬崩：如果解析不齐，就把剩余当作 1 段（保守输出）
                # 你可以之后用日志定位具体哪条样本的格式异常
                bodies = bodies[:total] + [""] * max(0, total - len(bodies))

            # 逐段输出训练样本
            prev_list: List[Tuple[int, str]] = []
            for i in range(1, total + 1):
                if i == 1:
                    user_content = build_initial_user_prompt(orig_user, args.end_marker)
                    asst_content = build_assistant_long_seg1(
                        summary or "",
                        bodies[0] if bodies else "",
                        chunk_summaries[0],
                        wrap_width=args.wrap_width,
                    )
                else:
                    user_content = build_continue_user_prompt(
                        summary or "",
                        prev_list,
                        seg_idx=i,
                        total=total,
                        end_marker=args.end_marker,
                        max_context_chars=args.max_context_chars,
                        wrap_width=args.wrap_width,
                    )
                    is_last = (i == total)
                    asst_content = build_assistant_long_segN(
                        bodies[i - 1] if i - 1 < len(bodies) else "",
                        chunk_summaries[i - 1],
                        args.end_marker if is_last else None,
                        wrap_width=args.wrap_width,
                    )

                out_rec = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": asst_content},
                    ],
                    "meta": {"orig_line": line_no, "segment_index": i, "segment_total": total, "mode": "long"},
                }

                if args.keep_extra_fields:
                    for k, v in rec.items():
                        if k != "messages":
                            out_rec.setdefault(k, v)

                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                total_out += 1

                prev_list.append((i, chunk_summaries[i - 1]))

    print(f"Done. in={total_in}, out={total_out}, skipped={skipped}, output={out_path}")


if __name__ == "__main__":
    main()
