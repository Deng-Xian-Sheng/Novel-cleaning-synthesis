# gradio_llama_segmented_infer.py
# 依赖：
#   pip install gradio requests
#
# 启动：
#   python gradio_llama_segmented_infer.py --base-url http://localhost:8080/v1 --model any-model
#
# llama-server 端点说明：
#   OpenAI兼容 ChatCompletions: POST {base_url}/chat/completions  (base_url 通常是 http://host:port/v1)
# 参考：llama.cpp server README 与示例 :contentReference[oaicite:4]{index=4}

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import requests
import gradio as gr


# ---------- 文本解析/构造（与分段数据集保持一致） ----------

_CN_NUM = "零一二三四五六七八九十"


def to_cn_ordinal(n: int) -> str:
    if n <= 0:
        return f"第{n}"
    if n < 10:
        return f"第{_CN_NUM[n]}"
    if n == 10:
        return "第十"
    if n < 20:
        return f"第十{_CN_NUM[n-10]}"
    tens = n // 10
    ones = n % 10
    if ones == 0:
        return f"第{_CN_NUM[tens]}十"
    return f"第{_CN_NUM[tens]}十{_CN_NUM[ones]}"


def quote_block(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return f"> **{title}**"
    lines = [f"> **{title}**", ">",]
    for ln in body.splitlines():
        ln = ln.rstrip()
        if not ln.strip():
            lines.append(">")
        else:
            lines.append("> " + ln)
    return "\n".join(lines).strip()


def extract_quote_block_body(text: str, title: str) -> Optional[str]:
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
        body_lines.append(ln[1:].lstrip())
        i += 1
    body = "\n".join(body_lines).strip()
    body = re.sub(r"^\s*\n+", "", body).strip()
    return body if body else ""


def remove_all_quote_blocks(text: str, title: str) -> str:
    """
    移除所有以 > **title** 开头的引用块（直到遇到首个非 '>' 行）。
    """
    lines = text.splitlines()
    out: List[str] = []
    pat = re.compile(rf"^\s*>\s*\*\*{re.escape(title)}\*\*\s*$")

    i = 0
    while i < len(lines):
        if pat.match(lines[i]):
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                i += 1
            # 顺便吃掉紧随其后的空行
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).strip()


def build_initial_user_prompt(user_prompt: str, end_marker: str) -> str:
    rule = (
        f"\n\n写作规则：先给出一段 Markdown 引用块 `> **综述**`；"
        f"若内容较长可分段输出，每次只写一段正文并在文末给出 `> **前文提要**`（概括这一段）。"
        f"只有当全文结束时，最后另起一行写“{end_marker}”。"
    )
    return (user_prompt or "").strip() + rule


def build_continue_user_prompt(
    summary: str,
    prev_summaries: List[Tuple[int, str]],
    seg_idx: int,
    *,
    end_marker: str,
    max_context_chars: int,
) -> str:
    plain = (summary or "") + "\n".join([s for _, s in prev_summaries])
    if len(plain) > max_context_chars and prev_summaries:
        prev_summaries = [prev_summaries[-1]]

    header = (
        f"继续写第{seg_idx}段。只写这一段正文，并在文末给出 `> **前文提要**`。"
        f"只有当全文结束时，最后另起一行写“{end_marker}”。\n\n"
    )
    blocks = [quote_block("综述", summary or "")]
    for idx, s in prev_summaries:
        blocks.append(quote_block(f"{to_cn_ordinal(idx)}段", s))
    return header + "\n\n".join(blocks)


# ---------- llama-server（OpenAI兼容）调用 ----------

@dataclass
class LlamaConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "any-model"
    timeout: int = 300
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 1024


def iter_chat_completions_stream(
    cfg: LlamaConfig,
    messages: List[Dict[str, str]],
) -> Generator[str, None, None]:
    """
    解析 SSE 流：兼容 OpenAI 风格 `data: {...}` 与少数实现的“裸 JSON 行”。
    """
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "stream": True,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
    }

    with requests.post(url, json=payload, stream=True, timeout=cfg.timeout) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            line = raw_line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
            else:
                data = line

            try:
                obj = json.loads(data)
            except Exception:
                continue

            choices = obj.get("choices") or []
            for ch in choices:
                delta = ch.get("delta")
                if isinstance(delta, dict):
                    piece = delta.get("content")
                    if piece:
                        yield piece
                    continue

                msg = ch.get("message")
                if isinstance(msg, dict):
                    piece = msg.get("content")
                    if piece:
                        yield piece


def chat_once_stream(
    cfg: LlamaConfig,
    system_prompt: str,
    user_prompt: str,
) -> Generator[str, None, None]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    yield from iter_chat_completions_stream(cfg, messages)


# ---------- Gradio 推理工程（自动分段续写） ----------

DEFAULT_SYSTEM_PROMPT = (
    "你是一个写作模型。输出可以使用 Markdown 引用块。"
    "请先给出 `> **综述**`。"
    "若内容较长：每次只输出一段正文，并在文末给出 `> **前文提要**`（概括这一段）。"
    "不要在前文提要里写段落编号。"
    "只有当全文结束时，最后另起一行写结束标记。"
)


def generate_with_auto_continue(
    user_prompt: str,
    base_url: str,
    model: str,
    system_prompt: str,
    end_marker: str,
    max_context_chars: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_segments: int,
    show_end_marker: bool,
    show_internal: bool,
) -> Generator[Tuple[str, str], None, None]:
    """
    产出 (visible_text, debug_text)
    """
    cfg = LlamaConfig(
        base_url=base_url,
        model=model,
        timeout=300,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    seg_idx = 1
    summary: Optional[str] = None
    prev_summaries: List[Tuple[int, str]] = []

    visible_all = ""
    debug_all = ""

    def render_visible(raw_segment: str, is_first: bool) -> str:
        # 默认对用户隐藏“前文提要”，让观感更像“一口气生成”
        text = raw_segment.strip()
        text = remove_all_quote_blocks(text, "前文提要")
        if not show_end_marker:
            text = text.replace(end_marker, "").rstrip()
        # 第一段保留综述；后续段通常没有综述块
        return text.strip()

    # 段1 prompt
    cur_user = build_initial_user_prompt(user_prompt, end_marker)

    while seg_idx <= max_segments:
        raw_seg = ""
        # streaming
        last_yield_t = 0.0
        for piece in chat_once_stream(cfg, system_prompt, cur_user):
            raw_seg += piece
            now = time.time()
            if now - last_yield_t > 0.12:
                # 频率节流：减少 Gradio 刷新压力
                vis_piece = render_visible(raw_seg, is_first=(seg_idx == 1))
                yield (visible_all + ("\n\n" if visible_all and vis_piece else "") + vis_piece,
                       debug_all + (f"\n\n---\n\n[段{seg_idx} RAW]\n{raw_seg}" if show_internal else ""))
                last_yield_t = now

        # 段结束一次最终刷新
        vis_seg = render_visible(raw_seg, is_first=(seg_idx == 1))
        if vis_seg:
            visible_all = visible_all + ("\n\n" if visible_all else "") + vis_seg

        if show_internal:
            debug_all = debug_all + (f"\n\n---\n\n[段{seg_idx} RAW]\n{raw_seg}")

        # 解析 summary / 本段前文提要
        if summary is None:
            summary = extract_quote_block_body(raw_seg, "综述")

        seg_summary = extract_quote_block_body(raw_seg, "前文提要")

        done = (end_marker in raw_seg)
        if done:
            yield (visible_all.strip(), debug_all.strip())
            return

        # 未结束却拿不到前文提要：无法继续
        if seg_summary is None or not seg_summary.strip():
            note = "\n\n（未检测到“前文提要”，已停止自动续写。）"
            yield (visible_all.strip() + note, debug_all.strip())
            return

        prev_summaries.append((seg_idx, seg_summary.strip()))
        seg_idx += 1

        # 构造下一段 prompt（上下文清空：只喂综述+之前提要）
        cur_user = build_continue_user_prompt(
            summary or "",
            prev_summaries,
            seg_idx=seg_idx,
            end_marker=end_marker,
            max_context_chars=max_context_chars,
        )

    yield (visible_all.strip() + "\n\n（已达到最大段数限制，停止。）", debug_all.strip())


def build_ui():
    with gr.Blocks(title="分段长文生成（llama-server + Gradio）") as demo:
        gr.Markdown("### 分段长文生成（自动续写直到结束标记）")

        with gr.Row():
            user_prompt = gr.Textbox(label="用户需求（原始写作要求）", lines=8, placeholder="例如：写一部长篇……")
        with gr.Row():
            base_url = gr.Textbox(label="llama-server base_url", value="http://localhost:8080/v1")
            model = gr.Textbox(label="model（llama-server 通常不严格使用，但建议填）", value="any-model")

        with gr.Row():
            end_marker = gr.Textbox(label="结束标记", value="（完）")
            max_context_chars = gr.Number(label="综述+提要字符上限（超过则只保留综述+最后一段提要）", value=1000, precision=0)

        with gr.Row():
            temperature = gr.Slider(label="temperature", minimum=0.0, maximum=2.0, value=0.8, step=0.05)
            top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.95, step=0.01)
            max_tokens = gr.Slider(label="max_tokens（每段上限）", minimum=64, maximum=4096, value=1024, step=64)

        with gr.Row():
            max_segments = gr.Slider(label="最大自动续写段数", minimum=1, maximum=50, value=12, step=1)
            show_end_marker = gr.Checkbox(label="显示结束标记给用户", value=False)
            show_internal = gr.Checkbox(label="显示内部 RAW 输出（含前文提要）", value=False)

        system_prompt = gr.Textbox(label="System Prompt（尽量短、明确）", value=DEFAULT_SYSTEM_PROMPT, lines=4)

        btn = gr.Button("开始生成", variant="primary")

        out_visible = gr.Markdown(label="生成结果（默认隐藏前文提要）")
        out_debug = gr.Textbox(label="内部信息（可选）", lines=12)

        def _run(
            _user_prompt, _base_url, _model, _system_prompt, _end_marker,
            _max_context_chars, _temperature, _top_p, _max_tokens, _max_segments,
            _show_end_marker, _show_internal
        ):
            gen = generate_with_auto_continue(
                user_prompt=_user_prompt,
                base_url=_base_url,
                model=_model,
                system_prompt=_system_prompt,
                end_marker=_end_marker,
                max_context_chars=int(_max_context_chars),
                temperature=float(_temperature),
                top_p=float(_top_p),
                max_tokens=int(_max_tokens),
                max_segments=int(_max_segments),
                show_end_marker=bool(_show_end_marker),
                show_internal=bool(_show_internal),
            )
            for vis, dbg in gen:
                yield vis, dbg

        btn.click(
            _run,
            inputs=[
                user_prompt, base_url, model, system_prompt, end_marker,
                max_context_chars, temperature, top_p, max_tokens, max_segments,
                show_end_marker, show_internal
            ],
            outputs=[out_visible, out_debug],
        )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="any-model")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    ui = build_ui()
    # 预填默认值
    ui.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
