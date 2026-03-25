import os
import re
import json
import time
import uuid
import random
import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except Exception:
    repair_json = None
    HAS_JSON_REPAIR = False


# =========================
# Config
# =========================

UPSTREAM_BASE = os.getenv("DPHN_BASE", "https://chat.dphn.ai").rstrip("/")
DEFAULT_TEMPLATE = os.getenv("DPHN_TEMPLATE", "creative")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# 逗号分隔的本地代理 key；为空则不校验
# 例如：PROXY_API_KEYS=sk-local-1,sk-local-2
PROXY_API_KEYS = {x.strip() for x in os.getenv("PROXY_API_KEYS", "").split(",") if x.strip()}

# 重试配置
UPSTREAM_RETRIES = int(os.getenv("UPSTREAM_RETRIES", "3"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "0.8"))
RETRY_BACKOFF_MAX = float(os.getenv("RETRY_BACKOFF_MAX", "5.0"))

# 超时配置
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "15"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "600"))
WRITE_TIMEOUT = float(os.getenv("WRITE_TIMEOUT", "60"))
POOL_TIMEOUT = float(os.getenv("POOL_TIMEOUT", "15"))

# 连接池
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "100"))
MAX_KEEPALIVE = int(os.getenv("MAX_KEEPALIVE", "20"))

BROWSER_UA = os.getenv(
    "DPHN_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

COMMON_HEADERS = {
    "origin": "https://chat.dphn.ai",
    "referer": "https://chat.dphn.ai/",
    "user-agent": BROWSER_UA,
}

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("openai-proxy")


# =========================
# Lifespan / App
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=READ_TIMEOUT,
        write=WRITE_TIMEOUT,
        pool=POOL_TIMEOUT,
    )
    limits = httpx.Limits(
        max_connections=MAX_CONNECTIONS,
        max_keepalive_connections=MAX_KEEPALIVE,
    )
    app.state.client = httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=True,
        http2=False,
    )

    if not HAS_JSON_REPAIR:
        logger.warning("json-repair 未安装，tool call JSON 修复能力不可用。建议安装: pip install json-repair")

    logger.info(
        "proxy started: upstream=%s default_template=%s retries=%s",
        UPSTREAM_BASE,
        DEFAULT_TEMPLATE,
        UPSTREAM_RETRIES,
    )
    try:
        yield
    finally:
        await app.state.client.aclose()
        logger.info("proxy stopped")


app = FastAPI(title="OpenAI-Compatible Proxy for dphn.ai", lifespan=lifespan)


# =========================
# Exceptions
# =========================

class UpstreamRetryableError(Exception):
    pass


class UpstreamNonRetryableError(Exception):
    pass


# =========================
# Utility
# =========================

def now_ts() -> int:
    return int(time.time())


def new_chat_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def sse_data(data: Any) -> str:
    if isinstance(data, str):
        return f"data: {data}\n\n"
    return f"data: {json_dumps(data)}\n\n"


def get_request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or uuid.uuid4().hex[:12]


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except Exception:
        return None


def compute_backoff(attempt: int, retry_after: Optional[float] = None) -> float:
    if retry_after is not None:
        return min(RETRY_BACKOFF_MAX, retry_after)
    base = RETRY_BACKOFF_BASE * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0, 0.25)
    return min(RETRY_BACKOFF_MAX, base + jitter)


def check_proxy_auth(request: Request) -> None:
    if not PROXY_API_KEYS:
        return

    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")

    token = auth[7:].strip()
    if token not in PROXY_API_KEYS:
        raise HTTPException(status_code=401, detail="invalid api key")


def upstream_json_headers() -> Dict[str, str]:
    return {
        **COMMON_HEADERS,
        "accept": "application/json",
    }


def upstream_sse_headers() -> Dict[str, str]:
    return {
        **COMMON_HEADERS,
        "accept": "text/event-stream",
        "content-type": "application/json",
        "cache-control": "no-cache",
    }


def split_text(text: str, size: int = 24):
    for i in range(0, len(text), size):
        yield text[i:i + size]


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("type")
                if t == "text":
                    parts.append(item.get("text", ""))
                elif t == "image_url":
                    url = (item.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        parts.append("[image:data-url]")
                    elif url:
                        parts.append(f"[image:{url}]")
                else:
                    parts.append(json_dumps(item))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    return str(content)


def normalize_upstream_content(content: Any) -> Any:
    """
    给上游使用：
    - str => str
    - list[text/image_url] => 原样保留 OpenAI 风格多模态 content
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "text":
                out.append({
                    "type": "text",
                    "text": item.get("text", "")
                })
            elif t == "image_url":
                image_url = item.get("image_url") or {}
                out.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url.get("url", "")
                    }
                })
        return out if out else content_to_text(content)
    return str(content)


def extract_forced_tool_name(tool_choice: Any) -> Optional[str]:
    if not isinstance(tool_choice, dict):
        return None
    fn = tool_choice.get("function") or {}
    if isinstance(fn, dict) and fn.get("name"):
        return fn["name"]
    if tool_choice.get("name"):
        return tool_choice["name"]
    return None


def build_tool_prompt(tools: List[Dict[str, Any]], tool_choice: Any) -> str:
    clean_tools = []
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        fn = tool.get("function") or {}
        clean_tools.append({
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {})
        })

    choice_rule = "如果不需要工具，就正常回答。"
    if tool_choice == "required":
        choice_rule = "你必须调用一个工具，不要直接回答。"
    else:
        forced_name = extract_forced_tool_name(tool_choice)
        if forced_name:
            choice_rule = f"你必须调用工具 `{forced_name}`，不要调用其他工具，也不要直接回答。"

    return (
        "你可以通过代理调用外部工具。\n"
        "当你需要调用工具时，严格遵守以下规则：\n"
        "1. 不要输出解释，不要输出 Markdown，不要输出代码块。\n"
        "2. 只输出严格 JSON。\n"
        "3. JSON 格式必须为：\n"
        '{"tool_calls":[{"name":"tool_name","arguments":{"arg1":"value1"}}]}\n'
        "4. arguments 必须是 JSON 对象。\n"
        "5. 工具名必须从可用工具列表中选择。\n"
        f"6. {choice_rule}\n\n"
        "可用工具列表：\n"
        f"{json.dumps(clean_tools, ensure_ascii=False, indent=2)}\n\n"
        "如果后续消息里出现 tool result，那是工具真实返回结果，请基于它继续完成回答。"
    )


def convert_messages_for_upstream(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Any = None
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    system_parts: List[str] = []

    # 收集所有系统消息内容（包括工具提示）
    if tools:
        system_parts.append(build_tool_prompt(tools, tool_choice))

    first_user_idx = -1
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role == "system":
            content = content_to_text(msg.get("content"))
            if content:
                system_parts.append(content)
        elif role == "user" and first_user_idx == -1:
            first_user_idx = idx

    system_prefix = "\n\n".join(system_parts) if system_parts else ""

    for idx, msg in enumerate(messages):
        role = msg.get("role")

        if role == "system":
            # 系统消息已收集，跳过
            continue

        if role in ("user", "assistant"):
            content = normalize_upstream_content(msg.get("content", ""))

            if role == "assistant" and msg.get("tool_calls"):
                tc_text = json.dumps(msg["tool_calls"], ensure_ascii=False)
                base_text = content_to_text(msg.get("content"))
                combined = []
                if base_text:
                    combined.append(base_text)
                combined.append("[assistant_tool_calls]")
                combined.append(tc_text)
                content = "\n".join(combined)

            # 第一条用户消息前加上系统提示
            if role == "user" and idx == first_user_idx and system_prefix:
                if isinstance(content, str):
                    content = system_prefix + "\n\n" + content
                else:
                    # content 是 list（多模态），在开头插入 text
                    content = [{"type": "text", "text": system_prefix}] + content

            out.append({
                "role": role,
                "content": content
            })

        elif role == "tool":
            tool_result_obj = {
                "tool_call_id": msg.get("tool_call_id"),
                "name": msg.get("name"),
                "content": msg.get("content"),
            }
            out.append({
                "role": "user",
                "content": "[tool_result]\n" + json.dumps(tool_result_obj, ensure_ascii=False)
            })

        else:
            out.append({
                "role": "user",
                "content": content_to_text(msg.get("content"))
            })

    return out


async def aiter_sse_data(response: httpx.Response):
    """
    更稳的 SSE 解析器：
    - 支持多行 data:
    - 以空行作为 event 结束
    """
    data_lines: List[str] = []
    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\r")
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue

    if data_lines:
        yield "\n".join(data_lines)


def extract_first_json_value(text: str) -> Optional[str]:
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not starts:
        return None

    start = min(starts)
    stack: List[str] = []
    in_string = False
    escape = False

    pairs = {"{": "}", "[": "]"}
    closers = {"}", "]"}

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in pairs:
            stack.append(ch)
            continue

        if ch in closers:
            if not stack:
                return None
            opener = stack.pop()
            if pairs[opener] != ch:
                return None
            if not stack:
                return text[start:i + 1]

    return None


def try_json_loads_with_repair(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        if not HAS_JSON_REPAIR:
            raise

    repaired = repair_json(text)
    if isinstance(repaired, (dict, list)):
        return repaired
    return json.loads(repaired)


def normalize_argument_string(arguments: Any) -> str:
    if arguments is None:
        return "{}"

    if isinstance(arguments, (dict, list, int, float, bool)):
        return json.dumps(arguments, ensure_ascii=False)

    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return "{}"

        try:
            parsed = json.loads(stripped)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

        if HAS_JSON_REPAIR:
            try:
                repaired = repair_json(stripped)
                if isinstance(repaired, (dict, list, int, float, bool)):
                    return json.dumps(repaired, ensure_ascii=False)
                parsed = json.loads(repaired)
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                pass

        return stripped

    return json.dumps(arguments, ensure_ascii=False)


def normalize_tool_calls(
    parsed: Any,
    tools: List[Dict[str, Any]],
    forced_tool_name: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    allowed_names = set()
    for tool in tools or []:
        if tool.get("type") == "function":
            fn = tool.get("function") or {}
            name = fn.get("name")
            if name:
                allowed_names.add(name)

    if forced_tool_name:
        allowed_names = {forced_tool_name} if forced_tool_name in allowed_names else set()

    items = None
    if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
        items = parsed["tool_calls"]
    elif isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and parsed.get("name"):
        items = [parsed]
    elif isinstance(parsed, dict) and isinstance(parsed.get("function"), dict):
        items = [parsed]

    if not items:
        return None

    out = []
    for item in items:
        if not isinstance(item, dict):
            continue

        if "function" in item and isinstance(item["function"], dict):
            fn = item["function"]
            name = fn.get("name")
            arguments = fn.get("arguments", {})
        else:
            name = item.get("name")
            arguments = item.get("arguments", {})

        if not name or name not in allowed_names:
            continue

        out.append({
            "id": item.get("id") or ("call_" + uuid.uuid4().hex[:24]),
            "type": "function",
            "function": {
                "name": name,
                "arguments": normalize_argument_string(arguments)
            }
        })

    return out or None


def try_parse_tool_response(
    text: str,
    tools: List[Dict[str, Any]],
    tool_choice: Any = None
) -> Optional[List[Dict[str, Any]]]:
    if not text or not tools:
        return None

    forced_tool_name = extract_forced_tool_name(tool_choice)
    candidates: List[str] = []

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    for m in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I):
        content = m.group(1).strip()
        if content:
            candidates.append(content)

    json_candidate = extract_first_json_value(text)
    if json_candidate:
        candidates.append(json_candidate)

    # 兜底：处理 [assistant_tool_calls] 标记包裹的内容
    # 例如：[assistant_tool_calls]\n    [{"id": "call_xxx", ...}]
    atc_match = re.search(r"\[assistant_tool_calls\]\s*(.+)", text, flags=re.S | re.I)
    if atc_match:
        content = atc_match.group(1).strip()
        if content:
            candidates.append(content)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        try:
            parsed = json.loads(candidate)
        except Exception:
            try:
                parsed = try_json_loads_with_repair(candidate)
            except Exception:
                continue

        normalized = normalize_tool_calls(
            parsed=parsed,
            tools=tools,
            forced_tool_name=forced_tool_name
        )
        if normalized:
            return normalized

    return None


def build_chat_completion_response(
    chat_id: str,
    created: int,
    model: str,
    text: str,
    finish_reason: str = "stop",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if tool_calls:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls
        }
        finish_reason = "tool_calls"
    else:
        message = {
            "role": "assistant",
            "content": text
        }

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


def synthetic_stream_response(
    chat_id: str,
    created: int,
    model: str,
    text: str,
    finish_reason: str = "stop",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
):
    yield sse_data({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }
        ]
    })

    if tool_calls:
        for idx, tc in enumerate(tool_calls):
            yield sse_data({
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": idx,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": tc["function"]["arguments"]
                                    }
                                }
                            ]
                        },
                        "finish_reason": None
                    }
                ]
            })

        yield sse_data({
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls"
                }
            ]
        })
        yield sse_data("[DONE]")
        return

    if text:
        for piece in split_text(text, 24):
            yield sse_data({
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None
                    }
                ]
            })

    yield sse_data({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }
        ]
    })
    yield sse_data("[DONE]")


# =========================
# Upstream HTTP helpers
# =========================

async def request_upstream_json(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    json_body: Optional[Dict[str, Any]] = None,
    request_id: str,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None

    for attempt in range(1, UPSTREAM_RETRIES + 1):
        try:
            logger.debug("[%s] upstream json request attempt=%s %s %s", request_id, attempt, method, url)
            resp = await app.state.client.request(
                method,
                url,
                headers=headers,
                json=json_body,
            )

            if resp.status_code >= 400:
                body_text = resp.text[:2000]
                msg = f"status={resp.status_code} body={body_text}"

                if resp.status_code in RETRYABLE_STATUS and attempt < UPSTREAM_RETRIES:
                    delay = compute_backoff(attempt, parse_retry_after(resp.headers.get("retry-after")))
                    logger.warning("[%s] retryable upstream json error attempt=%s %s", request_id, attempt, msg)
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code in RETRYABLE_STATUS:
                    raise UpstreamRetryableError(msg)
                raise UpstreamNonRetryableError(msg)

            try:
                return resp.json()
            except Exception as e:
                if attempt < UPSTREAM_RETRIES:
                    delay = compute_backoff(attempt)
                    logger.warning("[%s] invalid upstream json attempt=%s err=%s", request_id, attempt, e)
                    await asyncio.sleep(delay)
                    continue
                raise UpstreamRetryableError(f"invalid upstream json: {e}")

        except UpstreamNonRetryableError:
            raise
        except (httpx.TimeoutException, httpx.TransportError, UpstreamRetryableError) as e:
            last_error = e
            if attempt < UPSTREAM_RETRIES:
                delay = compute_backoff(attempt)
                logger.warning("[%s] upstream json retry attempt=%s err=%s delay=%.2f", request_id, attempt, e, delay)
                await asyncio.sleep(delay)
                continue
            break

    if isinstance(last_error, UpstreamNonRetryableError):
        raise last_error
    raise UpstreamRetryableError(str(last_error) if last_error else "upstream json request failed")


class OpenedUpstreamStream:
    def __init__(self, cm, response: httpx.Response, iterator, first_data: str):
        self.cm = cm
        self.response = response
        self.iterator = iterator
        self.first_data = first_data
        self.closed = False

    async def close(self):
        if not self.closed:
            self.closed = True
            try:
                await self.cm.__aexit__(None, None, None)
            except Exception:
                pass


async def open_upstream_chat_stream_with_retry(
    payload: Dict[str, Any],
    *,
    request_id: str
) -> OpenedUpstreamStream:
    url = f"{UPSTREAM_BASE}/api/chat"
    headers = upstream_sse_headers()
    last_error: Optional[Exception] = None

    for attempt in range(1, UPSTREAM_RETRIES + 1):
        cm = app.state.client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
        )
        resp = None
        try:
            logger.debug("[%s] open stream attempt=%s", request_id, attempt)
            resp = await cm.__aenter__()

            if resp.status_code >= 400:
                body = await resp.aread()
                body_text = body.decode("utf-8", errors="ignore")[:2000]
                msg = f"status={resp.status_code} body={body_text}"
                await cm.__aexit__(None, None, None)

                if resp.status_code in RETRYABLE_STATUS and attempt < UPSTREAM_RETRIES:
                    delay = compute_backoff(attempt, parse_retry_after(resp.headers.get("retry-after")))
                    logger.warning("[%s] retryable stream-open error attempt=%s %s", request_id, attempt, msg)
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code in RETRYABLE_STATUS:
                    raise UpstreamRetryableError(msg)
                raise UpstreamNonRetryableError(msg)

            iterator = aiter_sse_data(resp)
            first_data = None
            async for item in iterator:
                first_data = item
                break

            if first_data is None:
                await cm.__aexit__(None, None, None)
                raise UpstreamRetryableError("upstream stream closed without first event")

            return OpenedUpstreamStream(cm, resp, iterator, first_data)

        except UpstreamNonRetryableError:
            raise
        except (httpx.TimeoutException, httpx.TransportError, UpstreamRetryableError) as e:
            last_error = e
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass

            if attempt < UPSTREAM_RETRIES:
                delay = compute_backoff(attempt)
                logger.warning("[%s] stream-open retry attempt=%s err=%s delay=%.2f", request_id, attempt, e, delay)
                await asyncio.sleep(delay)
                continue
            break

    if isinstance(last_error, UpstreamNonRetryableError):
        raise last_error
    raise UpstreamRetryableError(str(last_error) if last_error else "failed to open upstream stream")


async def collect_upstream_text_with_retry(
    payload: Dict[str, Any],
    *,
    request_id: str
) -> Tuple[Dict[str, Any], str, str]:
    """
    非流式 / tools 模式流式使用：
    因为客户端还没收到任何数据，所以这里即便中途断流也可以整体重试。
    """
    url = f"{UPSTREAM_BASE}/api/chat"
    headers = upstream_sse_headers()
    last_error: Optional[Exception] = None

    for attempt in range(1, UPSTREAM_RETRIES + 1):
        meta = {
            "id": new_chat_id(),
            "created": now_ts(),
        }
        parts: List[str] = []
        finish_reason = "stop"
        got_any = False

        try:
            logger.debug("[%s] collect stream attempt=%s", request_id, attempt)
            async with app.state.client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    body_text = body.decode("utf-8", errors="ignore")[:2000]
                    msg = f"status={resp.status_code} body={body_text}"

                    if resp.status_code in RETRYABLE_STATUS and attempt < UPSTREAM_RETRIES:
                        delay = compute_backoff(attempt, parse_retry_after(resp.headers.get("retry-after")))
                        logger.warning("[%s] retryable collect error attempt=%s %s", request_id, attempt, msg)
                        await asyncio.sleep(delay)
                        continue

                    if resp.status_code in RETRYABLE_STATUS:
                        raise UpstreamRetryableError(msg)
                    raise UpstreamNonRetryableError(msg)

                async for data in aiter_sse_data(resp):
                    got_any = True

                    if data == "[DONE]":
                        return meta, "".join(parts), finish_reason

                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue

                    meta["id"] = obj.get("id", meta["id"])
                    meta["created"] = obj.get("created", meta["created"])

                    choices = obj.get("choices") or []
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        parts.append(piece)

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

            if got_any:
                return meta, "".join(parts), finish_reason

            raise UpstreamRetryableError("upstream stream ended without events")

        except UpstreamNonRetryableError:
            raise
        except (httpx.TimeoutException, httpx.TransportError, UpstreamRetryableError) as e:
            last_error = e
            if attempt < UPSTREAM_RETRIES:
                delay = compute_backoff(attempt)
                logger.warning("[%s] collect retry attempt=%s err=%s delay=%.2f", request_id, attempt, e, delay)
                await asyncio.sleep(delay)
                continue
            break

    if isinstance(last_error, UpstreamNonRetryableError):
        raise last_error
    raise UpstreamRetryableError(str(last_error) if last_error else "failed to collect upstream stream")


# =========================
# Routes
# =========================

@app.get("/")
async def root():
    return {
        "name": "OpenAI-Compatible Proxy for dphn.ai",
        "endpoints": ["/health", "/v1/models", "/v1/chat/completions"],
        "upstream": UPSTREAM_BASE,
        "json_repair": HAS_JSON_REPAIR,
    }


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/models")
async def list_models(request: Request):
    check_proxy_auth(request)
    request_id = get_request_id(request)

    try:
        raw = await request_upstream_json(
            "GET",
            f"{UPSTREAM_BASE}/api/models",
            headers=upstream_json_headers(),
            request_id=request_id,
        )
    except UpstreamNonRetryableError as e:
        logger.error("[%s] /v1/models non-retryable upstream error: %s", request_id, e)
        raise HTTPException(status_code=502, detail=f"upstream non-retryable error: {e}")
    except UpstreamRetryableError as e:
        logger.error("[%s] /v1/models retryable upstream exhausted: %s", request_id, e)
        raise HTTPException(status_code=502, detail=f"upstream request failed after retries: {e}")

    items = raw.get("data", [])
    created = now_ts()

    data = []
    for m in items:
        data.append({
            "id": m.get("id"),
            "object": "model",
            "created": created,
            "owned_by": "dphn"
        })

    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    check_proxy_auth(request)
    request_id = get_request_id(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")

    model = body.get("model")
    messages = body.get("messages")
    stream = bool(body.get("stream", False))

    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    tools = body.get("tools") or []
    tool_choice = body.get("tool_choice")

    # 兼容老的 functions/function_call
    legacy_functions = body.get("functions")
    legacy_function_call = body.get("function_call")
    if not tools and isinstance(legacy_functions, list):
        tools = [{"type": "function", "function": fn} for fn in legacy_functions if isinstance(fn, dict)]
        if legacy_function_call == "none":
            tool_choice = "none"
        elif legacy_function_call == "auto" or legacy_function_call is None:
            tool_choice = "auto"
        elif isinstance(legacy_function_call, dict) and legacy_function_call.get("name"):
            tool_choice = {
                "type": "function",
                "function": {"name": legacy_function_call["name"]}
            }

    use_tool_emulation = bool(tools) and tool_choice != "none"

    extra_body = body.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}

    template = body.get("template") or extra_body.get("template") or DEFAULT_TEMPLATE

    upstream_payload = {
        "model": model,
        "template": template,
        "messages": convert_messages_for_upstream(
            messages=messages,
            tools=tools if use_tool_emulation else None,
            tool_choice=tool_choice,
        )
    }

    logger.info(
        "[%s] chat request model=%s stream=%s tools=%s template=%s",
        request_id, model, stream, bool(tools), template
    )

    # 1) 无 tools 且 stream=true：尽量直接透传真实上游流式
    #    这里只能做到“首包前可重试”；一旦开始向客户端发送，若中途上游断流，就无法透明重试。
    if stream and not use_tool_emulation:
        try:
            opened = await open_upstream_chat_stream_with_retry(
                payload=upstream_payload,
                request_id=request_id,
            )
        except UpstreamNonRetryableError as e:
            logger.error("[%s] stream open non-retryable: %s", request_id, e)
            raise HTTPException(status_code=502, detail=f"upstream non-retryable error: {e}")
        except UpstreamRetryableError as e:
            logger.error("[%s] stream open exhausted retries: %s", request_id, e)
            raise HTTPException(status_code=502, detail=f"upstream stream failed after retries: {e}")

        async def passthrough():
            try:
                first = opened.first_data
                if first == "[DONE]":
                    yield sse_data("[DONE]")
                    return
                try:
                    obj = json.loads(first)
                    obj["model"] = model
                    yield sse_data(obj)
                except Exception:
                    yield sse_data(first)

                async for data in opened.iterator:
                    if data == "[DONE]":
                        yield sse_data("[DONE]")
                        return
                    try:
                        obj = json.loads(data)
                        obj["model"] = model
                        yield sse_data(obj)
                    except Exception:
                        yield sse_data(data)

                yield sse_data("[DONE]")

            except Exception as e:
                logger.exception("[%s] passthrough stream interrupted: %s", request_id, e)
            finally:
                await opened.close()

        return StreamingResponse(passthrough(), media_type="text/event-stream")

    # 2) 非流式 或 tools 模式流式：先完整收集，再返回
    try:
        meta, text, finish_reason = await collect_upstream_text_with_retry(
            payload=upstream_payload,
            request_id=request_id,
        )
    except UpstreamNonRetryableError as e:
        logger.error("[%s] collect non-retryable: %s", request_id, e)
        raise HTTPException(status_code=502, detail=f"upstream non-retryable error: {e}")
    except UpstreamRetryableError as e:
        logger.error("[%s] collect exhausted retries: %s", request_id, e)
        raise HTTPException(status_code=502, detail=f"upstream request failed after retries: {e}")

    chat_id = meta.get("id") or new_chat_id()
    created = meta.get("created") or now_ts()

    tool_calls = None
    if use_tool_emulation:
        tool_calls = try_parse_tool_response(
            text=text,
            tools=tools,
            tool_choice=tool_choice,
        )
        if tool_calls:
            logger.info("[%s] tool_calls parsed: %s", request_id, [x["function"]["name"] for x in tool_calls])
        else:
            logger.info("[%s] no tool_calls parsed, fallback to normal assistant text", request_id)

    if stream:
        # tools 模式下，这里是“伪流式”：先完整拿到结果，再合成 OpenAI SSE 返回
        return StreamingResponse(
            synthetic_stream_response(
                chat_id=chat_id,
                created=created,
                model=model,
                text=text,
                finish_reason="tool_calls" if tool_calls else finish_reason,
                tool_calls=tool_calls,
            ),
            media_type="text/event-stream"
        )

    return JSONResponse(
        build_chat_completion_response(
            chat_id=chat_id,
            created=created,
            model=model,
            text=text,
            finish_reason="tool_calls" if tool_calls else finish_reason,
            tool_calls=tool_calls,
        )
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "openai_proxy_prod:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level=LOG_LEVEL.lower(),
    )