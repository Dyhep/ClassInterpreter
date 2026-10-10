"""Minimal Claude API client (no extra dependencies)."""
import json
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"

# USD per million tokens (input, output)
PRICES = {"claude-haiku-4-5": (1.0, 5.0)}


SEARCH_USD = 0.01  # per web search


def cost_usd(model: str, tokens_in: int, tokens_out: int, cache_write=0, cache_read=0, searches=0) -> float:
    pin, pout = PRICES.get(model, (1.0, 5.0))
    return (tokens_in * pin + cache_write * pin * 1.25 + cache_read * pin * 0.1
            + tokens_out * pout) / 1e6 + searches * SEARCH_USD


def usage_usd(model: str, u: dict) -> float:
    return cost_usd(model, u.get("in", 0), u.get("out", 0), u.get("cw", 0), u.get("cr", 0), u.get("searches", 0))


class CloudError(Exception):
    pass


def claude_chat(api_key: str, model: str, system: str, user: str, max_tokens=1024, timeout=60,
                usage=None) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        if e.code == 401:
            raise CloudError("API key 无效，请在设置里检查") from e
        if e.code == 429:
            raise CloudError("请求太频繁或余额不足") from e
        raise CloudError(f"Claude API 错误 {e.code}: {detail}") from e
    except Exception as e:
        raise CloudError(f"连接 Claude 失败：{e}") from e
    if usage is not None:
        u = data.get("usage") or {}
        usage["in"] = usage.get("in", 0) + int(u.get("input_tokens") or 0)
        usage["out"] = usage.get("out", 0) + int(u.get("output_tokens") or 0)
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def claude_stream(api_key: str, model: str, system: str, user: str, max_tokens=600, timeout=15,
                  usage=None):
    """Yields text pieces as Claude writes them (low latency for live captions)."""
    body = json.dumps({
        "model": model, "max_tokens": max_tokens, "system": system, "stream": True,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json",
    })
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        if e.code == 401:
            raise CloudError("API key 无效，请在设置里检查") from e
        if e.code in (402, 429):
            raise CloudError("Claude 余额不足或请求太频繁") from e
        raise CloudError(f"Claude API 错误 {e.code}: {detail}") from e
    except Exception as e:
        raise CloudError(f"连接 Claude 失败：{e}") from e
    try:
        yield from _sse(r, usage if usage is not None else {})
    except CloudError:
        raise
    except Exception as e:
        raise CloudError(f"连接 Claude 中断：{e}") from e


def claude_stream_chat(api_key, model, system_blocks, messages, tools=None, max_tokens=1500,
                       timeout=60, usage=None, on_event=None):
    """Multi-turn streaming. system_blocks: list of {"type":"text","text":..., ["cache_control"]}.
    on_event(kind, data): "search" (query) and "results" ([{title,url}])."""
    payload = {"model": model, "max_tokens": max_tokens, "system": system_blocks, "stream": True,
               "messages": messages}
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(API_URL, data=json.dumps(payload).encode("utf-8"), method="POST", headers={
        "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json",
    })
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:400]
        if e.code == 401:
            raise CloudError("API key 无效，请在设置里检查") from e
        if e.code in (402, 429):
            raise CloudError("Claude 余额不足或请求太频繁") from e
        err = CloudError(f"Claude API 错误 {e.code}: {detail}")
        err.status, err.detail = e.code, detail
        raise err from e
    except Exception as e:
        raise CloudError(f"连接 Claude 失败：{e}") from e
    try:
        yield from _sse(r, usage if usage is not None else {}, on_event or (lambda k, d: None))
    except CloudError:
        raise
    except Exception as e:
        raise CloudError(f"连接 Claude 中断：{e}") from e


def _sse(r, usage, on_event=lambda k, d: None):
    query_buf = {}
    with r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            typ = ev.get("type")
            if typ == "message_start":
                u = (ev.get("message") or {}).get("usage") or {}
                usage["in"] = usage.get("in", 0) + int(u.get("input_tokens") or 0)
                usage["cw"] = usage.get("cw", 0) + int(u.get("cache_creation_input_tokens") or 0)
                usage["cr"] = usage.get("cr", 0) + int(u.get("cache_read_input_tokens") or 0)
            elif typ == "message_delta":
                u = ev.get("usage") or {}
                usage["out"] = usage.get("out", 0) + int(u.get("output_tokens") or 0)
                st = u.get("server_tool_use") or {}
                if st.get("web_search_requests") is not None:
                    usage["searches"] = int(st.get("web_search_requests") or 0)
            elif typ == "content_block_start":
                cb = ev.get("content_block") or {}
                if cb.get("type") == "server_tool_use":
                    query_buf[ev.get("index")] = ""
                elif cb.get("type") == "web_search_tool_result":
                    res = cb.get("content")
                    if isinstance(res, list):
                        on_event("results", [{"title": x.get("title", ""), "url": x.get("url", "")}
                                             for x in res if isinstance(x, dict) and x.get("url")])
                    elif isinstance(res, dict) and res.get("error_code"):
                        on_event("search_error", res.get("error_code"))
            elif typ == "content_block_delta":
                d = ev.get("delta", {})
                if d.get("type") == "input_json_delta" and ev.get("index") in query_buf:
                    query_buf[ev.get("index")] += d.get("partial_json", "")
                elif d.get("type") == "citations_delta":
                    c = d.get("citation") or {}
                    if c.get("url"):
                        on_event("citation", {"title": c.get("title", ""), "url": c.get("url")})
                else:
                    t = d.get("text")
                    if t:
                        yield t
            elif typ == "content_block_stop" and ev.get("index") in query_buf:
                try:
                    q = json.loads(query_buf.pop(ev.get("index")) or "{}").get("query", "")
                except ValueError:
                    q = ""
                on_event("search", q)
            elif ev.get("type") == "error":
                raise CloudError("Claude 出错：" + str(ev.get("error", {}).get("message", "")))
            elif ev.get("type") == "message_stop":
                return


def test_key(api_key: str, model: str) -> str:
    """Returns '' if OK, else an error message."""
    try:
        claude_chat(api_key, model, "Reply with OK.", "ping", max_tokens=5, timeout=20)
        return ""
    except CloudError as e:
        return str(e)
