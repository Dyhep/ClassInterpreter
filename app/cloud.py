"""Minimal Claude API client (no extra dependencies)."""
import json
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"


class CloudError(Exception):
    pass


def claude_chat(api_key: str, model: str, system: str, user: str, max_tokens=1024, timeout=60) -> str:
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
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def claude_stream(api_key: str, model: str, system: str, user: str, max_tokens=600, timeout=30):
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
        yield from _sse(r)
    except CloudError:
        raise
    except Exception as e:
        raise CloudError(f"连接 Claude 中断：{e}") from e


def _sse(r):
    with r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            if ev.get("type") == "content_block_delta":
                t = ev.get("delta", {}).get("text")
                if t:
                    yield t
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
