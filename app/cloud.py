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


def test_key(api_key: str, model: str) -> str:
    """Returns '' if OK, else an error message."""
    try:
        claude_chat(api_key, model, "Reply with OK.", "ping", max_tokens=5, timeout=20)
        return ""
    except CloudError as e:
        return str(e)
