"""Fake Anthropic Messages API (streaming + non-streaming) with realistic timing and token counts."""
import json, re, sys, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIRST_TOKEN = 0.6   # s, typical time to first token for a small model
TOK_PER_S = 120

def ntok(s):
    cjk = len(re.findall(r"[　-鿿＀-￯]", s))
    return cjk + max(1, (len(s) - cjk) // 4)

def fake_zh(en):
    n = max(2, int(len(en.split()) * 1.7))
    return ("译" * n)[:n] + "。"

def fake_fix(en):
    """'Proofread' English: put back one missed small word."""
    w = en.split()
    return " ".join(w[:1] + ["the"] + w[1:]) if len(w) >= 3 else en

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    CACHE = set()
    FAIL_WEB = False   # simulate "web search not enabled" (HTTP 400)
    META_REPLY = False # simulate the model talking instead of translating (batch requests)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        system = req.get("system", "")
        cached_tok = cw = cr = 0
        if isinstance(system, list):
            for b in system:
                if b.get("cache_control"):
                    t = ntok(b["text"])
                    key = hash(b["text"])
                    if key in H.CACHE: cr += t
                    else: H.CACHE.add(key); cw += t
                    cached_tok += t
            system = "\n".join(b["text"] for b in system)
        msgs = req["messages"]; user = msgs[-1]["content"]
        history_tok = sum(ntok(m["content"]) for m in msgs[:-1])
        tin = ntok(system) - cached_tok + ntok(user) + history_tok + 8
        tools = req.get("tools") or []
        web = any(t.get("type", "").startswith("web_search") for t in tools)
        if web and H.FAIL_WEB:
            b = json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                            "message": "web search is not enabled"}}).encode()
            self.send_response(400); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        chat = isinstance(req.get("system"), list)
        body = user.split("【请翻译这一句】")[-1] if "【请翻译这一句】" in user else user
        body = body.replace("<原文>", "").replace("</原文>", "").strip()
        lines = [l for l in body.split("\n") if re.match(r"^\d+\.\s", l)]
        if "专业术语表" in system:
            out = ("price elasticity of demand | 需求价格弹性 | How much quantity demanded responds to a price change\n"
                   "total revenue | 总收入 | Price times quantity sold\n"
                   "deadweight loss | 无谓损失 | Lost total surplus from a market distortion")
        elif chat:
            out = f"这是第{len(msgs)//2+1}轮回答：课上在 [00:05] 讲到了这一点。" + ("（参考了网上资料）" if web else "")
        elif lines and H.META_REPLY:
            out = "我准备好接收英文语音转写内容。请提供编号的英文文本，我将逐句译成简体中文。请粘贴需要翻译的内容"
        elif lines:
            out = "\n".join(f"{l.split('.',1)[0]}. {fake_zh(l.split('.',1)[1])}" for l in lines)
            if "英文校对" in system:
                out += "\n英文校对：\n" + "\n".join(f"{l.split('.',1)[0]}. {fake_fix(l.split('.',1)[1].strip())}" for l in lines)
        else:
            out = fake_zh(body.strip())
            if "英文校对" in system:
                out += "\n英文校对：" + fake_fix(body.strip())
        tout = ntok(out)
        if not req.get("stream"):
            time.sleep(FIRST_TOKEN + tout / TOK_PER_S)
            data = {"content": [{"type": "text", "text": out}], "usage": {"input_tokens": tin, "output_tokens": tout}}
            b = json.dumps(data).encode(); self.send_response(200)
            self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(b)))
            self.end_headers(); self.wfile.write(b); return
        self.send_response(200); self.send_header("content-type", "text/event-stream"); self.end_headers()
        def ev(e, d): self.wfile.write(f"event: {e}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n".encode()); self.wfile.flush()
        time.sleep(FIRST_TOKEN)
        ev("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": tin, "output_tokens": 1,
            "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr}}})
        idx = 0
        if web:
            ev("content_block_start", {"type": "content_block_start", "index": idx,
                "content_block": {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {}}})
            ev("content_block_delta", {"type": "content_block_delta", "index": idx,
                "delta": {"type": "input_json_delta", "partial_json": '{"query": "price elas'}})
            ev("content_block_delta", {"type": "content_block_delta", "index": idx,
                "delta": {"type": "input_json_delta", "partial_json": 'ticity definition"}'}})
            ev("content_block_stop", {"type": "content_block_stop", "index": idx}); idx += 1
            ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {
                "type": "web_search_tool_result", "tool_use_id": "srv_1", "content": [
                    {"type": "web_search_result", "title": "Price elasticity of demand", "url": "https://en.wikipedia.org/wiki/Price_elasticity_of_demand", "encrypted_content": "x"},
                    {"type": "web_search_result", "title": "Elasticity - Khan Academy", "url": "https://www.khanacademy.org/economics", "encrypted_content": "y"}]}})
            ev("content_block_stop", {"type": "content_block_stop", "index": idx}); idx += 1
        ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
        if web:
            ev("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "citations_delta",
                "citation": {"type": "web_search_result_location", "url": "https://en.wikipedia.org/wiki/Price_elasticity_of_demand", "title": "Price elasticity of demand", "cited_text": "..."}}})
        for i in range(0, len(out), 6):
            ev("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": out[i:i+6]}})
            time.sleep(min(6, len(out) - i) / TOK_PER_S)
        ev("content_block_stop", {"type": "content_block_stop", "index": idx})
        u = {"output_tokens": tout}
        if web: u["server_tool_use"] = {"web_search_requests": 1}
        ev("message_delta", {"type": "message_delta", "usage": u})
        ev("message_stop", {"type": "message_stop"})

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
