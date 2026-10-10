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

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        user = req["messages"][0]["content"]; system = req.get("system", "")
        tin = ntok(system) + ntok(user) + 8
        body = user.split("【请翻译这一句】")[-1] if "【请翻译这一句】" in user else user
        lines = [l for l in body.split("\n") if re.match(r"^\d+\.\s", l)]
        if lines:
            out = "\n".join(f"{l.split('.',1)[0]}. {fake_zh(l.split('.',1)[1])}" for l in lines)
        else:
            out = fake_zh(body.strip())
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
        ev("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": tin, "output_tokens": 1}}})
        for i in range(0, len(out), 6):
            ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": out[i:i+6]}})
            time.sleep(min(6, len(out) - i) / TOK_PER_S)
        ev("message_delta", {"type": "message_delta", "usage": {"output_tokens": tout}})
        ev("message_stop", {"type": "message_stop"})

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
