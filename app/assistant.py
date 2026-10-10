"""AI 助教: questions about one class, answered from its transcript and summary (optionally the web)."""
import json
from pathlib import Path

from cloud import CloudError, claude_stream_chat

MAX_TURNS = 10          # earlier turns are dropped from what is sent (still kept on disk)
MAX_TRANSCRIPT = 120000  # characters (~30k tokens); longer classes keep the most recent part

INSTRUCTIONS = (
    "你是这门课的助教，学生（母语中文）正在复习或旁听一节英文课。下面附有本节课的英文语音转写（可能有识别错误）和课堂总结。\n"
    "回答要求：\n"
    "1. 用简体中文回答，清楚、准确、不啰嗦；专业术语给出中文译名并括注英文。\n"
    "2. 优先依据本节课内容回答；引用课堂内容时标出时间点，格式严格为 [mm:ss]，例如 [12:05]，学生点一下就能跳回原文。\n"
    "3. 课堂里没讲到的，可以用你的知识补充，但要说明这是补充内容；不确定的要直说。\n"
    "4. 学生选中一段文字来问时，先解释它在这节课里的意思，再视需要展开（定义、例子、和其他概念的关系）。\n"
    "5. 如果转写明显识别错了某个词，指出你认为老师实际说的是什么。"
)


def _fmt_t(sec):
    sec = int(sec or 0)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def course_context(session) -> str:
    lines = [f"[{_fmt_t(s['t'])}] {s['en']}" for s in session.segments]
    text = "\n".join(lines)
    if len(text) > MAX_TRANSCRIPT:
        text = "（前面部分过长已省略）\n" + text[-MAX_TRANSCRIPT:]
    out = f"课程：{session.title}\n\n<课堂转写>\n{text}\n</课堂转写>"
    if session.summary:
        out += f"\n\n<课堂总结>\n{session.summary}\n</课堂总结>"
    return out


def chat_path(session) -> Path:
    return Path(session.folder) / "chat.json"


def load_chat(session):
    try:
        return json.loads(chat_path(session).read_text(encoding="utf-8"))
    except Exception:
        return []


def save_chat(session, history):
    p = chat_path(session)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def user_text(q: str, quote: str = "", quote_t=None) -> str:
    if not quote:
        return q
    where = f"（原文 [{_fmt_t(quote_t)}] 附近）" if quote_t is not None else ""
    return f"我选中的内容{where}：\n「{quote}」\n\n{q}"


EXPLAIN = "请解释这段内容在这节课里是什么意思。如果是专业术语，给出中文译名、简明定义和一个例子。"


def ask(api_key, model, session, history, question, quote="", quote_t=None, web=False,
        on_text=None, on_event=None, usage=None):
    """Streams the answer; returns (answer, sources, web_used)."""
    system = [
        {"type": "text", "text": INSTRUCTIONS},
        # the long, unchanging part is cached: follow-up questions cost ~1/10 for it
        {"type": "text", "text": course_context(session), "cache_control": {"type": "ephemeral"}},
    ]
    msgs = []
    for h in history[-MAX_TURNS * 2:]:
        content = user_text(h["q"], h.get("quote", ""), h.get("quote_t")) if h["role"] == "user" else h["text"]
        msgs.append({"role": h["role"], "content": content})
    if msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]
    msgs.append({"role": "user", "content": user_text(question, quote, quote_t)})

    sources, seen = [], set()

    def ev(kind, data):
        if kind in ("results", "citation"):
            for x in (data if isinstance(data, list) else [data]):
                if x["url"] not in seen:
                    seen.add(x["url"])
                    sources.append(x)
        if on_event:
            on_event(kind, data)

    def run(tools):
        answer = ""
        for piece in claude_stream_chat(api_key, model, system, msgs, tools=tools, max_tokens=1500,
                                        timeout=90, usage=usage, on_event=ev):
            answer += piece
            if on_text:
                on_text(answer)
        return answer

    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}] if web else None
    try:
        return run(tools), sources, bool(web)
    except CloudError as e:
        # web search switched off for the account, or not offered for this model: answer without it
        if web and getattr(e, "status", None) == 400:
            if on_event:
                on_event("web_unavailable", str(e))
            return run(None), sources, False
        raise
