"""Notes: one AI conversation ("thread") per selected passage, kept per class in notes.json.

thread = {
  id, created, saved (kept as a note?), key (one-line key point), title,
  anchor: {where: "transcript"|"summary"|"answer"|"none", text, seg_id, t},
  turns: [{q, a, sources, usd, web, keep}],
}
"""
import json
import re
import threading
import time
import uuid
from pathlib import Path

_lock = threading.Lock()


def _path(folder) -> Path:
    return Path(folder) / "notes.json"


def load(folder):
    p = _path(folder)
    with _lock:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return []
        return _migrate_chat(folder)


def _write(folder, threads):
    p = _path(folder)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(threads, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)
    _export_md(folder, threads)


def _migrate_chat(folder):
    """v0.4 kept one conversation per class in chat.json; turn it into a general thread."""
    old = Path(folder) / "chat.json"
    if not old.exists():
        return []
    try:
        hist = json.loads(old.read_text(encoding="utf-8"))
    except Exception:
        return []
    turns = []
    for i in range(0, len(hist) - 1, 2):
        u, a = hist[i], hist[i + 1]
        if u.get("role") == "user" and a.get("role") == "assistant":
            q = u.get("q", "")
            if u.get("quote"):
                q = f"「{u['quote']}」 {q}"
            turns.append({"q": q, "a": a.get("text", ""), "sources": a.get("sources", []),
                          "usd": a.get("usd"), "web": u.get("web", False), "keep": True})
    if not turns:
        return []
    threads = [new_thread({"where": "none", "text": ""}, turns=turns)]
    _write(folder, threads)
    return threads


def new_thread(anchor, turns=None):
    anchor = dict(anchor or {})
    anchor.setdefault("where", "none")
    anchor["text"] = (anchor.get("text") or "")[:1200]
    title = anchor["text"][:40] or "整节课提问"
    return {"id": uuid.uuid4().hex[:10], "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "saved": False, "key": "", "title": title, "anchor": anchor, "turns": turns or []}


def mutate(folder, fn):
    """Load, apply fn(threads) (which may return a value), save atomically."""
    with _lock:
        p = _path(folder)
        try:
            threads = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        except Exception:
            threads = []
        out = fn(threads)
        _write(folder, threads)
        return out


def find(threads, tid):
    return next((t for t in threads if t["id"] == tid), None)


def plain(md: str) -> str:
    s = re.sub(r"\[(\d{1,2}:\d{2})\]", r"\1", md or "")
    s = re.sub(r"[#*`>_]+", "", s)
    return " ".join(s.split())


def fallback_key(thread) -> str:
    for t in thread["turns"]:
        if t.get("keep", True) and t.get("a"):
            first = re.split(r"(?<=[。！？.!?])", plain(t["a"]))[0]
            return first[:60]
    return ""


KEY_PROMPT = ("下面是学生针对课堂上一段内容的提问和 AI 的回答。用一句简体中文（不超过 30 个字）写出最核心的要点，"
              "像笔记里的批注一样精炼，不要写“本笔记”“学生问”之类的话，只输出这一句。")


def key_input(thread) -> str:
    parts = []
    if thread["anchor"].get("text"):
        parts.append(f"原文：{thread['anchor']['text']}")
    for t in thread["turns"]:
        if t.get("keep", True):
            parts.append(f"问：{t['q']}\n答：{plain(t['a'])[:1500]}")
    return "\n\n".join(parts)


def _export_md(folder, threads):
    saved = [t for t in threads if t.get("saved")]
    p = Path(folder) / "笔记.md"
    if not saved:
        if p.exists():
            p.unlink()
        return
    out = ["# 课堂笔记", ""]
    for t in saved:
        a = t["anchor"]
        head = a.get("text") or "整节课提问"
        where = {"transcript": "原文", "summary": "总结"}.get(a.get("where"), "")
        stamp = ""
        if a.get("t") is not None:
            sec = int(a["t"])
            stamp = f" [{sec // 60:02d}:{sec % 60:02d}]"
        out += [f"## {head[:80]}{stamp}", ""]
        if where:
            out += [f"*出处：{where}*", ""]
        if t.get("key"):
            out += [f"**要点：{t['key']}**", ""]
        for turn in t["turns"]:
            if not turn.get("keep", True):
                continue
            out += [f"**问：** {turn['q']}", "", turn.get("a", ""), ""]
            if turn.get("sources"):
                out += ["来源：" + "；".join(f"[{s.get('title') or s['url']}]({s['url']})" for s in turn["sources"]), ""]
        out += ["---", ""]
    p.write_text("\n".join(out), encoding="utf-8")
