"""End-to-end check used by the build: audio file -> English text -> Chinese -> (optional) summary.

Env: CI_MODELS_DIR (models folder), LLAMA_SERVER (optional exe), TEST_WAV, SUMMARY=1 to test summary.
"""
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
tmp = tempfile.mkdtemp()
os.environ["LOCALAPPDATA"] = tmp
os.environ.setdefault("HOME", tmp)
os.environ["USERPROFILE"] = tmp

import engine  # noqa: E402
from translate import LocalTranslator  # noqa: E402

CJK = re.compile(r"[一-鿿]")


def check_translate():
    tr = LocalTranslator(threads=2)
    for en in ["When the price of a good goes up, the quantity demanded usually goes down.",
               "Remember, the problem set is due next Friday before class.",
               "Photosynthesis converts light energy into chemical energy."]:
        t = time.time()
        zh = tr.translate(en)
        print(f"  [{time.time() - t:.2f}s] {en}\n      -> {zh}")
        assert CJK.search(zh), "translation is not Chinese"
        assert not re.search(r"[繁體說會們個]", zh), "looks traditional"


def check_pipeline(wav):
    e = engine.Engine()
    e.settings.update({"api_key": "", "asr_choice": "moonshine"})  # don't download 630 MB in CI
    e.start("file:" + wav + "?speed=8", "smoke")
    t0 = time.time()
    while not any(ev["kind"] == "stopped" for ev in e.events):  # file sources stop by themselves
        time.sleep(0.3)
        assert time.time() - t0 < 600, "pipeline did not finish"
    segs = e.session.segments
    for s in segs:
        print(f"  {s['t']:6.1f}s  {s['en']}\n          {s['zh']}")
    assert segs, "no speech recognized"
    assert all(CJK.search(s["zh"] or "") for s in segs), "some sentences not translated"
    assert (e.session.folder / "转写与翻译.md").exists()
    return e


def check_cloud_tiers():
    """Both translation tiers against a fake Claude server (no real key needed)."""
    import threading
    from http.server import ThreadingHTTPServer
    sys.path.insert(0, str(ROOT / "tests"))
    import cloud
    import fake_claude
    fake_claude.FIRST_TOKEN = 0.05
    srv = ThreadingHTTPServer(("127.0.0.1", 0), fake_claude.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cloud.API_URL = f"http://127.0.0.1:{srv.server_address[1]}/v1/messages"
    lines = ["Okay.", "Today we talk about elasticity.", "Elasticity measures sensitivity to price.",
             "Right.", "Gasoline demand is inelastic in the short run.", "The midterm is in two weeks."]
    result = {}
    for tier in ("best", "saver"):
        e = engine.Engine()
        e.settings.update({"api_key": "sk-ant-fake", "use_cloud_translate": True, "translate_tier": tier})
        e.session = engine.Session.new(e.settings.sessions_dir(), "test")
        e.recording = True
        for i, l in enumerate(lines):
            e._on_final(l, float(i))
            time.sleep(0.4)
        e._flush_batch()
        e.tr_q.join()
        segs = e.session.segments
        assert all(s["zh"] for s in segs), f"{tier}: untranslated {segs}"
        assert e.usage["requests"] > 0 and e.usage["in"] > 0, f"{tier}: usage not counted"
        result[tier] = dict(e.usage)
        print(f"  {tier}: {e.usage}")
    assert result["saver"]["requests"] < result["best"]["requests"], "saver tier did not batch"
    # the model "talks back" instead of translating -> must be retried, never shown as a translation
    fake_claude.H.META_REPLY = True
    e = engine.Engine()
    e.settings.update({"api_key": "sk-ant-fake", "use_cloud_translate": True, "translate_tier": "saver"})
    e.session = engine.Session.new(e.settings.sessions_dir(), "meta")
    e.recording = True
    e._on_final("Different options for which coding agents to use and how to orchestrate them together.", 0.0)
    e._flush_batch()
    e.tr_q.join()
    fake_claude.H.META_REPLY = False
    zh = e.session.segments[0]["zh"]
    assert "请提供" not in zh and "准备好" not in zh and zh, f"meta reply leaked: {zh}"
    print(f"  meta reply caught and retried -> {zh[:20]}")
    assert result["saver"]["in"] < result["best"]["in"], "saver tier did not reduce input"
    # AI 助教: explain a selection, a follow-up (cached context), web search, and web-search fallback
    e = engine.Engine()
    e.settings.update({"api_key": "sk-ant-fake"})
    s = engine.Session.new(e.settings.sessions_dir(), "tutor")
    s.segments = [{"id": i, "t": i * 5.0, "en": l, "zh": "译"} for i, l in enumerate(lines * 40)]
    s.save()

    # an old v0.4 conversation is migrated into a general thread
    (s.folder / "chat.json").write_text(json.dumps([{"role": "user", "q": "old q"},
                                                   {"role": "assistant", "text": "old a"}]), encoding="utf-8")
    assert len(e.notes_list(str(s.folder))) == 1, "chat.json not migrated"
    folder = str(s.folder)

    def ask(tid, mid, **kw):
        e.ask(folder, tid, mid=mid, **kw)
        t0 = time.time()
        while True:
            for x in e.events:
                if x.get("mid") == mid and x["kind"] in ("chat_done", "chat_error"):
                    return x
            assert time.time() - t0 < 30, "tutor timed out"
            time.sleep(0.05)
    t = e.note_new(folder, {"where": "transcript", "text": "elasticity", "seg_id": 1, "t": 5})
    r1 = ask(t["id"], "a", question="")
    r2 = ask(t["id"], "b", question="follow up")
    r3 = ask(t["id"], "c", question="search", web=True)
    fake_claude.H.FAIL_WEB = True
    r4 = ask(t["id"], "d", question="search again", web=True)
    fake_claude.H.FAIL_WEB = False
    for r in (r1, r2, r3, r4):
        assert r["kind"] == "chat_done", r
    assert r3["sources"], "web search sources missing"
    assert r2["usd"] < r1["usd"], "follow-up should be cheaper (cached transcript)"
    e.note_keep(folder, t["id"], 1, False)
    e.note_update(folder, t["id"], {"saved": True})
    t0 = time.time()
    while not any(x["kind"] == "note_key" for x in e.events):
        assert time.time() - t0 < 20, "key point not generated"
        time.sleep(0.05)
    th = [x for x in e.notes_list(folder) if x["id"] == t["id"]][0]
    assert th["saved"] and th["key"] and len(th["turns"]) == 4 and th["turns"][1]["keep"] is False
    md = (s.folder / "笔记.md").read_text(encoding="utf-8")
    assert "follow up" not in md and "search again" in md, "unkept turn should not be exported"
    gone = e.note_delete(folder, t["id"])
    assert len(e.notes_list(folder)) == 1
    e.note_restore(folder, gone)
    assert len(e.notes_list(folder)) == 2
    print(f"  notes: explain ${r1['usd']}, follow-up ${r2['usd']}, web ${r3['usd']}, key「{th['key']}」")
    srv.shutdown()
    cloud.API_URL = "https://api.anthropic.com/v1/messages"


def check_summary(e):
    print("summary (local)…")
    t = time.time()
    r = e.summarize(str(e.session.folder))
    print(f"  took {time.time() - t:.0f}s")
    print(r.get("summary") or r)
    assert r.get("summary") and CJK.search(r["summary"]), "summary failed"


if __name__ == "__main__":
    print("translate…")
    check_translate()
    print("cloud tiers (fake server)…")
    check_cloud_tiers()
    wav = os.environ.get("TEST_WAV")
    if wav:
        print("pipeline…")
        e = check_pipeline(wav)
        if os.environ.get("SUMMARY") == "1":
            check_summary(e)
    print("SMOKE TEST PASSED")
