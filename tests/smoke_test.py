"""End-to-end check used by the build: audio file -> English text -> Chinese -> (optional) summary.

Env: CI_MODELS_DIR (models folder), LLAMA_SERVER (optional exe), TEST_WAV, SUMMARY=1 to test summary.
"""
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
    e.settings.update({"api_key": ""})
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
    wav = os.environ.get("TEST_WAV")
    if wav:
        print("pipeline…")
        e = check_pipeline(wav)
        if os.environ.get("SUMMARY") == "1":
            check_summary(e)
    print("SMOKE TEST PASSED")
