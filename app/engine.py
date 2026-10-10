"""Ties capture -> recognition -> translation -> storage -> summary together."""
import json
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from audio import Capture
from cloud import CloudError, cost_usd, usage_usd
from config import Settings
from models import Download, MODELS, is_installed
from summarize import Summarizer


def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip() or "未命名"


BATCH_MAX = 3        # economy tier: send after this many sentences…
BATCH_WORDS = 45     # …or this many words…
BATCH_WAIT = 2.5     # …or this many seconds after the first one, whichever comes first


class Session:
    def __init__(self, folder: Path, title: str, started: str):
        self.folder = folder
        self.title = title
        self.started = started
        self.segments = []   # {id, t, en, zh}
        self.summary = ""
        self.duration = 0.0
        self.usage = {}

    @classmethod
    def new(cls, root: Path, title: str):
        now = datetime.now()
        title = title or now.strftime("%m月%d日 %H:%M 课堂")
        folder = root / f"{now:%Y-%m-%d_%H%M%S}_{_safe(title)}"
        folder.mkdir(parents=True, exist_ok=True)
        return cls(folder, title, now.isoformat(timespec="seconds"))

    @classmethod
    def load(cls, folder: Path):
        d = json.loads((folder / "data.json").read_text(encoding="utf-8"))
        s = cls(folder, d.get("title", folder.name), d.get("started", ""))
        s.segments = d.get("segments", [])
        s.summary = d.get("summary", "")
        s.duration = d.get("duration", 0.0)
        s.usage = d.get("usage", {})
        return s

    def to_dict(self):
        return {"title": self.title, "started": self.started, "duration": self.duration,
                "segments": self.segments, "summary": self.summary, "folder": str(self.folder),
                "usage": self.usage}

    def save(self):
        tmp = self.folder / "data.json.tmp"
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.folder / "data.json")
        lines = [f"# {self.title}", "", f"开始时间：{self.started}", ""]
        for s in self.segments:
            m, sec = divmod(int(s["t"]), 60)
            lines += [f"**[{m:02d}:{sec:02d}]** {s['en']}", "", f"> {s.get('zh') or ''}", ""]
        (self.folder / "转写与翻译.md").write_text("\n".join(lines), encoding="utf-8")
        if self.summary:
            (self.folder / "课堂总结.md").write_text(f"# {self.title} · 课堂总结\n\n{self.summary}\n",
                                                  encoding="utf-8")


class Engine:
    def __init__(self):
        self.settings = Settings()
        self.events = []
        self.seq = 0
        self.ev_lock = threading.Lock()
        self.session = None
        self.capture = None
        self.recognizer = None
        self._rec_name = None
        self.local_tr = None
        self._tr_lock = threading.Lock()
        self.tr_q = queue.Queue()
        self.recording = False
        self.t_start = 0.0
        self.summarizer = Summarizer()
        self.summary_busy = False
        self.downloads = {}
        self._dirty = False
        self.level = 0.0
        self.batch = []            # economy tier: sentences waiting to be sent together
        self.batch_t0 = 0.0
        self.batch_lock = threading.Lock()
        self.usage = {"in": 0, "out": 0, "requests": 0}
        self.usage_lock = threading.Lock()
        self.state = "idle"        # idle | starting | recording | stopping
        self._ctl = threading.Lock()
        self._models = {}
        self._model_lock = threading.Lock()
        threading.Thread(target=self._batch_timer, daemon=True).start()
        threading.Thread(target=self._warmup, daemon=True).start()
        for _ in range(3):  # parallel workers so a slow reply never holds up later sentences
            threading.Thread(target=self._translate_loop, daemon=True).start()
        threading.Thread(target=self._autosave_loop, daemon=True).start()

    # ---------- events for the UI ----------
    def emit(self, kind, **data):
        with self.ev_lock:
            self.seq += 1
            self.events.append({"seq": self.seq, "kind": kind, **data})
            if len(self.events) > 2000:
                self.events = self.events[-1000:]

    def poll(self, after: int):
        with self.ev_lock:
            evs = [e for e in self.events if e["seq"] > after]
        return {"events": evs, "level": round(self.level, 4), "recording": self.recording, "state": self.state,
                "elapsed": (time.time() - self.t_start) if self.recording else 0}

    # ---------- models ----------
    def model_status(self):
        out = {}
        for name in MODELS:
            d = self.downloads.get(name)
            out[name] = {
                "installed": is_installed(name),
                "mb": MODELS[name][2],
                "downloading": bool(d and not d.done),
                "progress": d.progress if d else 0,
                "error": d.error if d else "",
            }
        out["summary"]["engine"] = self.summarizer.llm.available()
        return out

    def download(self, name):
        d = self.downloads.get(name)
        if d and not d.done:
            return
        self.downloads[name] = Download(name)

    # ---------- recording ----------
    # ---------- models are loaded once and kept ----------
    def _asr_name(self):
        name = self.settings.get("asr_model")
        return name if is_installed(name) else "moonshine"

    def _get_asr_model(self, name=None):
        from asr import make_recognizer
        name = name or self._asr_name()
        with self._model_lock:
            if name not in self._models:
                self._models = {name: make_recognizer(name, self.settings.threads())}
            return self._models[name]

    def _warmup(self):
        """Load the speech model (and the offline translator) right after launch, so 开始 is instant."""
        try:
            self._get_asr_model()
            import asr
            asr.make_vad()
        except Exception:
            pass
        try:
            self._local_translator()
        except Exception:
            pass

    # ---------- recording ----------
    def _set_state(self, st):
        self.state = st
        self.recording = st in ("starting", "recording")
        self.emit("state", state=st)

    def start(self, source_id: str, title: str = ""):
        """Returns at once; progress arrives as events. Ignored unless idle."""
        with self._ctl:
            if self.state != "idle":
                return False
            self._set_state("starting")
        try:
            self._start(source_id, title)
            return True
        except Exception as e:
            self.emit("error", text=f"无法开始：{e}")
            self._set_state("idle")
            return False

    def _start(self, source_id, title):
        from asr import Recognizer
        self.session = Session.new(self.settings.sessions_dir(), title)
        session = self.session
        self.summarizer.chunk_notes.clear()
        with self.usage_lock:
            self.usage = {"in": 0, "out": 0, "requests": 0}
        with self.batch_lock:
            self.batch = []
        self.emit("session", session=session.to_dict())
        self.settings.update({"source": source_id})
        loaded = self._asr_name() in self._models
        if not loaded:
            self.emit("status", text="正在加载语音识别模型…（已经在录音，这段时间说的话不会丢）")

        def ready():
            if self.state == "starting" and self.session is session:
                self._set_state("recording")
                self.emit("status", text="正在聆听…")

        rec = Recognizer(self._get_asr_model,
                         lambda t: self._on_partial(t) if self.session is session else None,
                         lambda t, st: self._on_final(t, st, session),
                         live_preview=bool(self.settings.get("live_preview")) and not source_id.startswith("file:"),
                         on_ready=ready, on_error=self._on_capture_error)
        self.recognizer = rec
        self.capture = Capture(source_id, rec.push, on_error=self._on_capture_error,
                               on_level=self._on_level,
                               on_progress=lambda p: self.emit("status", text=f"正在处理录音文件… {p}%"),
                               on_done=lambda: threading.Thread(target=self.stop, daemon=True).start(),
                               busy=lambda: rec.backlog() > 20)
        self.t_start = time.time()
        self.capture.start()
        threading.Thread(target=self._watch_lag, args=(rec, session), daemon=True).start()

    def _watch_lag(self, rec, session):
        """Warn once if the computer can't keep up (recognition falling behind real time)."""
        warned = False
        while self.session is session and self.state in ("starting", "recording"):
            time.sleep(2)
            behind = rec.backlog() / 10  # seconds
            if behind > 8 and not warned and self.state == "recording":
                warned = True
                self.emit("warning", text=f"电脑有点跟不上（落后约 {behind:.0f} 秒），可以在设置里关掉“实时预览”")
            elif behind < 2:
                warned = False

    def _on_level(self, v):
        self.level = v

    def _on_capture_error(self, msg):
        self.emit("error", text=msg)
        threading.Thread(target=self.stop, daemon=True).start()

    def stop(self):
        """Returns at once; finishing the last sentences happens in the background."""
        with self._ctl:
            if self.state not in ("starting", "recording"):
                return False
            self._set_state("stopping")
        self.emit("status", text="正在处理最后几句…")
        cap, rec, session = self.capture, self.recognizer, self.session
        if cap:
            cap.stop()
        self.level = 0.0
        try:
            if rec and not rec.stop(timeout=30):
                self.emit("warning", text="最后几句识别超时，已跳过")
            self.recognizer = None
            self._flush_batch()
            t0 = time.time()
            while self.tr_q.unfinished_tasks and time.time() - t0 < 25:
                time.sleep(0.1)
            if session:
                session.duration = time.time() - self.t_start
                session.save()
            self.summarizer.llm.stop()
        finally:
            self._set_state("idle")
            self.emit("stopped", session=session.to_dict() if session else None)
            self.emit("status", text="已停止，记录已保存")
        return True

    def _on_partial(self, text):
        self.emit("partial", text=text)

    def _on_final(self, text, start, session=None):
        s = self.session
        if s is None or (session is not None and session is not s):
            return  # a finished/abandoned recording must never write into the current one
        seg = {"id": len(s.segments), "t": round(start, 1), "en": text, "zh": ""}
        s.segments.append(seg)
        self.emit("final", seg=seg)
        self._dirty = True
        if self._cloud_tr() and self.settings.get("translate_tier") == "saver":
            from translate import filler_translation
            f = filler_translation(text)
            if f is not None:
                seg["zh"] = f
                self.emit("translated", id=seg["id"], zh=f)
            else:
                with self.batch_lock:
                    if not self.batch:
                        self.batch_t0 = time.monotonic()
                    self.batch.append(seg)
                    words = sum(len(x["en"].split()) for x in self.batch)
                    full = len(self.batch) >= BATCH_MAX or words >= BATCH_WORDS
                if full:
                    self._flush_batch()
        else:
            self.tr_q.put([seg])
        self._maybe_presummarize()

    # ---------- translation ----------
    def _cloud_tr(self):
        return bool(self.settings.get("api_key")) and bool(self.settings.get("use_cloud_translate"))

    def _flush_batch(self):
        with self.batch_lock:
            segs, self.batch = self.batch, []
        if segs:
            self.tr_q.put(segs)

    def _batch_timer(self):
        while True:
            time.sleep(0.2)
            with self.batch_lock:
                due = self.batch and time.monotonic() - self.batch_t0 >= BATCH_WAIT
            if due:
                self._flush_batch()

    def _add_usage(self, u):
        if not u:
            return
        with self.usage_lock:
            for k in ("in", "out", "cw", "cr", "searches"):
                self.usage[k] = self.usage.get(k, 0) + u.get(k, 0)
            self.usage["requests"] = self.usage.get("requests", 0) + 1
            d = dict(self.usage)
        d["usd"] = round(usage_usd(self.settings.get("cloud_model"), d), 4)
        if self.session:
            self.session.usage = d
        self.emit("cost", **d)

    def _charge(self, session, u):
        """Add an API call's usage to `session` (the live one or one opened from history)."""
        if not u:
            return
        if self.session is not None and session.folder == self.session.folder:
            self._add_usage(u)
            if self.session is not session:
                session.usage = dict(self.session.usage)
            return
        su = dict(session.usage or {})
        for k in ("in", "out", "cw", "cr", "searches"):
            su[k] = su.get(k, 0) + u.get(k, 0)
        su["requests"] = su.get("requests", 0) + 1
        su["usd"] = round(usage_usd(self.settings.get("cloud_model"), su), 4)
        session.usage = su
        session.save()
        self.emit("cost_session", folder=str(session.folder), **su)

    def _translator(self):
        key = self.settings.get("api_key")
        if key and self.settings.get("use_cloud_translate"):
            from translate import CloudTranslator
            topic = self.session.title if self.session else ""
            return CloudTranslator(key, self.settings.get("cloud_model"), topic)
        return self._local_translator()

    def _local_translator(self):
        with self._tr_lock:
            if self.local_tr is None:
                from translate import LocalTranslator
                self.local_tr = LocalTranslator(threads=2)
        return self.local_tr

    def _translate_loop(self):
        while True:
            segs = self.tr_q.get()
            try:
                self._translate_segs(segs)
            except Exception as e:
                self.emit("warning", text=f"翻译出错：{e}")
            finally:
                self.tr_q.task_done()

    def _context_before(self, seg_id, n):
        if not self.session:
            return []
        return [x["en"] for x in self.session.segments[max(0, seg_id - n):seg_id]]

    def _finish(self, seg, zh):
        seg["zh"] = zh
        self.emit("translated", id=seg["id"], zh=zh)
        self._dirty = True

    def _translate_segs(self, segs):
        tr = self._translator()
        if len(segs) > 1 or (self._cloud_tr() and self.settings.get("translate_tier") == "saver"):
            return self._translate_batch(segs)
        seg = segs[0]
        ctx = self._context_before(seg["id"], 4)
        usage = {}
        try:
            if hasattr(tr, "translate_stream"):
                zh, last = "", 0.0
                for zh in tr.translate_stream(seg["en"], ctx, usage=usage):
                    if time.monotonic() - last > 0.12:
                        last = time.monotonic()
                        self.emit("translating", id=seg["id"], zh=zh)
            else:
                zh = tr.translate(seg["en"], ctx)
        except CloudError as e:
            self.emit("warning", text=f"{e}（这句已改用本地翻译）")
            zh = self._local_translator().translate(seg["en"])
        finally:
            self._add_usage(usage)
        self._finish(seg, zh)

    def _translate_batch(self, segs):
        from translate import BatchTranslator
        bt = BatchTranslator(self.settings.get("api_key"), self.settings.get("cloud_model"),
                             self.session.title if self.session else "")
        ctx = self._context_before(segs[0]["id"], 2)
        usage, got, last = {}, {}, 0.0
        try:
            for got in bt.translate_stream([x["en"] for x in segs], ctx, usage=usage):
                if time.monotonic() - last > 0.12:
                    last = time.monotonic()
                    for i, zh in got.items():
                        self.emit("translating", id=segs[i]["id"], zh=zh)
        except CloudError as e:
            self.emit("warning", text=f"{e}（这几句已改用本地翻译）")
        finally:
            self._add_usage(usage)
        for i, seg in enumerate(segs):
            zh = got.get(i, "")
            if not zh:
                zh = self._local_translator().translate(seg["en"])
            self._finish(seg, zh)

    def _autosave_loop(self):
        while True:
            time.sleep(5)
            if self._dirty and self.session:
                self._dirty = False
                try:
                    if self.recording:
                        self.session.duration = time.time() - self.t_start
                    self.session.save()
                except Exception:
                    pass

    # ---------- summary ----------
    def _use_cloud_summary(self):
        return bool(self.settings.get("api_key")) and bool(self.settings.get("use_cloud_summary"))

    def _maybe_presummarize(self):
        if (self.summary_busy or self._use_cloud_summary()
                or not self.settings.get("background_summary") or not is_installed("summary")
                or not self.summarizer.llm.available()):
            return
        sentences = [x["en"] for x in self.session.segments]

        def run():
            self.summary_busy = True
            try:
                self.summarizer.presummarize(sentences, threads=2)
            except Exception:
                pass
            finally:
                self.summary_busy = False
        threading.Thread(target=run, daemon=True).start()

    def summarize(self, folder: str = ""):
        if self.summary_busy and not folder:
            # a background chunk is running; wait for it
            while self.summary_busy:
                time.sleep(0.5)
        session = Session.load(Path(folder)) if folder else self.session
        if session is None:
            return {"error": "还没有记录"}
        sentences = [x["en"] for x in session.segments]

        def progress(p, msg):
            self.emit("summary_progress", p=p, text=msg, folder=str(session.folder))

        self.summary_busy = True
        try:
            if self._use_cloud_summary():
                try:
                    u = {}
                    stamped = [f"[{int(x['t']) // 60:02d}:{int(x['t']) % 60:02d}] {x['en']}" for x in session.segments]
                    out = self.summarizer.summarize_cloud(stamped, self.settings.get("api_key"),
                                                          self.settings.get("cloud_model"), progress, usage=u)
                    self._charge(session, u)
                except CloudError as e:
                    self.emit("warning", text=f"{e}（改用本地总结）")
                    out = self._local_summary(sentences, progress)
            else:
                out = self._local_summary(sentences, progress)
            session.summary = out
            session.save()
            if self.session and session.folder == self.session.folder:
                self.session.summary = out
            self.emit("summary", text=out, folder=str(session.folder))
            return {"summary": out}
        except Exception as e:
            self.emit("summary_progress", p=0, text="", folder=str(session.folder))
            self.emit("error", text=f"总结失败：{e}")
            return {"error": str(e)}
        finally:
            self.summary_busy = False

    def _local_summary(self, sentences, progress):
        if not is_installed("summary"):
            raise RuntimeError("本地总结模型还没下载（设置 → 本地 AI 总结模型），或在设置里填写 Claude API key")
        threads = 2 if self.recording else max(2, self.settings.threads())
        return self.summarizer.summarize_local(sentences, threads, progress)

    # ---------- AI 助教 ----------
    def _session_for(self, folder):
        if self.session is not None and (not folder or Path(folder) == self.session.folder):
            return self.session
        return Session.load(Path(folder)) if folder else None

    def _folder(self, folder):
        s = self._session_for(folder)
        return (str(s.folder), s) if s else (None, None)

    def notes_list(self, folder=""):
        import notes
        f, _ = self._folder(folder)
        return notes.load(f) if f else []

    def note_new(self, folder, anchor):
        import notes
        f, _ = self._folder(folder)
        if not f:
            return None
        notes.load(f)  # migrates old chat.json if needed
        t = notes.new_thread(anchor)
        notes.mutate(f, lambda ts: ts.append(t))
        return t

    def note_update(self, folder, tid, fields):
        import notes
        f, _ = self._folder(folder)
        allowed = {k: v for k, v in (fields or {}).items() if k in ("saved", "key", "title")}

        def fn(ts):
            t = notes.find(ts, tid)
            if t:
                t.update(allowed)
            return t
        t = notes.mutate(f, fn)
        if t and allowed.get("saved") and not t.get("key"):
            self.note_keypoint(folder, tid)
        return t

    def note_keep(self, folder, tid, idx, keep):
        import notes
        f, _ = self._folder(folder)

        def fn(ts):
            t = notes.find(ts, tid)
            if t and 0 <= idx < len(t["turns"]):
                t["turns"][idx]["keep"] = bool(keep)
            return t
        return notes.mutate(f, fn)

    def note_delete(self, folder, tid):
        import notes
        f, _ = self._folder(folder)

        def fn(ts):
            t = notes.find(ts, tid)
            if t:
                ts.remove(t)
            return t
        return notes.mutate(f, fn)

    def note_restore(self, folder, thread):
        import notes
        f, _ = self._folder(folder)
        notes.mutate(f, lambda ts: None if notes.find(ts, thread["id"]) else ts.append(thread))
        return True

    def note_keypoint(self, folder, tid):
        """Ask Claude for a one-line key point of the kept Q&A (tiny request, no transcript)."""
        threading.Thread(target=self._keypoint, args=(folder, tid), daemon=True).start()
        return True

    def _keypoint(self, folder, tid):
        import notes
        from cloud import claude_chat
        f, session = self._folder(folder)
        t = notes.find(notes.load(f), tid) if f else None
        if not t:
            return
        key = notes.fallback_key(t)
        api_key = self.settings.get("api_key")
        if api_key and any(x.get("keep", True) for x in t["turns"]):
            u = {}
            try:
                key = claude_chat(api_key, self.settings.get("cloud_model"), notes.KEY_PROMPT,
                                  notes.key_input(t), max_tokens=80, timeout=30, usage=u).strip().strip("。") or key
            except CloudError:
                pass
            self._charge(session, u)

        def fn(ts):
            x = notes.find(ts, tid)
            if x:
                x["key"] = key
        notes.mutate(f, fn)
        self.emit("note_key", folder=f, tid=tid, key=key)

    def ask(self, folder, tid, question, web=False, mid=""):
        """Runs in the background; streams chat_text / chat_event / chat_done / chat_error."""
        threading.Thread(target=self._ask, args=(folder, tid, question, web, mid), daemon=True).start()
        return True

    def _ask(self, folder, tid, question, web, mid):
        import assistant
        import notes
        key = self.settings.get("api_key")
        if not key:
            self.emit("chat_error", mid=mid, tid=tid, text="AI 助教需要先在设置里填写 Claude API key")
            return
        f, session = self._folder(folder)
        if session is None or not session.segments:
            self.emit("chat_error", mid=mid, tid=tid, text="这节课还没有内容可以问")
            return
        thread = notes.find(notes.load(f), tid)
        if thread is None:
            self.emit("chat_error", mid=mid, tid=tid, text="找不到这条对话")
            return
        anchor = thread["anchor"]
        quote, quote_t = anchor.get("text", ""), anchor.get("t")
        question = (question or "").strip() or assistant.EXPLAIN
        history = []
        for i, turn in enumerate(thread["turns"]):
            history.append({"role": "user", "q": turn["q"], "quote": quote if i == 0 else "",
                            "quote_t": quote_t if i == 0 else None})
            history.append({"role": "assistant", "text": turn["a"]})
        first = not thread["turns"]
        usage, last = {}, [0.0]

        def on_text(t):
            if time.monotonic() - last[0] > 0.08:
                last[0] = time.monotonic()
                self.emit("chat_text", mid=mid, tid=tid, text=t)

        def on_event(kind, data):
            self.emit("chat_event", mid=mid, tid=tid, ev=kind, data=data)

        try:
            answer, sources, web_used = assistant.ask(
                key, self.settings.get("cloud_model"), session, history, question,
                quote if first else "", quote_t if first else None, web,
                on_text=on_text, on_event=on_event, usage=usage)
        except CloudError as e:
            self.emit("chat_error", mid=mid, tid=tid, text=str(e))
            return
        except Exception as e:
            self.emit("chat_error", mid=mid, tid=tid, text=f"出错了：{e}")
            return
        finally:
            self._charge(session, usage)
        usd = round(usage_usd(self.settings.get("cloud_model"), usage), 4)
        turn = {"q": question, "a": answer, "sources": sources, "usd": usd, "web": web_used, "keep": True}

        def fn(ts):
            t = notes.find(ts, tid)
            if t is not None:
                t["turns"].append(turn)
                return len(t["turns"]) - 1
            return -1
        idx = notes.mutate(f, fn)
        self.emit("chat_done", mid=mid, tid=tid, idx=idx, text=answer, sources=sources, usd=usd)

    # ---------- history ----------
    def list_sessions(self):
        out = []
        for f in sorted(self.settings.sessions_dir().glob("*/data.json"), reverse=True):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                out.append({"folder": str(f.parent), "title": d.get("title"), "started": d.get("started"),
                            "duration": d.get("duration", 0), "count": len(d.get("segments", [])),
                            "has_summary": bool(d.get("summary")),
                            "usd": (d.get("usage") or {}).get("usd", 0)})
            except Exception:
                pass
        return out

    def load_session(self, folder):
        return Session.load(Path(folder)).to_dict()

    def rename_session(self, folder, title):
        s = Session.load(Path(folder))
        s.title = title or s.title
        s.save()
        if self.session and self.session.folder == s.folder:
            self.session.title = s.title
        return s.to_dict()

    def delete_session(self, folder):
        import shutil
        p = Path(folder)
        if p.parent == self.settings.sessions_dir() and (p / "data.json").exists():
            if self.session and self.session.folder == p and self.recording:
                return False
            shutil.rmtree(p)
            return True
        return False
