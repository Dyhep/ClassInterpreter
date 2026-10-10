"""Ties capture -> recognition -> translation -> storage -> summary together."""
import json
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from audio import Capture
from cloud import CloudError
from config import Settings
from models import Download, MODELS, is_installed
from summarize import Summarizer


def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip() or "未命名"


class Session:
    def __init__(self, folder: Path, title: str, started: str):
        self.folder = folder
        self.title = title
        self.started = started
        self.segments = []   # {id, t, en, zh}
        self.summary = ""
        self.duration = 0.0

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
        return s

    def to_dict(self):
        return {"title": self.title, "started": self.started, "duration": self.duration,
                "segments": self.segments, "summary": self.summary, "folder": str(self.folder)}

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
        self.tr_q = queue.Queue()
        self.recording = False
        self.t_start = 0.0
        self.summarizer = Summarizer()
        self.summary_busy = False
        self.downloads = {}
        self._dirty = False
        self.level = 0.0
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
        return {"events": evs, "level": round(self.level, 4), "recording": self.recording,
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
    def start(self, source_id: str, title: str = ""):
        if self.recording:
            return
        self.emit("status", text="正在加载语音识别模型…")
        from asr import Recognizer
        name = self.settings.get("asr_model")
        if not is_installed(name):
            name = "moonshine"
        self.recognizer = Recognizer(name, self.settings.threads(), self._on_partial, self._on_final,
                                     live_preview=bool(self.settings.get("live_preview"))
                                     and not source_id.startswith("file:"))
        self.session = Session.new(self.settings.sessions_dir(), title)
        self.summarizer.chunk_notes.clear()
        self.emit("session", session=self.session.to_dict())
        self.settings.update({"source": source_id})
        rec = self.recognizer
        self.capture = Capture(source_id, rec.push, on_error=self._on_capture_error,
                               on_level=self._on_level,
                               on_progress=lambda p: self.emit("status", text=f"正在处理录音文件… {p}%"),
                               on_done=lambda: threading.Thread(target=self.stop, daemon=True).start(),
                               busy=lambda: rec.backlog() > 20)
        self.recording = True
        self.t_start = time.time()
        self.capture.start()
        self.emit("status", text="正在聆听…")

    def _on_level(self, v):
        self.level = v

    def _on_capture_error(self, msg):
        self.emit("error", text=msg)
        threading.Thread(target=self.stop, daemon=True).start()

    def stop(self):
        if not self.recording:
            return
        self.recording = False
        self.emit("status", text="正在处理最后几句…")
        if self.capture:
            self.capture.stop()
        if self.recognizer:
            self.recognizer.stop()
            self.recognizer = None
        self.tr_q.join()
        if self.session:
            self.session.duration = time.time() - self.t_start
            self.session.save()
        self.summarizer.llm.stop()
        self.level = 0.0
        self.emit("stopped", session=self.session.to_dict() if self.session else None)
        self.emit("status", text="已停止，记录已保存")

    def _on_partial(self, text):
        self.emit("partial", text=text)

    def _on_final(self, text, start):
        s = self.session
        if s is None:
            return
        seg = {"id": len(s.segments), "t": round(start, 1), "en": text, "zh": ""}
        s.segments.append(seg)
        self.emit("final", seg=seg)
        self.tr_q.put(seg)
        self._dirty = True
        self._maybe_presummarize()

    # ---------- translation ----------
    def _translator(self):
        key = self.settings.get("api_key")
        if key and self.settings.get("use_cloud_translate"):
            from translate import CloudTranslator
            return CloudTranslator(key, self.settings.get("cloud_model"))
        return self._local_translator()

    def _local_translator(self):
        if self.local_tr is None:
            from translate import LocalTranslator
            self.local_tr = LocalTranslator(threads=2)
        return self.local_tr

    def _translate_loop(self):
        while True:
            seg = self.tr_q.get()
            try:
                ctx = [x["en"] for x in self.session.segments[max(0, seg["id"] - 3):seg["id"]]] \
                    if self.session else []
                try:
                    zh = self._translator().translate(seg["en"], ctx)
                except CloudError as e:
                    self.emit("warning", text=f"{e}（这句已改用本地翻译）")
                    zh = self._local_translator().translate(seg["en"])
                seg["zh"] = zh
                self.emit("translated", id=seg["id"], zh=zh)
                self._dirty = True
            except Exception as e:
                self.emit("warning", text=f"翻译出错：{e}")
            finally:
                self.tr_q.task_done()

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
                    out = self.summarizer.summarize_cloud(sentences, self.settings.get("api_key"),
                                                          self.settings.get("cloud_model"), progress)
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

    # ---------- history ----------
    def list_sessions(self):
        out = []
        for f in sorted(self.settings.sessions_dir().glob("*/data.json"), reverse=True):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                out.append({"folder": str(f.parent), "title": d.get("title"), "started": d.get("started"),
                            "duration": d.get("duration", 0), "count": len(d.get("segments", [])),
                            "has_summary": bool(d.get("summary"))})
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
