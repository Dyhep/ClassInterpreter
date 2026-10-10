"""课堂同传 — desktop entry point (pywebview window + Python backend)."""
import os
import subprocess
import sys
import threading
from pathlib import Path

import webview

from cloud import test_key
from config import APP_TITLE, VERSION, resource_dir
from engine import Engine
import audio


class Api:
    def __init__(self, engine: Engine):
        self._e = engine
        self._window = None

    # ---- setup ----
    def init(self):
        return {"settings": self._e.settings.public(), "sources": audio.list_sources(),
                "models": self._e.model_status(), "version": VERSION,
                "sessions": self._e.list_sessions()}

    def sources(self):
        return audio.list_sources()

    def models(self):
        return self._e.model_status()

    def download(self, name):
        self._e.download(name)
        return True

    def save_settings(self, d):
        d = dict(d)
        if "api_key" in d and str(d["api_key"]).startswith("••••"):
            d.pop("api_key")  # unchanged masked value
        self._e.settings.update(d)
        return self._e.settings.public()

    def check_key(self, key):
        if str(key).startswith("••••"):
            key = self._e.settings.get("api_key")
        return test_key(key.strip(), self._e.settings.get("cloud_model"))

    # ---- recording ----
    def poll(self, after):
        return self._e.poll(int(after))

    def start(self, source_id, title=""):
        ok = self._e.start(source_id, title)
        return {"ok": True} if ok else {"ok": False, "busy": True}

    def stop(self):
        threading.Thread(target=self._e.stop, daemon=True).start()
        return True

    def import_file(self, title=""):
        types = ("音频文件 (*.wav;*.mp3;*.flac;*.ogg)",)
        mode = getattr(getattr(webview, "FileDialog", None), "OPEN", None) or webview.OPEN_DIALOG
        res = self._window.create_file_dialog(mode, file_types=types)
        if not res:
            return {"ok": False, "cancel": True}
        path = res[0] if isinstance(res, (list, tuple)) else res
        title = title or Path(path).stem
        return self.start(f"file:{path}?speed=30", title)

    def summarize(self, folder=""):
        return self._e.summarize(folder)

    # ---- AI 助教 ----
    def notes_list(self, folder=""):
        return self._e.notes_list(folder)

    def note_new(self, folder, anchor):
        return self._e.note_new(folder, anchor)

    def note_update(self, folder, tid, fields):
        return self._e.note_update(folder, tid, fields)

    def note_keep(self, folder, tid, idx, keep):
        return self._e.note_keep(folder, tid, int(idx), keep)

    def note_delete(self, folder, tid):
        return self._e.note_delete(folder, tid)

    def note_restore(self, folder, thread):
        return self._e.note_restore(folder, thread)

    def note_keypoint(self, folder, tid):
        return self._e.note_keypoint(folder, tid)

    def ask(self, folder, tid, question, web=False, mid=""):
        return self._e.ask(folder, tid, question, web, mid)

    def open_url(self, url):
        import webbrowser
        if str(url).startswith(("https://", "http://")):
            webbrowser.open(url)
        return True

    # ---- history ----
    def sessions(self):
        return self._e.list_sessions()

    def load_session(self, folder):
        return self._e.load_session(folder)

    def rename_session(self, folder, title):
        return self._e.rename_session(folder, title)

    def delete_session(self, folder):
        return self._e.delete_session(folder)

    def open_folder(self, folder=""):
        p = folder or str(self._e.settings.sessions_dir())
        if sys.platform == "win32":
            os.startfile(p)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])
        return True

    def choose_sessions_dir(self):
        mode = getattr(getattr(webview, "FileDialog", None), "FOLDER", None) or webview.FOLDER_DIALOG
        res = self._window.create_file_dialog(mode)
        if res:
            p = res[0] if isinstance(res, (list, tuple)) else res
            self._e.settings.update({"sessions_dir": p})
        return self._e.settings.public()

    def set_on_top(self, on):
        try:
            self._window.on_top = bool(on)
        except Exception:
            pass
        return bool(on)


def selftest():
    """Used by the build to make sure the packaged app can load everything."""
    from config import data_dir
    log = data_dir() / "selftest.log"
    lines = []
    try:
        from asr import make_recognizer, make_vad
        from summarize import LocalLLM
        from translate import LocalTranslator
        make_vad()
        make_recognizer("moonshine", 2)
        lines.append("asr ok")
        zh = LocalTranslator(2).translate("The lecture starts at nine o'clock.")
        lines.append("translate ok: " + zh)
        assert zh
        assert LocalLLM().available(), "llama-server missing"
        lines.append("llama-server ok")
        assert (resource_dir() / "app" / "ui" / "index.html").exists(), "ui missing"
        lines.append("ui ok")
        lines.append("sources: " + str(audio.list_sources()))
        log.write_text("\n".join(lines), encoding="utf-8")
        return 0
    except Exception as ex:
        import traceback
        lines.append("FAILED: " + repr(ex) + "\n" + traceback.format_exc())
        log.write_text("\n".join(lines), encoding="utf-8")
        return 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    engine = Engine()
    api = Api(engine)
    ui = resource_dir() / "app" / "ui" / "index.html"
    if not ui.exists():
        ui = Path(__file__).resolve().parent / "ui" / "index.html"
    window = webview.create_window(APP_TITLE, url=str(ui), js_api=api, width=1100, height=760,
                                   min_size=(420, 360), background_color="#F7F6F2",
                                   text_select=True)  # pywebview blocks text selection by default
    api._window = window

    def on_closing():
        threading.Thread(target=engine.stop, daemon=True).start()
        engine.summarizer.llm.stop()

    window.events.closing += on_closing
    webview.start(debug=bool(os.environ.get("CI_DEBUG")))
    engine.summarizer.llm.stop()


if __name__ == "__main__":
    main()
