"""Paths and user settings."""
import json
import os
import sys
from pathlib import Path

APP_NAME = "ClassInterpreter"
APP_TITLE = "课堂同传"
VERSION = "0.8.1"


def resource_dir() -> Path:
    """Folder holding bundled files (models shipped with the installer, ui, bin)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(Path.home(), ".local", "share")
    p = Path(base) / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def downloaded_models_dir() -> Path:
    p = data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def bundled_models_dir() -> Path:
    return resource_dir() / "models"


def default_sessions_dir() -> Path:
    docs = Path.home() / "Documents"
    if not docs.exists():
        docs = Path.home()
    return docs / "课堂同传记录"


DEFAULTS = {
    "api_key": "",                 # Claude API key; empty = fully local
    "use_cloud_translate": True,   # only used when api_key is set
    "proofread_en": True,          # Claude also fixes words the recognizer missed (with cloud translation)
    "translate_tier": "best",      # best (per sentence, instant) | saver (batched, ~half the cost)
    "use_cloud_summary": True,     # only used when api_key is set
    "cloud_model": "claude-haiku-4-5",
    "asr_model": "moonshine",      # (old setting, ignored unless asr_choice says otherwise)
    "asr_choice": "auto",          # auto (high-precision once downloaded) | moonshine | parakeet
    "save_audio": True,            # keep a compressed recording of each class (~17 MB/hour)
    "live_preview": True,          # show text while the speaker is still talking
    "background_summary": False,   # pre-summarize chunks during class (local mode)
    "threads": 0,                  # 0 = auto
    "sessions_dir": "",
    "display_mode": "both",        # both | zh | en
    "font_size": 20,
    "source": "",                  # last used audio device id
    "tour_done": False,            # the guided tour was offered / finished
}


class Settings:
    def __init__(self):
        self.path = data_dir() / "settings.json"
        self.values = dict(DEFAULTS)
        try:
            self.values.update(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            pass

    def get(self, k):
        return self.values.get(k, DEFAULTS.get(k))

    def update(self, d: dict):
        for k, v in d.items():
            if k in DEFAULTS:
                self.values[k] = v
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.values, ensure_ascii=False, indent=2), encoding="utf-8")

    def sessions_dir(self) -> Path:
        p = Path(self.get("sessions_dir") or default_sessions_dir())
        p.mkdir(parents=True, exist_ok=True)
        return p

    def threads(self) -> int:
        t = int(self.get("threads") or 0)
        if t > 0:
            return t
        n = os.cpu_count() or 2
        # leave headroom for the UI and the translator
        return max(1, min(4, n // 2))

    def public(self) -> dict:
        d = dict(self.values)
        key = d.get("api_key") or ""
        d["api_key"] = ("••••" + key[-4:]) if key else ""
        d["has_api_key"] = bool(key)
        return d
