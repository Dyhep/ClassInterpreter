"""Where each model lives, and downloading the optional ones."""
import os
import shutil
import tarfile
import threading
import urllib.request
from pathlib import Path

from config import bundled_models_dir, downloaded_models_dir

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"

# name -> (folder name, download url (tar.bz2 or single file), approx MB)
MODELS = {
    "vad": ("silero_vad.onnx", SHERPA + "silero_vad.onnx", 1),
    "moonshine": ("sherpa-onnx-moonshine-base-en-int8",
                  SHERPA + "sherpa-onnx-moonshine-base-en-int8.tar.bz2", 270),
    "parakeet": ("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
                 SHERPA + "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2", 630),
    "translate": ("opus-mt-en-zh-ct2", None, 80),  # converted at build time, always bundled
    "summary": ("qwen2.5-1.5b-instruct-q4_k_m.gguf", [
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ], 1100),
}


def model_path(name: str):
    """Return the path of an installed model, or None."""
    folder = MODELS[name][0]
    for base in (bundled_models_dir(), downloaded_models_dir()):
        p = base / folder
        if p.exists():
            return p
    env = os.environ.get("CI_MODELS_DIR")
    if env and (Path(env) / folder).exists():
        return Path(env) / folder
    return None


REQUIRED = {"parakeet": "tokens.txt", "moonshine": "tokens.txt"}


def is_installed(name: str) -> bool:
    p = model_path(name)
    if p is None:
        return False
    need = REQUIRED.get(name)
    return (p / need).exists() if need and p.is_dir() else True


class Download:
    """Background download with progress (0..1)."""

    def __init__(self, name: str):
        self.name = name
        self.progress = 0.0
        self.done = False
        self.error = ""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        folder, urls, _ = MODELS[self.name]
        urls = urls if isinstance(urls, list) else [urls]
        dest_dir = downloaded_models_dir()
        last_err = ""
        for url in urls:
            tmp = dest_dir / (folder + ".part")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ClassInterpreter"})
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                    total = int(r.headers.get("Content-Length") or 0)
                    got = 0
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
                        if total:
                            self.progress = got / total * (0.9 if url.endswith(".tar.bz2") else 1.0)
                if url.endswith(".tar.bz2"):
                    # unpack next to the target and rename only when complete, so a half-unpacked
                    # folder is never mistaken for an installed model
                    staging = dest_dir / (folder + ".unpack")
                    shutil.rmtree(staging, ignore_errors=True)
                    with tarfile.open(tmp, "r:bz2") as t:
                        t.extractall(staging, filter="data")
                    tmp.unlink()
                    inner = staging / folder
                    shutil.rmtree(dest_dir / folder, ignore_errors=True)
                    (inner if inner.exists() else staging).rename(dest_dir / folder)
                    shutil.rmtree(staging, ignore_errors=True)
                else:
                    shutil.move(str(tmp), str(dest_dir / folder))
                self.progress = 1.0
                self.done = True
                return
            except Exception as e:  # try next mirror
                last_err = str(e)
                try:
                    tmp.unlink()
                except Exception:
                    pass
        self.error = last_err or "下载失败"
        self.done = True
