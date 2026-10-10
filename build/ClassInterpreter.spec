# PyInstaller spec — run from repo root:  pyinstaller build/ClassInterpreter.spec --noconfirm
# Expects build assets in dist_assets/ (see build/prepare_models.py).
from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = SPECPATH + "/.."
datas, binaries, hidden = [], [], []
for pkg in ("sherpa_onnx", "ctranslate2", "webview", "pythonnet", "clr_loader"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hidden += h
datas += collect_data_files("soundcard")   # cffi header files read at runtime
datas += collect_data_files("opencc")
datas += [
    (ROOT + "/app/ui", "app/ui"),
    (ROOT + "/dist_assets/models", "models"),
    (ROOT + "/dist_assets/bin/llama", "bin/llama"),
]
hidden += ["sentencepiece", "soundfile", "opencc", "clr_loader", "pythonnet"]

a = Analysis(
    [ROOT + "/app/main.py"],
    pathex=[ROOT + "/app"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "torch", "transformers", "PyQt5", "PySide6", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ClassInterpreter",
    icon=ROOT + "/build/icon.ico",
    console=False,
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ClassInterpreter")
