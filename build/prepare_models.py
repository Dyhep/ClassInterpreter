"""Build-time: fetch speech models, convert the translation model, fetch llama-server.

Usage: python build/prepare_models.py <out_dir>
Creates <out_dir>/models/... and <out_dir>/bin/llama/...
Needs (build machine only): transformers, torch (cpu), huggingface_hub, ctranslate2, sentencepiece
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"


def fetch(url, headers=None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ClassInterpreter-build", **(headers or {})})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def speech(models: Path):
    if not (models / "silero_vad.onnx").exists():
        (models / "silero_vad.onnx").write_bytes(fetch(SHERPA + "silero_vad.onnx"))
    name = "sherpa-onnx-moonshine-base-en-int8"
    if not (models / name).exists():
        with tarfile.open(fileobj=io.BytesIO(fetch(SHERPA + name + ".tar.bz2")), mode="r:bz2") as t:
            t.extractall(models, filter="data")
        tw = models.parent / "test_wavs"
        if tw.exists():
            shutil.rmtree(tw)
        shutil.move(str(models / name / "test_wavs"), str(tw))  # used by the smoke test, not shipped
    print("speech models ok")


def translation(models: Path):
    out = models / "opus-mt-en-zh-ct2"
    if (out / "model.bin").exists():
        print("translation model exists")
        return
    from huggingface_hub import snapshot_download
    from transformers import MarianTokenizer
    import sentencepiece as spm

    repo = "Helsinki-NLP/opus-mt-en-zh"
    src = snapshot_download(repo)
    subprocess.check_call(["ct2-transformers-converter", "--model", src, "--output_dir", str(out),
                           "--quantization", "int8", "--force"])
    for f in ("source.spm", "target.spm"):
        shutil.copy(Path(src) / f, out / f)
    # Work out exactly which tokens the HF tokenizer adds around a sentence.
    tok = MarianTokenizer.from_pretrained(src)
    sample = "Hello world, this is a test."
    full = tok.convert_ids_to_tokens(tok(">>cmn_Hans<< " + sample).input_ids)
    body = spm.SentencePieceProcessor(model_file=str(out / "source.spm")).encode(sample, out_type=str)
    i = next(k for k in range(len(full)) if full[k:k + len(body)] == body) if body else 0
    meta = {"prefix": full[:i], "suffix": full[i + len(body):], "hf_tokens": full}
    print("tokenizer meta:", meta)
    if full[i:i + len(body)] != body:
        print("WARNING: sentencepiece and HF tokenization differ; using HF prefix/suffix anyway")
    (out / "ci_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    shutil.copy(Path(src) / "LICENSE", out / "LICENSE") if (Path(src) / "LICENSE").exists() else None
    print("translation model ok")


def llama(bin_dir: Path):
    dest = bin_dir / "llama"
    if (dest / "llama-server.exe").exists():
        return
    headers = {}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    rel = json.loads(fetch("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest", headers))
    assets = [a for a in rel["assets"] if a["name"].endswith(".zip") and "win" in a["name"]]
    pick = None
    for want in (("cpu", "x64"), ("avx2", "x64")):
        pick = next((a for a in assets if all(w in a["name"] for w in want)
                     and "arm" not in a["name"]), None)
        if pick:
            break
    if not pick:
        raise SystemExit("no windows cpu build found in llama.cpp release: " + ", ".join(a["name"] for a in assets))
    print("llama.cpp asset:", pick["name"])
    data = fetch(pick["browser_download_url"])
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.namelist():
            base = os.path.basename(m)
            # only what the server needs: the exe and its DLLs
            if base and (base == "llama-server.exe" or base.lower().endswith(".dll")):
                (dest / base).write_bytes(z.read(m))
    if not (dest / "llama-server.exe").exists():
        raise SystemExit("llama-server.exe missing from " + pick["name"])
    (dest / "VERSION.txt").write_text(rel.get("tag_name", ""), encoding="utf-8")
    print("llama-server ok")


def webview2(bin_dir: Path):
    p = bin_dir / "MicrosoftEdgeWebview2Setup.exe"
    if not p.exists():
        p.write_bytes(fetch("https://go.microsoft.com/fwlink/p/?LinkId=2124703"))
    print("webview2 bootstrapper ok")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "dist_assets")
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "bin").mkdir(parents=True, exist_ok=True)
    speech(out / "models")
    translation(out / "models")
    if sys.platform == "win32":
        llama(out / "bin")
        webview2(out / "bin")
