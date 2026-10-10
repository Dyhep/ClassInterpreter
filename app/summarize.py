"""Class summaries: local small LLM (llama.cpp server) or Claude API."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from cloud import claude_chat
from config import resource_dir
from models import model_path

CHUNK_CHARS = 5000  # English characters per local chunk (~1.3k tokens)

MAP_PROMPT = (
    "下面是一段课堂（或会议）录音的英文转写，可能有识别错误。"
    "请用简体中文列出这一段的要点（3-8条，每条一句话），"
    "保留关键术语的英文原词，如果提到作业、考试、截止日期、阅读材料，务必单独写出来。只输出要点列表。\n\n转写：\n"
)

REDUCE_PROMPT = (
    "下面是一堂课各个时间段的中文要点。请整理成一份完整的课堂笔记，使用 Markdown，按下面的结构：\n"
    "## 本节课主题\n（一两句话）\n## 主要内容\n（分小标题，条理清楚）\n"
    "## 重要概念 / 术语\n（中文 — English，附一句解释）\n"
    "## 作业、考试与截止日期\n（没有就写“未提及”）\n## 复习建议\n（2-3条）\n\n各段要点：\n"
)

CLOUD_SYSTEM = (
    "你是认真负责的助教。用户会给你一堂课（或一次会议）的英文语音转写，可能有识别错误，请结合上下文理解。"
    "用简体中文写一份清晰的课堂笔记（Markdown）：\n"
    "## 本节课主题\n## 主要内容（分小标题，条理清楚，可以较详细）\n"
    "## 重要概念 / 术语（中文 — English，附一句解释）\n"
    "## 作业、考试与截止日期（没有就写“未提及”）\n## 可能考到的点\n## 复习建议"
)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _llama_server_bin():
    env = os.environ.get("LLAMA_SERVER")
    if env:
        return env
    exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    p = resource_dir() / "bin" / "llama" / exe
    return str(p) if p.exists() else None


class LocalLLM:
    """Starts llama-server on demand and stops it when idle (frees ~1.3 GB RAM)."""

    def __init__(self):
        self.proc = None
        self.port = None
        self.lock = threading.Lock()

    def available(self):
        return _llama_server_bin() is not None

    def start(self, threads: int):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return
            exe = _llama_server_bin()
            model = model_path("summary")
            if not exe:
                raise RuntimeError("找不到本地总结程序（llama-server）")
            if not model:
                raise RuntimeError("本地总结模型未下载")
            self.port = _free_port()
            flags = 0
            if sys.platform == "win32":
                flags = 0x08000000 | 0x00004000  # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY
            self.proc = subprocess.Popen(
                [exe, "-m", str(model), "-c", "4096", "-t", str(threads), "--host", "127.0.0.1",
                 "--port", str(self.port), "-np", "1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
            t0 = time.time()
            while time.time() - t0 < 120:
                if self.proc.poll() is not None:
                    raise RuntimeError("本地总结程序启动失败（可能内存不足）")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                        if r.status == 200:
                            return
                except Exception:
                    pass
                time.sleep(0.5)
            raise RuntimeError("本地总结程序启动超时")

    def chat(self, prompt: str, max_tokens=600) -> str:
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.3, "repeat_penalty": 1.1,
        }).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(5)
                except Exception:
                    self.proc.kill()
            self.proc = None


def split_chunks(sentences, size=CHUNK_CHARS):
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) > size:
            chunks.append(cur)
            cur = ""
        cur += s + " "
    if cur.strip():
        chunks.append(cur)
    return chunks


class Summarizer:
    """Keeps per-chunk notes so background pre-summarizing can be reused at the end."""

    def __init__(self):
        self.llm = LocalLLM()
        self.chunk_notes = {}  # chunk text -> notes
        self.lock = threading.Lock()

    def summarize_local(self, sentences, threads, progress=lambda p, msg: None) -> str:
        chunks = split_chunks(sentences)
        if not chunks:
            return "（还没有内容可以总结）"
        progress(0.02, "正在启动本地 AI…")
        self.llm.start(threads)
        try:
            notes = []
            for i, c in enumerate(chunks):
                with self.lock:
                    n = self.chunk_notes.get(c)
                if n is None:
                    progress(0.05 + 0.8 * i / len(chunks), f"正在整理第 {i + 1}/{len(chunks)} 段…")
                    n = self.llm.chat(MAP_PROMPT + c, max_tokens=450)
                    with self.lock:
                        self.chunk_notes[c] = n
                notes.append(f"【第{i + 1}段】\n{n}")
            progress(0.88, "正在生成完整笔记…")
            joined = "\n\n".join(notes)
            # keep the reduce prompt inside the 4k context
            while len(joined) > 3500 and len(notes) > 1:
                merged = []
                for j in range(0, len(notes), 3):
                    group = "\n\n".join(notes[j:j + 3])
                    merged.append(self.llm.chat("把下面几段要点合并精简成一份要点列表（简体中文）：\n\n" + group,
                                                max_tokens=450))
                notes = merged
                joined = "\n\n".join(notes)
            result = self.llm.chat(REDUCE_PROMPT + joined, max_tokens=1200)
            progress(1.0, "完成")
            return result
        finally:
            self.llm.stop()

    def presummarize(self, sentences, threads):
        """Background: summarize finished chunks (all but the last, still-growing one)."""
        chunks = split_chunks(sentences)[:-1]
        todo = [c for c in chunks if c not in self.chunk_notes]
        if not todo:
            return
        self.llm.start(threads)
        for c in todo:
            n = self.llm.chat(MAP_PROMPT + c, max_tokens=450)
            with self.lock:
                self.chunk_notes[c] = n

    def summarize_cloud(self, sentences, api_key, model, progress=lambda p, msg: None) -> str:
        text = " ".join(sentences)
        if not text.strip():
            return "（还没有内容可以总结）"
        progress(0.1, "正在请 Claude 总结…")
        out = claude_chat(api_key, model, CLOUD_SYSTEM, "课堂转写：\n" + text, max_tokens=4000, timeout=180)
        progress(1.0, "完成")
        return out
