"""Audio capture: microphone or "what the computer is playing" (WASAPI loopback)."""
import sys
import threading
import time

import numpy as np

SR = 16000
BLOCK = 1600  # 0.1 s


def _co_init():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        except Exception:
            pass


def list_sources():
    """[{id, name, kind}] — kind is 'mic' or 'system'."""
    out = []
    try:
        import soundcard as sc
        try:
            dspk = sc.default_speaker()
            out.append({"id": "system:default", "name": f"电脑声音（{dspk.name}）", "kind": "system"})
        except Exception:
            pass
        try:
            dmic = sc.default_microphone()
            out.append({"id": "mic:default", "name": f"默认麦克风（{dmic.name}）", "kind": "mic"})
        except Exception:
            pass
        for m in sc.all_microphones(include_loopback=False):
            out.append({"id": "mic:" + str(m.id), "name": "麦克风：" + m.name, "kind": "mic"})
    except Exception as e:
        out.append({"id": "", "name": f"找不到音频设备：{e}", "kind": "none"})
    return out


class Capture:
    """Calls on_audio(float32 mono 16 kHz ndarray) from a background thread."""

    def __init__(self, source_id: str, on_audio, on_error=None, on_level=None,
                 on_progress=None, on_done=None, busy=None):
        self.source_id = source_id
        self.on_progress = on_progress or (lambda p: None)
        self.on_done = on_done or (lambda: None)
        self.busy = busy or (lambda: False)  # file mode: wait while the recognizer is behind
        self.on_audio = on_audio
        self.on_error = on_error or (lambda e: None)
        self.on_level = on_level or (lambda v: None)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=3)

    def _device(self):
        import soundcard as sc
        sid = self.source_id or "mic:default"
        if sid == "system:default":
            spk = sc.default_speaker()
            return sc.get_microphone(id=str(spk.name), include_loopback=True)
        if sid.startswith("system:"):
            return sc.get_microphone(id=sid[7:], include_loopback=True)
        if sid == "mic:default":
            return sc.default_microphone()
        return sc.get_microphone(id=sid[4:], include_loopback=False)

    def _run(self):
        _co_init()
        if self.source_id.startswith("file:"):
            try:
                self._run_file(self.source_id[5:])
            except Exception as e:
                self.on_error(f"无法读取录音文件：{e}")
                return
            if not self._stop.is_set():
                self.on_done()
            return
        try:
            dev = self._device()
            with dev.recorder(samplerate=SR, channels=1, blocksize=BLOCK) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=BLOCK)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    data = np.ascontiguousarray(data, dtype=np.float32)
                    self.on_level(float(np.sqrt(np.mean(data * data)) if len(data) else 0.0))
                    self.on_audio(data)
        except Exception as e:
            self.on_error(f"无法录音：{e}")

    def _run_file(self, path):
        """Test source: play a wav file in real time (or faster with ?speed=N)."""
        import soundfile as sf
        speed = 1.0
        if "?speed=" in path:
            path, s = path.split("?speed=")
            speed = float(s)
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != SR:
            n = int(len(a) * SR / sr)
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
        last = -1
        for i in range(0, len(a), BLOCK):
            if self._stop.is_set():
                return
            while self.busy() and not self._stop.is_set():
                time.sleep(0.05)
            self.on_audio(a[i:i + BLOCK])
            pct = int(i * 100 / max(1, len(a)))
            if pct != last:
                last = pct
                self.on_progress(pct)
            time.sleep(BLOCK / SR / speed)
        self.on_audio(np.zeros(SR * 2, dtype=np.float32))  # flush trailing speech
