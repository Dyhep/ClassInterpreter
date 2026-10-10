"""English speech -> text with sherpa-onnx (voice activity detection + offline model)."""
import queue
import threading
import time

import numpy as np
import sherpa_onnx

from models import model_path

SR = 16000
MAX_SEG = 7.0        # seconds; long monologues are cut here so captions keep flowing
PREVIEW_EVERY = 0.6  # seconds between live-preview decodes


def make_recognizer(name: str, threads: int):
    d = model_path(name)
    if d is None:
        raise RuntimeError(f"识别模型 {name} 未安装")
    d = str(d) + "/"
    if name == "moonshine":
        return sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=d + "preprocess.onnx", encoder=d + "encode.int8.onnx",
            uncached_decoder=d + "uncached_decode.int8.onnx",
            cached_decoder=d + "cached_decode.int8.onnx",
            tokens=d + "tokens.txt", num_threads=threads)
    if name == "parakeet":
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=d + "encoder.int8.onnx", decoder=d + "decoder.int8.onnx",
            joiner=d + "joiner.int8.onnx", tokens=d + "tokens.txt",
            model_type="nemo_transducer", num_threads=threads)
    raise ValueError(name)


def make_vad():
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(model_path("vad"))
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.3
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.max_speech_duration = MAX_SEG
    cfg.sample_rate = SR
    cfg.num_threads = 1
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120), cfg.silero_vad.window_size


def _clean(text: str) -> str:
    t = " ".join(text.split())
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t


class Recognizer:
    """Feed audio with push(); get callbacks:
       on_partial(text)            – words heard so far in the current sentence
       on_final(text, start_sec)   – a finished sentence
    """

    def __init__(self, get_model, on_partial, on_final, live_preview=True, on_ready=None, on_error=None):
        """get_model() returns a loaded OfflineRecognizer; it runs on the worker thread, and audio
        pushed meanwhile is kept, so nothing said while the model loads is lost."""
        self.get_model = get_model
        self.rec = None
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_ready = on_ready or (lambda: None)
        self.on_error = on_error or (lambda e: None)
        self.live_preview = live_preview
        self.q = queue.Queue()
        self._stop = threading.Event()
        self.samples_seen = 0
        self.thread = threading.Thread(target=self._run_safe, daemon=True)
        self.thread.start()

    def push(self, audio: np.ndarray):
        if not self._stop.is_set():
            self.q.put(audio)

    def stop(self, timeout=None):
        """Finish the sentences already heard, then end. Returns True if it finished in time."""
        self._stop.set()
        self.q.put(None)
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def backlog(self) -> int:
        """Audio waiting to be processed, in 0.1 s blocks."""
        return self.q.qsize()

    def _run_safe(self):
        try:
            self.rec = self.get_model()
            self.vad, self.win = make_vad()
        except Exception as e:
            self.on_error(f"语音识别模型加载失败：{e}")
            return
        self.on_ready()
        try:
            self._run()
        except Exception as e:
            self.on_error(f"语音识别出错：{e}")

    def _decode(self, samples) -> str:
        s = self.rec.create_stream()
        s.accept_waveform(SR, samples)
        self.rec.decode_stream(s)
        return _clean(s.result.text)

    def _run(self):
        buf = np.zeros(0, dtype=np.float32)
        speech = []          # audio since speech started (for preview)
        last_preview = 0.0
        preview_gap = PREVIEW_EVERY
        while True:
            item = self.q.get()
            if item is None:
                break
            buf = np.concatenate([buf, item])
            while len(buf) >= self.win:
                chunk, buf = buf[:self.win], buf[self.win:]
                self.vad.accept_waveform(chunk)
                self.samples_seen += self.win
                if self.vad.is_speech_detected():
                    speech.append(chunk)
                while not self.vad.empty():
                    seg = self.vad.front
                    samples = np.array(seg.samples, dtype=np.float32)
                    start = seg.start / SR
                    self.vad.pop()
                    speech = []
                    text = self._decode(samples)
                    if text:
                        self.on_final(text, start)
                    else:
                        self.on_partial("")
            # live preview of the sentence in progress
            # (skipped while behind, so a slow computer catches up instead of lagging further)
            if (self.live_preview and speech and self.q.qsize() <= 2
                    and time.monotonic() - last_preview > preview_gap):
                audio = np.concatenate(speech)[-int(MAX_SEG * SR):]
                if len(audio) > SR * 0.5:
                    t0 = time.monotonic()
                    self.on_partial(self._decode(audio))
                    # slow computer -> preview less often so final sentences are never delayed
                    preview_gap = max(PREVIEW_EVERY, 2.0 * (time.monotonic() - t0))
                last_preview = time.monotonic()
        # flush
        self.vad.flush()
        while not self.vad.empty():
            seg = self.vad.front
            samples = np.array(seg.samples, dtype=np.float32)
            start = seg.start / SR
            self.vad.pop()
            text = self._decode(samples)
            if text:
                self.on_final(text, start)
