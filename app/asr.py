"""English speech -> text with sherpa-onnx (voice activity detection + offline model)."""
import queue
import threading
import time

import numpy as np
import sherpa_onnx

from models import model_path

SR = 16000
MAX_SEG = 12.0       # seconds; long monologues are cut here
PREVIEW_EVERY = 1.5  # seconds between live-preview decodes


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
    cfg.silero_vad.min_silence_duration = 0.45
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

    def __init__(self, model_name, threads, on_partial, on_final, live_preview=True):
        self.rec = make_recognizer(model_name, threads)
        self.vad, self.win = make_vad()
        self.on_partial = on_partial
        self.on_final = on_final
        self.live_preview = live_preview
        self.q = queue.Queue()
        self._stop = threading.Event()
        self.samples_seen = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def push(self, audio: np.ndarray):
        self.q.put(audio)

    def stop(self):
        self._stop.set()
        self.q.put(None)
        self.thread.join()

    def backlog(self) -> int:
        return self.q.qsize()

    def _decode(self, samples) -> str:
        s = self.rec.create_stream()
        s.accept_waveform(SR, samples)
        self.rec.decode_stream(s)
        return _clean(s.result.text)

    def _run(self):
        buf = np.zeros(0, dtype=np.float32)
        speech = []          # audio since speech started (for preview)
        last_preview = 0.0
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
            if (self.live_preview and speech and self.q.empty()
                    and time.monotonic() - last_preview > PREVIEW_EVERY):
                last_preview = time.monotonic()
                audio = np.concatenate(speech)[-int(MAX_SEG * SR):]
                if len(audio) > SR * 0.8:
                    self.on_partial(self._decode(audio))
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
