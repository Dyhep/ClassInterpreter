"""English speech -> text with sherpa-onnx (voice activity detection + offline model)."""
import queue
import threading
import time

import numpy as np
import sherpa_onnx

from models import model_path

SR = 16000
PREVIEW_EVERY = 0.6  # seconds between live-preview decodes

# Segmentation & audio conditioning (tuned on synthetic quiet / speaker-in-room / far-field classroom audio)
CFG = {
    "max_seg": 7.0,       # longest sentence before a forced cut (s)
    "min_silence": 0.3,   # pause that ends a sentence (s)
    "threshold": 0.5,     # voice-activity probability threshold
    "pad_pre": 0.0,       # audio kept before the detected start (s) – catches clipped first syllables
    "pad_post": 0.0,      # audio kept after the detected end (s)
    "agc": False,         # automatic gain for quiet / far-away speech
}

# Measured word error rate (quiet / speaker-in-room / noisy far-field classroom):
#   moonshine, defaults            11.5% / 5.9% / 76.7%   (padding made the room case worse)
#   parakeet,  defaults             3.5% / 2.4% / 27.1%
#   parakeet,  settings below       2.3% / 1.6% / 28.6%   (no swallowed words in the room case)
MODEL_CFG = {
    "parakeet": {"pad_pre": 0.2, "pad_post": 0.1, "min_silence": 0.5},
    "moonshine": {},
}


def cfg_for(model: str) -> dict:
    c = dict(CFG)
    c.update(MODEL_CFG.get(model, {}))
    return c


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


def make_vad(c=None):
    c = c or CFG
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(model_path("vad"))
    cfg.silero_vad.threshold = c["threshold"]
    cfg.silero_vad.min_silence_duration = c["min_silence"]
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.max_speech_duration = c["max_seg"]
    cfg.sample_rate = SR
    cfg.num_threads = 1
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120), cfg.silero_vad.window_size


def _norm(w: str) -> str:
    return "".join(c for c in w.lower() if c.isalnum() or c == "'")


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

    def __init__(self, get_model, on_partial, on_final, live_preview=True, on_ready=None, on_error=None,
                 cfg=None):
        """get_model() returns a loaded OfflineRecognizer; it runs on the worker thread, and audio
        pushed meanwhile is kept, so nothing said while the model loads is lost."""
        self.get_model = get_model
        self.cfg = cfg  # None = use the module-wide CFG (as the test harness does)
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
            if self.cfg is None:
                self.cfg = dict(CFG)
            self.vad, self.win = make_vad(self.cfg)
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

    def _gain(self, block):
        """Slow automatic gain: bring the loud parts of recent audio to ~0.1 RMS (max x12)."""
        if not self.cfg["agc"]:
            return 1.0
        r = float(np.sqrt(np.mean(block * block))) if len(block) else 0.0
        self._levels.append(r)
        if len(self._levels) > 100:          # ~10 s of 0.1 s blocks
            self._levels.pop(0)
        loud = np.percentile(self._levels, 90) if len(self._levels) >= 5 else max(r, 1e-4)
        target = min(12.0, max(1.0, 0.1 / max(loud, 1e-4)))
        self._g += (target - self._g) * 0.15  # smooth
        return self._g

    def _segment_audio(self, seg):
        """The detected sentence plus a little audio before/after it, from the raw history."""
        start, n = seg.start, len(seg.samples)
        a = max(self._hist_start, start - int(self.cfg["pad_pre"] * SR))
        b = min(self._hist_start + len(self._hist), start + n + int(self.cfg["pad_post"] * SR))
        if self.cfg["pad_pre"] or self.cfg["pad_post"] or self.cfg["agc"]:
            x = self._hist[a - self._hist_start:b - self._hist_start] * self._g
        else:
            x = np.array(seg.samples, dtype=np.float32)
        if self.cfg["agc"]:
            r = float(np.sqrt(np.mean(x * x))) if len(x) else 0
            if r > 1e-4:
                x = x * min(12.0, 0.08 / r)
            x = np.clip(x, -1, 1)
        return x.astype(np.float32)

    # ---- committing whole sentences while the speaker keeps going ----
    # Fast speakers (news, policy talks) barely pause, so voice detection alone would wait 10+ s
    # before ending a "sentence". When two consecutive previews agree on a complete sentence
    # (and the speaker has already moved on to the next one), that sentence is finished early,
    # using the model's word timings to cut the audio exactly between the two sentences.
    COMMIT_FORCE = 12.0   # s of uncommitted speech before committing at a comma/agreed words

    def _decode_words(self, samples):
        """[(word, start_sec)] using token timings, or None if the model has none."""
        s = self.rec.create_stream()
        s.accept_waveform(SR, samples)
        self.rec.decode_stream(s)
        r = s.result
        toks, ts = list(getattr(r, "tokens", []) or []), list(getattr(r, "timestamps", []) or [])
        if not toks or len(toks) != len(ts):
            return _clean(r.text), None
        words = []
        for tok, t in zip(toks, ts):
            if tok.startswith(" ") or tok.startswith("\u2581") or not words:
                words.append([tok.strip().lstrip("\u2581"), t])
            else:
                words[-1][0] += tok
        words = [(w, t) for w, t in words if w]
        return _clean(r.text), words

    def _try_commit(self, a, words, prev, length):
        """Returns the number of words to commit now (0 = none)."""
        k = 0
        while k < len(prev) and k < len(words) and _norm(prev[k]) == _norm(words[k][0]):
            k += 1
        # the last agreed sentence end that already has a following word
        for j in range(min(k, len(words) - 1) - 1, -1, -1):
            if words[j][0][-1:] in ".?!" and j >= 2:
                return j + 1
        if length > self.COMMIT_FORCE and k >= 8:
            for j in range(min(k, len(words) - 1) - 1, 3, -1):
                if words[j][0][-1:] in ",;:":
                    return j + 1
            return k - 2
        return 0

    def _run(self):
        buf = np.zeros(0, dtype=np.float32)
        speech = False       # inside speech (for preview)
        last_preview = 0.0
        preview_gap = PREVIEW_EVERY
        self._hist = np.zeros(0, dtype=np.float32)   # raw audio of the last ~40 s
        self._hist_start = 0                          # absolute sample index of _hist[0]
        self._levels, self._g = [], 1.0
        sp_start = 0         # absolute sample where the current speech began
        cut = None           # absolute sample where uncommitted speech begins (after an early commit)
        prev = []            # previous preview's words (same start point)
        while True:
            item = self.q.get()
            if item is None:
                break
            self._hist = np.concatenate([self._hist, item])
            if len(self._hist) > SR * 40:
                drop = len(self._hist) - SR * 30
                self._hist = self._hist[drop:]
                self._hist_start += drop
            g = self._gain(item)
            buf = np.concatenate([buf, item * g if g != 1.0 else item])
            while len(buf) >= self.win:
                chunk, buf = buf[:self.win], buf[self.win:]
                self.vad.accept_waveform(np.clip(chunk, -1, 1) if g != 1.0 else chunk)
                self.samples_seen += self.win
                if self.vad.is_speech_detected() and not speech:
                    speech = True
                    sp_start = max(self._hist_start, self.samples_seen - int(0.5 * SR))
                while not self.vad.empty():
                    seg = self.vad.front
                    if cut is not None and cut > seg.start:
                        # part of this sentence group was already committed: finish the rest
                        end = seg.start + len(seg.samples) + int(self.cfg["pad_post"] * SR)
                        a = max(self._hist_start, cut)
                        samples = self._hist[a - self._hist_start:end - self._hist_start].astype(np.float32)
                        start = a / SR
                    else:
                        samples = self._segment_audio(seg)
                        start = seg.start / SR
                    self.vad.pop()
                    speech, cut, prev = False, None, []
                    text = self._decode(samples) if len(samples) > SR * 0.2 else ""
                    if text:
                        self.on_final(text, start)
                    else:
                        self.on_partial("")
            # live preview of the sentence in progress
            # (skipped while behind, so a slow computer catches up instead of lagging further)
            if (self.live_preview and speech and self.q.qsize() <= 2
                    and time.monotonic() - last_preview > preview_gap):
                a = max(self._hist_start, cut if cut is not None else sp_start)
                audio = self._hist[a - self._hist_start:][-int(20 * SR):]
                a = self._hist_start + len(self._hist) - len(audio)
                if len(audio) > SR * 0.5:
                    t0 = time.monotonic()
                    text, words = self._decode_words(audio)
                    if words:
                        n = self._try_commit(a, words, prev, len(audio) / SR)
                        if n:
                            sent = _clean(" ".join(w for w, _ in words[:n]))
                            nxt = words[n][1] if n < len(words) else len(audio) / SR
                            cut = a + int(max(0.0, nxt - 0.04) * SR)
                            self.on_final(sent, (a / SR) + max(0.0, words[0][1] - 0.1))
                            words = words[n:]
                            text = " ".join(w for w, _ in words)
                        prev = [w for w, _ in words]
                    self.on_partial(text)
                    # slow computer -> preview less often so final sentences are never delayed
                    preview_gap = max(PREVIEW_EVERY, 2.0 * (time.monotonic() - t0))
                last_preview = time.monotonic()
        # flush
        self.vad.flush()
        while not self.vad.empty():
            seg = self.vad.front
            if cut is not None and cut > seg.start:
                a = max(self._hist_start, cut)
                samples = self._hist[a - self._hist_start:seg.start + len(seg.samples) - self._hist_start]
                start = a / SR
            else:
                samples = self._segment_audio(seg)
                start = seg.start / SR
            cut = None
            self.vad.pop()
            text = self._decode(np.asarray(samples, dtype=np.float32)) if len(samples) > SR * 0.2 else ""
            if text:
                self.on_final(text, start)
