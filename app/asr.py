"""English speech -> text with sherpa-onnx (voice activity detection + offline model)."""
import difflib
import queue
import threading
import time

import numpy as np
import sherpa_onnx

from models import model_path

SR = 16000
LOOKBACK = 0.3  # s before detected speech that is included (if not already finished)
CUT_MARGIN = 0.1  # s before the next word's first sound where pieces are cut
RIGHT_CONTEXT = int(0.6 * 16000)  # audio heard after a sentence's end before finishing it
PREVIEW_EVERY = 0.4  # seconds between live-preview decodes (at least)

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

    # ---- committing whole sentences while the speaker keeps going ----
    # Fast speakers (news, policy talks) barely pause, so voice detection alone would wait 10+ s
    # before ending a "sentence". When two consecutive previews agree on a complete sentence
    # (and the speaker has already moved on to the next one), that sentence is finished early,
    # using the model's word timings to cut the audio exactly between the two sentences.
    COMMIT_FORCE = 12.0   # s of uncommitted speech before committing at a comma/agreed words

    def _decode_words(self, samples, a=None):
        """[(word, start_sec)] using token timings, or None if the model has none."""
        s = self.rec.create_stream()
        s.accept_waveform(SR, samples)
        self.rec.decode_stream(s)
        r = s.result
        toks, ts = list(getattr(r, "tokens", []) or []), list(getattr(r, "timestamps", []) or [])
        if not toks or len(toks) != len(ts):
            self._timed = False
            return _clean(r.text), None
        self._timed = True
        words = []
        for tok, t in zip(toks, ts):
            if tok.startswith(" ") or tok.startswith("\u2581") or not words:
                words.append([tok.strip().lstrip("\u2581"), t])
            else:
                words[-1][0] += tok
        words = [(w, t) for w, t in words if w]
        # the cut sits just before the next word, so the tail of the previous word can be heard
        # again at the start of the next piece: drop that echo
        if (a is not None and a == self._edge[0] and words and words[0][1] < 0.3
                and _norm(words[0][0]) == self._edge[1]):
            words = words[1:]
            return _clean(" ".join(w for w, _ in words)), words
        return _clean(r.text), words

    def _cut_after(self, a, words, n):
        """Absolute sample where the next piece starts, just before words[n]; remembers the edge."""
        cut = a + int(max(0.0, words[n][1] - CUT_MARGIN) * SR)
        self._edge = (cut, _norm(words[n - 1][0]) if n else "")
        return cut

    def _try_commit(self, a, words, prev, length):
        """Returns the number of words to commit now (0 = none).
        A sentence counts as settled when the previous preview heard (nearly) the same words: one word
        that keeps changing ("counter"/"counted") must not hold up everything after it."""
        if not prev:
            return 0
        cur = [_norm(w) for w, _ in words]
        sm = difflib.SequenceMatcher(None, [_norm(w) for w in prev], cur, autojunk=False)
        ok = [False] * len(cur)
        for blk in sm.get_matching_blocks():
            for t in range(blk.size):
                ok[blk.b + t] = True
        good, settled = 0, []
        for j in range(len(cur)):
            good += ok[j]
            settled.append(ok[j] and good >= 0.75 * (j + 1))
        last = len(words) - 1            # keep at least one word after the cut
        for j in range(last - 1, 1, -1):
            if settled[j] and words[j][0][-1:] in ".?!":
                return j + 1
        if length > self.COMMIT_FORCE:
            cands = [j for j in range(3, last - 1) if settled[j]]
            for j in reversed(cands):
                if words[j][0][-1:] in ",;:":
                    return j + 1
            if cands:
                return cands[-1] + 1
        return 0

    def _finish(self, a, target, hold=0, min_words=1, fallback=None):
        """Finish the speech from absolute sample `a` up to about `target`, cutting only between words.
        The model hears all the audio there is (right context helps the last words), and the words
        that start before `target` (minus the last `hold` words) become a finished sentence.
        Returns where the next piece starts, or None if nothing was finished."""
        now = self._hist_start + len(self._hist)
        x = self._hist[max(0, a - self._hist_start):].astype(np.float32)
        if len(x) < SR * 0.2:
            return None
        text, words = self._decode_words(x, a)
        if words is None:                       # model without word timings: cut where told
            if hold:
                if fallback is None:
                    return None
                target = fallback
            x = self._hist[max(0, a - self._hist_start):max(0, target - self._hist_start)]
            text = self._decode(x.astype(np.float32)) if len(x) > SR * 0.2 else ""
            if text:
                self.on_final(text, a / SR)
            return target
        tcut = (target - a) / SR
        n = sum(1 for _, t in words if t < tcut)
        if hold:
            n = min(n, len(words) - hold)
        if n < min_words:
            return None
        self.on_final(_clean(" ".join(w for w, _ in words[:n])), a / SR + max(0.0, words[0][1] - 0.1))
        if n < len(words):
            self.on_partial(" ".join(w for w, _ in words[n:]))
            return self._cut_after(a, words, n)
        self.on_partial("")
        return min(target, now)

    def _run(self):
        """Every bit of audio is decoded exactly once: `done` marks how far the audio has been turned
        into finished sentences, and each new piece starts exactly there. Voice detection only decides
        where to cut (in pauses), so a soft or fast word it misses is still heard by the recognizer."""
        buf = np.zeros(0, dtype=np.float32)
        last_preview = 0.0
        preview_gap = PREVIEW_EVERY
        self._hist = np.zeros(0, dtype=np.float32)   # raw audio of the last ~40 s
        self._hist_start = 0                          # absolute sample index of _hist[0]
        self._levels, self._g = [], 1.0
        done = 0             # absolute sample up to which audio is finished
        self._edge = (-1, "")  # (cut position, last word before it)
        self._timed = True     # the model gives word timings (cleared on the first decode if not)
        pend = None          # (start, end) of a sentence that ended, waiting for a little right context
        prev = []            # previous preview's words (for early commits)
        pad_post = int(self.cfg["pad_post"] * SR)
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
                while not self.vad.empty():
                    seg = self.vad.front
                    end = min(seg.start + len(seg.samples) + pad_post, self.samples_seen)
                    self.vad.pop()
                    # start where the last sentence ended (catching anything voice detection missed),
                    # but don't drag in a long silence
                    now = self._hist_start + len(self._hist)
                    if pend:
                        # the previous sentence is still waiting: what was just heard is its context
                        cut = self._finish(max(done, self._hist_start, pend[0]), pend[1])
                        done = max(done, cut if cut is not None else pend[1])
                        pend = None
                    a = max(done, self._hist_start, seg.start - (int(LOOKBACK * SR) if self._timed else 0))
                    if self.vad.is_speech_detected() and now - a < 25 * SR:
                        # the speaker is still going (a long sentence, or a pause too short to matter):
                        # a cut here can land mid-word and lose the words around it, so finish only up
                        # to a word boundary and leave the rest for the next piece
                        cut = self._finish(a, now, hold=2, min_words=3, fallback=end)
                        if cut is not None:
                            done, prev = cut, []
                        continue
                    if end > a:
                        # finish a moment later, so the recognizer also hears what follows the cut:
                        # without it, words right at the end (that voice detection missed) get dropped
                        pend = (a, end)
                    prev = []
            now = self._hist_start + len(self._hist)
            talking = self.vad.is_speech_detected()
            if pend and now >= pend[1] + RIGHT_CONTEXT:
                a = max(done, self._hist_start, pend[0])
                cut = self._finish(a, pend[1])
                done = max(done, cut if cut is not None else pend[1])
                pend, prev = None, []
            if not pend and not talking and self.vad.empty() and now - done > 2 * SR:
                # voice detection hears nothing for 2 s. It can be wrong (echo, a tablet speaker, a
                # soft voice), so let the recognizer check before skipping that audio
                cut = self._finish(done, now, hold=1)
                done = cut if cut is not None else now - SR // 2
                prev = []
            # live preview of the speech not finished yet
            # (skipped while behind, so a slow computer catches up instead of lagging further)
            if (self.live_preview and (talking or prev) and self.q.qsize() <= 2
                    and time.monotonic() - last_preview > preview_gap):
                a = max(done, self._hist_start, now - 20 * SR)
                audio = self._hist[a - self._hist_start:]
                if len(audio) > SR * 0.35:
                    t0 = time.monotonic()
                    text, words = self._decode_words(audio, a)
                    if words:
                        n = self._try_commit(a, words, prev, len(audio) / SR)
                        if n:
                            sent = _clean(" ".join(w for w, _ in words[:n]))
                            self.on_final(sent, a / SR + max(0.0, words[0][1] - 0.1))
                            done = self._cut_after(a, words, n)
                            words = words[n:]
                            text = " ".join(w for w, _ in words)
                        prev = [w for w, _ in words]
                    self.on_partial(text)
                    # slow computer -> preview less often so final sentences are never delayed
                    preview_gap = max(PREVIEW_EVERY, 1.5 * (time.monotonic() - t0))
                last_preview = time.monotonic()
        # flush: whatever was said after the last finished sentence
        if pend:
            cut = self._finish(max(done, self._hist_start, pend[0]), pend[1])
            done = max(done, cut if cut is not None else pend[1])
        self.vad.flush()
        end = self._hist_start + len(self._hist)
        last = None
        while not self.vad.empty():
            seg = self.vad.front
            last = seg.start + len(seg.samples)
            self.vad.pop()
        if last is not None and last > done:
            self._finish(max(done, self._hist_start), min(end, last + pad_post))
