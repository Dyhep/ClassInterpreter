"""English -> Simplified Chinese. Local (OPUS-MT via CTranslate2) or Claude API."""
import json
import re
from pathlib import Path

from cloud import claude_chat
from models import model_path

try:
    import opencc  # opencc-python-reimplemented
    _t2s = opencc.OpenCC("t2s")
except Exception:
    _t2s = None


def to_simplified(s: str) -> str:
    return _t2s.convert(s) if _t2s else s


_CJK = "\u4e00-\u9fff"
_PUNCT = {",": "，", "?": "？", "!": "！", ":": "：", ";": "；"}


def tidy_zh(s: str) -> str:
    """Full-width punctuation next to Chinese text, and a final full stop."""
    s = s.strip()
    for a, b in _PUNCT.items():
        s = re.sub(rf"(?<=[{_CJK}])\s*{re.escape(a)}\s*", b, s)
        s = re.sub(rf"\s*{re.escape(a)}\s*(?=[{_CJK}])", b, s)
    s = re.sub(rf"(?<=[{_CJK}])\.(?!\d)", "。", s)
    if s and re.match(rf"[{_CJK}）)]", s[-1]):
        s += "。"
    return s


class LocalTranslator:
    def __init__(self, threads: int = 2):
        import ctranslate2
        import sentencepiece as spm
        d = model_path("translate")
        if d is None:
            raise RuntimeError("翻译模型未安装")
        d = Path(d)
        meta = {}
        if (d / "ci_meta.json").exists():
            meta = json.loads((d / "ci_meta.json").read_text(encoding="utf-8"))
        self.prefix = meta.get("prefix", [">>cmn_Hans<<"])
        self.suffix = meta.get("suffix", ["</s>"])
        self.tr = ctranslate2.Translator(str(d), device="cpu", compute_type="int8",
                                         inter_threads=1, intra_threads=threads)
        self.sp_src = spm.SentencePieceProcessor(model_file=str(d / "source.spm"))
        self.sp_tgt = spm.SentencePieceProcessor(model_file=str(d / "target.spm"))

    def translate(self, text: str, context=None) -> str:
        pieces = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()] or [text]
        batch = [self.prefix + self.sp_src.encode(p, out_type=str) + self.suffix for p in pieces]
        res = self.tr.translate_batch(batch, beam_size=2, max_decoding_length=256,
                                      repetition_penalty=1.2)
        out = []
        for r in res:
            toks = [t for t in r.hypotheses[0] if not (t.startswith(">>") and t.endswith("<<"))]
            out.append(self.sp_tgt.decode_pieces(toks))
        s = "".join(out)
        s = re.sub(r"(?<=[　-〿一-鿿＀-￯])\s+|\s+(?=[　-〿一-鿿＀-￯])", "", s)
        return tidy_zh(to_simplified(s))


SYSTEM_TRANSLATE = (
    "你是课堂同声传译。把用户给出的英文（课堂或会议的语音转写，可能有识别错误）翻译成自然、准确的简体中文。"
    "专业术语第一次出现时在中文后用括号保留英文原词。只输出译文，不要解释。"
)


class CloudTranslator:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def translate(self, text: str, context=None) -> str:
        msg = text
        if context:
            msg = "（前文，仅供参考，不要翻译）\n" + "\n".join(context[-3:]) + "\n\n（请翻译）\n" + text
        return claude_chat(self.api_key, self.model, SYSTEM_TRANSLATE, msg, max_tokens=600).strip()
