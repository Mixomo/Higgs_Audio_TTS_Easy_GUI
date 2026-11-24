"""Checks of a generated take, carried over from Speech Studio.

* ``defect``: the synthesis went wrong as a whole (runaway, cut off, silence,
  hum, clipping) -- the take is redone.
* ``sound_flags``: cheap measures of the sound (clipping, little speech, a long
  inner pause, a cut edge, noise in the pauses, length and loudness against the
  voice's usual) plus two costly ones (a repeat by self-similarity, a whistle).
  Each flag has a role: a defect (redo), a suspect (recognition decides) or a
  note (shown only).
* ``check_text``: what recognition heard against the text, as a WER with a
  threshold that depends on the length and the strictness.
* ``VoiceEmbedder``: WeSpeaker ResNet34-LM (ONNX), the cosine of the take's
  voice and its reference.
"""

from __future__ import annotations

import json
import math
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .paths import CACHE_DIR, MODELS_DIR

SR = 24_000

# ---------------------------------------------------------------- catalogue
SOUND_CHECKS = {
    "defect": "Synthesis failure: ran on, cut off, silence, hum",
    "clipping": "Clipping",
    "length": "Longer or shorter than the text for this voice",
    "edge": "Cut off at an edge",
    "long_pause": "An inner pause longer than 1.2 s",
    "little_speech": "Little speech (silence padding)",
    "loudness": "Louder or quieter than the voice's usual (note)",
    "pause_noise": "Noise in the pauses (note)",
    "audio_repeat": "Repeated sound (slower)",
    "whistle": "Whistle (slower)",
}
CHEAP_CHECKS = ["defect", "clipping", "length", "edge", "long_pause", "little_speech", "loudness", "pause_noise"]
COSTLY_CHECKS = ["audio_repeat", "whistle"]

ISSUE_NAMES = {
    "runaway": "ran on past the text",
    "cutoff": "cut off",
    "silence": "silence",
    "hum": "hum",
    "clipping": "clipping",
    "click": "click at the end",
    "little_speech": "little speech",
    "long_pause": "long inner pause",
    "edge": "cut at an edge",
    "pause_noise": "noise in the pauses",
    "too_long": "too long for the text",
    "too_short": "too short for the text",
    "loud": "louder than usual",
    "quiet": "quieter than usual",
    "whistle": "whistle",
    "audio_repeat": "repeated sound",
    "text": "words differ from the text",
    "unlike": "voice unlike its reference",
    "repeat": "repeated words",
}

DEFECT_ROLE = {"runaway", "cutoff", "silence", "hum", "click", "clipping", "whistle"}
NOTE_ROLE = {"loud", "quiet", "pause_noise"}

# Quality presets (Speech Studio): draft = one unchecked take, standard = every take
# heard once, optimal = three takes, strict text check and voice similarity.
QUALITY_PRESETS = {
    "Draft": {"asr_check": False, "takes": 1, "similarity": False, "strictness": "Normal"},
    "Standard": {"asr_check": True, "takes": 1, "similarity": False, "strictness": "Normal"},
    "Optimal": {"asr_check": True, "takes": 3, "similarity": True, "strictness": "Strict"},
}

TAG_RE = re.compile(r"<\|[^|>]*\|>|</?SE[^>]*>|\[[^\]]*\]|<[^>]{1,40}>")


def visible(text: str) -> str:
    """The words that are said: without tags and markup."""
    return TAG_RE.sub(" ", text or "")


def is_cjk(c: str) -> bool:
    o = ord(c)
    return 0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF or 0xF900 <= o <= 0xFAFF


def syllables(text: str) -> int:
    vowels = set("aeiouyàáâãäåèéêëìíîïòóôõöùúûüýæœаеёиоуыэюяіїє")
    return sum(1 for c in visible(text).lower() if c in vowels or is_cjk(c))


# ---------------------------------------------------------------- the sound
def _db(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-9))


def trim_edges(x: np.ndarray, sr: int = SR, floor_db: float = -45.0, pad: float = 0.04) -> tuple[int, int]:
    """First and last sample of the speech (10 ms frames louder than the peak + floor)."""
    if len(x) == 0:
        return 0, 0
    w = max(1, sr // 100)
    n = len(x) // w
    if n == 0:
        return 0, len(x)
    rms = np.sqrt(np.mean(x[: n * w].reshape(n, w).astype(np.float64) ** 2, axis=1))
    peak = rms.max()
    if peak <= 1e-6:
        return 0, 0
    loud = np.nonzero(rms >= peak * 10 ** (floor_db / 20))[0]
    a = max(0, int(loud[0] * w - pad * sr))
    b = min(len(x), int((loud[-1] + 1) * w + pad * sr))
    return a, b


def runaway_secs(chars: int) -> float:
    return chars * 0.2 + 3.0


def auto_max_frames(text: str, frames_per_sec: float = 25.0) -> int:
    """Max frames when the setting is 0 (auto): a second past the runaway length."""
    chars = len(visible(text).strip())
    return int((runaway_secs(chars) + 1.0) * frames_per_sec)


def defect(x: np.ndarray, sr: int, text: str) -> str | None:
    """Speech Studio's synthesis defect guard (tts_host::defect)."""
    if sr <= 0 or len(x) == 0:
        return "silence"
    chars = len(visible(text).strip())
    dur = len(x) / sr
    if chars >= 1 and dur > runaway_secs(chars):
        return "runaway"
    if chars >= 4 and dur < chars * 0.045:
        return "cutoff"
    if np.mean(np.abs(x) >= 0.999) >= 0.01:
        return "clipping"
    w = sr // 40
    if w == 0 or len(x) < w * 4:
        return None
    n = len(x) // w
    rms = np.sqrt(np.mean(x[: n * w].reshape(n, w).astype(np.float64) ** 2, axis=1))
    peak = float(rms.max())
    if peak < 0.02:
        return "silence"
    silent = float(np.mean(rms < peak * 0.0316))
    nz = rms[rms > 1e-9]
    trough = float(nz.min()) if len(nz) else peak
    if _db(peak / max(trough, 1e-9)) < 16.0 and silent < 0.03:
        return "hum"
    return None


def tail_click(x: np.ndarray, sr: int) -> float:
    n30 = sr * 30 // 1000
    n200 = sr * 200 // 1000
    if len(x) < n30 + n200:
        return 0.0
    tail = x[-n30:]
    before = x[-(n30 + n200) : -n30]
    step = float(np.max(np.abs(np.diff(tail)))) if len(tail) > 1 else 0.0
    rms = float(np.sqrt(np.mean(before.astype(np.float64) ** 2)))
    return step / max(rms, 1e-6)


def soften_tail_click(x: np.ndarray, sr: int) -> bool:
    """A click at the very end is faded out (30 ms) instead of throwing the take away."""
    if tail_click(x, sr) <= 4.0:
        return False
    n = min(sr * 30 // 1000, len(x))
    x[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return True


def lufs(x: np.ndarray, sr: int) -> float | None:
    try:
        import pyloudnorm as pyln

        if len(x) < sr * 0.5:
            return None
        value = pyln.Meter(sr).integrated_loudness(x.astype(np.float64))
        return float(value) if np.isfinite(value) else None
    except Exception:
        return None


@dataclass
class Features:
    raw_s: float = 0.0
    trimmed_s: float = 0.0
    speech_frac: float = 0.0
    clip_share: float = 0.0
    lufs: float | None = None
    max_pause_s: float = 0.0
    edge_db: float = -99.0
    noise_db: float | None = None
    tonal_frac: float = 0.0
    rep_s: float = 0.0


def measure(x: np.ndarray, sr: int, whistle: bool = False, repeat: bool = False) -> Features:
    f = Features(raw_s=len(x) / sr)
    if len(x) == 0:
        return f
    a, b = trim_edges(x, sr)
    f.trimmed_s = (b - a) / sr
    f.clip_share = float(np.mean(np.abs(x) >= 0.999))
    f.lufs = lufs(x, sr)
    w = max(1, sr // 100)
    n = len(x) // w
    if n == 0:
        return f
    rms = np.sqrt(np.mean(x[: n * w].reshape(n, w).astype(np.float64) ** 2, axis=1))
    peak = float(rms.max())
    thr = peak * 10 ** (-35 / 20)
    loud = (rms >= thr) & (rms > 1e-6)
    f.speech_frac = float(loud.mean())
    idx = np.nonzero(loud)[0]
    quiet_frames = []
    if len(idx):
        lo, hi = idx[0], idx[-1]
        run = longest = 0
        for i in range(lo, hi + 1):
            if loud[i]:
                run = 0
            else:
                run += 1
                longest = max(longest, run)
                quiet_frames.append(i)
        f.max_pause_s = longest * w / sr
    n20 = min(sr // 50, len(x))
    n5 = min(sr // 200, len(x))
    peak_abs = float(np.max(np.abs(x)))
    edge = max(float(np.max(np.abs(x[-n20:]))), float(np.max(np.abs(x[:n5]))))
    f.edge_db = _db(edge / max(peak_abs, 1e-9))
    if quiet_frames:
        f.noise_db = _db(float(np.median(rms[quiet_frames])))
    if not (whistle or repeat):
        return f
    nfft, hop = 1024, 512
    limit = min(len(x), sr * 60)
    starts = np.arange(0, max(0, limit - nfft), hop)
    if len(starts) == 0:
        return f
    win = np.hanning(nfft)
    frames = np.stack([x[s : s + nfft] * win for s in starts])
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    frame_loud = np.array([loud[min(len(loud) - 1, (s + nfft // 2) // w)] for s in starts])
    if whistle:
        lo_b = int(2000 * nfft / sr)
        hi_b = min(int(10000 * nfft / sr), nfft // 2 - 1)
        peaks: dict[int, int] = {}
        counted = 0
        for p, is_loud in zip(spec, frame_loud):
            if not is_loud or hi_b <= lo_b:
                continue
            counted += 1
            band = p[lo_b : hi_b + 1]
            med = max(float(np.median(band)), 1e-12)
            bi = int(np.argmax(band))
            if 10 * math.log10(float(band[bi]) / med) >= 30:
                peaks[bi // 2] = peaks.get(bi // 2, 0) + 1
        if counted:
            f.tonal_frac = max(peaks.values(), default=0) / counted
    if repeat:
        import librosa

        mel = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=40, fmin=80, fmax=min(8000, sr // 2))
        m = np.log(np.maximum(spec @ mel.T, 1e-10))
        m = m - m.mean(axis=1, keepdims=True)
        m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        m[~frame_loud] = 0
        frame_s = hop / sr
        min_lag = int(math.ceil(0.4 / frame_s))
        best = 0
        nfr = len(m)
        for lag in range(min_lag, nfr):
            same = np.sum(m[: nfr - lag] * m[lag:], axis=1) >= 0.93
            if not same.any():
                continue
            # longest run of True
            padded = np.concatenate([[0], same.astype(np.int8), [0]])
            d = np.diff(padded)
            runs = np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]
            best = max(best, int(runs.max()))
        f.rep_s = best * frame_s
    return f


def _text_repeats(text: str) -> bool:
    words = [w[:4] for w in re.split(r"\W+", visible(text).lower().replace("ё", "е")) if len(w) >= 3]
    return len(words) != len(set(words))


def sound_flags(f: Features, want: list[str], text: str, norm: dict | None) -> list[str]:
    on = set(want)
    out: list[str] = []
    syl = syllables(text)
    if "clipping" in on and f.clip_share >= 0.01:
        out.append("clipping")
    if "little_speech" in on and f.raw_s >= 3.0 and f.speech_frac < 0.3:
        out.append("little_speech")
    if "long_pause" in on and f.max_pause_s >= 1.2:
        out.append("long_pause")
    if "edge" in on and f.edge_db >= -20.0:
        out.append("edge")
    if "pause_noise" in on and f.max_pause_s >= 0.25 and f.noise_db is not None and f.noise_db >= -45.0:
        out.append("pause_noise")
    rate = (norm or {}).get("secs_per_syllable")
    if "length" in on and rate:
        expected = max(syl, 1) * rate + 0.3
        ratio = f.trimmed_s / expected
        if ratio >= 1.3 and f.trimmed_s - expected >= 0.8 and syl >= 4:
            out.append("too_long")
        elif ratio < 0.6 and expected >= 1.5:
            out.append("too_short")
    usual = (norm or {}).get("lufs")
    if "loudness" in on and f.lufs is not None and usual is not None and abs(f.lufs - usual) >= 6.0:
        out.append("loud" if f.lufs > usual else "quiet")
    if "whistle" in on and f.trimmed_s >= 3.0 and f.tonal_frac >= 0.3:
        out.append("whistle")
    if "audio_repeat" in on and f.rep_s >= 0.2 and syl >= 4 and not _text_repeats(text):
        out.append("audio_repeat")
    return out


def role(flag: str) -> str:
    if flag in DEFECT_ROLE:
        return "defect"
    if flag in NOTE_ROLE:
        return "note"
    return "suspect"


# ---------------------------------------------------------------- the words
def normalize_words(text: str) -> list[str]:
    s = unicodedata.normalize("NFKC", visible(text)).lower().replace("ё", "е")
    words: list[str] = []
    cur = ""
    for c in s:
        if is_cjk(c):
            if cur:
                words.append(cur)
                cur = ""
            words.append(c)
        elif c.isalnum() or c == "'":
            cur += c
        elif cur:
            words.append(cur)
            cur = ""
    if cur:
        words.append(cur)
    out: list[str] = []
    for w in words:
        lead = re.match(r"\d+", w)
        if lead and w[lead.end():].isalpha():
            w = lead.group(0)
        if out and out[-1].isdigit() and w.isdigit():
            out[-1] += w
        else:
            out.append(w)
    # accents do not change the word for the check ("esta" / "está" heard alike)
    return ["".join(ch for ch in unicodedata.normalize("NFD", w) if unicodedata.category(ch) != "Mn") for w in out]


def word_ops(expected: list[str], heard: list[str]) -> tuple[int, int, int]:
    n, m = len(expected), len(heard)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if expected[i - 1] == heard[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    i, j = n, m
    ins = dele = sub = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if expected[i - 1] == heard[j - 1] else 1):
            sub += expected[i - 1] != heard[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return ins, dele, sub


def threshold(words: int, strictness: str) -> float:
    base, slack = {"strict": (0.10, 1.0), "lenient": (0.35, 3.0)}.get(strictness.lower(), (0.20, 2.0))
    return min(0.6, base + slack / max(words, 1))


def repeated(expected: list[str], heard: list[str]) -> bool:
    """Recognition heard a run of words twice that the text says once."""
    for size in (3, 2):
        for i in range(len(heard) - 2 * size + 1):
            a, b = heard[i : i + size], heard[i + size : i + 2 * size]
            if a == b:
                text = " ".join(expected)
                if text.count(" ".join(a) + " " + " ".join(a)) == 0:
                    return True
    return False


def check_text(expected: str, heard: str, strictness: str) -> dict:
    e, h = normalize_words(expected), normalize_words(heard)
    ins, dele, sub = word_ops(e, h)
    wer = (0.0 if not h else 1.0) if not e else (ins + dele + sub) / len(e)
    return {"wer": wer, "words": len(e), "inserted": ins, "deleted": dele, "substituted": sub,
            "passed": wer <= threshold(len(e), strictness), "repeat": repeated(e, h)}


# ---------------------------------------------------------------- the voice
WESPEAKER_URL = (
    "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/"
    "f0c48c298fd835726c27956a5d617bad7115627e/voxceleb_resnet34_LM.onnx"
)
WESPEAKER_PATH = MODELS_DIR / "wespeaker" / "voxceleb_resnet34_LM.onnx"
WESPEAKER_SIZE = 26_530_309
WESPEAKER_SHA256 = "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"


def wespeaker_installed() -> bool:
    return WESPEAKER_PATH.is_file() and WESPEAKER_PATH.stat().st_size == WESPEAKER_SIZE


def ensure_wespeaker(progress=None) -> Path:
    if wespeaker_installed():
        return WESPEAKER_PATH
    from .downloads import download_file

    return download_file(WESPEAKER_URL, WESPEAKER_PATH, WESPEAKER_SIZE, WESPEAKER_SHA256, progress=progress)


class VoiceEmbedder:
    """WeSpeaker ResNet34-LM on ONNX Runtime: a 256-d voice print of 16 kHz mono speech."""

    def __init__(self):
        self.session = None
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[float, np.ndarray]] = {}

    def load(self):
        if self.session is None:
            import onnxruntime as ort

            path = ensure_wespeaker()
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        return self.session

    def unload(self):
        self.session = None
        self.cache.clear()

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import torch
        import torchaudio

        with self.lock:
            session = self.load()
            x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
            if sr != 16000:
                x = torchaudio.functional.resample(x, sr, 16000)
            a, b = trim_edges(x[0].numpy(), 16000)
            if b - a > 16000:
                x = x[:, a:b]
            feats = torchaudio.compliance.kaldi.fbank(
                x * (1 << 15), num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                sample_frequency=16000, window_type="hamming", use_energy=False,
            )
            feats = feats - feats.mean(dim=0, keepdim=True)
            name = session.get_inputs()[0].name
            out = session.run(None, {name: feats.unsqueeze(0).numpy().astype(np.float32)})[0][0]
            return (out / max(float(np.linalg.norm(out)), 1e-9)).astype(np.float32)

    def embed_file(self, path: str) -> np.ndarray:
        import soundfile as sf

        p = Path(path)
        key = str(p.resolve())
        mtime = p.stat().st_mtime
        hit = self.cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
        data, sr = sf.read(key, dtype="float32", always_2d=True)
        e = self.embed(data.mean(axis=1), sr)
        self.cache[key] = (mtime, e)
        return e


EMBEDDER = VoiceEmbedder()


# ---------------------------------------------------------------- voice norms
NORMS_FILE = CACHE_DIR / "voice_norms.json"
_norm_lock = threading.Lock()


def _read_norms() -> dict:
    try:
        return json.loads(NORMS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def voice_norm(voice: str) -> dict | None:
    """The voice's usual pace and loudness: medians of its last checked takes (3 at least)."""
    with _norm_lock:
        entry = _read_norms().get(voice) or {}
    rates = entry.get("rates", [])
    louds = entry.get("lufs", [])
    if len(rates) < 3:
        return None
    return {"secs_per_syllable": float(np.median(rates)), "lufs": float(np.median(louds)) if len(louds) >= 3 else None}


def remember_take(voice: str, f: Features, text: str) -> None:
    syl = syllables(text)
    if not voice or syl < 4 or f.trimmed_s <= 0:
        return
    with _norm_lock:
        norms = _read_norms()
        entry = norms.setdefault(voice, {"rates": [], "lufs": []})
        entry["rates"] = (entry["rates"] + [f.trimmed_s / syl])[-40:]
        if f.lufs is not None:
            entry["lufs"] = (entry["lufs"] + [f.lufs])[-40:]
        NORMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        NORMS_FILE.write_text(json.dumps(norms), encoding="utf-8")


@dataclass
class TakeReport:
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    wer: float | None = None
    heard: str = ""
    similarity: float | None = None
    similar: bool | None = None
    passed_text: bool = True
    features: Features | None = None

    @property
    def ok(self) -> bool:
        return self.passed_text and self.similar is not False and not self.issues

    def rank(self) -> tuple:
        return (not self.ok, len(self.issues), self.wer or 0.0, -(self.similarity or 0.0))

    def summary(self) -> str:
        parts = []
        if self.wer is not None:
            parts.append(f"WER {self.wer:.0%}")
        if self.similarity is not None:
            parts.append(f"voice {self.similarity:.2f}")
        if self.issues:
            parts.append("issues: " + ", ".join(ISSUE_NAMES.get(i, i) for i in self.issues))
        if self.notes:
            parts.append("notes: " + ", ".join(ISSUE_NAMES.get(i, i) for i in self.notes))
        return ("OK" if self.ok else "REJECTED") + (" · " + " · ".join(parts) if parts else "")
