from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


SAMPLE_RATE = 24000
TARGET_LUFS = -16.0
PEAK_CEIL = 10 ** (-1.0 / 20.0)


def safe_name(text: str, fallback: str = "sample") -> str:
    name = re.sub(r"[^a-zA-Z0-9._ -]+", "_", (text or "").strip())
    name = re.sub(r"\s+", "_", name).strip("._- ")
    return name[:80] or fallback


def timestamped_name(prefix: str, suffix: str = ".wav") -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{safe_name(prefix, 'higgs')}_{stamp}{suffix}"


def normalize_audio(audio: np.ndarray, loudness: bool = True) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1 if x.shape[1] <= x.shape[0] else 0)
    if x.size == 0:
        return x
    if loudness:
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(SAMPLE_RATE)
            measured = meter.integrated_loudness(x)
            if np.isfinite(measured):
                gain = 10 ** ((TARGET_LUFS - measured) / 20.0)
                x = x * min(gain, 10.0)
        except Exception:
            rms = float(np.sqrt(np.mean(x**2)))
            if rms > 1e-6:
                x = x * min((10 ** (-20.0 / 20.0)) / rms, 10.0)
    peak = float(np.max(np.abs(x)))
    if peak > PEAK_CEIL:
        x = x * (PEAK_CEIL / peak)
    return x.astype(np.float32)


def dual_mono(audio: np.ndarray) -> np.ndarray:
    mono = normalize_audio(audio)
    if mono.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([mono, mono]).astype(np.float32)


def gradio_audio(audio: np.ndarray) -> np.ndarray:
    stereo = dual_mono(audio)
    if stereo.size == 0:
        return np.zeros((0, 2), dtype=np.int16)
    return np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)


def concatenate(chunks: Iterable[np.ndarray], sr: int = SAMPLE_RATE, gap_seconds: float = 0.35) -> np.ndarray:
    kept = [normalize_audio(c) for c in chunks if c is not None and len(c)]
    if not kept:
        return np.zeros(0, dtype=np.float32)
    silence = np.zeros(int(sr * float(gap_seconds)), dtype=np.float32)
    out = []
    for idx, chunk in enumerate(kept):
        if idx:
            out.append(silence)
        out.append(chunk)
    return normalize_audio(np.concatenate(out))


def save_wav(path: Path, sr: int, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), dual_mono(audio), int(sr))
    return path


def ensure_dual_mono_file(path: Path) -> bool:
    if not path.exists() or path.suffix.lower() not in {".wav", ".flac", ".ogg", ".mp3", ".m4a"}:
        return False
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        out = dual_mono(data)
        if data.ndim == 2 and data.shape[1] == 2:
            left = data[:, 0]
            right = data[:, 1]
            if np.allclose(left, right, atol=1e-5) and float(np.max(np.abs(data))) <= PEAK_CEIL + 1e-4:
                return False
        sf.write(str(path), out, sr)
        return True
    except Exception:
        return False


def split_long_text(text: str, mode: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if mode == "None":
        return [text]
    if mode == "Paragraph/Sentence Auto":
        return paragraph_sentence_split(text)
    if mode == "Periods":
        return split_by_periods(text)
    if mode == "Paragraphs":
        return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if mode == "Lines":
        return [p.strip() for p in text.splitlines() if p.strip()]
    if mode == "Speaker turns":
        chunks = []
        current = []
        for line in text.splitlines():
            if re.match(r"^\s*\[?SPEAKER\d+\]?", line, flags=re.IGNORECASE) and current:
                chunks.append("\n".join(current).strip())
                current = [line.strip()]
            elif line.strip():
                current.append(line.strip())
        if current:
            chunks.append("\n".join(current).strip())
        return chunks or [text]
    return [text]


def split_by_periods(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return [chunk.strip() for chunk in re.split(r"(?<=\.)\s+", text) if chunk.strip()]


def paragraph_sentence_split(text: str, max_chars: int = 120) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    for paragraph in paragraphs or [text]:
        paragraph = re.sub(r"\s+", " ", paragraph)
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?…])\s+", paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
    return chunks
