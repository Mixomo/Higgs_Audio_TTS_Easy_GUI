"""Dataset audio front-end: voice isolation (Inworld Voice Isolator) and natural
segmentation (Inworld SAVA) for building Higgs TTS training data.

Pipeline per source recording:
    load -> mono -> [isolate voice] -> [SAVA speech/turn curves -> segments]
         -> per-segment loudness normalisation -> 24 kHz mono WAV clips

SAVA closes a turn at end-of-turn or long pauses but has no maximum length, so
long turns are split here at the deepest internal pause, tiny fragments are
merged with their neighbours, and leftovers shorter than ``min_sec`` dropped.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

TARGET_SR = 24000
SAVA_SR = 16000
FRAME = 0.04  # SAVA frame length in seconds


@dataclass
class SegmentOptions:
    isolate: bool = True
    segment: bool = True
    min_sec: float = 2.0
    max_sec: float = 15.0
    target_lufs: float = -20.0
    pad_sec: float = 0.12


@dataclass
class Clip:
    path: Path
    source: Path
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or not len(audio):
        return audio.astype(np.float32, copy=False)
    import soxr

    return soxr.resample(audio, sr_in, sr_out).astype(np.float32)


def _normalize(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Loudness-normalise one clip, never letting peaks exceed -1 dBFS."""
    if not len(audio):
        return audio
    gain = 1.0
    if len(audio) >= int(0.5 * sr):
        try:
            import pyloudnorm

            loudness = pyloudnorm.Meter(sr).integrated_loudness(audio.astype(np.float64))
            if np.isfinite(loudness):
                gain = 10.0 ** ((target_lufs - loudness) / 20.0)
        except Exception:
            pass
    peak = float(np.max(np.abs(audio))) * gain
    ceiling = 10.0 ** (-1.0 / 20.0)
    if peak > ceiling:
        gain *= ceiling / peak
    return (audio * gain).astype(np.float32)


class DatasetAudioProcessor:
    def __init__(self, device: str | None = None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._isolator = None
        self._sava = None

    # ------------------------------------------------------------------ models
    def isolator(self):
        if self._isolator is None:
            from inworld import voice_isolation

            print(f"[dataset] Loading Inworld Voice Isolator on {self.device}", flush=True)
            self._isolator = voice_isolation.VoiceIsolator("inworld/voice-isolator", device=self.device)
        return self._isolator

    def sava(self):
        if self._sava is None:
            from inworld import sava

            print(f"[dataset] Loading Inworld SAVA on {self.device}", flush=True)
            self._sava = sava.load("inworld/sava", device=self.device)
        return self._sava

    def unload(self):
        self._isolator = None
        self._sava = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------ steps
    def isolate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch

        with torch.inference_mode():
            out = self.isolator().isolate(audio, sr, output_sample_rate=sr)
        if isinstance(out, torch.Tensor):
            out = out.detach().float().cpu().numpy()
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def segment(self, audio: np.ndarray, sr: int, opts: SegmentOptions) -> list[tuple[float, float]]:
        """Natural speech segments (seconds) of ``min_sec``..``max_sec``."""
        import torch
        from inworld import sava

        audio16 = _resample(audio, sr, SAVA_SR)
        model = self.sava()
        with torch.inference_mode():
            curves = sava.run(model, audio16)
        speech = np.asarray(curves.speech, dtype=np.float32)
        # Short pre-roll: the default 800 ms would add silence to every clip.
        policy = sava.Policy(pre_roll_ms=160, max_silence_ms=900, min_speech_ms=500)
        turns = [t for t in sava.segment(curves, policy, model.lookahead_frames) if t.kept]
        total = len(audio) / float(sr)
        spans: list[tuple[float, float]] = []
        for turn in turns:
            spans.extend(self._split_long(turn.start_seconds, min(turn.end_seconds, total), speech, opts))
        spans = self._merge_short(spans, opts)
        out = []
        for i, (start, end) in enumerate(spans):
            if end - start < opts.min_sec:
                continue
            prev_end = out[-1][1] if out else 0.0
            next_start = spans[i + 1][0] if i + 1 < len(spans) else total
            start = max(prev_end, start - opts.pad_sec, 0.0)
            end = min(next_start, end + opts.pad_sec, total)
            out.append((round(start, 3), round(end, 3)))
        return out

    @staticmethod
    def _split_long(start: float, end: float, speech: np.ndarray, opts: SegmentOptions) -> list[tuple[float, float]]:
        spans = []
        while end - start > opts.max_sec:
            lo = int((start + opts.min_sec) / FRAME)
            hi = min(int((start + opts.max_sec) / FRAME), len(speech) - 1)
            if hi <= lo:
                break
            window = speech[lo:hi]
            # Deepest pause: lowest smoothed speech probability, ties -> later cut.
            smooth = np.convolve(window, np.ones(5) / 5.0, mode="same")
            best = int(len(smooth) - 1 - np.argmin(smooth[::-1]))
            cut = (lo + best) * FRAME
            spans.append((start, cut))
            start = cut
        spans.append((start, end))
        return spans

    @staticmethod
    def _merge_short(spans: list[tuple[float, float]], opts: SegmentOptions) -> list[tuple[float, float]]:
        merged: list[tuple[float, float]] = []
        for start, end in spans:
            if merged:
                p_start, p_end = merged[-1]
                gap = start - p_end
                short = (end - start) < opts.min_sec or (p_end - p_start) < opts.min_sec
                if short and gap <= 0.8 and (end - p_start) <= opts.max_sec:
                    merged[-1] = (p_start, end)
                    continue
            merged.append((start, end))
        return merged

    # ------------------------------------------------------------------ files
    def process_file(
        self,
        src: Path,
        out_dir: Path,
        opts: SegmentOptions,
        keep_whole: bool = False,
    ) -> list[Clip]:
        """Clean/segment one recording into 24 kHz clips written under ``out_dir``."""
        audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
        audio = _mono(audio)
        if not len(audio):
            return []
        if opts.isolate:
            audio = self.isolate(audio, int(sr))
        audio = _resample(audio, int(sr), TARGET_SR)
        total = len(audio) / float(TARGET_SR)
        if opts.segment and not keep_whole:
            spans = self.segment(audio, TARGET_SR, opts)
        else:
            spans = [(0.0, total)]
        out_dir.mkdir(parents=True, exist_ok=True)
        clips = []
        stem = src.stem[:60]
        for i, (start, end) in enumerate(spans):
            piece = audio[int(start * TARGET_SR) : int(end * TARGET_SR)]
            if not len(piece):
                continue
            path = out_dir / (f"{stem}.wav" if len(spans) == 1 else f"{stem}_{i:04d}.wav")
            sf.write(str(path), _normalize(piece, TARGET_SR, opts.target_lufs), TARGET_SR, subtype="PCM_16")
            clips.append(Clip(path=path, source=src, start=start, end=end))
        return clips

    def process_sources(
        self,
        sources: list[Path],
        out_dir: Path,
        opts: SegmentOptions,
        keep_whole: Callable[[Path], bool] = lambda _p: False,
        progress: Callable[[tuple[int, int], str], None] | None = None,
    ) -> list[Clip]:
        clips: list[Clip] = []
        for idx, src in enumerate(sources, 1):
            if progress:
                progress((idx - 1, len(sources)), f"Cleaning/segmenting {src.name}")
            try:
                made = self.process_file(src, out_dir, opts, keep_whole=keep_whole(src))
            except Exception as exc:
                print(f"[dataset] Skipped {src.name}: {exc}", flush=True)
                continue
            seconds = sum(c.duration for c in made)
            print(f"[dataset] {src.name}: {len(made)} clip(s), {seconds:.1f}s kept", flush=True)
            clips.extend(made)
        return clips
