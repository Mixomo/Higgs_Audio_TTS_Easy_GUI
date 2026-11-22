"""Dataset audio front-end: voice isolation (Inworld Voice Isolator) and natural
segmentation (Inworld SAVA) for building Higgs TTS training data.

Voice clean-up engines: Inworld Voice Isolator or Mel-Band / BS-RoFormer
models via pymss, optionally chained ("a+b+c"). Every model is stored inside
this app's ``models`` folder.

Pipeline per source recording:
    load -> mono -> [clean-up engine] -> [SAVA speech/turn curves -> segments]
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

from .paths import MODELS_DIR

TARGET_SR = 24000
SAVA_SR = 16000
FRAME = 0.04  # SAVA frame length in seconds
MSS_SR = 44100
MSS_DIR = MODELS_DIR / "mss"
_MSS_CFG = Path(__file__).resolve().parent / "mss_configs"

# Checkpoints download once from their official Hugging Face repos into models/mss.
MSS_MODELS = {
    "becruily_deux": {
        "label": "Mel-Band RoFormer · voice / music (becruily deux)",
        "ckpt": MSS_DIR / "vocal" / "becruily_deux.ckpt",
        "yaml": _MSS_CFG / "becruily_deux.yaml",
        "repo": "becruily/mel-band-roformer-deux", "file": "becruily_deux.ckpt", "mb": 435,
        "voice": "Vocals",
    },
    "voc_fv7": {
        "label": "Mel-Band RoFormer · isolate voice (voc_fv7)",
        "ckpt": MSS_DIR / "vocal" / "voc_fv7.ckpt",
        "yaml": _MSS_CFG / "voc_fv7.yaml",
        "repo": "GaboxR67/MelBandRoformers", "file": "melbandroformers/vocals/voc_fv7.ckpt", "mb": 490,
        "voice": "Vocals",
    },
    "gabox_denoise": {
        "label": "Mel-Band RoFormer · remove noise / bleed (gabox)",
        "ckpt": MSS_DIR / "denoise" / "mel_band_roformer_denoise_debleed_gabox.ckpt",
        "yaml": _MSS_CFG / "gabox_denoise.yaml",
        "repo": "GaboxR67/MelBandRoformers", "file": "melbandroformers/instrumental/denoisedebleed.ckpt", "mb": 913,
        "voice": "Instrumental",  # in this model the "Instrumental" stem is the clean voice
    },
    "becruily_karaoke": {
        "label": "BS-RoFormer · lead voice only (becruily karaoke)",
        "ckpt": MSS_DIR / "vocal" / "bs_roformer_karaoke_frazer_becruily.ckpt",
        "yaml": _MSS_CFG / "becruily_karaoke.yaml",
        "repo": "becruily/bs-roformer-karaoke", "file": "bs_roformer_karaoke_frazer_becruily.ckpt", "mb": 195,
        "voice": "Vocals",
        "model_type": "bs_roformer",
    },
}

# Clean-up engines shown in the UI: (label, key). "off" = keep the audio as is.
CLEAN_ENGINES = [
    ("Off · keep audio as is", "off"),
    ("Voice Isolator (Inworld) · voice vs. everything else", "inworld"),
    ("voc_fv7 · isolate voice (music, ambience)", "voc_fv7"),
    ("gabox · remove noise and bleed", "gabox_denoise"),
    ("becruily deux · voice / music", "becruily_deux"),
    ("voc_fv7 + gabox · voice and noise", "voc_fv7+gabox_denoise"),
    ("becruily deux + gabox · music and noise", "becruily_deux+gabox_denoise"),
    ("karaoke · lead voice only", "becruily_karaoke"),
    ("karaoke + gabox · lead voice and noise", "becruily_karaoke+gabox_denoise"),
    ("voc_fv7 + gabox + karaoke · only the main voice", "voc_fv7+gabox_denoise+becruily_karaoke"),
]
CLEAN_ENGINE_KEYS = {key for _label, key in CLEAN_ENGINES}


def ensure_mss(key: str) -> None:
    """Download a RoFormer checkpoint the first time it is used."""
    m = MSS_MODELS[key]
    if m["ckpt"].exists():
        return
    import shutil
    import tempfile

    from huggingface_hub import hf_hub_download

    print(f"[dataset] Downloading {m['label']} (~{m['mb']} MB, only once) from huggingface.co/{m['repo']}", flush=True)
    m["ckpt"].parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=m["ckpt"].parent) as tmp:
        path = hf_hub_download(m["repo"], m["file"], local_dir=tmp)
        shutil.move(path, m["ckpt"])


def _fit(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def _process_blocks(
    x: np.ndarray, sr: int, fn: Callable[[np.ndarray], np.ndarray], block_s: float = 300.0, overlap_s: float = 4.0
) -> np.ndarray:
    """Apply ``fn`` to overlapping blocks of a mono signal, joined with linear crossfades."""
    n = len(x)
    block = max(1, int(block_s * sr))
    if n <= block:
        return _fit(np.asarray(fn(x), dtype=np.float32).reshape(-1), n)
    ov = min(int(overlap_s * sr), block // 2)
    step = block - ov
    out = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    for start in range(0, n, step):
        end = min(n, start + block)
        y = _fit(np.asarray(fn(x[start:end]), dtype=np.float32).reshape(-1), end - start)
        w = np.ones(end - start, dtype=np.float32)
        if start > 0 and ov:
            w[:ov] = np.linspace(0.0, 1.0, ov, dtype=np.float32)
        if end < n and ov:
            w[-ov:] = np.minimum(w[-ov:], np.linspace(1.0, 0.0, ov, dtype=np.float32))
        out[start:end] += y * w
        weight[start:end] += w
        if end >= n:
            break
    return out / np.maximum(weight, 1e-6)


@dataclass
class CleanOptions:
    """Clean-up for ready-made clips (audio + transcript): never re-cuts the text."""

    engine: str = "off"
    vad_trim: bool = False
    min_speech_ratio: float = 0.0  # drop clips whose speech covers less than this share
    normalize: bool = True
    target_lufs: float = -20.0
    pad_sec: float = 0.15


@dataclass
class SegmentOptions:
    isolate: bool = True
    engine: str = "inworld"
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
        self._mss: dict[str, object] = {}

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

    def mss(self, key: str):
        if key not in self._mss:
            import logging

            from pymss import MSSeparator, get_separation_logger
            from pymss.logger import set_log_level

            m = MSS_MODELS[key]
            ensure_mss(key)
            print(f"[dataset] Loading {m['label']} on {self.device}", flush=True)
            log = get_separation_logger(console_level=logging.ERROR)
            set_log_level(log, logging.ERROR)
            log.setLevel(logging.ERROR)
            sep = MSSeparator(
                model_type=m.get("model_type", "mel_band_roformer"),
                model_path=str(m["ckpt"]),
                config_path=str(m["yaml"]),
                device=self.device,
                device_ids=[0],
                output_format="wav",
                use_tta=False,
                store_dirs={},
                logger=log,
                debug=False,
                inference_params={
                    "batch_size": 1, "overlap_size": None, "chunk_size": None,
                    "standardize": None, "normalize": False, "mask_mode": None,
                },
            )
            set_log_level(sep.logger, logging.ERROR)
            self._mss[key] = sep
        return self._mss[key]

    def unload(self):
        self._isolator = None
        self._sava = None
        self._mss = {}
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

        def run(block: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                out = self.isolator().isolate(np.ascontiguousarray(block), sr, output_sample_rate=sr)
            if isinstance(out, torch.Tensor):
                out = out.detach().float().cpu().numpy()
            return np.asarray(out, dtype=np.float32).reshape(-1)

        return _process_blocks(audio, sr, run, block_s=120.0, overlap_s=3.0)

    def separate_mss(self, audio: np.ndarray, sr: int, key: str) -> np.ndarray:
        m = MSS_MODELS[key]
        sep = self.mss(key)

        def run(block: np.ndarray) -> np.ndarray:
            x44 = _resample(np.ascontiguousarray(block), sr, MSS_SR)
            res = sep.separate(x44, pbar=False, stems=[m["voice"]])
            v = np.asarray(next(iter(res.values())), dtype=np.float32)
            if v.ndim == 2:
                v = v.mean(axis=1) if v.shape[0] > v.shape[1] else v.mean(axis=0)
            return _resample(v.reshape(-1), MSS_SR, sr)

        return _process_blocks(audio, sr, run, block_s=300.0, overlap_s=4.0)

    def clean(self, audio: np.ndarray, sr: int, engine: str) -> np.ndarray:
        """Run one clean-up engine or a chain ``a+b+c`` (mono in, mono out, same rate)."""
        engine = (engine or "off").strip()
        if engine in ("", "off"):
            return audio
        for key in engine.split("+"):
            if key == "inworld":
                audio = self.isolate(audio, sr)
            elif key in MSS_MODELS:
                audio = self.separate_mss(audio, sr, key)
            else:
                raise ValueError(f"Unknown clean-up engine: {key}")
        return audio

    def speech_curve(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """SAVA voice probability per 40 ms frame."""
        import torch
        from inworld import sava

        audio16 = _resample(audio, sr, SAVA_SR)
        with torch.inference_mode():
            curves = sava.run(self.sava(), audio16)
        return np.asarray(curves.speech, dtype=np.float32)

    def clean_clip(self, src: Path, dst: Path, opts: CleanOptions, threshold: float = 0.5) -> dict | None:
        """Clean one ready-made clip. Returns info, or None when the clip is dropped.

        Only leading/trailing non-speech is trimmed (with padding), so the
        transcript always stays aligned with the audio."""
        audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
        audio = _mono(audio)
        if not len(audio):
            return None
        audio = self.clean(audio, int(sr), opts.engine)
        audio = _resample(audio, int(sr), TARGET_SR)
        info: dict = {"trimmed": 0.0, "speech_ratio": None, "dropped": ""}
        if opts.vad_trim or opts.min_speech_ratio > 0:
            speech = self.speech_curve(audio, TARGET_SR)
            voiced = np.flatnonzero(speech >= threshold)
            if not len(voiced):
                return {**info, "dropped": "no speech detected"}
            ratio = float(len(voiced)) / float(max(len(speech), 1))
            info["speech_ratio"] = round(ratio, 3)
            if ratio < opts.min_speech_ratio:
                return {**info, "dropped": f"speech covers only {ratio:.0%}"}
            if opts.vad_trim:
                total = len(audio) / float(TARGET_SR)
                start = max(0.0, voiced[0] * FRAME - opts.pad_sec)
                end = min(total, (voiced[-1] + 1) * FRAME + opts.pad_sec)
                info["trimmed"] = round(total - (end - start), 3)
                audio = audio[int(start * TARGET_SR) : int(end * TARGET_SR)]
        if opts.normalize:
            audio = _normalize(audio, TARGET_SR, opts.target_lufs)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), audio, TARGET_SR, subtype="PCM_16")
        info["duration"] = len(audio) / float(TARGET_SR)
        return info

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
            audio = self.clean(audio, int(sr), opts.engine or "inworld")
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
