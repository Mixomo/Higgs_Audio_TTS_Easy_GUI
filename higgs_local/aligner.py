"""Text-to-audio forced alignment for dataset clips (Qwen3-ForcedAligner 0.6B).

Every clip of a built dataset is aligned against its own transcript: the model
predicts the start and end time of each word (80 ms steps). That is used to

* trim the clip edges tightly to the first and last word (plus padding), so
  there is no leading/trailing silence or stray sound the text does not cover;
* drop clips whose transcript does not match the audio: on a mismatch the
  aligner collapses many words to zero length or piles them at the clip end.

The model is downloaded once into ``models/qwen3-forced-aligner-0.6b``.
Supported languages: zh, en, yue, fr, de, it, ja, ko, pt, ru, es.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .paths import MODELS_DIR

ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
ALIGNER_DIR = MODELS_DIR / "qwen3-forced-aligner-0.6b"
ALIGNER_SR = 16000
MAX_CLIP_SEC = 180.0
LANGUAGES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "fr": "French", "de": "German", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "ru": "Russian", "es": "Spanish",
}
_NAMES = {name.lower(): code for code, name in LANGUAGES.items()}


@dataclass
class AlignOptions:
    trim: bool = True              # trim clip edges to the first / last aligned word
    drop_mismatch: bool = True     # drop clips whose text does not match the audio
    pad_sec: float = 0.15          # silence kept before the first word
    pad_end_sec: float = 0.25      # kept after the last word (final syllables decay softly)
    max_bad_ratio: float = 0.12    # share of misplaced words that marks a mismatch


def resolve_language(language: str | None) -> tuple[str | None, bool]:
    """(code or None for auto, supported)."""
    value = (language or "").strip().lower()
    if not value or value.startswith("auto"):
        return None, True
    code = value if value in LANGUAGES else _NAMES.get(value) or value.split("-")[0].split("_")[0]
    return (code, True) if code in LANGUAGES else (code, False)


def ensure_aligner() -> Path:
    """Download the aligner the first time it is used."""
    if (ALIGNER_DIR / "config.json").exists():
        return ALIGNER_DIR
    from huggingface_hub import snapshot_download

    print(f"[dataset] Downloading Qwen3-ForcedAligner 0.6B (~1.8 GB, only once) from huggingface.co/{ALIGNER_REPO}", flush=True)
    ALIGNER_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(ALIGNER_REPO, local_dir=str(ALIGNER_DIR))
    return ALIGNER_DIR


class QwenAligner:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = str(ensure_aligner())
        print(f"[dataset] Loading Qwen3-ForcedAligner 0.6B on {self.device}", flush=True)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path, dtype=dtype).to(self.device).eval()

    def align(self, audio16: np.ndarray, text: str, language: str | None) -> list[tuple[str, float, float]]:
        import torch

        inputs, word_lists = self.proc.prepare_forced_aligner_inputs(
            audio=audio16.astype(np.float32), transcript=text, language=language
        )
        if not word_lists or not word_lists[0]:
            return []
        inputs = inputs.to(self.model.device)
        for key, value in inputs.items():
            if torch.is_floating_point(value):
                inputs[key] = value.to(self.model.dtype)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        stamps = self.proc.decode_forced_alignment(
            logits=logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.model.config.timestamp_token_id,
        )[0]
        return [(s["text"], float(s["start_time"]), float(s["end_time"])) for s in stamps]

    def unload(self):
        self.model = None
        self.proc = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _frame_energy(audio16: np.ndarray) -> tuple[np.ndarray, float]:
    """RMS per 20 ms frame and a loud-speech reference level (90th percentile)."""
    hop = ALIGNER_SR // 50
    n = len(audio16) // hop
    if n == 0:
        return np.zeros(1, dtype=np.float32), 1.0
    frames = audio16[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return rms, float(np.percentile(rms, 90)) or 1.0


def judge(
    words: list[tuple[str, float, float]],
    duration: float,
    max_bad_ratio: float,
    audio16: np.ndarray | None = None,
) -> tuple[bool, str, float]:
    """(matches, reason, bad ratio). A word is misplaced when it collapses (2+ letters under 30 ms), piles up at
    the clip end, or is stretched far beyond its spoken length over audio that is not silent (a pause after a
    word is absorbed into it and is fine; extra speech the text does not cover is not)."""
    if not words:
        return False, "no alignable words", 1.0
    rms, ref = _frame_energy(audio16) if audio16 is not None else (None, 1.0)
    bad = 0
    for text, start, end in words:
        letters = sum(ch.isalnum() for ch in text)
        collapsed = letters >= 2 and (end - start) < 0.03
        piled = start >= duration - 0.05
        plausible = 0.4 + 0.12 * max(letters, 1)
        stretched = (end - start) > plausible
        if stretched and rms is not None:
            tail = rms[int((start + plausible) * 50) : int(end * 50)]
            stretched = bool(len(tail)) and float(np.mean(tail > 0.1 * ref)) > 0.3
        bad += int(collapsed or piled or stretched)
    ratio = bad / len(words)
    if len(words) >= 3 and ratio > max_bad_ratio:
        return False, f"text does not match the audio ({ratio:.0%} of words could not be placed)", ratio
    return True, "", ratio


def align_dataset(
    split_dirs: list[Path],
    language: str | None,
    opts: AlignOptions,
    progress: Callable[[tuple[int, int], str], None] | None = None,
) -> tuple[list[str], int]:
    """Align every clip listed in each split's metadata.json, trimming / dropping in place.

    Rewrites the WAV files and each split's metadata.json. Returns (report lines, clips kept)."""
    import json

    from .dataset_audio import _resample
    from .training_utils import write_metadata

    default_code, default_ok = resolve_language(language)
    splits = []
    for split_dir in split_dirs:
        meta_path = split_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        splits.append((split_dir, meta))
    total = sum(len(meta.get("samples") or []) for _d, meta in splits)
    if not total:
        return [], 0

    aligner = QwenAligner()
    report: list[str] = []
    dropped: list[str] = []
    skipped_lang: set[str] = set()
    trimmed = 0.0
    kept_total = 0
    done = 0
    try:
        for split_dir, meta in splits:
            kept = []
            for sample in meta.get("samples") or []:
                done += 1
                wav = split_dir / str(sample.get("audio_file") or "")
                txt = split_dir / str(sample.get("transcript_file") or "")
                if progress:
                    progress((done - 1, total), f"Aligning text to audio {wav.name}")
                if not wav.exists() or not txt.exists():
                    kept.append(sample)
                    continue
                code, ok = resolve_language(sample.get("language")) if sample.get("language") else (default_code, default_ok)
                if not ok:
                    skipped_lang.add(str(code))
                    kept.append(sample)
                    continue
                text = txt.read_text(encoding="utf-8").strip()
                audio, sr = sf.read(str(wav), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                duration = len(audio) / float(sr)
                if not text or duration > MAX_CLIP_SEC:
                    kept.append(sample)
                    continue
                audio16 = _resample(audio, int(sr), ALIGNER_SR)
                try:
                    words = aligner.align(audio16, text, LANGUAGES.get(code or "", None))
                except Exception as exc:
                    print(f"[dataset-align] {wav.name}: {exc}", flush=True)
                    kept.append(sample)
                    continue
                matches, reason, _ratio = judge(words, duration, opts.max_bad_ratio, audio16)
                if not matches and opts.drop_mismatch:
                    dropped.append(f"{wav.name}: {reason}")
                    wav.unlink(missing_ok=True)
                    txt.unlink(missing_ok=True)
                    continue
                if opts.trim and matches:
                    start = max(0.0, words[0][1] - opts.pad_sec)
                    end = min(duration, words[-1][2] + opts.pad_end_sec)
                    if end - start > 0.3 and (start > 0.02 or duration - end > 0.02):
                        piece = audio[int(start * sr) : int(end * sr)]
                        sf.write(str(wav), piece, int(sr), subtype="PCM_16")
                        trimmed += duration - len(piece) / float(sr)
                        sample["duration"] = round(len(piece) / float(sr), 3)
                kept.append(sample)
            meta["samples"] = kept
            kept_total += len(kept)
            write_metadata(split_dir, kept, list((meta.get("dataset_info") or {}).get("created_from") or []))
    finally:
        aligner.unload()

    line = f"Text alignment (Qwen3-ForcedAligner): {total} clips checked"
    if opts.trim:
        line += f" · {trimmed:.1f} s trimmed to the first / last word"
    report.append(line)
    if skipped_lang:
        report.append(f"Alignment skipped for unsupported language(s): {', '.join(sorted(skipped_lang))}")
    if dropped:
        report.append(
            f"Dropped {len(dropped)} clips whose text does not match the audio:\n  "
            + "\n  ".join(dropped[:20])
            + ("\n  ..." if len(dropped) > 20 else "")
        )
    elif opts.drop_mismatch:
        report.append("All transcripts match their audio.")
    return report, kept_total
