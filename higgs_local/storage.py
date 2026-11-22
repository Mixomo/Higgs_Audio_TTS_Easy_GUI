"""Disk usage of the app's models, LoRA runs and datasets, with safe deletion (Utils tab).

Everything listed lives inside the app folder; deletion refuses any path outside the
``models``, ``exp``, ``data`` and ``downloads`` folders.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .paths import MODELS_DIR, ROOT

EXP_DIR = ROOT / "exp"
DATA_DIR = ROOT / "data"
DOWNLOADS_DIR = ROOT / "downloads"
_SAFE_ROOTS = (MODELS_DIR, EXP_DIR, DATA_DIR, DOWNLOADS_DIR)

# What each model folder is used for (shown in the table).
_MODEL_INFO = {
    "higgs-audio-v3-tts-4b-transformers": ("Higgs Audio V3 TTS 4B", "TTS · inference and training"),
    "higgs-audio-v2-tokenizer": ("Higgs audio tokenizer", "TTS · audio codec"),
    "crisperwhisper2.0-large": ("CrisperWhisper 2.0 large", "ASR · transcription"),
    "crisperwhisper2.0-turbo": ("CrisperWhisper 2.0 turbo", "ASR · transcription"),
    "qwen3-forced-aligner-0.6b": ("Qwen3-ForcedAligner 0.6B", "Dataset · text alignment"),
    "inworld--sava": ("Inworld SAVA", "Dataset · voice activity / segmentation"),
    "inworld--voice-isolator": ("Inworld Voice Isolator", "Dataset · voice clean-up"),
    "Systran--faster-whisper-large-v3": ("faster-whisper large-v3", "ASR · transcription"),
    "Systran--faster-whisper-tiny": ("faster-whisper tiny", "ASR · transcription"),
    "Qwen--Qwen3-ForcedAligner-0.6B-hf": ("Qwen3-ForcedAligner (download cache)", "Dataset · text alignment"),
    "nyralabs--CrisperWhisper2.0_large": ("CrisperWhisper 2.0 large (download cache)", "ASR · transcription"),
    "nyralabs--CrisperWhisper2.0_turbo": ("CrisperWhisper 2.0 turbo (download cache)", "ASR · transcription"),
    "bosonai--higgs-audio-v2-tokenizer": ("Higgs audio tokenizer (download cache)", "TTS · audio codec"),
    "multimodalart--higgs-audio-v3-tts-4b-transformers": ("Higgs Audio V3 TTS (download cache)", "TTS"),
}
_CACHE_INFO = {
    "uv": "Python package cache (installs)",
    "xet": "Hugging Face download chunks",
    "tmp": "Temporary files",
    "torch": "PyTorch kernels / hub cache",
    "triton": "Compiled GPU kernels",
    "hf_modules": "Hugging Face remote code",
    "final": "Temporary export folder",
}


@dataclass
class Item:
    key: str        # path relative to the app folder (stable id)
    name: str
    kind: str
    size: int
    detail: str = ""

    @property
    def path(self) -> Path:
        return ROOT / self.key


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def dir_size(path: Path) -> int:
    """Bytes on disk of a file or folder (symlinks not followed)."""
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


# ------------------------------------------------------------------ models
def scan_models() -> list[Item]:
    items: list[Item] = []
    if not MODELS_DIR.exists():
        return items
    for path in sorted(MODELS_DIR.iterdir()):
        if path.name.startswith(".") or path.name in ("hub", "xet", "mss") or not path.is_dir():
            continue
        name, use = _MODEL_INFO.get(path.name, (path.name, "model"))
        items.append(Item(_rel(path), name, "Model", dir_size(path), use))
    for ckpt in sorted((MODELS_DIR / "mss").rglob("*.ckpt")) if (MODELS_DIR / "mss").exists() else []:
        items.append(Item(_rel(ckpt), f"RoFormer · {ckpt.stem}", "Model", dir_size(ckpt), "Dataset · voice separation"))
    for hub in (MODELS_DIR / "hub", MODELS_DIR / ".cache" / "huggingface"):
        if not hub.exists():
            continue
        for path in sorted(hub.glob("models--*")):
            repo = path.name[len("models--"):]
            name, use = _MODEL_INFO.get(repo, (repo.replace("--", "/"), "model"))
            items.append(Item(_rel(path), name, "Model (HF cache)", dir_size(path), use))
    caches = [MODELS_DIR / "xet"] + ([p for p in sorted((MODELS_DIR / ".cache").iterdir()) if p.is_dir() and p.name != "huggingface"]
                                     if (MODELS_DIR / ".cache").exists() else [])
    for path in caches:
        if path.exists():
            items.append(Item(_rel(path), f"cache · {path.name}", "Cache", dir_size(path), _CACHE_INFO.get(path.name, "cache")))
    return [i for i in items if i.size > 0]


# ------------------------------------------------------------------ LoRA runs
def _checkpoint_number(path: Path) -> int:
    m = re.match(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def _best_step(run: Path) -> int | None:
    try:
        return int(json.loads((run / "checkpoint-best" / "best_info.json").read_text(encoding="utf-8"))["step"])
    except Exception:
        return None


def scan_loras() -> list[Item]:
    items = []
    if not EXP_DIR.exists():
        return items
    for run in sorted(p for p in EXP_DIR.iterdir() if p.is_dir()):
        checkpoints = [p for p in run.glob("checkpoint-*") if p.is_dir() and _checkpoint_number(p) >= 0]
        best = _best_step(run)
        detail = f"{len(checkpoints)} checkpoints" + (f" · best step {best}" if best is not None else "")
        if not (run / "qwen3_lora").exists() and not checkpoints:
            detail += " · no adapter"
        items.append(Item(_rel(run), run.name, "LoRA run", dir_size(run), detail))
    return items


def scan_run_parts(run_key: str) -> list[Item]:
    """Checkpoints and other folders of one LoRA run."""
    run = _safe(run_key)
    if not run.is_dir():
        return []
    best = _best_step(run)
    items = []
    for path in sorted(run.glob("checkpoint-*"), key=lambda p: (_checkpoint_number(p) < 0, _checkpoint_number(p))):
        if not path.is_dir():
            continue
        n = _checkpoint_number(path)
        if path.name == "checkpoint-best":
            detail = f"best eval (step {best})" if best is not None else "best eval"
        else:
            resumable = (path / "trainer_state.pt").exists()
            detail = ("same step as best · " if n == best else "") + ("resumable" if resumable else "adapter only")
        items.append(Item(_rel(path), path.name, "Checkpoint", dir_size(path), detail))
    final = run / "qwen3_lora"
    if final.exists():
        items.append(Item(_rel(final), "final adapter", "Final", dir_size(final), "exported LoRA (inference 'Final')"))
    for name, detail in (
        ("audio_code_cache", "cached audio codes · rebuilt automatically when training again"),
        ("eval_audio", "eval previews of the training monitor"),
        ("tensorboard", "TensorBoard logs"),
        ("checkpoint_compare", "audio from Quick checkpoint compare"),
    ):
        path = run / name
        if path.exists():
            items.append(Item(_rel(path), name, "Extra", dir_size(path), detail))
    return items


def keep_only_best_targets(run_key: str) -> list[Item]:
    """Every numbered checkpoint of a run (checkpoint-best and the final adapter stay)."""
    run = _safe(run_key)
    if not (run / "checkpoint-best").exists():
        return []
    return [i for i in scan_run_parts(run_key) if i.kind == "Checkpoint" and i.name != "checkpoint-best"]


# ------------------------------------------------------------------ datasets
def scan_datasets() -> list[Item]:
    items = []
    if DATA_DIR.exists():
        for ds in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
            counts = []
            for split in ("train", "eval"):
                meta = ds / split / "metadata.json"
                if meta.exists():
                    try:
                        n = len(json.loads(meta.read_text(encoding="utf-8")).get("samples") or [])
                        counts.append(f"{split} {n}")
                    except Exception:
                        pass
            detail = " · ".join(counts) or ("unfinished import" if ds.name.startswith("_import_") else "no metadata")
            items.append(Item(_rel(ds), ds.name, "Dataset", dir_size(ds), detail))
    if DOWNLOADS_DIR.exists():
        for dl in sorted(p for p in DOWNLOADS_DIR.iterdir() if p.is_dir()):
            n = len(list(dl.glob("*.wav")))
            items.append(Item(_rel(dl), dl.name, "YouTube download", dir_size(dl), f"{n} audio files"))
    return items


# ------------------------------------------------------------------ delete
def _safe(key: str) -> Path:
    path = (ROOT / str(key)).resolve()
    for base in _SAFE_ROOTS:
        base = base.resolve()
        if path != base and base in path.parents:
            return path
    raise ValueError(f"Refusing to touch a path outside the app's models/exp/data/downloads folders: {key}")


def delete(keys: list[str]) -> tuple[int, list[str]]:
    """Delete the given items. Returns (bytes freed, errors)."""
    freed, errors = 0, []
    for key in keys:
        try:
            path = _safe(key)
            if not path.exists():
                continue
            size = dir_size(path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            freed += size
        except Exception as exc:  # noqa: BLE001 - report and continue
            errors.append(f"{key}: {exc}")
    return freed, errors


def table(items: list[Item]) -> list[list[str]]:
    return [[i.name, i.kind, human(i.size), i.detail, i.key] for i in items]


def total(items: list[Item]) -> int:
    return sum(i.size for i in items)
