"""Pick an audio + transcript pair from any Higgs dataset (or another folder / file) as a reference voice.

A clip qualifies when it has a transcript: a sidecar ``.txt`` (or ``.json`` / ``.lab``) with the same
name, or a row in the folder's ``metadata.jsonl`` / ``metadata.csv``.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .paths import ROOT, TEMP_DIR
from .training_utils import AUDIO_EXTS, _sidecar_transcript, list_train_datasets

_SCAN_CACHE: dict[str, tuple[float, list[tuple[str, str]]]] = {}


def dataset_choices() -> list[str]:
    """Every Higgs dataset split under data/ (data/<project>/train, data/<project>/eval)."""
    return list_train_datasets()


def _resolve(folder: str) -> Path:
    path = Path(str(folder or "").strip().strip('"'))
    if path.is_absolute():
        return path
    for candidate in (ROOT / path, ROOT / "data" / path):  # "Dross/eval" works like "data/Dross/eval"
        if candidate.is_dir():
            return candidate
    return ROOT / path


def _transcript(audio: Path) -> str:
    text = _sidecar_transcript(audio)
    if not text:
        lab = audio.with_suffix(".lab")
        if lab.exists():
            text = lab.read_text(encoding="utf-8", errors="ignore").strip()
    return text


def clip_choices(folder: str) -> tuple[list[tuple[str, str]], str]:
    """([(label, audio path)], status) for a dataset folder. Labels show name, length and the text start."""
    path = _resolve(folder)
    if not str(folder or "").strip():
        return [], "Pick a dataset or a folder."
    if not path.is_dir():
        return [], f"Folder not found: `{path}`"
    try:
        stamp = max((p.stat().st_mtime for p in path.iterdir()), default=0.0)
    except OSError:
        stamp = 0.0
    cached = _SCAN_CACHE.get(str(path))
    if cached and cached[0] == stamp:
        choices = cached[1]
    else:
        import soundfile as sf

        from .ready_dataset import scan_ready_dataset

        scan = scan_ready_dataset(path)
        choices = []
        for entry in scan.entries:
            try:
                seconds = f"{sf.info(str(entry.audio)).duration:.1f} s"
            except Exception:
                seconds = "? s"
            text = " ".join(entry.text.split())
            short = text[:70] + ("…" if len(text) > 70 else "")
            try:
                name = entry.audio.relative_to(path).as_posix()
            except ValueError:
                name = entry.audio.name
            choices.append((f"{name} · {seconds} · {short}", str(entry.audio)))
        _SCAN_CACHE[str(path)] = (stamp, choices)
    if not choices:
        return [], f"No audio + transcript pairs in `{path}`."
    return choices, f"{len(choices)} clips with transcript in `{path}`."


def text_for_clip(audio_path: str, folder: str = "") -> str:
    """Transcript of a clip: its sidecar file, else the folder's metadata."""
    audio = Path(audio_path)
    text = _transcript(audio)
    if text or not folder:
        return text
    from .ready_dataset import scan_ready_dataset

    for entry in scan_ready_dataset(_resolve(folder)).entries:
        if entry.audio.resolve() == audio.resolve():
            return entry.text
    return ""


def servable_copy(audio_path: str) -> str:
    """Audio outside the app folder is copied to the app's temp dir so the UI is allowed to play it."""
    src = Path(audio_path)
    try:
        src.resolve().relative_to(ROOT.resolve())
        return str(src)
    except ValueError:
        pass
    digest = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:10]
    dst = TEMP_DIR / "imported_refs" / f"{src.stem[:60]}_{digest}{src.suffix.lower()}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
    return str(dst)


def pick_pair_dialog(initial_dir: str = "") -> tuple[str, str, str]:
    """Native file picker: choose an audio file (its .txt is found automatically) or select the audio and
    its .txt together. Returns (audio path, transcript, status)."""
    import tkinter as tk
    from tkinter import filedialog

    exts = " ".join(f"*{e}" for e in sorted(AUDIO_EXTS))
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        picked = filedialog.askopenfilenames(
            title="Pick an audio file (and optionally its .txt)",
            initialdir=initial_dir or str(ROOT / "data"),
            filetypes=[("Audio + transcript", f"{exts} *.txt"), ("Audio", exts), ("Transcript", "*.txt"), ("All files", "*.*")],
        )
        files = [Path(p) for p in picked]
        audios = [p for p in files if p.suffix.lower() in AUDIO_EXTS]
        texts = [p for p in files if p.suffix.lower() == ".txt"]
        if not files:
            return "", "", "Selection cancelled."
        if len(audios) != 1:
            return "", "", "Pick exactly one audio file (plus, optionally, its .txt)."
        audio = audios[0]
        if texts:
            text = texts[0].read_text(encoding="utf-8", errors="ignore").strip()
        else:
            text = _transcript(audio)
            if not text:
                chosen = filedialog.askopenfilename(
                    title=f"No transcript next to {audio.name}: pick its .txt",
                    initialdir=str(audio.parent),
                    filetypes=[("Transcript", "*.txt"), ("All files", "*.*")],
                )
                if not chosen:
                    return "", "", f"`{audio.name}` has no transcript; pick its .txt too."
                text = Path(chosen).read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return "", "", "The transcript file is empty."
        return str(audio), text, f"Loaded `{audio.name}`."
    finally:
        root.destroy()
