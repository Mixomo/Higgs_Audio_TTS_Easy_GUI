"""Import ready-made TTS datasets (audio + transcript pairs) into the Higgs layout.

Accepted sources (a folder or a ``.zip`` with the same content):

* ``metadata.jsonl`` (optional): one JSON object per clip with at least an audio
  path and its text. When present it is the source of truth: ``keep: false`` rows
  are skipped and a ``split`` column (train / eval) is honoured.
* ``metadata.csv`` in LJSpeech style (``file|text[|normalized]``), optional.
* Otherwise every audio file with a sidecar ``.txt`` / ``.json`` / ``.lab``
  transcript of the same name is used.

License / readme files (``LICENSE*``, ``*.md``) are optional and copied along.
An optional clean-up stage (voice separation engine + SAVA voice-activity trim /
filter) runs per clip without ever re-cutting the text.
"""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .paths import ROOT
from .training_utils import (
    AUDIO_EXTS,
    DatasetSummary,
    _sample_metadata,
    _sidecar_transcript,
    iter_audio_files,
    normalize_transcript,
    slugify,
    write_metadata,
)

AUDIO_KEYS = ("audio", "audio_file", "audio_filepath", "file", "path", "wav", "filename")
TEXT_KEYS = ("text", "transcript", "transcription", "sentence", "normalized_text")
EVAL_SPLITS = {"eval", "val", "valid", "validation", "dev", "test"}
LANG_CODES = {
    "English": "en", "Spanish": "es", "Mandarin": "zh", "Chinese": "zh",
    "Korean": "ko", "German": "de", "Auto-detect": "unknown",
}


@dataclass
class ReadyEntry:
    audio: Path
    text: str
    split: str = ""  # "train" / "eval" / "" (unknown)
    language: str = ""


@dataclass
class ReadyScan:
    root: Path
    kind: str  # "metadata.jsonl" / "metadata.csv" / "audio + txt pairs"
    entries: list[ReadyEntry] = field(default_factory=list)
    skipped_keep: int = 0
    missing_audio: int = 0
    missing_text: int = 0
    license_files: list[Path] = field(default_factory=list)


def _clean_path(value: str) -> Path:
    return Path(str(value or "").strip().strip('"').strip("'")).expanduser()


def _lang_code(language: str) -> str:
    return LANG_CODES.get(language, language[:2].lower() if language else "unknown")


def extract_zip(zip_path: Path, target: Path) -> Path:
    """Extract a dataset archive (path-traversal safe) and return its content root."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    base = target.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            dest = (target / member.filename).resolve()
            if base not in dest.parents and dest != base:
                raise ValueError(f"Unsafe path inside zip: {member.filename}")
        zf.extractall(target)
    return _content_root(target)


def _content_root(folder: Path) -> Path:
    """Descend through single wrapper folders (zip/dataset/... -> real content)."""
    current = folder
    for _ in range(4):
        children = [p for p in current.iterdir() if not p.name.startswith((".", "__MACOSX"))]
        has_payload = any(p.is_file() and (p.suffix.lower() in AUDIO_EXTS or p.suffix.lower() in (".jsonl", ".csv")) for p in children)
        dirs = [p for p in children if p.is_dir()]
        if has_payload or len(dirs) != 1:
            return current
        current = dirs[0]
    return current


def _find_meta(root: Path, suffix: str) -> Path | None:
    candidates = [p for p in root.rglob(f"*{suffix}") if p.is_file() and "__MACOSX" not in p.parts]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.name.lower() != f"metadata{suffix}", len(p.relative_to(root).parts), str(p)))
    return candidates[0]


def _resolve_audio(value: str, base: Path, root: Path, by_name: dict[str, Path]) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for candidate in (Path(raw), base / raw, root / raw):
        if candidate.is_file():
            return candidate
    name = Path(raw.replace("\\", "/")).name
    hit = by_name.get(name.lower())
    if hit:
        return hit
    if not Path(name).suffix:
        for ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
            hit = by_name.get((name + ext).lower())
            if hit:
                return hit
    return None


def scan_ready_dataset(root: Path) -> ReadyScan:
    """Find the clips of a ready dataset folder (already extracted)."""
    root = Path(root)
    audio_files = iter_audio_files(root, recursive=True)
    by_name = {p.name.lower(): p for p in audio_files}
    license_files = sorted(
        p for p in root.iterdir() if p.is_file() and (p.name.upper().startswith("LICENSE") or p.suffix.lower() == ".md")
    )
    jsonl = _find_meta(root, ".jsonl")
    if jsonl is not None:
        scan = ReadyScan(root=root, kind=jsonl.relative_to(root).as_posix(), license_files=license_files)
        for line in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("keep") is False:
                scan.skipped_keep += 1
                continue
            audio_value = next((row[k] for k in AUDIO_KEYS if row.get(k)), "")
            if isinstance(audio_value, dict):  # HF-style {"path": ...}
                audio_value = audio_value.get("path", "")
            audio = _resolve_audio(str(audio_value), jsonl.parent, root, by_name)
            if audio is None:
                scan.missing_audio += 1
                continue
            text = next((str(row[k]) for k in TEXT_KEYS if row.get(k)), "") or _sidecar_transcript(audio)
            if not normalize_transcript(text):
                scan.missing_text += 1
                continue
            split = str(row.get("split") or "").lower()
            split = "eval" if split in EVAL_SPLITS else ("train" if split == "train" else "")
            scan.entries.append(ReadyEntry(audio=audio, text=text, split=split, language=str(row.get("language") or "")))
        return scan

    csv_meta = _find_meta(root, ".csv")
    if csv_meta is not None:
        scan = ReadyScan(root=root, kind=csv_meta.relative_to(root).as_posix(), license_files=license_files)
        with csv_meta.open(encoding="utf-8", errors="ignore", newline="") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            delimiter = "|" if "|" in sample else ","
            for row in csv.reader(fh, delimiter=delimiter):
                if len(row) < 2 or row[0].strip().lower() in ("file", "audio", "path", "id"):
                    continue
                audio = _resolve_audio(row[0], csv_meta.parent, root, by_name) or _resolve_audio(
                    f"wavs/{row[0]}", csv_meta.parent, root, by_name
                )
                if audio is None:
                    scan.missing_audio += 1
                    continue
                text = (row[2] if len(row) > 2 and row[2].strip() else row[1]).strip()
                if not normalize_transcript(text):
                    scan.missing_text += 1
                    continue
                scan.entries.append(ReadyEntry(audio=audio, text=text))
        if scan.entries:
            return scan

    scan = ReadyScan(root=root, kind="audio + txt pairs", license_files=license_files)
    for audio in audio_files:
        text = _sidecar_transcript(audio)
        if normalize_transcript(text):
            scan.entries.append(ReadyEntry(audio=audio, text=text))
        else:
            scan.missing_text += 1
    return scan


def _durations(entries: list[ReadyEntry]) -> list[float]:
    out = []
    for entry in entries:
        try:
            out.append(float(sf.info(str(entry.audio)).duration))
        except Exception:
            out.append(0.0)
    return out


def open_ready_source(source: str, work_dir: Path) -> tuple[Path, bool]:
    """Return (content root, extracted?) for a folder or .zip path."""
    path = _clean_path(source)
    if not str(path) or str(path) == ".":
        raise ValueError("Enter the path of a dataset folder or .zip.")
    if not path.exists():
        raise ValueError(f"Path not found: {path}")
    if path.is_file():
        if path.suffix.lower() != ".zip":
            raise ValueError("Ready datasets must be a folder or a .zip file.")
        return extract_zip(path, work_dir), True
    return _content_root(path), False


def describe_ready_source(source: str) -> str:
    """Markdown summary of what an import would use (for the 🔍 Scan button)."""
    path = _clean_path(source)
    work = ROOT / "data" / "_ready_scan"
    try:
        if path.is_file() and path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
            audio = [n for n in names if Path(n).suffix.lower() in AUDIO_EXTS]
            texts = [n for n in names if n.lower().endswith(".txt")]
            meta = [n for n in names if n.lower().endswith((".jsonl", "metadata.csv"))]
            lic = [n for n in names if Path(n).name.upper().startswith("LICENSE")]
            return (
                f"**📦 Zip:** `{path.name}` · {len(audio)} audio · {len(texts)} .txt"
                f" · metadata: {', '.join(Path(m).name for m in meta) or 'none (pairs)'}"
                f"{' · license note found' if lic else ''}\n\nThe archive is extracted when you import it."
            )
        root, extracted = open_ready_source(source, work)
        scan = scan_ready_dataset(root)
        durations = _durations(scan.entries)
        minutes = sum(durations) / 60.0
        splits = {"train": 0, "eval": 0}
        for entry in scan.entries:
            if entry.split in splits:
                splits[entry.split] += 1
        longest = max(durations or [0.0])
        lines = [
            f"**📦 {len(scan.entries)} clips · {minutes:.1f} min** · source: {scan.kind}",
            f"Median {float(np.median(durations)) if durations else 0:.1f} s · longest {longest:.1f} s",
        ]
        if splits["eval"]:
            lines.append(f"Split from metadata: {splits['train']} train / {splits['eval']} eval")
        notes = []
        if scan.skipped_keep:
            notes.append(f"{scan.skipped_keep} rows marked keep=false")
        if scan.missing_audio:
            notes.append(f"{scan.missing_audio} rows without audio file")
        if scan.missing_text:
            notes.append(f"{scan.missing_text} clips without transcript")
        if notes:
            lines.append("Skipped: " + ", ".join(notes))
        long_clips = sum(1 for d in durations if d > 30.0)
        if long_clips:
            lines.append(f"⚠️ {long_clips} clips are longer than 30 s (training skips them by default).")
        if scan.license_files:
            lines.append("License / notes: " + ", ".join(f"`{p.name}`" for p in scan.license_files))
        if extracted:
            shutil.rmtree(work, ignore_errors=True)
        return "  \n".join(lines)
    except Exception as exc:
        return f"⚠️ {exc}"


def _eval_indices(n: int, ratio: float) -> set[int]:
    ratio = max(min(float(ratio), 0.5), 0.0)
    if ratio <= 0 or n < 2:
        return set()
    k = min(max(1, int(round(n * ratio))), n - 1)
    # Spread eval clips evenly over the dataset instead of taking the tail.
    return {min(n - 1, int((i + 0.5) * n / k)) for i in range(k)}


def import_ready_dataset(
    source: str,
    dataset_name: str,
    val_split: float = 0.1,
    language: str = "Auto-detect",
    clean_options=None,
    use_metadata_split: bool = True,
    progress: Callable[[tuple[int, int] | float, str], None] | None = None,
) -> tuple[DatasetSummary, str]:
    """Copy a ready dataset into data/<name>/{train,eval}. Returns (summary, report)."""
    dataset_slug = slugify(dataset_name)
    dataset_dir = ROOT / "data" / dataset_slug
    work_dir = ROOT / "data" / f"_import_{dataset_slug}"
    src_path = _clean_path(source).resolve()
    target = dataset_dir.resolve()
    if src_path == target or target in src_path.parents:
        raise ValueError("The source is inside this project's own dataset folder; choose another project name.")
    if progress:
        progress(0.0, "Reading the dataset")
    root, extracted = open_ready_source(source, work_dir)
    try:
        scan = scan_ready_dataset(root)
        if not scan.entries:
            raise RuntimeError("No usable clips found (need audio files with a transcript, or a metadata.jsonl).")

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        train_dir = dataset_dir / "train"
        eval_dir = dataset_dir / "eval"
        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        entries = scan.entries
        meta_split = use_metadata_split and any(e.split == "eval" for e in entries) and any(e.split != "eval" for e in entries)
        eval_idx = set() if meta_split else _eval_indices(len(entries), val_split)

        from .dataset_audio import TARGET_SR, CleanOptions, DatasetAudioProcessor, _mono, _resample

        opts = clean_options or CleanOptions()
        active = opts.engine not in ("", "off") or opts.vad_trim or opts.min_speech_ratio > 0
        processor = DatasetAudioProcessor() if active else None
        speaker_id = f"{dataset_slug}_speaker"
        speaker_name = dataset_slug.replace("_", " ").title()
        default_lang = _lang_code(language)
        train_samples: list[dict] = []
        eval_samples: list[dict] = []
        dropped: list[str] = []
        trimmed = 0.0
        failed = 0
        kept = 0
        try:
            for idx, entry in enumerate(entries):
                if progress:
                    label = "Cleaning" if active else "Copying"
                    progress((idx, len(entries)), f"{label} {entry.audio.name}")
                split = entry.split if meta_split else ("eval" if idx in eval_idx else "train")
                split = "eval" if split == "eval" else "train"
                out_dir = eval_dir if split == "eval" else train_dir
                sample_id = f"{dataset_slug}_{kept:06d}"
                dst = out_dir / f"{sample_id}.wav"
                try:
                    if processor is not None:
                        info = processor.clean_clip(entry.audio, dst, opts)
                        if info is None or info.get("dropped"):
                            dropped.append(f"{entry.audio.name}: {(info or {}).get('dropped', 'empty audio')}")
                            continue
                        duration = float(info["duration"])
                        trimmed += float(info.get("trimmed") or 0.0)
                    else:
                        audio, sr = sf.read(str(entry.audio), dtype="float32", always_2d=True)
                        audio = _resample(_mono(audio), int(sr), TARGET_SR)
                        if not len(audio):
                            dropped.append(f"{entry.audio.name}: empty audio")
                            continue
                        sf.write(str(dst), audio, TARGET_SR, subtype="PCM_16")
                        duration = len(audio) / float(TARGET_SR)
                except Exception as exc:
                    failed += 1
                    print(f"[dataset-import] Skipped {entry.audio.name}: {exc}", flush=True)
                    continue
                transcript = normalize_transcript(entry.text)
                (out_dir / f"{sample_id}.txt").write_text(transcript, encoding="utf-8")
                sample = _sample_metadata(
                    sample_id,
                    dst.name,
                    f"{sample_id}.txt",
                    duration,
                    speaker_id,
                    speaker_name,
                    (entry.language or default_lang)[:8],
                    str(entry.audio if not extracted else Path(source) / entry.audio.relative_to(root)),
                    split,
                )
                sample["training_task_type"] = "single_speaker_smart_voice"
                (eval_samples if split == "eval" else train_samples).append(sample)
                kept += 1
        finally:
            if processor is not None:
                processor.unload()

        if not train_samples:
            raise RuntimeError("No train clips were written (all clips were dropped or failed).")
        created_from = [str(_clean_path(source))]
        if eval_samples:
            write_metadata(eval_dir, eval_samples, created_from)
            eval_final = eval_dir
        else:
            shutil.rmtree(eval_dir, ignore_errors=True)
            eval_final = None
        write_metadata(train_dir, train_samples, created_from)
        all_samples = train_samples + eval_samples
        write_metadata(dataset_dir, all_samples, created_from)
        for lic in scan.license_files:
            shutil.copy2(lic, dataset_dir / lic.name)
        if progress:
            progress((len(entries), len(entries)), "Dataset ready")

        total = sum(s["duration"] for s in all_samples)
        summary = DatasetSummary(
            dataset_dir=dataset_dir,
            train_dir=train_dir,
            eval_dir=eval_final,
            train_samples=len(train_samples),
            eval_samples=len(eval_samples),
            total_duration=total,
            avg_duration=total / max(len(all_samples), 1),
            languages=sorted({s["language"] for s in all_samples}),
            speakers=sorted({s["speaker_id"] for s in all_samples}),
        )
        report = [
            f"Imported from: {_clean_path(source)} ({scan.kind})",
            f"Split: {'from metadata' if meta_split else f'{val_split:.0%} evenly spread'}",
        ]
        if active:
            report.append(
                f"Clean-up: engine={opts.engine}, VAD trim={'on' if opts.vad_trim else 'off'}, "
                f"min speech={opts.min_speech_ratio:.0%} · {trimmed:.1f} s of edge silence removed"
            )
        if scan.skipped_keep:
            report.append(f"Skipped {scan.skipped_keep} rows marked keep=false")
        if scan.missing_audio or scan.missing_text:
            report.append(f"Skipped {scan.missing_audio} rows without audio, {scan.missing_text} clips without text")
        if dropped:
            report.append(f"Dropped {len(dropped)} clips:\n  " + "\n  ".join(dropped[:20]) + ("\n  ..." if len(dropped) > 20 else ""))
        if failed:
            report.append(f"{failed} clips failed to load (see console)")
        if scan.license_files:
            report.append("Copied: " + ", ".join(p.name for p in scan.license_files))
        return summary, "\n".join(report)
    finally:
        if extracted:
            shutil.rmtree(work_dir, ignore_errors=True)
