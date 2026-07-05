from __future__ import annotations

import json
import shutil
from pathlib import Path

import soundfile as sf

from .audio_utils import ensure_dual_mono_file, safe_name, save_wav
from .paths import SAMPLES_DIR, VOICES_DIR


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _roots() -> tuple[Path, Path]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    return VOICES_DIR, SAMPLES_DIR


def list_voice_names() -> list[str]:
    voices, samples = _roots()
    names = {"None"}
    for root in (voices, samples):
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
                names.add(path.stem)
    return sorted(names, key=lambda x: (x != "None", x.lower()))


def _read_sidecar_text(audio: Path) -> str:
    txt = audio.with_suffix(".txt")
    if txt.exists():
        return txt.read_text(encoding="utf-8").strip()
    meta = audio.with_suffix(".json")
    if not meta.exists():
        return ""
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return ""
    for key in ("Text", "text", "Transcript", "transcript", "ReferenceText", "reference_text"):
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_voice(name: str | None) -> tuple[str | None, str]:
    if not name or name == "None":
        return None, ""
    voices, samples = _roots()
    for root in (samples, voices):
        if not root.exists():
            continue
        for ext in AUDIO_EXTS:
            audio = root / f"{name}{ext}"
            if audio.exists():
                return str(audio), _read_sidecar_text(audio)
    return None, ""


def save_voice_sample(audio_path: str | None, sample_name: str, transcript: str) -> str:
    if not audio_path:
        return "No audio file selected."
    source = Path(audio_path)
    if not source.exists():
        return f"Audio file not found: {source}"
    name = safe_name(sample_name or source.stem, "voice")
    dest = SAMPLES_DIR / f"{name}.wav"
    try:
        audio, sr = sf.read(str(source), dtype="float32", always_2d=False)
        save_wav(dest, sr, audio)
    except Exception:
        dest = SAMPLES_DIR / f"{name}{source.suffix or '.wav'}"
        shutil.copy2(source, dest)
        ensure_dual_mono_file(dest)
    transcript = (transcript or "").strip()
    dest.with_suffix(".txt").write_text(transcript, encoding="utf-8")
    dest.with_suffix(".json").write_text(
        json.dumps({"Type": "Sample", "Text": transcript}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return f"Saved voice sample: {dest.name}"


def normalize_voice_library() -> int:
    voices, samples = _roots()
    changed = 0
    for root in (voices, samples):
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
                changed += int(ensure_dual_mono_file(path))
    return changed
