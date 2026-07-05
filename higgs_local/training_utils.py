from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Iterable

import soundfile as sf

from .model_registry import TOKENIZER_MODELS, TTS_MODELS
from .paths import LOGS_DIR, MODELS_DIR, ROOT


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


@dataclass
class DatasetSummary:
    dataset_dir: Path
    train_dir: Path
    eval_dir: Path | None
    train_samples: int
    eval_samples: int
    total_duration: float
    avg_duration: float
    languages: list[str]
    speakers: list[str]


@dataclass
class TrainingDatasetAnalysis:
    dataset_dir: Path
    source: str
    trainable: bool
    sample_count: int
    total_duration: float | None
    avg_duration: float | None
    median_duration: float | None
    min_duration: float | None
    max_duration: float | None
    transcript_coverage: float | None
    languages: list[str]
    speakers: list[str]
    warnings: list[str]


def slugify(value: str, fallback: str = "higgs_dataset") -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", (value or "").strip()).strip("._-")
    return value or fallback


def normalize_v2_event_tags(text: str) -> str:
    replacements = [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE_s>[Humming]</SE_s>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]
    for tag, replacement in replacements:
        text = text.replace(tag, replacement)
    return text


def normalize_transcript(text: str, convert_v2_tags: bool = True) -> str:
    text = (text or "").strip()
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")
    if convert_v2_tags:
        text = normalize_v2_event_tags(text)
    lines = text.splitlines()
    text = "\n".join(" ".join(line.split()) for line in lines if line.strip()).strip()
    if text and not any(text.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]):
        text += "."
    return text


def iter_audio_files(source_dir: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(source_dir)
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in AUDIO_EXTS)


def _resolve_dataset_path(dataset_dir: str | Path) -> Path:
    path = Path(str(dataset_dir or "")).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _duration_from_audio_file(path: Path) -> float | None:
    try:
        info = sf.info(str(path))
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                return float(handle.getnframes()) / float(rate) if rate else None
        except Exception:
            return None
    return None


def _sidecar_exists(audio_path: Path) -> bool:
    for suffix in (".txt", ".json", ".lab"):
        path = audio_path.with_suffix(suffix)
        if path.exists() and path.read_text(encoding="utf-8", errors="ignore").strip():
            return True
    return False


def _clamp_int(value: int | float, lower: int, upper: int) -> int:
    return int(max(lower, min(upper, int(value))))


def _format_minutes(seconds: float | None) -> str:
    return "unknown" if seconds is None else f"{seconds / 60.0:.1f} min"


def analyze_training_dataset(dataset_dir: str | Path) -> TrainingDatasetAnalysis:
    path = _resolve_dataset_path(dataset_dir)
    warnings: list[str] = []
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return TrainingDatasetAnalysis(path, "metadata.json", False, 0, None, None, None, None, None, None, [], [], [f"metadata.json read failed: {exc}"])

        info = metadata.get("dataset_info") or {}
        samples = [sample for sample in (metadata.get("samples") or []) if isinstance(sample, dict)]
        valid_samples = []
        durations: list[float] = []
        transcript_hits = 0
        transcript_file_mode = bool(samples) and all(sample.get("transcript_file") for sample in samples)

        for sample in samples:
            audio_file = sample.get("audio_file")
            audio_path = path / str(audio_file) if audio_file else None
            if audio_path and audio_path.exists():
                valid_samples.append(sample)
            elif audio_file:
                warnings.append(f"Missing audio file: {audio_file}")
                continue

            raw_duration = sample.get("duration", sample.get("audio_duration"))
            try:
                if raw_duration is not None:
                    durations.append(float(raw_duration))
            except (TypeError, ValueError):
                warnings.append(f"Invalid duration in sample: {sample.get('id', audio_file or 'unknown')}")

            if transcript_file_mode:
                transcript_file = sample.get("transcript_file")
                transcript_path = path / str(transcript_file)
                if transcript_path.exists() and transcript_path.read_text(encoding="utf-8", errors="ignore").strip():
                    transcript_hits += 1
            elif str(sample.get("text") or sample.get("transcript") or "").strip():
                transcript_hits += 1

        sample_count = len(valid_samples) or int(info.get("total_samples") or len(samples) or 0)
        total_duration = sum(durations) if durations else _safe_float(info.get("total_duration"))
        avg_duration = (sum(durations) / len(durations)) if durations else _safe_float(info.get("avg_duration"))
        transcript_coverage = (transcript_hits / len(samples)) if samples else None
        languages = sorted(
            {
                str(value)
                for value in list(info.get("languages") or []) + [sample.get("language") for sample in samples]
                if value
            }
        )
        speakers = sorted(
            {
                str(value)
                for value in list(info.get("speakers") or [])
                + [sample.get("speaker_id") or sample.get("speaker_name") for sample in samples]
                if value
            }
        )
        return TrainingDatasetAnalysis(
            dataset_dir=path,
            source="metadata.json",
            trainable=metadata_path.exists() and sample_count > 0 and bool(valid_samples or samples),
            sample_count=sample_count,
            total_duration=total_duration,
            avg_duration=avg_duration,
            median_duration=median(durations) if durations else None,
            min_duration=min(durations) if durations else None,
            max_duration=max(durations) if durations else None,
            transcript_coverage=transcript_coverage,
            languages=languages,
            speakers=speakers,
            warnings=warnings,
        )

    warnings.append(
        "metadata.json not found; this folder can be analyzed but Higgs trainers require a prepared dataset. Build Higgs Dataset first."
    )
    audio_files = iter_audio_files(path, recursive=True) if path.exists() else []
    durations = []
    sidecars = 0
    for audio_path in audio_files:
        duration = _duration_from_audio_file(audio_path)
        if duration is None:
            warnings.append(f"Could not read duration: {audio_path.name}")
        else:
            durations.append(duration)
        sidecars += int(_sidecar_exists(audio_path))

    sample_count = len(audio_files)
    total_duration = sum(durations) if durations else None
    return TrainingDatasetAnalysis(
        dataset_dir=path,
        source="audio scan",
        trainable=False,
        sample_count=sample_count,
        total_duration=total_duration,
        avg_duration=(total_duration / sample_count) if total_duration is not None and sample_count else None,
        median_duration=median(durations) if durations else None,
        min_duration=min(durations) if durations else None,
        max_duration=max(durations) if durations else None,
        transcript_coverage=(sidecars / sample_count) if sample_count else None,
        languages=[],
        speakers=[],
        warnings=warnings,
    )


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_audio(path: Path) -> tuple[object, int, float]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    duration = float(audio.shape[0]) / float(sr) if sr else 0.0
    return audio, int(sr), duration


def _write_wav_mono(src: Path, dst: Path, target_sr: int = 24000) -> float:
    import librosa

    audio, sr = librosa.load(str(src), sr=target_sr, mono=True)
    sf.write(str(dst), audio, target_sr)
    return float(len(audio)) / float(target_sr) if target_sr else 0.0


def _sidecar_transcript(audio_path: Path) -> str:
    txt_path = audio_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    json_path = audio_path.with_suffix(".json")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        if isinstance(data, dict):
            for key in ("Text", "text", "Transcript", "transcript", "ReferenceText", "reference_text"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


def _prepare_reference_audio(
    ref_audio_path: str | Path | None,
    out_dir: Path,
    dataset_slug: str,
    split: str,
) -> tuple[str | None, float]:
    if not ref_audio_path:
        return None, 0.0
    src = Path(ref_audio_path)
    if not src.exists() or not src.is_file():
        raise ValueError(f"Reference audio not found: {src}")
    ref_name = f"{dataset_slug}_{split}_reference.wav"
    duration = _write_wav_mono(src, out_dir / ref_name)
    return ref_name, duration


def _sample_metadata(
    sample_id: str,
    audio_filename: str,
    transcript_filename: str,
    duration: float,
    speaker_id: str,
    speaker_name: str,
    language: str,
    original_audio_path: str,
    split: str,
) -> dict:
    return {
        "id": sample_id,
        "audio_file": audio_filename,
        "transcript_file": transcript_filename,
        "duration": round(duration, 3),
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "scene": "quiet_room",
        "emotion": "neutral",
        "language": language,
        "gender": "unknown",
        "quality_score": 1.0,
        "original_audio_path": original_audio_path,
        "user_instruction": "Generate audio following instruction.",
        "task_type": "audio_generation",
        "split": split,
    }


def write_metadata(dataset_dir: Path, samples: list[dict], created_from: list[str]) -> Path:
    total_duration = sum(float(sample.get("duration", 0.0)) for sample in samples)
    metadata = {
        "dataset_info": {
            "total_samples": len(samples),
            "speakers": sorted({sample.get("speaker_id", "speaker") for sample in samples}),
            "languages": sorted({sample.get("language", "unknown") for sample in samples}),
            "total_duration": round(total_duration, 3),
            "avg_duration": round(total_duration / max(len(samples), 1), 3),
            "created_from": created_from,
            "format": "higgs_audio_v2_metadata",
        },
        "samples": samples,
    }
    path = dataset_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _load_metadata_samples(dataset_dir: Path) -> tuple[list[dict], list[str]]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return [], []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return list(metadata.get("samples") or []), list((metadata.get("dataset_info") or {}).get("created_from") or [])


def _copy_reference_from_train(
    train_dir: Path,
    eval_dir: Path,
    dataset_slug: str,
    task_type: str,
    fallback_audio: Path,
    fallback_transcript: str,
) -> dict:
    train_samples, _ = _load_metadata_samples(train_dir)
    first_sample = train_samples[0] if train_samples else {}

    if task_type == "zero_shot_voice_cloning":
        ref_audio = first_sample.get("ref_audio_file")
        ref_transcript = first_sample.get("ref_transcript") or fallback_transcript
        ref_src = train_dir / ref_audio if ref_audio else fallback_audio
        ref_name = f"{dataset_slug}_eval_reference.wav"
        _write_wav_mono(ref_src, eval_dir / ref_name)
        return {"ref_audio_file": ref_name, "ref_transcript": normalize_transcript(ref_transcript)}

    if task_type == "multi_speaker_voice_cloning":
        ref_speakers = first_sample.get("ref_speakers") if isinstance(first_sample, dict) else None
        if isinstance(ref_speakers, list) and ref_speakers:
            copied = []
            for index, ref_info in enumerate(ref_speakers):
                if not isinstance(ref_info, dict):
                    continue
                ref_file = ref_info.get("ref_audio_file")
                ref_src = train_dir / ref_file if ref_file else fallback_audio
                speaker_tag = ref_info.get("speaker_tag", f"[SPEAKER{index}]")
                ref_name = f"{dataset_slug}_eval_reference_speaker{index}.wav"
                _write_wav_mono(ref_src, eval_dir / ref_name)
                copied.append(
                    {
                        "speaker_tag": speaker_tag,
                        "ref_audio_file": ref_name,
                        "ref_transcript": normalize_transcript(ref_info.get("ref_transcript") or fallback_transcript),
                    }
                )
            if copied:
                return {"ref_speakers": copied}

        ref_name = f"{dataset_slug}_eval_reference_speaker0.wav"
        _write_wav_mono(fallback_audio, eval_dir / ref_name)
        return {
            "ref_speakers": [
                {
                    "speaker_tag": "[SPEAKER0]",
                    "ref_audio_file": ref_name,
                    "ref_transcript": normalize_transcript(fallback_transcript),
                }
            ]
        }

    return {}


def append_eval_sample(
    train_dataset_dir: str | Path,
    eval_dataset_dir: str | Path | None,
    target_text: str,
    target_audio_path: str | Path,
    reference_transcript: str = "",
) -> tuple[Path, str]:
    train_dir = ROOT / train_dataset_dir if not os.path.isabs(str(train_dataset_dir)) else Path(train_dataset_dir)
    if not train_dir.exists():
        raise ValueError(f"Train dataset not found: {train_dir}")
    if not target_audio_path:
        raise ValueError("Eval target audio is required.")
    target_audio = Path(target_audio_path)
    if not target_audio.exists():
        raise ValueError(f"Eval target audio not found: {target_audio}")

    target_text = normalize_transcript(target_text)
    if not target_text:
        raise ValueError("Eval text is required.")

    if eval_dataset_dir:
        eval_dir = ROOT / eval_dataset_dir if not os.path.isabs(str(eval_dataset_dir)) else Path(eval_dataset_dir)
    else:
        eval_dir = train_dir.parent / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_samples, train_created_from = _load_metadata_samples(train_dir)
    if not train_samples:
        raise ValueError(f"Train dataset has no metadata samples: {train_dir}")
    first_train = train_samples[0]
    task_type = dataset_task_type(train_dir) or "single_speaker_smart_voice"
    dataset_slug = train_dir.parent.name

    eval_samples, eval_created_from = _load_metadata_samples(eval_dir)
    next_index = len(eval_samples)
    sample_id = f"{dataset_slug}_eval_{next_index:06d}"
    audio_filename = f"{sample_id}.wav"
    transcript_filename = f"{sample_id}.txt"
    duration = _write_wav_mono(target_audio, eval_dir / audio_filename)
    (eval_dir / transcript_filename).write_text(target_text, encoding="utf-8")

    sample = _sample_metadata(
        sample_id=sample_id,
        audio_filename=audio_filename,
        transcript_filename=transcript_filename,
        duration=duration,
        speaker_id=first_train.get("speaker_id", f"{dataset_slug}_speaker"),
        speaker_name=first_train.get("speaker_name", dataset_slug.replace("_", " ").title()),
        language=first_train.get("language", "unknown"),
        original_audio_path=str(target_audio.resolve()),
        split="eval",
    )
    sample["training_task_type"] = task_type
    sample.update(
        _copy_reference_from_train(
            train_dir=train_dir,
            eval_dir=eval_dir,
            dataset_slug=dataset_slug,
            task_type=task_type,
            fallback_audio=target_audio,
            fallback_transcript=reference_transcript or target_text,
        )
    )
    eval_samples.append(sample)
    created_from = sorted(set(eval_created_from + train_created_from + [str(target_audio.resolve())]))
    write_metadata(eval_dir, eval_samples, created_from)
    ok, report = validate_higgs_dataset(eval_dir)
    if not ok:
        raise RuntimeError(f"Eval sample was written but validation failed.\n{report}")
    return eval_dir, report


def build_higgs_dataset(
    source_folder: str,
    dataset_name: str,
    asr_transcribe: Callable[[str], str],
    asr_transcribe_many: Callable[[list[str]], dict[str, str]] | None = None,
    language: str = "Spanish",
    dataset_task_type: str = "single_speaker_smart_voice",
    ref_audio_path: str | Path | None = None,
    ref_transcript: str = "",
    val_split: float = 0.1,
    recursive: bool = True,
    progress: Callable[[float | tuple[int, int], str], None] | None = None,
) -> DatasetSummary:
    source = Path(source_folder)
    if not source.exists() or not source.is_dir():
        raise ValueError("Choose a valid source folder.")
    audio_paths = iter_audio_files(source, recursive=recursive)
    if not audio_paths:
        raise ValueError("No audio files found in source folder.")

    dataset_slug = slugify(dataset_name)
    dataset_dir = ROOT / "data" / dataset_slug
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    train_dir = dataset_dir / "train"
    eval_dir = dataset_dir / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    ref_transcript = normalize_transcript(ref_transcript or "This is a voice sample for cloning.")
    valid_task_types = {
        "single_speaker_smart_voice",
        "zero_shot_voice_cloning",
        "multi_speaker_smart_voice",
        "multi_speaker_voice_cloning",
    }
    if dataset_task_type not in valid_task_types:
        raise ValueError(f"Unsupported Higgs dataset task type: {dataset_task_type}")
    if dataset_task_type in {"zero_shot_voice_cloning", "multi_speaker_voice_cloning"} and not ref_audio_path:
        raise ValueError(f"{dataset_task_type} requires a reference audio.")

    train_ref_audio, _ = _prepare_reference_audio(ref_audio_path, train_dir, dataset_slug, "train")
    eval_ref_audio, _ = _prepare_reference_audio(ref_audio_path, eval_dir, dataset_slug, "eval")

    lang_code = {
        "English": "en",
        "Spanish": "es",
        "Mandarin": "zh",
        "Chinese": "zh",
        "Korean": "ko",
        "German": "de",
        "Auto-detect": "unknown",
    }.get(language, language[:2].lower() if language else "unknown")
    speaker_id = f"{dataset_slug}_speaker"
    speaker_name = dataset_slug.replace("_", " ").title()
    transcripts: dict[Path, str] = {}
    missing_transcripts: list[Path] = []
    for idx, audio_path in enumerate(audio_paths, 1):
        if progress:
            progress((idx - 1, len(audio_paths)), f"Scanning {audio_path.name}")
        transcript = _sidecar_transcript(audio_path)
        if transcript:
            transcripts[audio_path] = transcript
        else:
            missing_transcripts.append(audio_path)

    if missing_transcripts:
        if progress:
            progress((0, len(missing_transcripts)), f"Transcribing {len(missing_transcripts)} files")
        if asr_transcribe_many:
            batch_results = asr_transcribe_many([str(path) for path in missing_transcripts])
            for audio_path in missing_transcripts:
                transcripts[audio_path] = batch_results.get(str(audio_path), "")
        else:
            for idx, audio_path in enumerate(missing_transcripts, 1):
                if progress:
                    progress((idx - 1, len(missing_transcripts)), f"Transcribing {audio_path.name}")
                transcripts[audio_path] = asr_transcribe(str(audio_path))

    valid_audio_paths = [path for path in audio_paths if normalize_transcript(transcripts.get(path, ""))]
    if not valid_audio_paths:
        raise RuntimeError("No valid samples were created. Add .txt/.json transcripts or enable ASR.")
    split_ratio = max(min(float(val_split), 0.5), 0.0)
    if split_ratio > 0 and len(valid_audio_paths) > 1:
        eval_count_target = min(max(1, int(round(len(valid_audio_paths) * split_ratio))), len(valid_audio_paths) - 1)
    else:
        eval_count_target = 0
    eval_start = len(valid_audio_paths) - eval_count_target

    train_samples: list[dict] = []
    eval_samples: list[dict] = []
    kept = 0
    skipped = 0
    for idx, audio_path in enumerate(valid_audio_paths, 1):
        if progress:
            progress((idx - 1, len(valid_audio_paths)), f"Writing {audio_path.name}")
        transcript = transcripts.get(audio_path, "")
        transcript = normalize_transcript(transcript)
        if not transcript:
            skipped += 1
            continue

        split = "eval" if idx > eval_start else "train"
        out_dir = eval_dir if split == "eval" else train_dir
        sample_id = f"{dataset_slug}_{kept:06d}"
        audio_filename = f"{sample_id}.wav"
        transcript_filename = f"{sample_id}.txt"
        duration = _write_wav_mono(audio_path, out_dir / audio_filename)
        (out_dir / transcript_filename).write_text(transcript, encoding="utf-8")
        sample = _sample_metadata(
            sample_id,
            audio_filename,
            transcript_filename,
            duration,
            speaker_id,
            speaker_name,
            lang_code,
            str(audio_path.resolve()),
            split,
        )
        sample["training_task_type"] = dataset_task_type
        if dataset_task_type == "zero_shot_voice_cloning":
            sample["ref_audio_file"] = eval_ref_audio if split == "eval" else train_ref_audio
            sample["ref_transcript"] = ref_transcript
        elif dataset_task_type == "multi_speaker_voice_cloning":
            sample["ref_speakers"] = [
                {
                    "speaker_tag": "[SPEAKER0]",
                    "ref_audio_file": eval_ref_audio if split == "eval" else train_ref_audio,
                    "ref_transcript": ref_transcript,
                }
            ]
        if split == "eval":
            eval_samples.append(sample)
        else:
            train_samples.append(sample)
        kept += 1

    if not train_samples:
        raise RuntimeError("No valid train samples were created.")
    if not eval_samples:
        shutil.rmtree(eval_dir, ignore_errors=True)
        eval_dir_final = None
    else:
        write_metadata(eval_dir, eval_samples, [str(source.resolve())])
        eval_dir_final = eval_dir
    write_metadata(train_dir, train_samples, [str(source.resolve())])
    all_samples = train_samples + eval_samples
    write_metadata(dataset_dir, all_samples, [str(source.resolve())])
    if progress:
        progress((len(audio_paths), len(audio_paths)), "Dataset ready")

    total_duration = sum(sample["duration"] for sample in all_samples)
    return DatasetSummary(
        dataset_dir=dataset_dir,
        train_dir=train_dir,
        eval_dir=eval_dir_final,
        train_samples=len(train_samples),
        eval_samples=len(eval_samples),
        total_duration=total_duration,
        avg_duration=total_duration / max(len(all_samples), 1),
        languages=sorted({sample["language"] for sample in all_samples}),
        speakers=sorted({sample["speaker_id"] for sample in all_samples}),
    )


def validate_higgs_dataset(dataset_dir: str | Path) -> tuple[bool, str]:
    dataset_path = Path(dataset_dir)
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.exists():
        return False, f"metadata.json not found: {metadata_path}"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = metadata.get("samples", [])
    valid = 0
    invalid = 0
    total_duration = 0.0
    messages = [f"Validating {dataset_path}", f"Samples in metadata: {len(samples)}"]
    for sample in samples:
        audio_file = dataset_path / sample.get("audio_file", "")
        transcript_file = dataset_path / sample.get("transcript_file", "")
        if not audio_file.exists() or not transcript_file.exists():
            invalid += 1
            messages.append(f"Missing pair: {sample.get('id', 'unknown')}")
            continue
        try:
            _, _, duration = _read_audio(audio_file)
        except Exception as exc:
            invalid += 1
            messages.append(f"Audio load failed {audio_file.name}: {exc}")
            continue
        text = transcript_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            invalid += 1
            messages.append(f"Empty transcript: {transcript_file.name}")
            continue
        ref_audio = sample.get("ref_audio_file")
        if ref_audio and not (dataset_path / ref_audio).exists():
            invalid += 1
            messages.append(f"Missing reference audio for {sample.get('id', 'unknown')}: {ref_audio}")
            continue
        ref_speakers = sample.get("ref_speakers") or []
        if ref_speakers and not isinstance(ref_speakers, list):
            invalid += 1
            messages.append(f"Invalid ref_speakers for {sample.get('id', 'unknown')}: expected list")
            continue
        missing_ref = False
        for ref_info in ref_speakers:
            ref_file = ref_info.get("ref_audio_file") if isinstance(ref_info, dict) else None
            if not ref_file or not (dataset_path / ref_file).exists():
                invalid += 1
                missing_ref = True
                messages.append(f"Missing speaker reference for {sample.get('id', 'unknown')}: {ref_file}")
                break
        if missing_ref:
            continue
        valid += 1
        total_duration += duration
    messages.append(f"Valid samples: {valid}")
    messages.append(f"Invalid samples: {invalid}")
    messages.append(f"Total duration: {total_duration / 3600:.3f} h")
    messages.append(f"Average duration: {total_duration / max(valid, 1):.2f} s")
    return invalid == 0 and valid > 0, "\n".join(messages)


def list_train_datasets() -> list[str]:
    data_root = ROOT / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    choices = []
    for metadata in sorted(data_root.rglob("metadata.json")):
        rel = metadata.parent.relative_to(ROOT).as_posix()
        if rel.endswith("/train") or rel.endswith("/eval"):
            choices.append(rel)
    return choices


def dataset_task_type(dataset_dir: str | Path) -> str | None:
    dataset_path = ROOT / dataset_dir if dataset_dir and not os.path.isabs(str(dataset_dir)) else Path(dataset_dir)
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    samples = metadata.get("samples") or []
    for sample in samples:
        value = sample.get("training_task_type")
        if value:
            return value
        if sample.get("ref_speakers"):
            return "multi_speaker_voice_cloning"
        if sample.get("ref_audio_file"):
            return "zero_shot_voice_cloning"
    return None


DEFAULT_TRAINING_24GB = {
    "batch": 1,
    "grad_accum": 8,
    "lr": 2e-5,
    "lora_rank": 16,
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 250,
    "eval_steps": 250,
}


def build_training_command(
    train_data_dir: str,
    eval_data_dir: str,
    output_name: str,
    epochs: int,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    lora_rank: int,
    use_lora: bool,
    bf16: bool,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    resume_checkpoint: str = "",
    enable_eval_audio: bool = False,
    eval_text: str = "",
    eval_audio_max_new_tokens: int = 1000,
) -> tuple[list[str], str]:
    train_dir = ROOT / train_data_dir if not os.path.isabs(train_data_dir) else Path(train_data_dir)
    eval_dir = ROOT / eval_data_dir if eval_data_dir and not os.path.isabs(eval_data_dir) else Path(eval_data_dir) if eval_data_dir else None
    out_dir = ROOT / "exp" / slugify(output_name, "higgs_v2_lora")
    script = ROOT / "train-higgs-audio" / "trainer" / "v2_hf_trainer.py"

    args = [
        sys.executable,
        str(script),
        "--model_path",
        str(TTS_MODELS["Higgs V2 TTS"].local_dir),
        "--train_data_dir",
        str(train_dir),
        "--output_dir",
        str(out_dir),
        "--num_train_epochs",
        str(int(epochs)),
        "--batch_size",
        str(int(batch_size)),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--learning_rate",
        str(float(learning_rate)),
        "--logging_steps",
        str(int(logging_steps)),
        "--save_steps",
        str(int(save_steps)),
        "--eval_steps",
        str(int(eval_steps)),
    ]
    if eval_dir:
        args.extend(["--eval_data_dir", str(eval_dir)])
    if use_lora:
        args.extend(["--use_lora", "--lora_rank", str(int(lora_rank)), "--lora_alpha", str(int(lora_rank) * 2)])
    if bf16:
        args.append("--bf16")
    if max_steps > 0:
        args.extend(["--max_steps", str(int(max_steps))])
    if resume_checkpoint:
        ckpt = ROOT / resume_checkpoint if not os.path.isabs(resume_checkpoint) else Path(resume_checkpoint)
        args.extend(["--resume_checkpoint", str(ckpt)])
    if enable_eval_audio:
        args.append("--enable_eval_audio")
        args.extend(["--eval_text", eval_text or "This is my voice evolution during training."])
        args.extend(["--eval_audio_max_new_tokens", str(int(eval_audio_max_new_tokens))])

    pretty = subprocess.list2cmdline(args)
    pretty += f"\n\n# Native Transformers V2 trainer from HF README: single-speaker only, use_text_head=True, output_labels=True. Effective batch: {int(batch_size)} x grad_accum {int(grad_accum)}. The project root is kept as latest; checkpoint-* folders keep snapshots. Both include LoRA/model, processor, optimizer, scheduler, RNG, and optimizer_step."
    return args, pretty


def build_v3_training_command(
    train_data_dir: str,
    eval_data_dir: str,
    output_name: str,
    task_type: str,
    epochs: int,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    lora_rank: int,
    use_lora: bool,
    bf16: bool,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    freeze_audio_head: bool,
    resume_checkpoint: str = "",
    enable_eval_audio: bool = False,
    eval_text: str = "",
    eval_audio_max_new_tokens: int = 1000,
) -> tuple[list[str], str]:
    if task_type not in {"single_speaker_smart_voice", "zero_shot_voice_cloning"}:
        raise ValueError("Custom V3 trainer currently supports single_speaker_smart_voice and zero_shot_voice_cloning only.")
    train_dir = ROOT / train_data_dir if not os.path.isabs(train_data_dir) else Path(train_data_dir)
    eval_dir = ROOT / eval_data_dir if eval_data_dir and not os.path.isabs(eval_data_dir) else Path(eval_data_dir) if eval_data_dir else None
    out_dir = ROOT / "exp" / slugify(output_name, "higgs_v3_lora")
    script = ROOT / "train-higgs-audio" / "trainer" / "v3_trainer.py"
    args = [
        sys.executable,
        str(script),
        "--model_path",
        str(TTS_MODELS["Higgs V3 TTS"].local_dir),
        "--audio_tokenizer_path",
        str(TOKENIZER_MODELS["Higgs V2 tokenizer"].local_dir),
        "--train_data_dir",
        str(train_dir),
        "--output_dir",
        str(out_dir),
        "--task_type",
        task_type,
        "--num_train_epochs",
        str(int(epochs)),
        "--batch_size",
        str(int(batch_size)),
        "--gradient_accumulation_steps",
        str(int(grad_accum)),
        "--learning_rate",
        str(float(learning_rate)),
        "--logging_steps",
        str(int(logging_steps)),
        "--save_steps",
        str(int(save_steps)),
        "--eval_steps",
        str(int(eval_steps)),
        "--lora_rank",
        str(int(lora_rank)),
        "--lora_alpha",
        str(int(lora_rank) * 2),
    ]
    if eval_dir:
        args.extend(["--eval_data_dir", str(eval_dir)])
    if max_steps > 0:
        args.extend(["--max_steps", str(int(max_steps))])
    if resume_checkpoint:
        ckpt = ROOT / resume_checkpoint if not os.path.isabs(resume_checkpoint) else Path(resume_checkpoint)
        args.extend(["--resume_checkpoint", str(ckpt)])
    if enable_eval_audio:
        args.append("--enable_eval_audio")
        if eval_text:
            args.extend(["--eval_text", eval_text])
        args.extend(["--eval_audio_max_new_tokens", str(int(eval_audio_max_new_tokens))])
    if use_lora:
        args.append("--use_lora")
    if bf16:
        args.append("--bf16")
    if freeze_audio_head:
        args.append("--freeze_audio_head")
    pretty = subprocess.list2cmdline(args)
    pretty += f"\n\n# Custom Higgs V3 trainer: teacher-forced CE over delayed 8-codebook audio rows. Effective batch: {int(batch_size)} x grad_accum {int(grad_accum)}. The project root is kept as latest; checkpoint-* folders keep snapshots. LoRA adapters are saved as qwen3_lora with optimizer, scheduler, RNG, and optimizer_step."
    return args, pretty
