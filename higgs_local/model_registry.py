from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from .paths import CACHE_DIR, MODELS_DIR


os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("HF_XET_CACHE", str(CACHE_DIR / "xet"))


@dataclass(frozen=True)
class ModelSpec:
    label: str
    repo_id: str
    local_name: str
    kind: str

    @property
    def local_dir(self) -> Path:
        return MODELS_DIR / self.local_name


TTS_MODELS: dict[str, ModelSpec] = {
    "Higgs V3 TTS": ModelSpec(
        "Higgs V3 TTS",
        "multimodalart/higgs-audio-v3-tts-4b-transformers",
        "higgs-audio-v3-tts-4b-transformers",
        "tts",
    ),
    "Higgs V2 TTS": ModelSpec(
        "Higgs V2 TTS",
        "bosonai/higgs-audio-v2-generation-3B-base",
        "higgs-audio-v2-generation-3B-base",
        "tts",
    ),
}

TOKENIZER_MODELS: dict[str, ModelSpec] = {
    "Higgs V2 tokenizer": ModelSpec(
        "Higgs V2 tokenizer",
        "bosonai/higgs-audio-v2-tokenizer",
        "higgs-audio-v2-tokenizer",
        "tokenizer",
    ),
}

PROCESSOR_MODELS: dict[str, ModelSpec] = {
    "Whisper large-v3 processor": ModelSpec(
        "Whisper large-v3 processor",
        "openai/whisper-large-v3",
        "whisper-large-v3",
        "processor",
    ),
}

HIGGS_ASR_MODELS: dict[str, ModelSpec] = {}

WHISPER_MODELS: dict[str, ModelSpec] = {
    "Faster-Whisper tiny": ModelSpec("Faster-Whisper tiny", "Systran/faster-whisper-tiny", "faster-whisper-tiny", "asr"),
    "Faster-Whisper base": ModelSpec("Faster-Whisper base", "Systran/faster-whisper-base", "faster-whisper-base", "asr"),
    "Faster-Whisper small": ModelSpec("Faster-Whisper small", "Systran/faster-whisper-small", "faster-whisper-small", "asr"),
    "Faster-Whisper medium": ModelSpec("Faster-Whisper medium", "Systran/faster-whisper-medium", "faster-whisper-medium", "asr"),
    "Faster-Whisper large-v2": ModelSpec(
        "Faster-Whisper large-v2",
        "Systran/faster-whisper-large-v2",
        "faster-whisper-large-v2",
        "asr",
    ),
    "Faster-Whisper large-v3": ModelSpec(
        "Faster-Whisper large-v3",
        "Systran/faster-whisper-large-v3",
        "faster-whisper-large-v3",
        "asr",
    ),
    "Faster-Whisper distil-large-v3": ModelSpec(
        "Faster-Whisper distil-large-v3",
        "Systran/faster-distil-whisper-large-v3",
        "faster-distil-whisper-large-v3",
        "asr",
    ),
}

ASR_MODELS: dict[str, ModelSpec] = dict(WHISPER_MODELS)


def model_status(spec: ModelSpec) -> str:
    return "installed" if spec.local_dir.exists() and any(spec.local_dir.iterdir()) else "missing"


def ensure_model(spec: ModelSpec, force: bool = False, progress_cb=None) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not force and model_status(spec) == "installed":
        if progress_cb:
            progress_cb(1.0, f"{spec.label} already installed")
        return spec.local_dir
    print(f"[hf] Downloading {spec.label} from Hugging Face into {spec.local_dir}", flush=True)
    kwargs = {
        "repo_id": spec.repo_id,
        "local_dir": str(spec.local_dir),
        "local_dir_use_symlinks": False,
    }
    if progress_cb:
        progress_cb(None, f"Downloading {spec.label}; Hugging Face progress is shown in CMD")
    try:
        snapshot_download(**kwargs)
    except TypeError:
        snapshot_download(**kwargs)
    if progress_cb:
        progress_cb(1.0, f"{spec.label} download ready")
    print(f"[hf] Ready: {spec.label}", flush=True)
    return spec.local_dir


def require_model(spec: ModelSpec) -> Path:
    if model_status(spec) != "installed":
        raise FileNotFoundError(
            f"{spec.label} is not installed in {spec.local_dir}. Select it in the GUI flow to download it on demand."
        )
    return spec.local_dir


def ensure_tts_model(label: str, progress_cb=None) -> Path:
    if label not in TTS_MODELS:
        raise ValueError(f"Unknown TTS model: {label}")
    path = ensure_model(TTS_MODELS[label], progress_cb=progress_cb)
    if label == "Higgs V2 TTS":
        ensure_model(TOKENIZER_MODELS["Higgs V2 tokenizer"], progress_cb=progress_cb)
    return path


def require_tts_model(label: str) -> Path:
    if label not in TTS_MODELS:
        raise ValueError(f"Unknown TTS model: {label}")
    path = require_model(TTS_MODELS[label])
    if label == "Higgs V2 TTS":
        require_model(TOKENIZER_MODELS["Higgs V2 tokenizer"])
    return path


def ensure_asr_model(label: str, progress_cb=None) -> Path:
    if label not in ASR_MODELS:
        raise ValueError(f"Unknown ASR model: {label}")
    return ensure_model(ASR_MODELS[label], progress_cb=progress_cb)


def install_default_models(include_asr: bool = True) -> list[str]:
    installed = []
    for spec in [
        TTS_MODELS["Higgs V3 TTS"],
        TTS_MODELS["Higgs V2 TTS"],
        TOKENIZER_MODELS["Higgs V2 tokenizer"],
        PROCESSOR_MODELS["Whisper large-v3 processor"],
    ]:
        ensure_model(spec)
        installed.append(f"{spec.label}: {spec.local_dir}")
    if include_asr:
        ensure_model(WHISPER_MODELS["Faster-Whisper large-v3"])
        installed.append(f"Faster-Whisper large-v3: {WHISPER_MODELS['Faster-Whisper large-v3'].local_dir}")
    return installed


def describe_models() -> str:
    groups = [
        ("TTS", TTS_MODELS.values()),
        ("Tokenizer", TOKENIZER_MODELS.values()),
        ("Processor", PROCESSOR_MODELS.values()),
        ("Faster-Whisper ASR", WHISPER_MODELS.values()),
    ]
    lines = []
    for title, specs in groups:
        lines.append(f"[{title}]")
        for spec in specs:
            lines.append(f"{spec.label}: {model_status(spec)} -> {spec.local_dir}")
    return "\n".join(lines)
