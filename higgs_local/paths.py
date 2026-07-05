from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CACHE_DIR = MODELS_DIR / ".cache"
VOICES_DIR = ROOT / "voices"
SAMPLES_DIR = ROOT / "samples"
OUTPUTS_DIR = ROOT / "outputs"
LOGS_DIR = ROOT / "logs"
TEMP_DIR = CACHE_DIR / "tmp"

V3_MODEL_PATH = MODELS_DIR / "higgs-audio-v3-tts-4b-transformers"
V2_DEFAULT_MODEL = "bosonai/higgs-audio-v2-generation-3B-base"
V2_DEFAULT_TOKENIZER = "bosonai/higgs-audio-v2-tokenizer"
V3_TTS_REPO = "bosonai/higgs-audio-v3-tts-4b"


def ensure_local_dirs() -> None:
    for path in (MODELS_DIR, CACHE_DIR, SAMPLES_DIR, OUTPUTS_DIR, LOGS_DIR, TEMP_DIR):
        path.mkdir(parents=True, exist_ok=True)
