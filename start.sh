#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv was not found. Run ./install.sh first."
  exit 1
fi

APP_CACHE="$PWD/models/.cache"
export TMPDIR="$APP_CACHE/tmp"
export GRADIO_TEMP_DIR="$APP_CACHE/tmp"
export HF_HOME="$PWD/models"
export HUGGINGFACE_HUB_CACHE="$APP_CACHE/huggingface"
export HF_XET_CACHE="$APP_CACHE/xet"
export TRANSFORMERS_CACHE="$PWD/models"
export TORCH_HOME="$APP_CACHE/torch"
export XDG_CACHE_HOME="$APP_CACHE"
export UV_CACHE_DIR="$APP_CACHE/uv"
export HF_MODULES_CACHE="$APP_CACHE/hf_modules"
export PYTHONIOENCODING="utf-8"
export PYTHONUNBUFFERED="1"
export HIGGS_CPU_THREADS="${HIGGS_CPU_THREADS:-8}"
export OMP_NUM_THREADS="$HIGGS_CPU_THREADS"
export MKL_NUM_THREADS="$HIGGS_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$HIGGS_CPU_THREADS"

mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$HF_MODULES_CACHE"
uv run --no-sync python app.py
