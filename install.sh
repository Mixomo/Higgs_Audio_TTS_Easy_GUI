#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    uv --version
    return
  fi
  echo "[install] uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || {
    echo "[ERROR] uv was installed but is not visible in PATH. Open a new terminal and rerun ./install.sh."
    exit 1
  }
}

detect_backend() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo cpu
    return
  fi
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  echo "[install] Detected GPU: ${gpu:-unknown}" >&2
  case "$gpu" in
    *"GTX 10"*) echo cu118 ;;
    *"RTX 20"*|*"RTX 30"*) echo cu126 ;;
    *) echo cu128 ;;
  esac
}

install_torch() {
  case "$1" in
    cpu) index="cpu" ;;
    cu118|cu126|cu128) index="$1" ;;
    *) echo "[ERROR] Unknown backend: $1"; exit 1 ;;
  esac
  echo "[install] Installing PyTorch $1 wheels..."
  uv pip install --reinstall torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url "https://download.pytorch.org/whl/$index"
}

ensure_uv

APP_CACHE="$PWD/models/.cache"
mkdir -p models samples outputs logs data exp config "$APP_CACHE/uv" "$APP_CACHE/tmp" "$APP_CACHE/huggingface" "$APP_CACHE/xet"
export UV_CACHE_DIR="$APP_CACHE/uv"
export XDG_CACHE_HOME="$APP_CACHE"
export HUGGINGFACE_HUB_CACHE="$APP_CACHE/huggingface"
export HF_XET_CACHE="$APP_CACHE/xet"
export TMPDIR="$APP_CACHE/tmp"
export UV_LINK_MODE=copy

cat <<'EOF'

Select PyTorch backend:
  1. Auto-detect NVIDIA / CPU
  2. NVIDIA GTX 10xx Pascal - CUDA 11.8
  3. NVIDIA RTX 20xx/30xx - CUDA 12.6
  4. NVIDIA RTX 40xx/50xx - CUDA 12.8
  5. CPU only
EOF
read -r -p "Choose backend (1-5, default 1): " choice

case "${choice:-1}" in
  1) TORCH_BACKEND="$(detect_backend)" ;;
  2) TORCH_BACKEND=cu118 ;;
  3) TORCH_BACKEND=cu126 ;;
  4) TORCH_BACKEND=cu128 ;;
  5) TORCH_BACKEND=cpu ;;
  *) TORCH_BACKEND=cpu ;;
esac

echo "$TORCH_BACKEND" > torch_backend.txt
echo "[install] Selected backend: $TORCH_BACKEND"

echo "[install] Installing app dependencies..."
uv sync --inexact --no-install-package torch --no-install-package torchaudio --no-install-package torchvision
install_torch "$TORCH_BACKEND"

if [[ "$TORCH_BACKEND" == cu* ]]; then
  echo "[install] Installing Triton for torch.compile..."
  uv pip install "triton>=3.0.0,<3.4"
else
  echo "[install] Skipping Triton; torch.compile acceleration is CUDA-only in this installer."
fi

echo "[install] Verifying Python/Torch runtime..."
uv run --no-sync python -c "import sys, torch; backend='$TORCH_BACKEND'; cuda=torch.cuda.is_available(); print('[torch]', torch.__version__); print('[cuda_available]', cuda); print('[cuda_version]', torch.version.cuda); print('[device]', torch.cuda.get_device_name(0) if cuda else 'cpu'); sys.exit(1 if backend.startswith('cu') and not cuda else 0)"

echo
echo "[DONE] Install completed. Run ./start.sh"
