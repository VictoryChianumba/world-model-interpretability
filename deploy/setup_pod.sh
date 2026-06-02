#!/usr/bin/env bash
# Build the Python env on the RunPod. Run ON THE POD from $REMOTE_DIR (default /root/wm-causal).
# Reproduces iris/.venv310: torch CUDA wheel + the frozen deps + Atari ROMs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3.10}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
echo ">> using $($PY --version 2>&1)"
case "$($PY --version 2>&1)" in
  *3.10*) ;;
  *) echo "!! WARNING: ale-py 0.7.4 / gym 0.25.2 expect Python 3.10 — 3.11+ may build from source and fail." ;;
esac

# Detect CUDA → pick the matching PyTorch wheel index.
CUDA_TAG=cu121
if command -v nvidia-smi >/dev/null 2>&1; then
  CVER=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
  echo ">> driver CUDA: ${CVER:-unknown}"
  case "$CVER" in
    12.4*|12.5*|12.6*|12.7*|12.8*|13.*) CUDA_TAG=cu124 ;;
    12.*) CUDA_TAG=cu121 ;;
    11.*) CUDA_TAG=cu118 ;;
  esac
fi
echo ">> torch wheel tag: $CUDA_TAG"

$PY -m venv venv
. venv/bin/activate
pip install --upgrade pip wheel

# torch CUDA first (match local versions; fall back to latest cu wheel if the exact pin is
# missing for this CUDA tag).
pip install "torch==2.10.0" "torchvision==0.25.0" --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
  || pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

pip install -r deploy/requirements-cloud.txt

# Atari ROMs for ale-py (gym 0.25 + ale-py 0.7.4).
AutoROM --accept-license || python -m AutoROM --accept-license || true

echo ">> smoke test:"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
import gym, ale_py, hydra, einops  # noqa
print("gym/ale-py/hydra/einops import OK")
PY
echo ">> setup complete. Run: SEEDS=2 NSTEPS=10 SCALE=5 bash deploy/run_causal.sh"
