#!/usr/bin/env bash
# Launch the causal-importance run on the pod (CUDA). Run ON THE POD from $REMOTE_DIR.
# Continues from the transferred 644-feature cache via --resume.
#   Tunables: SEEDS (2) NSTEPS (10) SCALE (5) FEATURES (0=all) DEVICE (cuda)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
. venv/bin/activate

export IRIS_ROOT="$ROOT/iris"
export IRIS_SRC="$ROOT/iris/src"

SEEDS="${SEEDS:-2}"; NSTEPS="${NSTEPS:-10}"; SCALE="${SCALE:-5}"
FEATURES="${FEATURES:-0}"; DEVICE="${DEVICE:-cuda}"

CMD="python backend/scripts/causal_importance.py \
  --checkpoint iris/checkpoints/Breakout.pt \
  --sae iris/checkpoints/sae_L5.pt \
  --out-dir iris/checkpoints \
  --device $DEVICE --seeds $SEEDS --n-steps $NSTEPS --scale $SCALE \
  --features $FEATURES --save-every 50 --resume"

echo ">> $CMD"
nohup $CMD > "$ROOT/causal.log" 2>&1 &
echo ">> started (pid $!). tail -f $ROOT/causal.log"
