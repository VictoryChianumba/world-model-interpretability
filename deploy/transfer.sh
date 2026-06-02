#!/usr/bin/env bash
# Push code + IRIS + checkpoints to the RunPod. Run LOCALLY (this machine has the venv +
# checkpoints). Driven by env vars:
#   REMOTE=root@host  PORT=22  SSH_KEY=~/.ssh/id_ed25519  REMOTE_DIR=/root/wm-causal
set -euo pipefail

: "${REMOTE:?set REMOTE=user@host}"
: "${PORT:=22}"
: "${SSH_KEY:?set SSH_KEY=path to private key}"
: "${REMOTE_DIR:=/root/wm-causal}"

# Resolve repo + iris roots relative to this script.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
IRIS="${IRIS_ROOT:-$(cd "$REPO/../iris" && pwd)}"

SSH="ssh -p $PORT -i $SSH_KEY"
RSYNC="rsync -avz --progress -e \"$SSH\""

echo ">> remote: $REMOTE  dir: $REMOTE_DIR  (iris=$IRIS)"
$SSH "$REMOTE" "mkdir -p $REMOTE_DIR/iris/checkpoints"

# 1) backend code + deploy scripts (small)
eval $RSYNC --exclude '__pycache__' --exclude '*.pyc' \
  "$REPO/backend" "$REPO/deploy" "$REMOTE:$REMOTE_DIR/"

# 2) IRIS source + config (small)
eval $RSYNC --exclude '__pycache__' "$IRIS/src" "$IRIS/config" "$REMOTE:$REMOTE_DIR/iris/"

# 3) checkpoints needed by the causal run (121M + 4M)
eval $RSYNC "$IRIS/checkpoints/Breakout.pt" "$IRIS/checkpoints/sae_L5.pt" \
  "$REMOTE:$REMOTE_DIR/iris/checkpoints/"

# 4) the 644-feature snapshot so the pod run can --resume from it
if [ -f "$REPO/data/causal_L5.json" ]; then
  eval $RSYNC "$REPO/data/causal_L5.json" "$REMOTE:$REMOTE_DIR/iris/checkpoints/causal_L5.json"
  echo ">> seeded resume cache (644 features)"
fi

echo ">> transfer complete."
