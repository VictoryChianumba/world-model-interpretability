#!/usr/bin/env bash
# Pull the causal_L5.json result back from the pod. Run LOCALLY. Updates both the live
# cache (iris/checkpoints) and the committed snapshot (data/). Driven by the same env vars
# as transfer.sh: REMOTE, PORT, SSH_KEY, REMOTE_DIR.
set -euo pipefail

: "${REMOTE:?set REMOTE=user@host}"
: "${PORT:=22}"
: "${SSH_KEY:?set SSH_KEY=path to private key}"
: "${REMOTE_DIR:=/root/wm-causal}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
IRIS="${IRIS_ROOT:-$(cd "$REPO/../iris" && pwd)}"
SSH="ssh -p $PORT -i $SSH_KEY"

rsync -avz -e "$SSH" "$REMOTE:$REMOTE_DIR/iris/checkpoints/causal_L5.json" "$IRIS/checkpoints/causal_L5.json"
cp "$IRIS/checkpoints/causal_L5.json" "$REPO/data/causal_L5.json"

python3 - "$REPO/data/causal_L5.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f">> fetched {len(d['scores'])} features (layer {d['layer']}, seeds {d.get('seeds')}, "
      f"n_steps {d.get('n_steps')}, scale {d.get('scale')})")
PY
echo ">> updated iris/checkpoints/causal_L5.json and data/causal_L5.json"
echo ">> remember to: git add data/causal_L5.json data/README.md && git commit"
