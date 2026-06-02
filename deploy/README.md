# Cloud-GPU causal run (RunPod)

Run the full causal-importance pipeline (`scripts/causal_importance.py`) on a CUDA GPU
instead of local CPU/MPS. The local run managed ~70 features/hr (contended); a GPU should
do the full ~2048 features in a couple of hours.

Nothing here is committed-into-git data — it's the *recipe*. The 121 MB `Breakout.pt`
checkpoint is rsynced straight to the pod, not into the repo.

## What runs where

- **Locally** (this machine, has the working `iris/.venv310` + checkpoints): `transfer.sh`
  pushes code + IRIS + checkpoints to the pod.
- **On the pod** (fresh RunPod CUDA box): `setup_pod.sh` builds the env, `run_causal.sh`
  runs the job.

## Prereqs / assumptions

- Pod image: a CUDA box with **Python 3.10** (ale-py 0.7.4 / gym 0.25.2 have cp310 wheels;
  3.11+ may force source builds — prefer a 3.10 image, e.g. RunPod "PyTorch 2.x / py3.10").
- The working dependency set is frozen in `requirements-cloud.txt` (from `iris/.venv310`,
  torch/torchvision excluded — those install as CUDA wheels in `setup_pod.sh`).
- torch **2.10.0** / torchvision **0.25.0** (match the local versions; CUDA wheel chosen by
  the pod's CUDA version, auto-detected in `setup_pod.sh`).

## Steps

```bash
# 0) Get the RunPod SSH details. Export them locally (example):
export REMOTE=root@213.181.x.x          # or the runpod user@host
export PORT=22                           # RunPod-assigned SSH port
export SSH_KEY=~/.ssh/id_ed25519         # the key RunPod has
export REMOTE_DIR=/root/wm-causal        # where everything lands on the pod

# 1) Push code + IRIS + checkpoints + the 644-feature resume cache (run locally):
bash deploy/transfer.sh

# 2) Build the env on the pod (run on the pod, or via ssh):
ssh -p $PORT -i $SSH_KEY $REMOTE 'cd /root/wm-causal && bash deploy/setup_pod.sh'

# 3) Launch the causal run on the pod (nohup; logs to causal.log):
ssh -p $PORT -i $SSH_KEY $REMOTE 'cd /root/wm-causal && SEEDS=2 NSTEPS=10 SCALE=5 bash deploy/run_causal.sh'

# 4) Watch progress:
ssh -p $PORT -i $SSH_KEY $REMOTE 'tail -f /root/wm-causal/causal.log'

# 5) When done, pull the result back (run locally):
bash deploy/fetch_result.sh
#   → updates iris/checkpoints/causal_L5.json AND data/causal_L5.json
```

## Notes

- `run_causal.sh` uses `--resume`, so it continues from the transferred 644-feature cache
  (same params: seeds 2, n-steps 10, scale 5). To do a cleaner, more robust full run
  instead, delete the cache on the pod and bump `SEEDS=3` — but that discards the 644 and
  starts fresh.
- The pipeline scores features most-active-first and saves every `--save-every` (50 here),
  so it is crash-safe and `fetch_result.sh` can be run any time for a partial.
- If the SAE (`sae_L5.pt`) is ever retrained, the cache is stale — regenerate.
