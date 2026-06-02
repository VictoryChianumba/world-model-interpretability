# Precomputed caches

Generated artifacts checked in so a fresh clone / demo has data without re-running the
hours-long offline pipelines. **These are derived data, not source** — they are tied to a
specific SAE checkpoint and go stale if that checkpoint is retrained.

## `causal_L5.json` — causal-importance ranking snapshot

Per-feature causal importance (mean token divergence under ±scale intervention rollouts),
read by `GET /ranking/causal` and the discovery panel's **causal** toggle.

| field | value |
|---|---|
| SAE checkpoint | `sae_L5.pt` (sha256 `ee5ee63ad564d85d…`) |
| layer | 5 |
| env | `BreakoutNoFrameskip-v4` |
| params | `--seeds 2 --n-steps 10 --scale 5` |
| features scored | **2048 (full run)** |
| compute | RunPod A100-80GB, ~3h (CUDA) |
| generated | 2026-06-02 |

**Full coverage.** All 2048 features, scored against one consistent set of seed-states in a
single GPU run (top by score: #1364 15.0, #286 11.0, #497 10.4 — the air-flight tracker
#1364 remains #1, matching the earlier CPU sample). Regenerate with:

```bash
python backend/scripts/causal_importance.py \
  --checkpoint <iris>/checkpoints/Breakout.pt --sae <iris>/checkpoints/sae_L5.pt \
  --seeds 2 --n-steps 10 --scale 5 --device mps --resume
```

**To use this snapshot**, copy it next to the SAE artifact (the backend reads from
`SAE_DIR`, default = the checkpoint dir):

```bash
cp data/causal_L5.json <iris>/checkpoints/causal_L5.json
```

If the SAE is retrained (hash changes), delete and regenerate — the scores won't correspond
to the new dictionary.
