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
| features scored | 644 (most-active-first; partial run, stopped early) |
| generated | 2026-06-01 |

**Partial by design.** The pipeline scores features in descending activation order and was
stopped at 644/2048 — the silent majority of the remaining features have ≈0 activation and
≈0 causal effect, so the meaningful features are covered. Regenerate / extend with:

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
