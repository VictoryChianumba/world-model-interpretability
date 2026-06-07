# Causal-importance research artifacts

These are **research artifacts, not a shipped feature**. Causal importance was investigated
as a third discovery-panel ranking and **deliberately not integrated** — fixed-norm causal
importance reached only cross-set Spearman ≈ 0.49 (below the pre-committed 0.6 bar), so it is
not reproducible enough to rank features in the UI. See `REPORT.md` §4 ("Clean negative result")
for the full story (magnitude confound → single-state limitation → robustness plateau → decision). The pipeline survives as a *characterization* tool
(`backend/scripts/causal_importance.py` + the `/feature/{id}` endpoint).

> The earlier `causal_L5.json` (a 2048-feature **magnitude-relative** run) was removed: it was
> confounded — its scores covaried with activation, so it was a magnitude ranking in disguise.

## `causal_fixed_norm_18state_{A,B}.json`

The two disjoint-state-set runs behind the final verdict. Magnitude-**independent** (fixed-norm)
injection, 18 time-sampled diverse states each, A = warmup 30 / B = warmup 45.

| field | value |
|---|---|
| SAE | `sae_L5.pt` (sha256 `ee5ee63ad564d85d…`), layer 5, Breakout |
| injection | `fixed_norm` (`scale × unit_direction`; activation-independent) |
| params | `--seeds 18 --state-stride 30 --n-steps 10 --scale 5` |
| features | 80 candidates (∪ of top-50 magnitude / stability / collision-Δ + 6 case-study) |
| per feature | `score`, `pos`/`neg` (per-sign mean token divergence), `act`, `trace` (per-step) |
| compute | RunPod A100, CUDA |

**Verdict:** A vs B Spearman **+0.49**, top-10 overlap 3/10. Confound gone
(`corr(score, act) ≈ +0.2`). The top handful is reproducible (#1364 air-tracker rank 1,
#120 collision rank 2), the tail is not. Regenerate / extend with more states:

```bash
python backend/scripts/causal_importance.py --checkpoint <iris>/checkpoints/Breakout.pt \
  --sae <iris>/checkpoints/sae_L5.pt --fixed-norm \
  --features-file results/causal_candidate_ids.json \
  --seeds 18 --state-stride 30 --n-steps 10 --scale 5 --warmup 30 --device cuda
```

Note: the scorer is CPU-dispatch-bound (autoregressive rollouts in a Python loop), so a GPU
barely helps — batching the rollouts is the real speedup. See FINDINGS.
