# World Model Interpretability

An investigation into **SAE-based interpretability on world models** — specifically [IRIS](https://github.com/eloialonso/iris), a transformer world model trained on Atari. The question driving it: the tooling and conventions that have matured around sparse autoencoders on *language* models are now being imported into other domains; **which of them actually transfer to a world model, and which break?**

The answer, found by building the tooling and then characterizing what went wrong, is that several core conventions don't transfer cleanly. **The findings are the artifact; the interactive tool is the substrate that made them visible.**

> **New here? Start with [`REPORT.md`](REPORT.md)** — the synthesis of what was built, the headline result, and the clean negative. For the project-independent lessons read [`FINDINGS.md`](FINDINGS.md).

---

## The findings, in one paragraph

LLM-SAE conventions transfer worse than expected to world models, in ways that are individually subtle and collectively substantial. **Magnitude ranking** — the default discovery sort — surfaces *persistent* features (a ball-tracker that fires throughout the ball's flight, the single most-active feature `#1364`) over *event* features (collision detectors `#1199`/`#120`, which exist and are robust but rank lower), so a user scanning the top of the list for "the collision feature" is shown the wrong thing. **Pixel-space state extraction** fails because the tokenizer drops small objects — the Breakout ball is sub-pixel even in the model's own 64×64 input. **Single-frame-primed rollouts** are near-static (~0.8/255 mean pixel change over 20 frames) because one frame gives position but not velocity. **Argmax decoding** makes intervention response a step function, so the decoder, not the SAE, sets the granularity. **Causal-importance ranking** silently collapses into a magnitude ranking unless you inject magnitude-independently, and even fixed it is too single-state-fragile to rank reproducibly (cross-set Spearman plateaued at +0.49, below the pre-committed 0.6 bar). And the compute lever you'd reach for to fix that doesn't pull: **autoregressive rollouts are CPU-dispatch-bound, not GPU-bound.**

Of three proposed substrate-adapted importance axes, two shipped (magnitude, temporal stability) and one (causal) was demoted to a characterization tool after its confound and single-state fragility were found and measured. Documenting *why* causal didn't make it is the more transferable result.

---

## Documents

| File | What it is |
|---|---|
| [`REPORT.md`](REPORT.md) | Synthesis report: what was built, the headline result, the clean negative, a process retrospective — the place to start. |
| [`FINDINGS.md`](FINDINGS.md) | Transferable, project-independent methodological lessons. |
| [`CLAUDE.md`](CLAUDE.md) | Developer-facing architecture, data flow, and design decisions. |

---

## The tool (the substrate)

A FastAPI backend runs IRIS inference and extracts residual-stream activations through PyTorch forward hooks; a trained SAE turns those into features; a Next.js frontend lets you discover, pin, label, and **intervene** on features and watch the model's imagined rollout diverge.

The current UI (`frontend/`, **v2**) is a **canvas of pinned feature cards**:

- **Canvas of feature cards** (react-konva) — each card is a feature with a sparkline, an editable label, and an intervention-scale slider; layout is persisted server-side so a working set survives reloads.
- **Search-first discovery** — jump to a feature by id or label; a side discovery panel ranks candidates by **firing magnitude** or **temporal stability** (the two axes that survived).
- **Multi-feature intervention** — non-zero-scale cards jointly steer a paused, N-step imagined rollout; the rollout sums their directions, so steering N features costs the same as one.
- **N-step imagined rollouts** — baseline vs. intervened frames with a **token-divergence trajectory** beneath (the load-bearing measurement, since pixel-space object extraction is unreliable on this substrate).
- **Model Internals tab** (hidden by default) — the original attention heatmaps and residual-norm bars, demoted once feature intervention became the workflow.

The SAE is a plain **ReLU/L1** sparse autoencoder trained on layer-5 residual-stream activations from Breakout (`d_in=256`, `d_hidden=2048`, `l1_coeff=2.0`, L0 ≈ 14–15). The legacy v1 panel-grid UI is preserved and runnable at `frontend-v1/`; both connect to the same backend.

IRIS source (`iris/src/`) is **never modified** — it is added to `sys.path` and driven through hooks and its own imagination path.

---

## Results & research artifacts

- **`results/`** — the feature-characterization diagnostic battery: frame/activation sync (`test1`), extraction determinism (`test2`), collision correlation (`test3`), and per-event timing traces (`test5`, with plots). These are what resolved the apparent feature–event "decoupling."
- **`data/`** — the fixed-norm causal-importance runs behind the demotion decision (two disjoint 18-state sets, A and B). See [`data/README.md`](data/README.md). **Research artifacts, not a shipped feature.**
- **`deploy/`** — RunPod cloud-GPU scaffold used for the (ultimately dispatch-bound) causal run.

---

## Repo structure

```
world-model-interpretability/
├── REPORT.md  FINDINGS.md          # the writeups
├── backend/
│   ├── hooks.py            # IrisHookExtractor — forward hooks on the WM residual stream
│   ├── inference.py        # InferenceEngine — background thread, queues, SAE, rollouts, stability ranking
│   ├── main.py             # FastAPI app — /ws, /rollout, /pinned, /feature, /ranking/*, /control
│   ├── sae.py              # ReLU/L1 SparseAutoencoder
│   ├── pinned.py           # v2 canvas state store
│   ├── ranking_store.py    # causal-importance score store (characterization tool)
│   ├── autointerp_store.py # vision-LLM feature-label cache (built, not run by default)
│   ├── bookmarks.py  state_extract.py
│   └── scripts/
│       ├── train_sae.py            # SAE training pipeline
│       ├── causal_importance.py    # offline causal scoring (fixed-norm injection)
│       ├── autointerp.py  analyze_sae.py  collect_episodes.py  label_features.py
│       └── diagnostics/            # the feature-characterization test battery (test1/2/3/5)
├── frontend/        # v2 — canvas of pinned feature cards (react-konva), discovery, rollout
├── frontend-v1/     # legacy v1 panel-grid UI (runnable fallback)
├── results/  data/  deploy/        # diagnostic outputs, causal artifacts, cloud scaffold
├── tests/           # test_backend.py, test_sae.py, test_visualizer.py
├── visualizer.py    # separate standalone Ha & Schmidhuber WM viewer (not the IRIS work)
└── docker-compose.yml
```

---

## Running it

### Prerequisites

- Python 3.10 with the IRIS venv set up (`iris/.venv310`)
- Node.js 20+
- IRIS checkpoints in `iris/checkpoints/` (e.g. `Breakout.pt`) and a trained SAE (`sae_L5.pt`)
- Atari ROMs installed: `ale-import-roms /path/to/roms/`

### Backend

```bash
cd world-model-interpretability/backend
export IRIS_ROOT=/Users/temp/Desktop/projects/world_models/iris
export IRIS_SRC=$IRIS_ROOT/src
export CHECKPOINT_DIR=$IRIS_ROOT/checkpoints
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Set `DEFAULT_DEVICE=mps` (or `cuda`) to use an accelerator; the activation-harvest path gets ~2× on MPS over CPU.

### Frontend

```bash
cd world-model-interpretability/frontend   # or frontend-v1/ for the legacy UI
npm install
npm run dev        # → http://localhost:3000
```

REST calls are proxied to `http://localhost:8000` via the Next.js rewrite in `next.config.js`.

### Tests

```bash
# Backend (from repo root)
IRIS_ROOT=/path/to/iris /path/to/iris/.venv310/bin/pytest tests/test_backend.py -v

# Frontend
cd frontend && npm test -- --watchAll=false
```

### Docker

```bash
docker-compose up --build   # backend → :8000, frontend → :3000
```

`docker-compose.yml` mounts the IRIS repo as a **read-only** volume; nothing under `iris/src/` is ever written.

---

## Training an SAE / regenerating artifacts

```bash
# Train a layer-5 SAE on harvested Breakout activations
python backend/scripts/train_sae.py \
  --episodes-dir <iris>/outputs/.../media/episodes/test \
  --checkpoint <iris>/checkpoints/Breakout.pt \
  --layers 5 --expansion 8 --l1 2.0 --out-dir <iris>/checkpoints

# Causal-importance characterization (CPU-dispatch-bound — a bigger GPU barely helps; batch the rollouts)
python backend/scripts/causal_importance.py --checkpoint <iris>/checkpoints/Breakout.pt \
  --sae <iris>/checkpoints/sae_L5.pt --fixed-norm \
  --features-file results/causal_candidate_ids.json \
  --seeds 18 --state-stride 30 --n-steps 10 --scale 5 --device cuda
```

---

## Limitations

These are observations from one substrate, **not laws**:

- **One game, one model** — Breakout on IRIS only; no cross-game or cross-model replication. The SAE is trained at a single layer (5).
- **Small SAE** — 2048 features from ~80k vectors (~5k frames); on the low side of the sample-to-feature ratio for clean recovery.
- **No architecture bake-off** — one plain ReLU/L1 SAE, not a comparison across SAE variants.
- **Autointerp labels** are built but not run by default, so most features show "unlabeled" (the "stability beats magnitude" claim rests on a collision-correlation proxy, not read labels).
- **Causal importance** is a *not-yet* as much as a *no* — below the robustness bar at the state counts that were affordable; batched rollouts might clear it.
- **Tooling caveats** — single shared engine instance across browser tabs; no KV cache in the hook-extraction pass (attention is always 17×17, not the true autoregressive context); single-frame-primed rollouts are near-static; frames are dropped (not queued) when the consumer falls behind (`drop_rate`).
