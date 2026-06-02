# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend

```bash
# Start backend (from world-model-interpretability/backend/)
export IRIS_ROOT=/Users/temp/Desktop/projects/world_models/iris
export IRIS_SRC=$IRIS_ROOT/src
export CHECKPOINT_DIR=$IRIS_ROOT/checkpoints
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run backend tests (from world-model-interpretability/)
IRIS_ROOT=/Users/temp/Desktop/projects/world_models/iris \
  /Users/temp/Desktop/projects/world_models/iris/.venv310/bin/pytest tests/test_backend.py -v

# Run a single test class or test
IRIS_ROOT=... pytest tests/test_backend.py::TestHookExtraction -v
IRIS_ROOT=... pytest tests/test_backend.py::TestHookExtraction::test_attn_shape_uncached -v
```

### Frontend

`frontend/` is the **v2** UI (react-konva canvas of pinned SAE-feature cards). The legacy
v1 panel-grid UI is preserved at `frontend-v1/` as a runnable fallback — both connect to the
same backend; run the dev server from whichever directory you want.

```bash
# From world-model-interpretability/frontend/   (or frontend-v1/ for the legacy UI)
npm run dev       # dev server → http://localhost:3000
npm test          # Jest test suite (watch mode)
npm test -- --watchAll=false   # single run, no watch
npm run build     # production build (also type-checks)
```

### Docker

```bash
# From world-model-interpretability/
docker-compose up --build
```

## Architecture

### Data flow

```
Atari env → InferenceEngine (background thread) → bounded Queue(5)
                 │                                      │
           PyTorch hooks                        FastAPI WebSocket /ws
           fire per-layer                       polls queue @ 0.05s timeout
                 │                                      │
           IrisHookExtractor                   JSON FrameData → browser
           stores tensors                      useVisualizerSocket hook
           (no .item() calls                   updates React state
            inside hook)
```

The inference thread **never blocks** — frames are dropped (not queued) when the consumer falls behind. Monitor `drop_rate` in the log pane.

### Key design decisions

**Hook extraction runs on a no-KV-cache single-block WM forward pass.** This gives a consistent `(1, nh, 17, 17)` attention shape every step, regardless of episode position. The agent itself runs its normal cached forward for `act()`. These are two separate passes.

**`.item()` is deferred.** `IrisHookExtractor` stores 0-d tensors from the norm hook without calling `.item()`. `inference.py` calls `.item()` once after the full forward pass to avoid N GPU→CPU syncs (one per layer) inside the forward.

**`token_layout` reference stability.** `useVisualizerSocket` preserves the previous `token_layout` object reference when labels haven't changed (compared by joining with `\0`). A new object every frame would cause `MorphingGraph`'s Effect 1 to rebuild the entire SVG/force simulation each frame.

**Latent subscriber gating.** Reconstruction decode (tokenizer codebook lookup + decode) is expensive. `InferenceEngine` only runs it when `latent_subscriber_count > 0`, i.e., when a client is connected to `/ws/latent`.

**Fan-out queues.** Both `/ws` and `/ws/latent` read from separate bounded queues (`_queue` and `_latent_queue`) fed from the same inference loop. Latent drops are not counted against `drop_rate`.

### Backend modules

- `hooks.py` — `IrisHookExtractor`: attach/detach forward hooks on `block.attn.attn_drop` (attention) and `block` (residual norm). Hooks store `.detach()` tensors only; no sync.
- `inference.py` — `InferenceEngine`: background thread lifecycle, `FrameData` dataclass, `_decode_reconstruction()` helper, `_load_agent()` (uses Hydra compose API to load IRIS checkpoints without modifying IRIS source).
- `main.py` — FastAPI app: WebSocket `/ws` and `/ws/latent`, REST `GET /agents`, `GET /config`, `GET /devices`, `POST /control` (loop/restart/pause/resume/switch_agent), `POST /rollout`, `GET/POST/DELETE /bookmarks`, `GET/POST/DELETE /pinned`, `GET /feature/{id}` + `GET /features`.
- `pinned.py` — `PinnedStore`: JSON-backed v2 canvas state (one card = `{feature_id, custom_label, intervention_scale, x, y}` keyed by env+layer). `/pinned` POST has **merge semantics** (partial update writes only the fields sent; `custom_label=""` clears).
- `autointerp_store.py` + `scripts/autointerp.py` — offline vision-LLM feature labeling. The script renders each feature's top-activating frames into a 4×4 grid and asks Claude what they share; labels + firing stats are cached under `<SAE_DIR>/autointerp_L{layer}.json` and served read-only by `/feature/{id}` and `/features`. Build-only: needs `ANTHROPIC_API_KEY`; until run, labels are null and the UI shows "unlabeled". `--no-llm`/`--limit`/`--resume` for cheap dry runs.

**`/rollout` accepts multiple interventions.** The v2 form is `interventions: [{feature_id, scale}]`; the legacy single-feature `{feature_id, scale}` form still works. Because every SAE feature is read at the same layer, the per-feature raw-space directions are **summed** into one vector added to the residual — so N simultaneous interventions cost the same as one and `_rollout` itself is unchanged. Zero-scale cards are observation-only and dropped.

### Frontend

The v2 main view (`frontend/src/app/page.tsx`) is a **canvas of pinned feature cards**:
`FeatureCanvas` (react-konva, dynamically imported `ssr:false`) draws draggable cards with a
sparkline + Konva scale-slider; `SearchBar` pins by id/label; `DiscoveryPanel` lists top-firing
features to pin; `RolloutComparison` runs the multi-feature rollout read from cards with non-zero
scale; `ModelInternals` holds the demoted attention/norms/log behind a default-hidden tab. New
hooks: `usePinned`, `useFeatureIndex` (autointerp labels), `useActivationHistory` (sparklines).
Card label resolves: per-card override → autointerp → bookmark → "unlabeled". The list below is
shared/legacy infrastructure (also the whole v1 UI in `frontend-v1/`):

- `useVisualizerSocket` (`hooks/`) — single hook managing WebSocket lifecycle, auto-reconnect (2s), intentional-close flag to prevent duplicate reconnect, `sendControl()` POSTs to `/control`.
- `AttentionHeatmap` — visx heatmap, one subplot per head, axes labelled from `token_layout.labels`.
- `MorphingGraph` — D3 force-directed graph, nodes = tokens, edges = attention weights above threshold (0.02), action token pinned to right side to prevent rotation drift.
- `ActivationNorms` — visx bar chart, plasma colour scale by magnitude.
- `ErrorMapHeatmap` — renders the base64 grayscale error-map PNG from `/ws/latent`.
- `GameFrame` — canvas with `imageSmoothingEnabled = false` for pixel-accurate Atari frames.

### WebSocket message types

| `type` | When | Key fields |
|--------|------|-----------|
| `frame` | Every inference step | `frame`, `attention` (layer→[nh][T_q][T_k]), `norms`, `metrics`, `token_layout`, `reconstruction`, `error_map`, `reconstruction_error` |
| `config` | Once on connect | `num_layers`, `num_heads`, `embed_dim`, `tokens_per_block`, `max_blocks`, `agents` |
| `event` | Agent loaded, episode start/end, errors | `event`, `data` |

### IRIS dependency

IRIS source (`iris/src/`) is never modified. It is added to `sys.path` at `InferenceEngine` construction time. Tests add it via `sys.path.insert` at module load. The `IRIS_ROOT` env var (default: `../../iris` relative to backend) controls the path.

Token labels are always derived from `tokens_per_block` at runtime — never hardcoded. Standard IRIS config: `tokens_per_block=17` (16 obs tokens `o0`–`o15` + 1 action token `act`).

## Known limitations

- Single engine instance — all browser tabs share one inference loop.
- No KV cache in the WM hook-extraction pass — attention is always `17×17`, not the full autoregressive context.
- MPS device support is untested with IRIS models.
