# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend

```bash
# Start backend (from wm-visualizer/backend/)
export IRIS_ROOT=/Users/temp/Desktop/projects/world_models/iris
export IRIS_SRC=$IRIS_ROOT/src
export CHECKPOINT_DIR=$IRIS_ROOT/checkpoints
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run backend tests (from wm-visualizer/)
IRIS_ROOT=/Users/temp/Desktop/projects/world_models/iris \
  /Users/temp/Desktop/projects/world_models/iris/.venv310/bin/pytest tests/test_backend.py -v

# Run a single test class or test
IRIS_ROOT=... pytest tests/test_backend.py::TestHookExtraction -v
IRIS_ROOT=... pytest tests/test_backend.py::TestHookExtraction::test_attn_shape_uncached -v
```

### Frontend

```bash
# From wm-visualizer/frontend/
npm run dev       # dev server → http://localhost:3000
npm test          # Jest test suite (watch mode)
npm test -- --watchAll=false   # single run, no watch
npm run build     # production build (also type-checks)
```

### Docker

```bash
# From wm-visualizer/
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
- `main.py` — FastAPI app: WebSocket `/ws` and `/ws/latent`, REST `GET /agents`, `GET /config`, `GET /devices`, `POST /control` (loop/restart/pause/resume/switch_agent).

### Frontend

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
