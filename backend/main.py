"""
FastAPI application and WebSocket server for the WM Visualizer.

Endpoints
---------
  WS  /ws                   — stream FrameData JSON; accepts query params
                              ?agent=<name>&env_id=<id>&device=<cpu|mps|cuda>
  GET /agents               — list available checkpoints
  GET /config               — current model config
  POST /control             — episode control (loop, restart, pause, resume,
                              switch_agent)
"""

import asyncio
import logging
import os
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import InferenceEngine
from bookmarks import BookmarkStore
from pinned import PinnedStore, _UNSET
from autointerp_store import AutoInterpStore, resolve_layer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

IRIS_ROOT = os.environ.get(
    "IRIS_ROOT",
    str(Path(__file__).parent.parent.parent / "iris"),
)
IRIS_SRC = os.environ.get("IRIS_SRC", str(Path(IRIS_ROOT) / "src"))
CHECKPOINT_DIR = os.environ.get(
    "CHECKPOINT_DIR",
    str(Path(IRIS_ROOT) / "checkpoints"),
)
# Directory scanned for trained sae_L*.pt artifacts (defaults to CHECKPOINT_DIR)
SAE_DIR = os.environ.get("SAE_DIR", CHECKPOINT_DIR)
DEFAULT_DEVICE = os.environ.get("DEFAULT_DEVICE", "cpu")
# JSON store for SAE feature bookmarks (the backend's only persistence)
BOOKMARKS_PATH = os.environ.get("BOOKMARKS_PATH", str(Path(SAE_DIR) / "bookmarks.json"))
# JSON store for v2 canvas pinned features (source of truth for the canvas layout)
PINNED_PATH = os.environ.get("PINNED_PATH", str(Path(SAE_DIR) / "pinned.json"))

# Map checkpoint stem → Atari env ID (fallback: append NoFrameskip-v4)
_KNOWN_ENV_IDS: Dict[str, str] = {
    "Alien": "AlienNoFrameskip-v4",
    "Breakout": "BreakoutNoFrameskip-v4",
    "Pong": "PongNoFrameskip-v4",
    "SpaceInvaders": "SpaceInvadersNoFrameskip-v4",
    "Seaquest": "SeaquestNoFrameskip-v4",
    "Freeway": "FreewayNoFrameskip-v4",
    "MsPacman": "MsPacmanNoFrameskip-v4",
    "Qbert": "QbertNoFrameskip-v4",
}


def _infer_env_id(checkpoint_stem: str) -> str:
    return _KNOWN_ENV_IDS.get(checkpoint_stem, f"{checkpoint_stem}NoFrameskip-v4")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="WM Visualizer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InferenceEngine(iris_src=IRIS_SRC, iris_root=IRIS_ROOT, sae_dir=SAE_DIR)
bookmarks = BookmarkStore(BOOKMARKS_PATH)
pinned = PinnedStore(PINNED_PATH)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/agents")
async def list_agents() -> List[dict]:
    """
    Scan the checkpoint directory and return all available agents.

    Each entry::

        {"id": "Breakout", "name": "Breakout",
         "path": "/abs/path/to/Breakout.pt",
         "env_id": "BreakoutNoFrameskip-v4"}
    """
    ckpt_dir = Path(CHECKPOINT_DIR)
    agents = []
    if ckpt_dir.is_dir():
        for pt in sorted(ckpt_dir.glob("*.pt")):
            stem = pt.stem
            # Skip SAE artifacts that share the checkpoint dir — they are not agents.
            if stem.startswith("sae_"):
                continue
            agents.append({
                "id": stem,
                "name": stem,
                "path": str(pt.resolve()),
                "env_id": _infer_env_id(stem),
            })
    return agents


@app.get("/devices")
async def list_devices() -> dict:
    """
    Return the compute devices available on this machine and the current default.

    Response::

        {"available": ["cpu", "mps"], "default": "mps"}
    """
    import torch
    available = ["cpu"]
    if torch.backends.mps.is_available():
        available.append("mps")
    if torch.cuda.is_available():
        available.append("cuda")
    return {"available": available, "default": DEFAULT_DEVICE}


@app.get("/config")
async def get_config() -> dict:
    """Return the current model config or an empty dict if no agent is loaded."""
    return engine.get_config()


class ControlCommand(BaseModel):
    command: str                          # loop | restart | pause | resume | step | set_intervention | switch_agent
    payload: Optional[Dict[str, Any]] = None


@app.post("/control")
async def control(cmd: ControlCommand) -> dict:
    """
    Send a control command to the inference engine.

    switch_agent payload::

        {"checkpoint_path": "/path/to/Agent.pt",
         "env_id": "BreakoutNoFrameskip-v4",
         "device": "cpu|mps|cuda"}
    """
    loop = asyncio.get_event_loop()

    if cmd.command == "loop":
        engine.set_loop(bool((cmd.payload or {}).get("enabled", True)))

    elif cmd.command == "restart":
        engine.restart_episode()

    elif cmd.command == "pause":
        engine.pause()

    elif cmd.command == "resume":
        engine.resume()

    elif cmd.command == "step":
        engine.step_once()

    elif cmd.command == "set_intervention":
        p = cmd.payload or {}
        fid = p.get("feature_id")
        engine.set_intervention(
            int(fid) if fid is not None else None,
            float(p.get("scale", 0.0)),
        )

    elif cmd.command == "switch_agent":
        p = cmd.payload or {}
        ckpt = Path(p["checkpoint_path"])
        env_id = p.get("env_id") or _infer_env_id(ckpt.stem)
        device = p.get("device", DEFAULT_DEVICE)
        # Run blocking operation in thread pool
        await loop.run_in_executor(
            None, engine.switch_agent, ckpt, env_id, device
        )

    else:
        return {"status": "error", "detail": f"Unknown command: {cmd.command!r}"}

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Intervention rollout (paused-only, on-demand N-step experiment)
# ---------------------------------------------------------------------------

class InterventionItem(BaseModel):
    feature_id: int
    scale: float


class RolloutCommand(BaseModel):
    # New multi-feature form: a list of interventions summed in the residual stream.
    interventions: Optional[List[InterventionItem]] = None
    # Legacy single-feature form (v1 frontend) — used only when `interventions` is absent.
    feature_id: Optional[int] = None
    scale: Optional[float] = None
    n_steps: int = 20
    n_seeds: int = 2


@app.post("/rollout")
async def run_rollout(cmd: RolloutCommand) -> dict:
    """Run paired baseline vs intervened N-step imagined rollouts from the current frame.

    Accepts either the v2 multi-feature form (``interventions: [{feature_id, scale}, ...]``)
    or the legacy single-feature form (``feature_id`` + ``scale``). Zero-scale entries are
    observation-only and dropped. Heavy (~n_seeds * 2 * n_steps * 17 WM passes) and
    paused-only — runs off the event loop in a thread.
    """
    from fastapi import HTTPException
    loop = asyncio.get_event_loop()

    # Normalize both request forms to a list of (feature_id, scale) tuples.
    if cmd.interventions:
        ivs = [(int(i.feature_id), float(i.scale)) for i in cmd.interventions]
    elif cmd.feature_id is not None:
        ivs = [(int(cmd.feature_id), float(cmd.scale or 0.0))]
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide `interventions` or legacy `feature_id` + `scale`",
        )
    # Drop observation-only (zero-scale) interventions before running.
    ivs = [(f, s) for (f, s) in ivs if s != 0.0]
    if not ivs:
        raise HTTPException(status_code=422, detail="No non-zero interventions to run")

    # Clamp to sane bounds (cost grows linearly in both).
    n_steps = max(1, min(int(cmd.n_steps), 40))
    n_seeds = max(1, min(int(cmd.n_seeds), 4))
    try:
        result = await loop.run_in_executor(
            None, engine.run_rollout, ivs, n_steps, n_seeds
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return result


# ---------------------------------------------------------------------------
# Bookmarks (SAE feature labels — the only persisted state)
# ---------------------------------------------------------------------------

class BookmarkModel(BaseModel):
    env_id: str
    layer: int
    feature_id: int
    label: str
    notes: str = ""
    source: str = "user"


@app.get("/bookmarks")
async def list_bookmarks(
    env_id: Optional[str] = Query(None),
    layer: Optional[int] = Query(None),
) -> List[dict]:
    """Return saved feature bookmarks, optionally filtered by env_id and/or layer."""
    return bookmarks.list(env_id=env_id, layer=layer)


@app.post("/bookmarks")
async def upsert_bookmark(item: BookmarkModel) -> dict:
    """Create or update one feature bookmark (keyed by env_id+layer+feature_id)."""
    from datetime import datetime, timezone
    rec = bookmarks.upsert(
        env_id=item.env_id,
        layer=item.layer,
        feature_id=item.feature_id,
        label=item.label,
        notes=item.notes,
        source=item.source,
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return {"status": "ok", "bookmark": rec}


@app.delete("/bookmarks")
async def delete_bookmark(
    env_id: str = Query(...),
    layer: int = Query(...),
    feature_id: int = Query(...),
) -> dict:
    """Delete one feature bookmark."""
    existed = bookmarks.delete(env_id, layer, feature_id)
    return {"status": "ok", "deleted": existed}


# ---------------------------------------------------------------------------
# Pinned features (v2 canvas state — feature cards with scale + position)
# ---------------------------------------------------------------------------

class PinnedModel(BaseModel):
    env_id: str
    layer: int
    feature_id: int
    # All optional: a partial update changes only the fields it includes (the
    # store keeps the rest). Omit a field to leave it untouched; send custom_label=""
    # to clear a label. intervention_scale 0 = pinned for observation, not steering.
    custom_label: Optional[str] = None
    intervention_scale: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None


@app.get("/pinned")
async def list_pinned(
    env_id: Optional[str] = Query(None),
    layer: Optional[int] = Query(None),
) -> List[dict]:
    """Return pinned canvas features, optionally filtered by env_id and/or layer."""
    return pinned.list(env_id=env_id, layer=layer)


@app.post("/pinned")
async def upsert_pinned(item: PinnedModel) -> dict:
    """Pin a feature or partially update an existing pin (add / rename / move / set scale).

    Merge semantics: only fields actually present in the request body are applied, so the
    canvas can PATCH-style update one aspect (drag → x/y, slider → scale) without clobbering
    the others.
    """
    from datetime import datetime, timezone
    sent = item.model_dump(exclude_unset=True)
    rec = pinned.upsert(
        env_id=item.env_id,
        layer=item.layer,
        feature_id=item.feature_id,
        custom_label=sent.get("custom_label", _UNSET),
        intervention_scale=sent.get("intervention_scale", _UNSET),
        x=sent.get("x", _UNSET),
        y=sent.get("y", _UNSET),
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return {"status": "ok", "pinned": rec}


@app.delete("/pinned")
async def delete_pinned(
    env_id: str = Query(...),
    layer: int = Query(...),
    feature_id: int = Query(...),
) -> dict:
    """Unpin one feature (remove its card from the canvas)."""
    existed = pinned.delete(env_id, layer, feature_id)
    return {"status": "ok", "deleted": existed}


# ---------------------------------------------------------------------------
# Autointerp feature labels (read-only; populated offline by scripts/autointerp.py)
# ---------------------------------------------------------------------------

@app.get("/feature/{feature_id}")
async def get_feature(
    feature_id: int,
    layer: Optional[int] = Query(None),
) -> dict:
    """Return one SAE feature's autointerp record: label + firing stats + example frames.

    Reads the cache written by ``scripts/autointerp.py``. If the pipeline hasn't been run
    (or this feature is dead/unlabeled), returns ``label: null`` and no examples — the UI
    shows 'unlabeled'. ``layer`` is inferred when a single cache exists; otherwise pass it.
    """
    from fastapi import HTTPException
    resolved = resolve_layer(SAE_DIR, layer if layer is not None else engine.sae_layer)
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail="No autointerp cache found — run scripts/autointerp.py, or pass ?layer=",
        )
    return AutoInterpStore(SAE_DIR, resolved).read_feature(feature_id)


@app.get("/features")
async def list_features(
    layer: Optional[int] = Query(None),
    labeled_only: bool = Query(False),
) -> dict:
    """Return the whole autointerp index (light: labels + stats, no images).

    Drives the v2 label-keyword search. ``{layer, env_id, features: {id: {...}}}`` or an
    empty ``features`` map when the pipeline hasn't run.
    """
    resolved = resolve_layer(SAE_DIR, layer if layer is not None else engine.sae_layer)
    if resolved is None:
        return {"layer": None, "features": {}}
    index = AutoInterpStore(SAE_DIR, resolved).load_index()
    if labeled_only:
        index = {**index, "features": {
            k: v for k, v in (index.get("features") or {}).items() if v.get("label")
        }}
    return index


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_handler(
    ws: WebSocket,
    agent: Optional[str] = Query(None),
    env_id: Optional[str] = Query(None),
    device: Optional[str] = Query(None),
) -> None:
    """
    Single WebSocket endpoint.

    On connect (if agent is provided and no inference is running) the engine is
    started with the requested agent.  Frames are streamed at inference rate.
    Events (agent_loaded, episode_start/end, error) are interleaved as they occur.
    """
    await ws.accept()
    loop = asyncio.get_event_loop()

    # Thread-safe event queue: inference thread → async WS sender
    from queue import Queue as SyncQueue
    event_q: SyncQueue = SyncQueue()

    def event_cb(event_name: str, data: dict) -> None:
        event_q.put_nowait({"type": "event", "event": event_name, "data": data})

    engine.register_event_callback(event_cb)

    # Start inference if not running and an agent was requested
    if agent and not engine.is_running:
        # Find checkpoint path
        ckpt_path = _resolve_checkpoint(agent)
        resolved_env = env_id or _infer_env_id(agent)
        resolved_dev = device or DEFAULT_DEVICE
        try:
            await loop.run_in_executor(
                None, engine.start, ckpt_path, resolved_env, resolved_dev
            )
        except Exception as exc:
            await ws.send_json({
                "type": "event",
                "event": "error",
                "data": {"message": str(exc)},
            })
            engine.unregister_event_callback(event_cb)
            return

    # Send initial config snapshot
    cfg = engine.get_config()
    agents_list = await list_agents()
    await ws.send_json({
        "type": "config",
        **cfg,
        "agents": agents_list,
    })

    try:
        while True:
            # Flush any queued events first (non-blocking)
            while not event_q.empty():
                try:
                    await ws.send_json(event_q.get_nowait())
                except Empty:
                    break

            # Fetch next frame (0.05 s timeout)
            frame = await loop.run_in_executor(None, engine.get_frame, 0.05)
            if frame is not None:
                await ws.send_json(frame)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
    finally:
        engine.unregister_event_callback(event_cb)


@app.websocket("/ws/latent")
async def websocket_latent_handler(
    ws: WebSocket,
    agent: Optional[str] = Query(None),
    env_id: Optional[str] = Query(None),
    device: Optional[str] = Query(None),
) -> None:
    """
    WebSocket endpoint for the latent-space reconstruction page.

    Identical frame stream to /ws, but registers the client as a latent
    subscriber so the inference loop computes reconstruction decode only
    while at least one client is connected here.
    """
    await ws.accept()
    engine.add_latent_subscriber()
    loop = asyncio.get_event_loop()

    from queue import Queue as SyncQueue
    event_q: SyncQueue = SyncQueue()

    def event_cb(event_name: str, data: dict) -> None:
        event_q.put_nowait({"type": "event", "event": event_name, "data": data})

    engine.register_event_callback(event_cb)

    if agent and not engine.is_running:
        ckpt_path = _resolve_checkpoint(agent)
        resolved_env = env_id or _infer_env_id(agent)
        resolved_dev = device or DEFAULT_DEVICE
        try:
            await loop.run_in_executor(
                None, engine.start, ckpt_path, resolved_env, resolved_dev
            )
        except Exception as exc:
            await ws.send_json({
                "type": "event",
                "event": "error",
                "data": {"message": str(exc)},
            })
            engine.unregister_event_callback(event_cb)
            engine.remove_latent_subscriber()
            return

    cfg = engine.get_config()
    agents_list = await list_agents()
    await ws.send_json({
        "type": "config",
        **cfg,
        "agents": agents_list,
    })

    try:
        while True:
            while not event_q.empty():
                try:
                    await ws.send_json(event_q.get_nowait())
                except Empty:
                    break

            frame = await loop.run_in_executor(None, engine.get_latent_frame, 0.05)
            if frame is not None:
                await ws.send_json(frame)

    except WebSocketDisconnect:
        logger.info("WebSocket latent client disconnected")
    except Exception as exc:
        logger.error("WebSocket latent error: %s", exc)
    finally:
        engine.unregister_event_callback(event_cb)
        engine.remove_latent_subscriber()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint(agent_name: str) -> Path:
    """
    Resolve agent name to absolute checkpoint path.
    Accepts bare stem ("Breakout") or absolute path.
    """
    p = Path(agent_name)
    if p.is_absolute() and p.exists():
        return p
    # Look in checkpoint dir
    candidate = Path(CHECKPOINT_DIR) / f"{agent_name}.pt"
    if candidate.exists():
        return candidate
    # Maybe the name already includes .pt
    candidate2 = Path(CHECKPOINT_DIR) / agent_name
    if candidate2.exists():
        return candidate2
    raise FileNotFoundError(
        f"Checkpoint not found for agent {agent_name!r} in {CHECKPOINT_DIR}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
