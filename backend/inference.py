"""
IRIS inference engine.

Architecture
------------
InferenceEngine runs one background daemon thread that:
  1. Loads an IRIS Agent from a checkpoint.
  2. Steps the agent through a real Atari environment.
  3. Runs a single-block, no-KV-cache WorldModel forward pass for hook extraction
     (gives a consistent (1, nh, 17, 17) attention shape every step).
  4. Encodes the raw RGB frame as a base64 PNG.
  5. Pushes FrameData onto a bounded Queue.

If the queue is full the frame is dropped — inference is never blocked by a slow
consumer.  Agent switching stops the thread, flushes the queue, and restarts with
the new agent.
"""

import base64
import io
import logging
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from PIL import Image

from hooks import IrisHookExtractor
from sae import load_artifact

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_token_labels(num_tokens: int, tokens_per_block: int) -> List[str]:
    """Token labels derived from model config — never hardcoded."""
    return [
        "act" if i % tokens_per_block == tokens_per_block - 1
        else f"o{i % tokens_per_block}"
        for i in range(num_tokens)
    ]


class _FpsCounter:
    """Rolling 1-second FPS counter (thread-safe for single writer)."""

    def __init__(self) -> None:
        self._ts: deque = deque()
        self.fps: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        self._ts.append(now)
        cutoff = now - 1.0
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()
        self.fps = float(len(self._ts))
        return self.fps


class _LatentSubscriberCounter:
    """Thread-safe count of active /ws/latent WebSocket clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count: int = 0

    def add(self) -> None:
        with self._lock:
            self._count += 1

    def remove(self) -> None:
        with self._lock:
            if self._count > 0:
                self._count -= 1

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


def _encode_frame(obs_np: np.ndarray) -> str:
    """Encode (H, W, 3) uint8 array → base64 PNG string."""
    img = Image.fromarray(obs_np.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# FrameData
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    """One complete inference step, ready for JSON serialisation."""

    type: str = "frame"
    frame: str = ""                              # base64 PNG
    attention: Dict[str, List] = field(default_factory=dict)
    # str(layer_idx) → list[nh][T_q][T_k]
    norms: List[float] = field(default_factory=list)   # per layer
    metrics: Dict[str, Any] = field(default_factory=dict)
    token_layout: Dict[str, Any] = field(default_factory=dict)
    reconstruction: Optional[str] = None         # base64 PNG or null
    error_map: Optional[str] = None              # base64 grayscale PNG or null
    reconstruction_error: Optional[float] = None # MAE in [0, 255] or null
    imagined_next: Optional[str] = None          # base64 PNG: WM-predicted next frame (step mode only)
    sae_features: Optional[List[Dict[str, float]]] = None  # top-K firing: [{"id", "mag"}], or null
    sae_layer: Optional[int] = None              # WM layer the SAE reads, or null if no SAE loaded
    imagined_intervened: Optional[str] = None    # base64 PNG: next frame with feature intervention (step mode)
    intervention_diff: Optional[str] = None      # base64 grayscale PNG: |baseline - intervened|
    intervention: Optional[Dict[str, Any]] = None  # {"feature_id", "scale"} active when imagined (or null)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "frame": self.frame,
            "attention": self.attention,
            "norms": self.norms,
            "metrics": self.metrics,
            "token_layout": self.token_layout,
            "reconstruction": self.reconstruction,
            "error_map": self.error_map,
            "reconstruction_error": self.reconstruction_error,
            "imagined_next": self.imagined_next,
            "sae_features": self.sae_features,
            "sae_layer": self.sae_layer,
            "imagined_intervened": self.imagined_intervened,
            "intervention_diff": self.intervention_diff,
            "intervention": self.intervention,
        }


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

def _decode_obs_tokens_to_pixels(tokenizer, obs_tokens: "torch.Tensor") -> "torch.Tensor":
    """Decode observation token indices → (1, C, H, W) pixels in [0, 1], on device.

    Mirrors IRIS WorldModelEnv.decode_obs_tokens: embedding lookup → reshape to a
    square latent grid → tokenizer decoder.  Shared by the tokenizer reconstruction
    (current frame) and the world-model imagination (predicted next frame).
    """
    K = obs_tokens.shape[1]
    hw = math.isqrt(K)
    if hw * hw != K:
        raise ValueError(f"obs token count {K} is not a perfect square")
    embed_dim = tokenizer.embedding.embedding_dim
    emb = tokenizer.embedding(obs_tokens)                              # (1, K, E)
    z_q = emb.view(1, hw, hw, embed_dim).permute(0, 3, 1, 2).contiguous()
    return tokenizer.decode(z_q, should_postprocess=True)             # (1, C, H, W)


def _rec_tensor_to_rgb(
    rec: "torch.Tensor",
    target_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """(1, C, H, W) float [0,1] → (H, W, 3) uint8, optionally NEAREST-upscaled."""
    rec_np = (
        rec.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).detach().cpu().numpy() * 255
    ).astype("uint8")
    if target_hw is not None and rec_np.shape[:2] != target_hw:
        rec_pil = Image.fromarray(rec_np).resize((target_hw[1], target_hw[0]), Image.NEAREST)
        rec_np = np.asarray(rec_pil)
    return rec_np


def _decode_reconstruction(
    tokenizer,
    obs_tokens: "torch.Tensor",     # (1, K) long, on device
    obs_tensor: "torch.Tensor",     # (1, C, H, W) float [0, 1], on device
    device: "torch.device",
    target_hw: Optional[Tuple[int, int]] = None,  # (H, W) to upscale rec to
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Decode observation token indices back to pixel space via the tokenizer codebook.

    Steps:
      1. Look up each token index in the embedding to get quantised embeddings z_q.
      2. Reshape from (1, K, E) → (1, E, hw, hw) where hw = sqrt(K).
      3. Call tokenizer.decode(z_q, should_postprocess=True) — stays on device.
      4. Compute per-pixel MAE between obs_tensor and reconstruction.
      5. Encode both images as base64 PNGs for the WebSocket message.

    All tensor operations stay on `device` until the final CPU→numpy conversion
    required for PIL image encoding. If anything fails, returns (None, None, None)
    and logs a warning — the inference loop is never interrupted.
    """
    try:
        # Decode to [0, 1] pixel space: (1, C, H_out, W_out) — on device
        rec = _decode_obs_tokens_to_pixels(tokenizer, obs_tokens)

        # Encode reconstruction as PNG, upscaled to match the raw frame size
        rec_b64 = _encode_frame(_rec_tensor_to_rgb(rec, target_hw))

        # Per-pixel MAE averaged across channels (H, W) — computed on device
        err = (obs_tensor - rec).abs().mean(dim=1).squeeze(0)          # (H, W)
        err_np = (err.clamp(0.0, 1.0).detach().cpu().numpy() * 255).astype("uint8")

        err_img = Image.fromarray(err_np, mode="L")
        err_buf = io.BytesIO()
        err_img.save(err_buf, format="PNG", optimize=False)
        error_map_b64 = base64.b64encode(err_buf.getvalue()).decode("ascii")

        mae = float(err.mean().item()) * 255.0  # scale to [0, 255] display range
        return rec_b64, error_map_b64, mae

    except Exception as exc:
        logger.warning("Reconstruction decode failed: %s", exc)
        return None, None, None


@torch.no_grad()
def _imagine_next_rgb(
    wm,
    tokenizer,
    obs_tokens: "torch.Tensor",     # (1, K) long — CURRENT frame's obs tokens, on device
    action_int: int,
    device: "torch.device",
    target_hw: Optional[Tuple[int, int]] = None,
    intervention: Optional[Tuple[int, "torch.Tensor"]] = None,  # (layer, raw-space direction (E,))
    deterministic: bool = False,
) -> Optional[np.ndarray]:
    """Roll the world model one step forward; return the imagined next frame (H, W, 3) uint8.

    A single-step "what does the WM predict next" probe (IRIS WorldModelEnv.step
    semantics): prime a fresh KV cache with the current observation tokens, feed the
    chosen action, then autoregressively generate the K next-observation tokens and
    decode them to pixels.  Not a multi-step dream — it carries only the single
    current observation, not the running episode's long context.

    ``deterministic`` selects argmax (vs sampling) for next-token generation — use it
    for the baseline/intervened pair so their pixel diff reflects only the
    intervention, not sampling noise.

    ``intervention=(layer, direction)`` adds ``direction`` (already in raw residual
    space, scaled) to **every** token-position residual at ``layer`` — during the
    priming pass (which fills the KV cache all generated tokens attend to) and every
    generation step. Injecting on all positions (not just the action token) gives the
    intervention far more leverage per unit scale. The hook is registered only for
    this rollout and removed in ``finally`` — it never pollutes the Phase-2 read hook,
    which runs on the separate no-cache extraction pass.

    Returns ``(rgb, gen_tokens)`` — the imagined frame (H,W,3 uint8) and the (1,K) long
    tensor of generated obs tokens — or ``(None, None)`` on failure (never interrupts
    the inference loop). ``gen_tokens`` lets the caller count how many tokens an
    intervention flipped, a reliable signal even when the pixel diff is tiny.
    """
    handle = None
    try:
        if intervention is not None:
            iv_layer, iv_dir = intervention

            def iv_hook(_mod, _inp, out):
                # Add the direction to all T positions (broadcasts over the T axis).
                return out + iv_dir

            handle = wm.transformer.blocks[iv_layer].register_forward_hook(iv_hook)

        K = obs_tokens.shape[1]
        kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=wm.config.max_tokens)
        wm(obs_tokens, past_keys_values=kv)    # prime cache (hook active if intervening)

        token = torch.tensor([[action_int]], dtype=torch.long, device=device)  # (1, 1)
        gen: List[torch.Tensor] = []
        for k in range(1 + K):                 # 1 action pass + K obs-token passes
            out = wm(token, past_keys_values=kv)
            if k < K:
                logits = out.logits_observations
                token = (
                    logits.argmax(dim=-1) if deterministic
                    else Categorical(logits=logits).sample()
                )                              # (1, 1)
                gen.append(token)
        gen_tokens = torch.cat(gen, dim=1)      # (1, K)

        rec = _decode_obs_tokens_to_pixels(tokenizer, gen_tokens)
        return _rec_tensor_to_rgb(rec, target_hw), gen_tokens
    except Exception as exc:
        logger.warning("Imagined-frame rollout failed: %s", exc)
        return None, None
    finally:
        if handle is not None:
            handle.remove()


def _abs_diff_gray_png(a: np.ndarray, b: np.ndarray) -> Optional[str]:
    """Per-pixel |a - b| averaged across channels → base64 grayscale PNG (or None)."""
    try:
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
        diff = diff.clip(0, 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(diff, mode="L").save(buf, format="PNG", optimize=False)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        logger.warning("Intervention diff encode failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Agent loader
# ---------------------------------------------------------------------------

def _load_agent(iris_root: Path, checkpoint_path: Path, env_id: str, device_str: str):
    """
    Load an IRIS Agent from a checkpoint.

    Uses Hydra's compose API (not @hydra.main) so it can be called at runtime.
    GlobalHydra state is cleared before and after to allow repeated calls.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            config_dir=str(iris_root / "config"),
        ):
            cfg = compose(
                config_name="trainer",
                overrides=[
                    f"env.train.id={env_id}",
                    f"env.test.id={env_id}",
                    f"initialization.path_to_checkpoint={checkpoint_path}",
                    f"common.device={device_str}",
                    "wandb.mode=disabled",
                ],
            )
    finally:
        GlobalHydra.instance().clear()

    # Import IRIS modules (iris/src must already be on sys.path)
    from agent import Agent
    from envs import SingleProcessEnv
    from models.actor_critic import ActorCritic
    from models.world_model import WorldModel

    device = torch.device(device_str)
    env_fn = partial(instantiate, config=cfg.env.test)
    env = SingleProcessEnv(env_fn)

    tokenizer = instantiate(cfg.tokenizer)
    world_model = WorldModel(
        obs_vocab_size=tokenizer.vocab_size,
        act_vocab_size=env.num_actions,
        config=instantiate(cfg.world_model),
    )
    actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=env.num_actions)
    agent = Agent(tokenizer, world_model, actor_critic).to(device)

    ckpt = cfg.initialization.path_to_checkpoint
    if ckpt is None:
        ckpt = str(iris_root / "checkpoints" / "last.pt")
    agent.load(Path(ckpt), device)
    agent.eval()

    return agent, env, cfg


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Manages one IRIS inference loop in a background daemon thread.

    Usage::

        engine = InferenceEngine(iris_src="/iris/src", iris_root="/iris")
        engine.start(Path("Breakout.pt"), "BreakoutNoFrameskip-v4")
        frame_dict = engine.get_frame()          # None on timeout
        engine.switch_agent(Path("Alien.pt"), "AlienNoFrameskip-v4")
        engine.stop()

    Thread safety: all public methods are safe to call from any thread.
    """

    def __init__(self, iris_src: str, iris_root: str, sae_dir: Optional[str] = None) -> None:
        self._iris_src = Path(iris_src)
        self._iris_root = Path(iris_root)
        # Directory scanned for sae_L*.pt artifacts (default: <iris_root>/checkpoints)
        self._sae_dir = Path(sae_dir) if sae_dir else (self._iris_root / "checkpoints")

        # Add IRIS src/ to path once at construction time
        if str(self._iris_src) not in sys.path:
            sys.path.insert(0, str(self._iris_src))

        self._queue: Queue = Queue(maxsize=_QUEUE_MAXSIZE)
        self._latent_queue: Queue = Queue(maxsize=_QUEUE_MAXSIZE)
        self._latent_sub = _LatentSubscriberCounter()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()          # not paused initially
        self._reset_requested = threading.Event()
        # Single-step gate (alongside pause, never overloading it):
        #   _single_step_mode set  → loop is paused and advances one frame per step_once()
        #   _step_event            → released by step_once(); consumed (cleared) per frame
        self._single_step_mode = threading.Event()
        self._step_event = threading.Event()
        self._loop_episodes = True

        self._hooks = IrisHookExtractor()
        self._agent = None
        self._env = None
        self._cfg = None

        # SAE (loaded per agent in start(); None disables the feature panel)
        self._sae = None
        self._sae_layer: Optional[int] = None
        self._sae_norm_mean: Optional[torch.Tensor] = None
        self._sae_norm_std: Optional[torch.Tensor] = None
        self._sae_topk = 12

        # Intervention state (set from control thread, read in inference thread)
        self._iv_lock = threading.Lock()
        self._iv_feature_id: Optional[int] = None
        self._iv_scale: float = 0.0
        # Last frame's full SAE feature vector (d_hidden,), for magnitude-relative
        # intervention scaling. Written by _compute_sae_features (inference thread).
        self._sae_last_feats: Optional[torch.Tensor] = None
        # Floor activation so an off (zero) feature can still be driven by the slider.
        self._sae_mag_floor = 1.0

        # Counters (written only from inference thread)
        self._step_count = 0
        self._episode_count = 0
        self._drop_count = 0
        self._total_frames = 0
        self._infer_fps = _FpsCounter()

        # Event callbacks: called from inference thread with (event_name, data)
        self._event_callbacks: List[Callable] = []
        self._cb_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        checkpoint_path: Path,
        env_id: str,
        device_str: str = "cpu",
        loop: bool = True,
    ) -> None:
        """Load agent and start background inference thread."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Engine already running; call stop() or switch_agent() first")

        self._stop_event.clear()
        self._pause_event.set()
        self._reset_requested.clear()
        self._single_step_mode.clear()
        self._step_event.clear()
        self._loop_episodes = loop
        self._step_count = 0
        self._episode_count = 0
        self._drop_count = 0
        self._total_frames = 0
        self._infer_fps = _FpsCounter()
        # Clear any SAE from a previous agent; _load_sae repopulates below.
        self._sae = None
        self._sae_layer = None
        self._sae_norm_mean = None
        self._sae_norm_std = None
        with self._iv_lock:
            self._iv_feature_id = None
            self._iv_scale = 0.0
        self._sae_last_feats = None

        logger.info("Loading agent from %s (env=%s, device=%s)", checkpoint_path, env_id, device_str)
        try:
            self._agent, self._env, self._cfg = _load_agent(
                self._iris_root, checkpoint_path, env_id, device_str
            )
        except Exception as exc:
            logger.error("Agent load failed: %s", exc)
            self._emit_event("error", {"message": f"Agent load failed: {exc}"})
            raise

        # Load an SAE for this env (if one exists) before attaching hooks, so the
        # residual-capture hook is registered at the SAE's layer.
        self._load_sae(env_id, device_str, num_layers=len(self._agent.world_model.transformer.blocks))
        self._hooks.attach(self._agent.world_model, capture_resid_layer=self._sae_layer)

        num_layers = len(self._agent.world_model.transformer.blocks)
        num_heads = self._agent.world_model.transformer.config.num_heads
        tpb = self._agent.world_model.config.tokens_per_block

        self._emit_event("agent_loaded", {
            "agent": Path(checkpoint_path).stem,
            "env_id": env_id,
            "layers": num_layers,
            "heads": num_heads,
            "tokens_per_block": tpb,
        })

        self._thread = threading.Thread(
            target=self._run_safe, name="iris-inference", daemon=True
        )
        self._thread.start()
        logger.info("Inference started (%d layers, %d heads, tpb=%d)", num_layers, num_heads, tpb)

    def stop(self) -> None:
        """Signal the thread to stop, wait up to 10 s, then clean up."""
        self._stop_event.set()
        self._pause_event.set()  # unblock if paused
        self._step_event.set()   # unblock if parked in the single-step gate
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logger.warning("Inference thread did not stop within 10 s")
            self._thread = None

        if self._hooks.num_layers > 0:
            self._hooks.detach()
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
        self._agent = None

    def switch_agent(
        self,
        checkpoint_path: Path,
        env_id: str,
        device_str: str = "cpu",
        loop: bool = True,
    ) -> None:
        """Stop current agent, flush queue, start fresh with new agent."""
        self.stop()
        self._drain_queue()
        self.start(checkpoint_path, env_id, device_str, loop)

    # ------------------------------------------------------------------
    # Runtime controls
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        # Leave single-step mode cleanly and let the loop run freely again.
        self._single_step_mode.clear()
        self._step_event.set()       # release a thread parked in the step gate
        self._pause_event.set()

    def step_once(self) -> None:
        """Advance the inference loop by exactly one frame while paused.

        Enters single-step mode (so the loop parks on _step_event rather than
        free-running) and releases one iteration.  Has no visible effect while
        the loop is running free — the step gate is only reached when paused.
        """
        self._single_step_mode.set()
        self._step_event.set()

    def set_intervention(self, feature_id: Optional[int], scale: float) -> None:
        """Set (or clear) the SAE feature intervention applied during imagination.

        ``feature_id=None`` or ``scale=0`` disables it.  Takes effect on the next
        single-step; the live free-running loop is unaffected (imagination is
        step-mode only).  Thread-safe.
        """
        with self._iv_lock:
            self._iv_feature_id = feature_id
            self._iv_scale = float(scale)

    def restart_episode(self) -> None:
        self._reset_requested.set()

    def set_loop(self, enabled: bool) -> None:
        self._loop_episodes = enabled

    def add_latent_subscriber(self) -> None:
        """Increment the count of active /ws/latent clients (thread-safe)."""
        self._latent_sub.add()

    def remove_latent_subscriber(self) -> None:
        """Decrement the count of active /ws/latent clients (thread-safe, clamps at 0)."""
        self._latent_sub.remove()

    @property
    def latent_subscriber_count(self) -> int:
        """Current number of active /ws/latent clients."""
        return self._latent_sub.count

    def get_frame(self, timeout: float = 0.05) -> Optional[dict]:
        """Return the next FrameData dict or None if queue empty within timeout."""
        try:
            return self._queue.get(timeout=timeout).to_dict()
        except Empty:
            return None

    def get_latent_frame(self, timeout: float = 0.05) -> Optional[dict]:
        """Return the next latent-page FrameData dict or None if empty within timeout."""
        try:
            return self._latent_queue.get(timeout=timeout).to_dict()
        except Empty:
            return None

    def register_event_callback(self, cb: Callable) -> None:
        with self._cb_lock:
            self._event_callbacks.append(cb)

    def unregister_event_callback(self, cb: Callable) -> None:
        with self._cb_lock:
            try:
                self._event_callbacks.remove(cb)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def drop_rate(self) -> float:
        if self._total_frames == 0:
            return 0.0
        return self._drop_count / self._total_frames

    def get_config(self) -> dict:
        """Current model config (returns empty dict if no agent loaded)."""
        if self._agent is None:
            return {}
        wm = self._agent.world_model
        return {
            "num_layers": len(wm.transformer.blocks),
            "num_heads": wm.transformer.config.num_heads,
            "embed_dim": wm.transformer.config.embed_dim,
            "tokens_per_block": wm.config.tokens_per_block,
            "max_blocks": wm.config.max_blocks,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_sae(self, env_id: str, device_str: str, num_layers: int) -> None:
        """Discover and load an SAE artifact for this env from the SAE dir.

        Picks the first ``sae_L*.pt`` whose stored env_id matches; on any problem
        (no file, arch/layer mismatch, load error) the SAE stays disabled and the
        feature panel simply won't render — never fatal to inference.
        """
        try:
            if not self._sae_dir.is_dir():
                return
            candidates = sorted(self._sae_dir.glob("sae_L*.pt"))
            for path in candidates:
                try:
                    sae, meta = load_artifact(path, device=device_str)
                except Exception as exc:
                    logger.warning("SAE %s failed to load: %s", path.name, exc)
                    continue
                if meta.get("env_id") not in (None, env_id):
                    continue  # trained for a different game
                layer = int(meta["layer"])
                if not (0 <= layer < num_layers):
                    logger.warning("SAE %s layer %d out of range for this model", path.name, layer)
                    continue
                if int(meta["d_in"]) != self._agent.world_model.transformer.config.embed_dim:
                    logger.warning("SAE %s d_in mismatch", path.name)
                    continue
                device = torch.device(device_str)
                self._sae = sae
                self._sae_layer = layer
                self._sae_norm_mean = meta["norm"]["mean"].to(device)
                self._sae_norm_std = meta["norm"]["std"].to(device).clamp_min(1e-6)
                logger.info(
                    "Loaded SAE %s (layer=%d, d_hidden=%d, env=%s)",
                    path.name, layer, int(meta["d_hidden"]), meta.get("env_id"),
                )
                return
        except Exception as exc:
            logger.warning("SAE discovery failed: %s", exc)
            self._sae = None
            self._sae_layer = None

    def _compute_sae_features(self) -> Optional[List[Dict[str, float]]]:
        """Encode the last-token residual through the SAE → top-K firing features.

        Reads the full residual captured by the hook on the SAE layer's block,
        normalises the action-token (index -1) vector with the training stats, and
        returns the top-K active features as ``[{"id", "mag"}]``.  None if no SAE is
        loaded or no residual was captured this pass.
        """
        if self._sae is None:
            return None
        resid = self._hooks.get_resid()      # (1, T, E) or None
        if resid is None:
            return None
        try:
            with torch.no_grad():
                x = (resid[0, -1] - self._sae_norm_mean) / self._sae_norm_std   # (E,)
                feats = self._sae.encode(x.unsqueeze(0)).squeeze(0)             # (d_hidden,)
                self._sae_last_feats = feats.detach()  # for magnitude-relative interventions
                k = min(self._sae_topk, feats.numel())
                vals, idx = torch.topk(feats, k)
                return [
                    {"id": int(i), "mag": round(float(v), 4)}
                    for v, i in zip(vals.tolist(), idx.tolist())
                    if v > 0.0
                ]
        except Exception as exc:
            logger.debug("SAE feature computation failed: %s", exc)
            return None

    def _intervention_direction(self, feature_id: int, scale: float) -> Optional["torch.Tensor"]:
        """Raw residual-space vector to add for a feature intervention, or None.

        ``scale`` is a **magnitude-relative multiplier** (comparable across features):
        the injected vector is ``scale * ref * W_dec[feature_id] * std`` where ``ref``
        is the feature's own current activation (floored so an off feature can still
        be driven). ``W_dec[feature_id]`` is the unit-norm decoder row (feature
        direction in normalised space); multiplying by ``std`` maps it back to the raw
        residual the WM block emits. So ``scale=0`` ≈ leave as-is, ``scale=-1`` ≈
        cancel one unit of the feature (suppress), ``scale>0`` amplifies.
        """
        if self._sae is None or feature_id is None or scale == 0.0:
            return None
        try:
            if not (0 <= feature_id < self._sae.d_hidden):
                return None
            with torch.no_grad():
                ref = self._sae_mag_floor
                if self._sae_last_feats is not None and feature_id < self._sae_last_feats.numel():
                    ref = max(float(self._sae_last_feats[feature_id]), self._sae_mag_floor)
                direction = self._sae.W_dec[feature_id] * self._sae_norm_std   # (E,) raw space
                return (scale * ref * direction).detach()
        except Exception as exc:
            logger.debug("Intervention direction failed: %s", exc)
            return None

    def _drain_queue(self) -> None:
        for q in (self._queue, self._latent_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break

    def _emit_event(self, event_name: str, data: dict) -> None:
        with self._cb_lock:
            cbs = list(self._event_callbacks)
        for cb in cbs:
            try:
                cb(event_name, data)
            except Exception as exc:
                logger.debug("Event callback error: %s", exc)

    def _run_safe(self) -> None:
        try:
            self._run_inner()
        except Exception as exc:
            logger.error("Inference thread crashed: %s", exc, exc_info=True)
            self._emit_event("error", {"message": str(exc)})

    def _run_inner(self) -> None:
        agent = self._agent
        env = self._env
        hooks = self._hooks
        wm = agent.world_model
        tokenizer = agent.tokenizer
        device = agent.device

        tpb = wm.config.tokens_per_block
        num_layers = len(wm.transformer.blocks)
        token_layout = {
            "tokens_per_block": tpb,
            "obs_per_block": tpb - 1,
            "labels": get_token_labels(tpb, tpb),
        }

        # --- Episode init ---
        obs = env.reset()                          # (1, H, W, C) uint8
        obs_tensor = _to_tensor(obs, device)       # (1, C, H, W) float [0,1]
        agent.actor_critic.reset(n=1)
        self._episode_count += 1
        ep_return = 0.0
        self._emit_event("episode_start", {"episode": self._episode_count})

        while not self._stop_event.is_set():
            # Pause / single-step gate.  While paused, the loop parks on
            # _step_event, which resume(), step_once() and stop() all set.  After
            # each wake we decide from the flags whether to run the body:
            #   - stop()       → break
            #   - resume()     → _pause_event set, fall through to free running
            #   - step_once()  → _single_step_mode set, run exactly one frame and
            #                    re-park on the next iteration
            #   - spurious     → still paused and not stepping → re-park
            if not self._pause_event.is_set():
                self._step_event.wait()
                self._step_event.clear()       # consume exactly one release
                if self._stop_event.is_set():
                    break
                if not self._pause_event.is_set() and not self._single_step_mode.is_set():
                    continue                   # still paused, no step pending → re-park
            if self._stop_event.is_set():
                break

            # Manual episode reset
            if self._reset_requested.is_set():
                self._reset_requested.clear()
                obs = env.reset()
                obs_tensor = _to_tensor(obs, device)
                agent.actor_critic.reset(n=1)
                self._episode_count += 1
                ep_return = 0.0
                self._emit_event("episode_start", {"episode": self._episode_count})

            # --- Agent step ---
            with torch.no_grad():
                act = agent.act(obs_tensor, should_sample=True).cpu().numpy()  # (1,)

            next_obs, reward, done, _ = env.step(act)
            r = float(reward[0]) if hasattr(reward, "__len__") else float(reward)
            ep_return += r

            # --- World model forward (single block, no KV cache) for hooks ---
            t_wm = time.perf_counter()
            obs_tokens_enc: Optional[torch.Tensor] = None
            attn_data: Optional[Dict[int, torch.Tensor]] = None
            norms_data: Optional[Dict[int, torch.Tensor]] = None  # 0-d tensors
            try:
                with torch.no_grad():
                    enc_out = tokenizer.encode(obs_tensor, should_preprocess=True)
                    obs_tokens_enc = enc_out.tokens  # (1, K=16)
                    act_tensor = torch.tensor(
                        [[act[0]]], dtype=torch.long, device=device
                    )  # (1, 1)
                    tokens = torch.cat([obs_tokens_enc, act_tensor], dim=1)  # (1, 17)
                    wm(tokens, past_keys_values=None)
                attn_data, norms_data = hooks.get_data()
            except Exception as exc:
                logger.debug("WM forward failed: %s", exc)
            hook_ms = (time.perf_counter() - t_wm) * 1000.0

            # --- Encode raw frame (fetch before reconstruction so we know target size) ---
            raw = _get_raw_frame(env)
            frame_b64 = _encode_frame(raw)

            # --- Decode reconstruction (only when latent clients are connected) ---
            reconstruction_b64: Optional[str] = None
            error_map_b64: Optional[str] = None
            reconstruction_error: Optional[float] = None
            if obs_tokens_enc is not None and self._latent_sub.count > 0:
                with torch.no_grad():
                    reconstruction_b64, error_map_b64, reconstruction_error = (
                        _decode_reconstruction(
                            tokenizer, obs_tokens_enc, obs_tensor, device,
                            target_hw=(raw.shape[0], raw.shape[1]),
                        )
                    )

            # --- Imagined next frame (world-model prediction) — step mode only ---
            # ~1 + K autoregressive WM passes; far heavier than the live path, so
            # only run when single-stepping (paused) and the user is studying one
            # transition.  Skipped entirely while free-running (zero live-cost).
            # If an intervention is set, also roll out an intervened copy and diff
            # the two (~2*(1+K) passes) — both use argmax so the diff reflects only
            # the intervention, not sampling noise.
            imagined_next_b64: Optional[str] = None
            imagined_iv_b64: Optional[str] = None
            iv_diff_b64: Optional[str] = None
            iv_meta: Optional[Dict[str, Any]] = None
            if self._single_step_mode.is_set() and obs_tokens_enc is not None:
                with self._iv_lock:
                    iv_fid, iv_scale = self._iv_feature_id, self._iv_scale
                iv_dir = self._intervention_direction(iv_fid, iv_scale)
                deterministic = iv_dir is not None
                a = int(act[0])
                target_hw = (raw.shape[0], raw.shape[1])
                base_rgb, base_tokens = _imagine_next_rgb(
                    wm, tokenizer, obs_tokens_enc, a, device,
                    target_hw=target_hw, deterministic=deterministic,
                )
                if base_rgb is not None:
                    imagined_next_b64 = _encode_frame(base_rgb)
                if iv_dir is not None and base_rgb is not None and self._sae_layer is not None:
                    iv_rgb, iv_tokens = _imagine_next_rgb(
                        wm, tokenizer, obs_tokens_enc, a, device,
                        target_hw=target_hw,
                        intervention=(self._sae_layer, iv_dir),
                        deterministic=True,
                    )
                    if iv_rgb is not None:
                        imagined_iv_b64 = _encode_frame(iv_rgb)
                        iv_diff_b64 = _abs_diff_gray_png(base_rgb, iv_rgb)
                        # Token-change count: a reliable signal even when the pixel
                        # diff is sub-visible (argmax decoding is discrete).
                        n_changed = None
                        if base_tokens is not None and iv_tokens is not None:
                            n_changed = int((base_tokens != iv_tokens).sum().item())
                        iv_meta = {
                            "feature_id": int(iv_fid),
                            "scale": float(iv_scale),
                            "n_changed": n_changed,
                        }

            # --- SAE feature firing (rides the Regime-A residual capture) ---
            sae_features = self._compute_sae_features()

            # --- Update obs ---
            obs = next_obs
            obs_tensor = _to_tensor(obs, device)
            self._step_count += 1
            self._total_frames += 1
            self._infer_fps.tick()

            # --- Build attention payload ---
            attention_payload: Dict[str, list] = {}
            if attn_data is not None:
                for li, attn_t in attn_data.items():
                    # (1, nh, T_q, T_k) → nested Python list [nh][T_q][T_k]
                    attention_payload[str(li)] = attn_t.squeeze(0).cpu().tolist()

            # .item() is called here — once, outside the forward pass, after all
            # layers have completed — rather than per-layer inside the hook callback.
            # This avoids 10× GPU→CPU syncs interleaved with the transformer blocks.
            norms_payload = [
                norms_data[i].item() if norms_data and i in norms_data else 0.0
                for i in range(num_layers)
            ]

            frame_data = FrameData(
                frame=frame_b64,
                attention=attention_payload,
                norms=norms_payload,
                metrics={
                    "infer_fps": round(self._infer_fps.fps, 1),
                    "step": self._step_count,
                    "episode": self._episode_count,
                    "queue_depth": self._queue.qsize(),
                    "hook_latency_ms": round(hook_ms, 2),
                    "drop_rate": round(self.drop_rate, 3),
                    "return": round(ep_return, 1),
                },
                token_layout=token_layout,
                reconstruction=reconstruction_b64,
                error_map=error_map_b64,
                reconstruction_error=reconstruction_error,
                imagined_next=imagined_next_b64,
                sae_features=sae_features,
                sae_layer=self._sae_layer,
                imagined_intervened=imagined_iv_b64,
                intervention_diff=iv_diff_b64,
                intervention=iv_meta,
            )

            # Fan-out to main and latent queues — drop if consumer is behind
            try:
                self._queue.put_nowait(frame_data)
            except Full:
                self._drop_count += 1
            try:
                self._latent_queue.put_nowait(frame_data)
            except Full:
                pass  # latent drops are not counted against drop_rate

            # --- Episode end ---
            done_val = bool(done[0]) if hasattr(done, "__len__") else bool(done)
            if done_val:
                self._emit_event("episode_end", {
                    "episode": self._episode_count,
                    "return": round(ep_return, 1),
                    "steps": self._step_count,
                })
                ep_return = 0.0
                if self._loop_episodes:
                    obs = env.reset()
                    obs_tensor = _to_tensor(obs, device)
                    agent.actor_critic.reset(n=1)
                    self._episode_count += 1
                    self._emit_event("episode_start", {"episode": self._episode_count})
                else:
                    break


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """(1, H, W, C) uint8  →  (1, C, H, W) float32 [0, 1]."""
    import torch
    from einops import rearrange
    return rearrange(torch.FloatTensor(obs).div(255), "n h w c -> n c h w").to(device)


def _get_raw_frame(env) -> np.ndarray:
    """
    Return the original human-readable RGB frame (H, W, 3) uint8.

    Tries env.env.unwrapped.original_obs (set by IRIS ALE wrappers) first,
    then falls back to ALE render, then to a placeholder.
    """
    try:
        raw = env.env.unwrapped.original_obs
        if raw is not None:
            return np.asarray(raw, dtype=np.uint8)
    except AttributeError:
        pass
    try:
        raw = env.env.unwrapped.render("rgb_array")
        if raw is not None:
            return np.asarray(raw, dtype=np.uint8)
    except Exception:
        pass
    return np.zeros((210, 160, 3), dtype=np.uint8)
