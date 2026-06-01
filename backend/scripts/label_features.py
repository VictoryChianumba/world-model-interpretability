#!/usr/bin/env python
"""
Automated DESCRIPTIVE labeling pass for SAE features → seed bookmarks.

For the most-frequently-firing features, this measures what an intervention DOES to
the world model's imagined next frame — purely descriptive facts:
  - n_tokens_changed (of 16): how many predicted obs tokens the intervention flipped
  - diff region: bounding box + centroid of the changed pixels (where the effect is)
  - mean diff magnitude: how strong the pixel change is

It writes these as seed bookmarks (source="auto-descriptive") that a human renames.
It is explicitly NOT semantic — it reports *where/how much* a feature's intervention
changes the prediction, never *what game concept* it represents.

Offline tool; reuses the engine's _load_agent / _imagine_next_rgb and the SAE artifact.

Example
-------
    python backend/scripts/label_features.py \
        --checkpoint /path/to/iris/checkpoints/Breakout.pt \
        --env-id BreakoutNoFrameskip-v4 \
        --sae /path/to/iris/checkpoints/sae_L5.pt \
        --bookmarks /path/to/iris/checkpoints/bookmarks.json \
        --top 20 --scale 6 --frames 60 --device mps
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_BACKEND = Path(__file__).resolve().parents[1]
_IRIS_ROOT = Path(os.environ.get("IRIS_ROOT", Path(__file__).resolve().parents[3] / "iris"))
_IRIS_SRC = Path(os.environ.get("IRIS_SRC", _IRIS_ROOT / "src"))
for _p in (str(_BACKEND), str(_IRIS_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference import _load_agent, _imagine_next_rgb  # noqa: E402
from sae import load_artifact  # noqa: E402
from bookmarks import BookmarkStore  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


@torch.no_grad()
def run(args) -> None:
    device = torch.device(args.device)
    log(f"Loading agent {args.checkpoint} + SAE {args.sae} (device={args.device})")
    agent, env, _cfg = _load_agent(_IRIS_ROOT, args.checkpoint, args.env_id, args.device)
    agent.eval()
    wm, tokenizer = agent.world_model, agent.tokenizer

    sae, meta = load_artifact(str(args.sae), device=args.device)
    layer = int(meta["layer"])
    norm_mean = meta["norm"]["mean"].to(device)
    norm_std = meta["norm"]["std"].to(device).clamp_min(1e-6)
    mag_floor = 1.0
    log(f"SAE layer={layer} d_hidden={sae.d_hidden} env={meta.get('env_id')} "
        f"token_policy={meta.get('token_policy')}")

    # Residual capture hook at the SAE layer (last-token, matching the live engine).
    captured = {}
    h = wm.transformer.blocks[layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("resid", o.detach())
    )

    from einops import rearrange

    # --- Collect frames: obs tokens, action, last-token feature vector ---
    frames = []  # list of dicts: {obs_tokens (1,16), action int, feats (d_hidden,)}
    firing_counts = torch.zeros(sae.d_hidden)
    obs = env.reset()
    agent.actor_critic.reset(n=1)
    for _ in range(args.frames):
        obs_t = rearrange(torch.FloatTensor(obs).div(255), "n h w c -> n c h w").to(device)
        act = agent.act(obs_t, should_sample=True).cpu().numpy()
        enc = tokenizer.encode(obs_t, should_preprocess=True)
        obs_tokens = enc.tokens                                   # (1, 16)
        act_tensor = torch.tensor([[int(act[0])]], dtype=torch.long, device=device)
        tokens = torch.cat([obs_tokens, act_tensor], dim=1)      # (1, 17)
        captured.clear()
        wm(tokens, past_keys_values=None)                        # fills captured["resid"]
        resid = captured["resid"]                                # (1, 17, E)
        x = (resid[0, -1] - norm_mean) / norm_std
        feats = sae.encode(x.unsqueeze(0)).squeeze(0)            # (d_hidden,)
        firing_counts += (feats > 0).float().cpu()
        frames.append({"obs_tokens": obs_tokens, "action": int(act[0]), "feats": feats.cpu()})
        obs, _r, done, _ = env.step(act)
        d = bool(done[0]) if hasattr(done, "__len__") else bool(done)
        if d:
            obs = env.reset()
            agent.actor_critic.reset(n=1)
    h.remove()

    # --- Pick top-N most-frequently-firing features ---
    n_top = min(args.top, sae.d_hidden)
    top_ids = torch.topk(firing_counts, n_top).indices.tolist()
    log(f"Collected {len(frames)} frames; labeling top {n_top} features by firing freq")

    store = BookmarkStore(str(args.bookmarks))
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    written = 0
    for fid in top_ids:
        # Pick the frame where this feature fires most strongly.
        best = max(frames, key=lambda fr: float(fr["feats"][fid]))
        ref = max(float(best["feats"][fid]), mag_floor)
        direction = (args.scale * ref * sae.W_dec[fid] * norm_std).detach()
        obs_tokens = best["obs_tokens"]
        a = best["action"]

        base_rgb, base_tok = _imagine_next_rgb(
            wm, tokenizer, obs_tokens, a, device, deterministic=True
        )
        iv_rgb, iv_tok = _imagine_next_rgb(
            wm, tokenizer, obs_tokens, a, device,
            intervention=(layer, direction), deterministic=True,
        )
        if base_rgb is None or iv_rgb is None:
            continue

        n_changed = int((base_tok != iv_tok).sum().item())
        diff = np.abs(base_rgb.astype(np.int16) - iv_rgb.astype(np.int16)).mean(axis=2)
        ys, xs = np.where(diff > args.diff_thresh)
        if len(xs) == 0:
            region = "no visible effect"
            mean_diff = 0.0
        else:
            H, W = diff.shape
            cy, cx = ys.mean() / H, xs.mean() / W
            vert = "top" if cy < 0.4 else "bottom" if cy > 0.6 else "mid"
            horiz = "left" if cx < 0.4 else "right" if cx > 0.6 else "center"
            region = f"{vert}-{horiz}"
            mean_diff = float(diff[diff > args.diff_thresh].mean())

        label = f"effect: {region}, {n_changed}/16 tok"
        notes = (f"auto: scale×{args.scale}, mean_diff={mean_diff:.1f}, "
                 f"changed_px={len(xs)}, peak_act={ref:.2f}")
        store.upsert(meta.get("env_id") or args.env_id, layer, fid, label,
                     notes=notes, source="auto-descriptive", updated_at=ts)
        written += 1
        log(f"  #{fid}: {label}  ({notes})")

    log(f"Wrote {written} descriptive seed bookmarks → {args.bookmarks}")
    try:
        env.close()
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--sae", type=Path, required=True, help="Trained sae_L*.pt artifact")
    p.add_argument("--bookmarks", type=Path, required=True, help="bookmarks.json to write")
    p.add_argument("--top", type=int, default=20, help="Number of top features to label")
    p.add_argument("--scale", type=float, default=6.0, help="Magnitude-relative intervention scale")
    p.add_argument("--frames", type=int, default=60, help="Frames to sample for firing stats")
    p.add_argument("--diff-thresh", type=float, default=8.0, help="Per-pixel diff threshold for region")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
