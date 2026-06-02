#!/usr/bin/env python
"""
Test 5 — activation-event timing trace.

Over a full episode, does a feature's activation trace align temporally with on-screen
events? For 5 features (collision detectors + the air-flight tracker, from Test 3) we plot
the activation trace and mark programmatically-detected events:
  - paddle-ball collision: ball at the paddle, x-overlapping it
  - ball-brick collision:  the red brick-pixel count drops (a brick was destroyed)
  - ball direction change: vertical velocity of the ball flips sign
For each (feature, event-type) we quantify an **event lift** = mean activation in a ±1
window around events / mean activation over the whole episode. Lift >> 1 means the feature
fires at that event; ~1 means no alignment. Run on >=2 episodes; a feature is only an event
detector if the lift holds in both.

Run:
    python backend/scripts/diagnostics/test5_timing_trace.py \
        --checkpoint /path/.../Breakout.pt --sae /path/.../sae_L5.pt \
        --device cpu --episodes 2 --max-frames 1200
Writes results/test5_timing_trace.json and results/test5_timing_ep{N}.png.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BACKEND = Path(__file__).resolve().parents[2]
_IRIS_ROOT = Path(os.environ.get("IRIS_ROOT", Path(__file__).resolve().parents[4] / "iris"))
_IRIS_SRC = Path(os.environ.get("IRIS_SRC", _IRIS_ROOT / "src"))
for _p in (str(_BACKEND), str(_IRIS_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference import _load_agent, _get_raw_frame  # noqa: E402
from sae import load_artifact  # noqa: E402

# Default features (Test 3 roles): collision detectors + the air-flight tracker + anti-corr.
DEFAULT_FEATURES = [1199, 120, 1773, 1364, 316]
ROLE = {1199: "collision", 120: "collision", 1773: "collision (stable)",
        1364: "air-flight tracker", 316: "anti-collision (air)"}

_WALL_L, _WALL_R = 8, 152
_BRICK_ROWS = (57, 92)
_BALL_ROWS = (93, 184)
_PADDLE_ROWS = (185, 200)


def detect(raw):
    """Return (ball_xy, paddle_x, brick_px) from the 210x160 human frame."""
    r, g, b = raw[..., 0].astype(int), raw[..., 1].astype(int), raw[..., 2].astype(int)
    red = (r > 160) & (g < 110) & (b < 110)
    red[:, :_WALL_L] = False
    red[:, _WALL_R:] = False
    ball = red.copy(); ball[: _BALL_ROWS[0]] = False; ball[_BALL_ROWS[1]:] = False
    pad = red.copy(); pad[: _PADDLE_ROWS[0]] = False; pad[_PADDLE_ROWS[1]:] = False
    brick_px = int(red[_BRICK_ROWS[0]:_BRICK_ROWS[1]].sum())
    by, bx = np.nonzero(ball); py, px = np.nonzero(pad)
    ball_xy = (float(bx.mean()), float(by.mean())) if len(bx) >= 1 else None
    paddle_x = float(px.mean()) if len(px) >= 1 else None
    return ball_xy, paddle_x, brick_px


@torch.no_grad()
def feature_vector(wm, sae, layer, norm_mean, norm_std, obs_tokens, action, device):
    captured = {}
    h = wm.transformer.blocks[layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("resid", o.detach()))
    try:
        tokens = torch.cat(
            [obs_tokens, torch.tensor([[int(action)]], dtype=torch.long, device=device)], dim=1)
        wm(tokens, past_keys_values=None)
        x = (captured["resid"][0, -1] - norm_mean) / norm_std
        feats = sae.encode(x.unsqueeze(0)).squeeze(0)
    finally:
        h.remove()
    return feats.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def run_episode(agent, env, wm, tokenizer, sae, layer, norm_mean, norm_std, device, feats_ids,
                max_frames, collision_y, x_overlap):
    from einops import rearrange
    obs = env.reset(); agent.actor_critic.reset(n=1)
    acts, ball_y, brick, coll, resets = [], [], [], [], []
    for t in range(max_frames):
        obs_t = rearrange(torch.FloatTensor(obs).div(255), "n h w c -> n c h w").to(device)
        a = agent.act(obs_t, should_sample=True).cpu().numpy()
        tok = tokenizer.encode(obs_t, should_preprocess=True).tokens
        fv = feature_vector(wm, sae, layer, norm_mean, norm_std, tok, int(a[0]), device)
        raw = _get_raw_frame(env)
        ball, paddle, brick_px = detect(raw)
        acts.append(fv[feats_ids])
        ball_y.append(ball[1] if ball else np.nan)
        brick.append(brick_px)
        coll.append(1 if (ball and paddle is not None and ball[1] >= collision_y
                          and abs(ball[0] - paddle) <= x_overlap) else 0)
        obs, _r, done, _ = env.step(a)
        if (bool(done[0]) if hasattr(done, "__len__") else bool(done)):
            obs = env.reset(); agent.actor_critic.reset(n=1); resets.append(t)
    return (np.array(acts), np.array(ball_y), np.array(brick), np.array(coll), set(resets))


def detect_events(ball_y, brick, coll, resets):
    """Return dict of event-type -> sorted list of frame indices (resets excluded)."""
    n = len(coll)
    skip = resets | {r + 1 for r in resets}
    # paddle-ball collision: rising edge of the collision flag
    pball = [t for t in range(1, n) if coll[t] and not coll[t - 1] and t not in skip]
    # ball-brick collision: brick pixel count drops (a brick removed)
    brick_evt = [t for t in range(1, n) if (brick[t] < brick[t - 1] - 4) and t not in skip]
    # ball vertical direction change: sign flip of dy (needs ball visible on both sides)
    dirchg = []
    for t in range(2, n):
        if t in skip:
            continue
        a, b, c = ball_y[t - 2], ball_y[t - 1], ball_y[t]
        if np.isnan(a) or np.isnan(b) or np.isnan(c):
            continue
        if np.sign(b - a) != 0 and np.sign(c - b) != 0 and np.sign(b - a) != np.sign(c - b):
            dirchg.append(t)
    return {"paddle_collision": pball, "brick_destruction": brick_evt, "ball_dir_change": dirchg}


def event_lift(act, events, win=1):
    """mean activation in ±win around event frames / mean activation overall (per feature)."""
    base = act.mean(axis=0)
    base[base < 1e-6] = 1e-6
    out = {}
    for name, idxs in events.items():
        if not idxs:
            out[name] = None
            continue
        rows = []
        for t in idxs:
            lo, hi = max(0, t - win), min(len(act), t + win + 1)
            rows.append(act[lo:hi].mean(axis=0))
        out[name] = (np.array(rows).mean(axis=0) / base).round(3).tolist()
    return out


def make_plot(act, events, feats_ids, ep, out):
    fig, axes = plt.subplots(len(feats_ids), 1, figsize=(13, 1.7 * len(feats_ids)), sharex=True)
    colors = {"paddle_collision": "#ec4899", "brick_destruction": "#22c55e",
              "ball_dir_change": "#f59e0b"}
    for i, fid in enumerate(feats_ids):
        ax = axes[i]
        ax.plot(act[:, i], color="#6366f1", lw=0.8)
        for name, idxs in events.items():
            for t in idxs:
                ax.axvline(t, color=colors[name], alpha=0.35, lw=0.7)
        ax.set_ylabel(f"#{fid}", fontsize=8)
        ax.set_title(f"#{fid} — {ROLE.get(fid, '?')}", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    handles = [plt.Line2D([0], [0], color=c, lw=2) for c in colors.values()]
    axes[0].legend(handles, list(colors.keys()), fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel("frame", fontsize=8)
    fig.suptitle(f"Test 5 — activation traces vs events (episode {ep})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--features", type=int, nargs="*", default=DEFAULT_FEATURES)
    p.add_argument("--collision-y", type=float, default=176)
    p.add_argument("--x-overlap", type=float, default=12)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=Path, default=_BACKEND.parent / "results")
    args = p.parse_args()

    agent, env, _cfg = _load_agent(_IRIS_ROOT, args.checkpoint, args.env_id, args.device)
    agent.eval()
    wm, tokenizer = agent.world_model, agent.tokenizer
    device = torch.device(args.device)
    sae, meta = load_artifact(str(args.sae), device=args.device)
    layer = int(meta["layer"])
    norm_mean = meta["norm"]["mean"].to(device)
    norm_std = meta["norm"]["std"].to(device).clamp_min(1e-6)
    feats_ids = list(args.features)

    episodes = []
    for ep in range(args.episodes):
        act, ball_y, brick, coll, resets = run_episode(
            agent, env, wm, tokenizer, sae, layer, norm_mean, norm_std, device,
            feats_ids, args.max_frames, args.collision_y, args.x_overlap)
        events = detect_events(ball_y, brick, coll, resets)
        lifts = event_lift(act, events)
        png = args.out_dir / f"test5_timing_ep{ep}.png"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        make_plot(act, events, feats_ids, ep, png)
        episodes.append({
            "episode": ep, "frames": int(len(coll)),
            "n_events": {k: len(v) for k, v in events.items()},
            "event_lift_per_feature": {  # feature_id -> {event_type -> lift}
                str(fid): {ev: (lifts[ev][i] if lifts[ev] else None) for ev in lifts}
                for i, fid in enumerate(feats_ids)},
            "plot": str(png),
        })
        print(f"ep {ep}: {len(coll)} frames, events {episodes[-1]['n_events']}")
        for i, fid in enumerate(feats_ids):
            lf = {ev: (lifts[ev][i] if lifts[ev] else None) for ev in lifts}
            print(f"   #{fid} ({ROLE.get(fid,'?')}): paddle×{lf['paddle_collision']} "
                  f"brick×{lf['brick_destruction']} dir×{lf['ball_dir_change']}")

    # cross-episode consistency of the paddle-collision lift (the cleanest event)
    summary = {}
    if len(episodes) >= 2:
        rows = []
        for fid in feats_ids:
            l0 = episodes[0]["event_lift_per_feature"][str(fid)]["paddle_collision"]
            l1 = episodes[1]["event_lift_per_feature"][str(fid)]["paddle_collision"]
            rows.append({"feature": fid, "paddle_lift_ep0": l0, "paddle_lift_ep1": l1,
                         "consistent": (l0 is not None and l1 is not None
                                        and (l0 > 1.5) == (l1 > 1.5))})
        summary["paddle_collision_lift"] = rows
    with open(args.out_dir / "test5_timing_trace.json", "w") as f:
        json.dump({"features": feats_ids, "roles": {str(k): v for k, v in ROLE.items()},
                   "episodes": episodes, "summary": summary}, f, indent=2)
    print("wrote", args.out_dir / "test5_timing_trace.json")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
