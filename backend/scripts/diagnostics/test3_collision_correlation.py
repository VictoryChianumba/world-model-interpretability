#!/usr/bin/env python
"""
Test 3 — paddle-ball collision correlation.

Do features fire more strongly during paddle-ball collisions than during ball-in-air
states? For each frame we compute the full SAE feature vector AND extract ball/paddle
positions (state_extract on the real obs), then label the frame:
  - collision: ball low (near the paddle band) AND horizontally over the paddle
  - air:       ball clearly above the paddle
For the top-N features by overall activity, compare mean activation across collision
frames (mu_c) vs air frames (mu_a): Delta = mu_c - mu_a. Large +Delta = collision-
correlated; ~0 = uncorrelated; -Delta = anti-correlated (fires more in mid-air).

Run on >=2 episodes; a feature is only "really" collision-correlated if it tops the
Delta ranking in both. Single-measurement discipline is mandatory here.

Run:
    python backend/scripts/diagnostics/test3_collision_correlation.py \
        --checkpoint /path/.../Breakout.pt --sae /path/.../sae_L5.pt \
        --device cpu --episodes 2 --target 20 --topN 50
Writes results/test3_collision_correlation.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_BACKEND = Path(__file__).resolve().parents[2]
_IRIS_ROOT = Path(os.environ.get("IRIS_ROOT", Path(__file__).resolve().parents[4] / "iris"))
_IRIS_SRC = Path(os.environ.get("IRIS_SRC", _IRIS_ROOT / "src"))
for _p in (str(_BACKEND), str(_IRIS_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference import _load_agent, _get_raw_frame  # noqa: E402
from sae import load_artifact  # noqa: E402

# Breakout 210x160 human-frame layout (calibrated empirically — see WRITEUP Test 3):
# ball AND paddle are red (200,72,72); gray (142,142,142) is the side walls/score, so we
# detect by colour+region, not brightness. The 64x64 model obs is too coarse to resolve
# the ball at all (sub-pixel after resize) — itself a finding — so we label from the full-
# resolution human frame, which depicts the same moment as the activations.
_WALL_L, _WALL_R = 8, 152          # interior columns between the gray side walls
_BALL_ROWS = (93, 184)             # below the brick band, above the paddle: only the ball
_PADDLE_ROWS = (185, 200)


def detect_ball_paddle(raw):
    """Return (ball_xy_px or None, paddle_x_px or None) from the 210x160 human frame."""
    r, g, b = raw[..., 0].astype(int), raw[..., 1].astype(int), raw[..., 2].astype(int)
    red = (r > 160) & (g < 110) & (b < 110)
    red[:, :_WALL_L] = False
    red[:, _WALL_R:] = False
    ball = red.copy(); ball[: _BALL_ROWS[0]] = False; ball[_BALL_ROWS[1]:] = False
    pad = red.copy(); pad[: _PADDLE_ROWS[0]] = False; pad[_PADDLE_ROWS[1]:] = False
    by, bx = np.nonzero(ball)
    py, px = np.nonzero(pad)
    ball_xy = (float(bx.mean()), float(by.mean())) if len(bx) >= 1 else None
    paddle_x = float(px.mean()) if len(px) >= 1 else None
    return ball_xy, paddle_x


def classify(raw, collision_y, air_y, x_overlap):
    """Return 'collision', 'air', or None (pixel thresholds on the human frame)."""
    ball, paddle = detect_ball_paddle(raw)
    if ball is None:
        return None
    bx, by = ball
    if by >= collision_y and paddle is not None and abs(bx - paddle) <= x_overlap:
        return "collision"
    if by <= air_y:
        return "air"
    return None


@torch.no_grad()
def feature_vector(wm, sae, layer, norm_mean, norm_std, obs_tokens, action, device):
    captured = {}
    h = wm.transformer.blocks[layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("resid", o.detach())
    )
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
def collect_episode(agent, env, wm, tokenizer, sae, layer, norm_mean, norm_std, device, args):
    from einops import rearrange
    obs = env.reset(); agent.actor_critic.reset(n=1)
    coll, air, allv = [], [], []
    frames = 0
    while (len(coll) < args.target or len(air) < args.target) and frames < args.max_frames:
        obs_t = rearrange(torch.FloatTensor(obs).div(255), "n h w c -> n c h w").to(device)
        act = agent.act(obs_t, should_sample=True).cpu().numpy()
        obs_tokens = tokenizer.encode(obs_t, should_preprocess=True).tokens
        fv = feature_vector(wm, sae, layer, norm_mean, norm_std, obs_tokens, int(act[0]), device)
        allv.append(fv)
        # Capture the display frame BEFORE env.step so the label matches the obs the
        # feature vector was computed from (same alignment fix as Test 1).
        raw = _get_raw_frame(env)
        label = classify(raw, args.collision_y, args.air_y, args.x_overlap)
        if label == "collision" and len(coll) < args.target:
            coll.append(fv)
        elif label == "air" and len(air) < args.target:
            air.append(fv)
        obs, _r, done, _ = env.step(act)
        frames += 1
        if (bool(done[0]) if hasattr(done, "__len__") else bool(done)):
            obs = env.reset(); agent.actor_critic.reset(n=1)
    return np.array(coll), np.array(air), np.array(allv), frames


def analyze(coll, air, allv, topN):
    activity = allv.mean(axis=0)                       # overall activity per feature
    top = np.argsort(-activity)[:topN]
    mu_c = coll.mean(axis=0); mu_a = air.mean(axis=0)
    sd_c = coll.std(axis=0);  sd_a = air.std(axis=0)
    rows = []
    for fid in top:
        rows.append({
            "feature_id": int(fid),
            "mu_c": round(float(mu_c[fid]), 4),
            "mu_a": round(float(mu_a[fid]), 4),
            "delta": round(float(mu_c[fid] - mu_a[fid]), 4),
            "std_c": round(float(sd_c[fid]), 4),
            "std_a": round(float(sd_a[fid]), 4),
            "activity": round(float(activity[fid]), 4),
        })
    rows.sort(key=lambda r: r["delta"], reverse=True)
    return rows


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--target", type=int, default=20, help="Collision/air frames to collect each")
    p.add_argument("--topN", type=int, default=50)
    p.add_argument("--max-frames", type=int, default=6000)
    p.add_argument("--collision-y", type=float, default=176, help="ball_y px >= this & x-overlap → collision")
    p.add_argument("--air-y", type=float, default=150, help="ball_y px <= this → air (mid-flight)")
    p.add_argument("--x-overlap", type=float, default=12, help="|ball_x - paddle_x| px for collision")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=Path, default=_BACKEND.parent / "results" / "test3_collision_correlation.json")
    args = p.parse_args()

    agent, env, _cfg = _load_agent(_IRIS_ROOT, args.checkpoint, args.env_id, args.device)
    agent.eval()
    wm, tokenizer = agent.world_model, agent.tokenizer
    device = torch.device(args.device)
    sae, meta = load_artifact(str(args.sae), device=args.device)
    layer = int(meta["layer"])
    norm_mean = meta["norm"]["mean"].to(device)
    norm_std = meta["norm"]["std"].to(device).clamp_min(1e-6)

    episodes = []
    for ep in range(args.episodes):
        coll, air, allv, frames = collect_episode(
            agent, env, wm, tokenizer, sae, layer, norm_mean, norm_std, device, args)
        if len(coll) < 3 or len(air) < 3:
            print(f"ep {ep}: only {len(coll)} collision / {len(air)} air frames in {frames} "
                  f"frames — not enough to analyze")
            episodes.append({"episode": ep, "n_collision": len(coll), "n_air": len(air),
                             "frames_scanned": frames, "table": []})
            continue
        rows = analyze(coll, air, allv, args.topN)
        episodes.append({
            "episode": ep, "n_collision": int(len(coll)), "n_air": int(len(air)),
            "frames_scanned": frames, "table": rows,
        })
        pos = sum(1 for r in rows if r["delta"] > 0.1)
        neg = sum(1 for r in rows if r["delta"] < -0.1)
        zero = len(rows) - pos - neg
        print(f"ep {ep}: {len(coll)} collision / {len(air)} air in {frames} frames | "
              f"top-{args.topN}: {pos} collision-corr (Δ>0.1), {zero} flat, {neg} anti-corr")
        print(f"   top-5 by Δ: " + ", ".join(
            f"#{r['feature_id']}(Δ{r['delta']:+.2f})" for r in rows[:5]))

    # Cross-episode stability on the shared top-N feature set.
    summary = {}
    tabs = [e["table"] for e in episodes if e["table"]]
    if len(tabs) >= 2:
        d0 = {r["feature_id"]: r["delta"] for r in tabs[0]}
        d1 = {r["feature_id"]: r["delta"] for r in tabs[1]}
        common = sorted(set(d0) & set(d1))
        if len(common) >= 3:
            sp = spearman([d0[f] for f in common], [d1[f] for f in common])
            top0 = {r["feature_id"] for r in sorted(tabs[0], key=lambda r: -r["delta"])[:10]}
            top1 = {r["feature_id"] for r in sorted(tabs[1], key=lambda r: -r["delta"])[:10]}
            summary = {
                "shared_features": len(common),
                "delta_spearman_across_episodes": round(sp, 3),
                "top10_delta_overlap": len(top0 & top1),
            }
            print(f"\ncross-episode: Δ Spearman {sp:+.3f} on {len(common)} shared features, "
                  f"top-10 Δ overlap {len(top0 & top1)}/10")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"params": vars(args) | {"checkpoint": str(args.checkpoint),
                   "sae": str(args.sae), "out": str(args.out)},
                   "episodes": episodes, "stability": summary}, f, indent=2, default=str)
    print("wrote", args.out)
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
