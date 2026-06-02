#!/usr/bin/env python
"""
Select the candidate feature set for the fixed-norm causal re-run.

One local harvest computes all three rankings whose union we want to score causally
(scoring all 2048 is wasteful — most are inactive). For each frame we compute the full
2048-feature vector and a collision/air label (human-frame red detector, as in Test 3):
  - magnitude:       top-50 by mean activation over all frames
  - temporal stability: top-50 by ascending coefficient of variation among features
                        firing in >= min_firing of frames
  - collision Δ:     top-50 by (mean activation at collisions − at air frames)
plus the 6 case-study features. Dedup → ~100-150 candidates.

Run:
    python backend/scripts/diagnostics/select_causal_candidates.py \
        --checkpoint .../Breakout.pt --sae .../sae_L5.pt --device mps --frames 1800
Writes results/causal_candidates.json (candidate ids + each ranking's top-50, for the
cross-ranking comparison later).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_BACKEND = Path(__file__).resolve().parents[2]
_IRIS_ROOT = Path(os.environ.get("IRIS_ROOT", Path(__file__).resolve().parents[4] / "iris"))
_IRIS_SRC = Path(os.environ.get("IRIS_SRC", _IRIS_ROOT / "src"))
for _p in (str(_BACKEND), str(_IRIS_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference import _load_agent, _get_raw_frame  # noqa: E402
from sae import load_artifact  # noqa: E402

CASE_STUDY = [1364, 120, 27, 1773, 1199, 316]
_WALL_L, _WALL_R = 8, 152
_BALL_ROWS = (93, 184)
_PADDLE_ROWS = (185, 200)


def classify(raw, collision_y=176, air_y=150, x_overlap=12):
    """'collision' / 'air' / None from the 210x160 human frame (ball+paddle are red)."""
    r, g, b = raw[..., 0].astype(int), raw[..., 1].astype(int), raw[..., 2].astype(int)
    red = (r > 160) & (g < 110) & (b < 110)
    red[:, :_WALL_L] = False
    red[:, _WALL_R:] = False
    ball = red.copy(); ball[: _BALL_ROWS[0]] = False; ball[_BALL_ROWS[1]:] = False
    pad = red.copy(); pad[: _PADDLE_ROWS[0]] = False; pad[_PADDLE_ROWS[1]:] = False
    by, bx = np.nonzero(ball); py, px = np.nonzero(pad)
    if len(bx) < 1:
        return None
    bx_m, by_m = float(bx.mean()), float(by.mean())
    if by_m >= collision_y and len(px) >= 1 and abs(bx_m - float(px.mean())) <= x_overlap:
        return "collision"
    if by_m <= air_y:
        return "air"
    return None


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
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--frames", type=int, default=1800)
    p.add_argument("--topN", type=int, default=50)
    p.add_argument("--min-firing", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=Path, default=_BACKEND.parent / "results" / "causal_candidates.json")
    args = p.parse_args()

    from einops import rearrange
    agent, env, _cfg = _load_agent(_IRIS_ROOT, args.checkpoint, args.env_id, args.device)
    agent.eval()
    wm, tokenizer = agent.world_model, agent.tokenizer
    device = torch.device(args.device)
    sae, meta = load_artifact(str(args.sae), device=args.device)
    layer = int(meta["layer"])
    norm_mean = meta["norm"]["mean"].to(device)
    norm_std = meta["norm"]["std"].to(device).clamp_min(1e-6)

    obs = env.reset(); agent.actor_critic.reset(n=1)
    vecs, labels = [], []
    for i in range(args.frames):
        obs_t = rearrange(torch.FloatTensor(obs).div(255), "n h w c -> n c h w").to(device)
        act = agent.act(obs_t, should_sample=True).cpu().numpy()
        tok = tokenizer.encode(obs_t, should_preprocess=True).tokens
        vecs.append(feature_vector(wm, sae, layer, norm_mean, norm_std, tok, int(act[0]), device))
        labels.append(classify(_get_raw_frame(env)))
        obs, _r, done, _ = env.step(act)
        if (bool(done[0]) if hasattr(done, "__len__") else bool(done)):
            obs = env.reset(); agent.actor_critic.reset(n=1)
        if (i + 1) % 500 == 0:
            print(f"  harvested {i + 1}/{args.frames}", flush=True)

    fv = np.stack(vecs)  # (F, d_hidden)
    labels = np.array(labels)
    n_coll = int((labels == "collision").sum()); n_air = int((labels == "air").sum())
    print(f"frames {len(fv)} | collision {n_coll} | air {n_air}")

    # --- three rankings ---
    magnitude = fv.mean(0)
    top_mag = np.argsort(-magnitude)[: args.topN]

    firing = (fv > 0).mean(0)
    cv = np.sqrt(fv.var(0)) / (fv.mean(0) + 1e-6)
    gated = np.where(firing >= args.min_firing)[0]
    top_stab = gated[np.argsort(cv[gated])][: args.topN]

    delta = fv[labels == "collision"].mean(0) - fv[labels == "air"].mean(0)
    top_delta = np.argsort(-delta)[: args.topN]

    rankings = {
        "magnitude": [int(x) for x in top_mag],
        "stability": [int(x) for x in top_stab],
        "collision_delta": [int(x) for x in top_delta],
    }
    candidates = sorted(set(rankings["magnitude"]) | set(rankings["stability"])
                        | set(rankings["collision_delta"]) | set(CASE_STUDY))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n_frames": len(fv), "n_collision": n_coll, "n_air": n_air,
                   "case_study": CASE_STUDY, "rankings": rankings,
                   "candidates": candidates}, f, indent=2)
    # also write a bare id-list the scorer's --features-file consumes
    (args.out.parent / "causal_candidate_ids.json").write_text(json.dumps(candidates))
    print(f"candidates: {len(candidates)} (mag50 ∪ stab50 ∪ Δ50 ∪ {len(CASE_STUDY)} case-study)")
    print("wrote", args.out, "and causal_candidate_ids.json")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
