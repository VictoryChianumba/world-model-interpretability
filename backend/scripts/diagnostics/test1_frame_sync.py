#!/usr/bin/env python
"""
Test 1 — frame/activation synchronization (empirical lag measurement).

The viewer shows a game frame next to the SAE activations "for that frame". The SAE
activations are computed from the obs the agent acted on (frame t). The question: is the
DISPLAYED frame also frame t, or is it offset?

This replicates the engine's per-step sequence (act → capture display frame → env.step)
and measures, over many steps and >=2 episodes, the lag between the displayed frame and
the activation obs, for two captures:
  - POST-step  (what the code did originally): _get_raw_frame AFTER env.step
  - PRE-step   (the fix):                       _get_raw_frame BEFORE env.step
The displayed human frame (210x160) and the model obs (64x64) are the same scene at
different resolutions, so we downsample both to 64x64 grayscale and, for each candidate
lag L, compute the mean abs difference between display[t] and obs[t+L]. The argmin lag is
the offset: 0 = aligned, +1 = display is one step ahead of its activations.

Run:
    python backend/scripts/diagnostics/test1_frame_sync.py \
        --checkpoint /path/iris/checkpoints/Breakout.pt --device cpu --episodes 2 --steps 150
Writes results/test1_frame_sync.json.
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

LAGS = [-2, -1, 0, 1, 2]


def gray64(img):
    """Any HxWx3 uint8 array → 64x64 grayscale float in [0,1]."""
    im = Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("L").resize((64, 64))
    return np.asarray(im, dtype=np.float32) / 255.0


def best_lag(display, obs):
    """argmin over candidate lags L of mean|display[t] - obs[t+L]|; returns (lag, table)."""
    n = len(obs)
    table = {}
    for L in LAGS:
        diffs = []
        for t in range(n):
            j = t + L
            if 0 <= j < n:
                diffs.append(np.abs(display[t] - obs[j]).mean())
        table[L] = round(float(np.mean(diffs)), 5) if diffs else None
    valid = {L: v for L, v in table.items() if v is not None}
    return min(valid, key=valid.get), table


@torch.no_grad()
def run_episode(agent, env, device, steps):
    """Return per-step (obs_gray, pre_frame_gray, post_frame_gray) sequences."""
    from einops import rearrange
    obs = env.reset()
    agent.actor_critic.reset(n=1)
    obs_seq, pre_seq, post_seq = [], [], []
    for _ in range(steps):
        # obs the activations would be computed from (frame t):
        obs_seq.append(gray64(obs[0]))
        obs_t = rearrange(torch.FloatTensor(obs).div(255), "n h w c -> n c h w").to(device)
        act = agent.act(obs_t, should_sample=True).cpu().numpy()
        # PRE-step display frame (the fix): env's current original_obs == frame t
        pre_seq.append(gray64(_get_raw_frame(env)))
        obs, _r, done, _ = env.step(act)
        # POST-step display frame (the original bug): now advanced to frame t+1
        post_seq.append(gray64(_get_raw_frame(env)))
        if (bool(done[0]) if hasattr(done, "__len__") else bool(done)):
            obs = env.reset()
            agent.actor_critic.reset(n=1)
    return np.array(obs_seq), np.array(pre_seq), np.array(post_seq)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=Path, default=_BACKEND.parent / "results" / "test1_frame_sync.json")
    args = p.parse_args()

    agent, env, _cfg = _load_agent(_IRIS_ROOT, args.checkpoint, args.env_id, args.device)
    agent.eval()
    device = torch.device(args.device)

    episodes = []
    for ep in range(args.episodes):
        obs_seq, pre_seq, post_seq = run_episode(agent, env, device, args.steps)
        pre_lag, pre_tab = best_lag(pre_seq, obs_seq)
        post_lag, post_tab = best_lag(post_seq, obs_seq)
        episodes.append({
            "episode": ep, "steps": len(obs_seq),
            "pre_step_capture": {"best_lag": pre_lag, "mean_abs_diff_by_lag": pre_tab},
            "post_step_capture": {"best_lag": post_lag, "mean_abs_diff_by_lag": post_tab},
        })
        print(f"ep {ep}: PRE-step (fix) best lag = {pre_lag:+d}  | "
              f"POST-step (old) best lag = {post_lag:+d}")

    pre_lags = [e["pre_step_capture"]["best_lag"] for e in episodes]
    post_lags = [e["post_step_capture"]["best_lag"] for e in episodes]
    summary = {
        "pre_step_lags": pre_lags, "post_step_lags": post_lags,
        "verdict": (
            "POST-step capture is offset by +1 (display leads activations by one frame); "
            "PRE-step capture is aligned (lag 0)."
            if all(l == 1 for l in post_lags) and all(l == 0 for l in pre_lags)
            else "see per-episode lags"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"episodes": episodes, "summary": summary}, f, indent=2)
    print("summary:", summary["verdict"])
    print("wrote", args.out)
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
