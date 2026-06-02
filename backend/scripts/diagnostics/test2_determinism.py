#!/usr/bin/env python
"""
Test 2 — activation determinism.

Given the same input frame, does the SAE produce the same activation vector on repeated
forward passes? This isolates extraction stochasticity (dropout left on at inference,
nondeterministic ops, hidden state) from the env/policy sampling that drives the live loop.

Method: replicate the engine's _compute_sae_features exactly (encode obs → WM forward with
a residual hook on the SAE layer → normalise the action-token vector → SAE.encode), on a
FIXED (obs_tokens, action) input, repeated N times. The action is fixed too, since the SAE
reads the action-token position. Diff every repeat against the first across all ~2048
features: max abs, mean abs, # features with |diff| > 1e-6. Repeated on several frames.

Run:
    python backend/scripts/diagnostics/test2_determinism.py \
        --checkpoint /path/iris/checkpoints/Breakout.pt --sae /path/.../sae_L5.pt \
        --device cpu --frames 3 --repeats 10
Writes results/test2_determinism.json.
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

from inference import _load_agent  # noqa: E402
from sae import load_artifact  # noqa: E402


@torch.no_grad()
def extract_once(wm, sae, layer, norm_mean, norm_std, obs_tokens, action, device):
    """One full extraction: WM forward on [obs_tokens, action], SAE-encode the action-token
    residual. Returns the (d_hidden,) feature vector as float64 numpy (exact diffing)."""
    captured = {}
    h = wm.transformer.blocks[layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("resid", o.detach())
    )
    try:
        tokens = torch.cat(
            [obs_tokens, torch.tensor([[int(action)]], dtype=torch.long, device=device)], dim=1
        )
        wm(tokens, past_keys_values=None)
        x = (captured["resid"][0, -1] - norm_mean) / norm_std
        feats = sae.encode(x.unsqueeze(0)).squeeze(0)
    finally:
        h.remove()
    return feats.detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--frames", type=int, default=3, help="Distinct game states to test")
    p.add_argument("--repeats", type=int, default=10, help="Repeated extractions per state")
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=Path, default=_BACKEND.parent / "results" / "test2_determinism.json")
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

    train_flags = {
        "world_model.training": bool(wm.training),
        "tokenizer.training": bool(tokenizer.training),
        "sae.training": bool(sae.training),
    }

    # Gather a few distinct states.
    obs = env.reset(); agent.actor_critic.reset(n=1)
    states = []
    for _ in range(args.frames):
        for _ in range(args.warmup):
            obs_t = rearrange(torch.FloatTensor(obs).div(255), "n h w c -> n c h w").to(device)
            act = agent.act(obs_t, should_sample=True).cpu().numpy()
            obs, _r, done, _ = env.step(act)
            if (bool(done[0]) if hasattr(done, "__len__") else bool(done)):
                obs = env.reset(); agent.actor_critic.reset(n=1)
        obs_t = rearrange(torch.FloatTensor(obs).div(255), "n h w c -> n c h w").to(device)
        a = int(agent.act(obs_t, should_sample=False).item())  # fixed (argmax) action
        obs_tokens = tokenizer.encode(obs_t, should_preprocess=True).tokens
        states.append((obs_tokens, a))

    frames_report = []
    for fi, (obs_tokens, a) in enumerate(states):
        ref = extract_once(wm, sae, layer, norm_mean, norm_std, obs_tokens, a, device)
        max_abs = mean_abs = 0.0
        n_diff = 0
        for _ in range(args.repeats - 1):
            v = extract_once(wm, sae, layer, norm_mean, norm_std, obs_tokens, a, device)
            d = np.abs(v - ref)
            max_abs = max(max_abs, float(d.max()))
            mean_abs = max(mean_abs, float(d.mean()))
            n_diff = max(n_diff, int((d > 1e-6).sum()))
        frames_report.append({
            "frame": fi, "n_features": int(ref.size), "repeats": args.repeats,
            "max_abs_diff": max_abs, "mean_abs_diff": mean_abs,
            "n_features_diff_gt_1e-6": n_diff,
            "n_active_in_ref": int((ref > 0).sum()),
        })
        print(f"frame {fi}: max|Δ|={max_abs:.3e} mean|Δ|={mean_abs:.3e} "
              f"#(|Δ|>1e-6)={n_diff}/{ref.size}  (active={int((ref>0).sum())})")

    deterministic = all(f["max_abs_diff"] == 0.0 for f in frames_report)
    report = {
        "train_flags": train_flags,
        "deterministic_bit_exact": deterministic,
        "frames": frames_report,
        "verdict": (
            "bit-exact deterministic across all repeats and frames"
            if deterministic else
            "nonzero diff detected — investigate (see per-frame max_abs_diff)"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("train flags:", train_flags)
    print("verdict:", report["verdict"])
    print("wrote", args.out)
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
