#!/usr/bin/env python
"""
Causal-importance ranking of SAE features in the IRIS world model.

For each feature, measure how much its intervention changes the model's *imagined
dynamics*: run an N-step imagined rollout under the feature pushed to +scale and to
-scale, and compare the obs-token stream to a baseline rollout (same seed, same frozen
action sequence). The score is the mean token divergence vs baseline, averaged over seeds
and the two signs. Ranking by this score = "which features, when steered, actually move
the rollout" — the most defensible notion of feature importance for a world model, and the
one with no published prior art I've found for this substrate.

Only token divergence is needed (not pixels), so this skips the tokenizer decode that the
live /rollout does — each feature is just WM forward passes. Cost per feature ≈
2 signs * seeds * n_steps * (1+K) passes; the baseline is computed once per seed.

This is an offline pipeline (full ~2K-feature run is hours). Built to sample and resume:
--features limits to the most-active features, --resume skips already-scored ones, the
store is written incrementally. Single-measurement discipline: use >=2 seeds, and compare
two independent runs before trusting "feature X is more important than Y".

Example
-------
    # Small sample for findings (most-active 24 features, 2 seeds), on MPS:
    python backend/scripts/causal_importance.py \
        --checkpoint /path/iris/checkpoints/Breakout.pt \
        --sae /path/iris/checkpoints/sae_L5.pt \
        --features 24 --seeds 2 --n-steps 10 --scale 5 --device mps

    # Full run, resumable:
    python backend/scripts/causal_importance.py \
        --checkpoint /path/iris/checkpoints/Breakout.pt \
        --sae /path/iris/checkpoints/sae_L5.pt --seeds 2 --device mps --resume
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_BACKEND = Path(__file__).resolve().parents[1]
_IRIS_ROOT = Path(os.environ.get("IRIS_ROOT", Path(__file__).resolve().parents[3] / "iris"))
_IRIS_SRC = Path(os.environ.get("IRIS_SRC", _IRIS_ROOT / "src"))
for _p in (str(_BACKEND), str(_IRIS_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference import _load_agent, _rollout, _decode_obs_tokens_to_pixels  # noqa: E402
from sae import load_artifact  # noqa: E402
from ranking_store import CausalRankingStore  # noqa: E402

MAG_FLOOR = 1.0  # match the live engine's _sae_mag_floor


def log(msg: str) -> None:
    print(msg, flush=True)


@torch.no_grad()
def gen_actions(agent, wm, tokenizer, init, device, n_steps, seed):
    """Frozen, self-consistent action sequence (policy on imagined frames). Mirrors
    InferenceEngine._rollout_actions; falls back to all-NOOP if it misbehaves."""
    from torch.distributions.categorical import Categorical
    try:
        torch.manual_seed(seed)
        K = init.shape[1]
        max_tokens = wm.config.max_tokens
        kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=max_tokens)
        wm(init, past_keys_values=kv)
        obs_tokens = init
        agent.actor_critic.reset(n=1)
        actions = []
        for _ in range(n_steps):
            rec = _decode_obs_tokens_to_pixels(tokenizer, obs_tokens)
            a = int(agent.act(rec, should_sample=False).item())
            actions.append(a)
            num_passes = 1 + K
            if kv.size + num_passes > max_tokens:
                kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=max_tokens)
                wm(obs_tokens, past_keys_values=kv)
            token = torch.tensor([[a]], dtype=torch.long, device=device)
            gen = []
            for k in range(num_passes):
                out = wm(token, past_keys_values=kv)
                if k < K:
                    token = Categorical(logits=out.logits_observations).sample()
                    gen.append(token)
            obs_tokens = torch.cat(gen, dim=1)
        return actions
    except Exception as exc:
        log(f"  action-gen fell back to NOOP ({exc})")
        return [0] * n_steps


@torch.no_grad()
def sample_states(agent, wm, tokenizer, sae, layer, norm_mean, norm_std, env,
                  n, warmup, stride, device):
    """Capture `n` DIVERSE seed states by time-sampling one playthrough at steps
    warmup, warmup+stride, ..., warmup+(n-1)*stride.

    The env reset + the (confident) policy are effectively deterministic, so re-seeding
    torch does NOT vary the seed state — verified empirically. Diversity therefore has to
    come from sampling different *points* along the trajectory. Each captured state is the
    obs_tokens + its SAE feature vector. Different `warmup` offsets give disjoint state sets
    (used for the cross-state-set robustness check)."""
    from einops import rearrange
    captured = {}
    h = wm.transformer.blocks[layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("resid", o.detach()))
    obs = env.reset(); agent.actor_critic.reset(n=1)
    capture_at = {warmup + i * stride for i in range(n)}
    total = warmup + (n - 1) * stride
    states = []
    for step in range(total + 1):
        obs_t = rearrange(torch.FloatTensor(obs).div(255), "n h w c -> n c h w").to(device)
        act = agent.act(obs_t, should_sample=True).cpu().numpy()
        if step in capture_at:
            obs_tokens = tokenizer.encode(obs_t, should_preprocess=True).tokens
            tokens = torch.cat([obs_tokens, torch.tensor([[int(act[0])]], device=device)], dim=1)
            captured.clear(); wm(tokens, past_keys_values=None)
            x = (captured["resid"][0, -1] - norm_mean) / norm_std
            states.append((obs_tokens, sae.encode(x.unsqueeze(0)).squeeze(0)))
        obs, _r, done, _ = env.step(act)
        if (bool(done[0]) if hasattr(done, "__len__") else bool(done)):
            obs = env.reset(); agent.actor_critic.reset(n=1)
    h.remove()
    return states


def per_step_divergence(baseline_tokens, iv_tokens):
    """Per-step number of obs tokens that differ baseline vs intervened (length n_steps)."""
    return [int((b != v).sum().item()) for b, v in zip(baseline_tokens, iv_tokens)]


def mean_token_divergence(baseline_tokens, iv_tokens):
    """Mean over steps of the number of obs tokens that differ baseline vs intervened."""
    return float(np.mean(per_step_divergence(baseline_tokens, iv_tokens)))


@torch.no_grad()
def run(args) -> None:
    device = torch.device(args.device)
    # Seed states come from a deterministic Atari reset + warmup, so without this every
    # run/process samples the SAME starting state. --state-seed varies the warmup RNG so we
    # can assess robustness to the choice of seed states (not just rollout sampling).
    torch.manual_seed(args.state_seed)
    log(f"Loading agent {args.checkpoint} + SAE {args.sae} (device={args.device})")
    agent, env, _cfg = _load_agent(_IRIS_ROOT, args.checkpoint, args.env_id, args.device)
    agent.eval()
    wm, tokenizer = agent.world_model, agent.tokenizer

    sae, meta = load_artifact(str(args.sae), device=args.device)
    layer = int(meta["layer"])
    env_id = meta.get("env_id") or args.env_id
    norm_mean = meta["norm"]["mean"].to(device)
    norm_std = meta["norm"]["std"].to(device).clamp_min(1e-6)
    log(f"SAE layer={layer} d_hidden={sae.d_hidden} env={env_id}")

    # N DIVERSE seed states (time-sampled along one playthrough), each with a frozen action
    # sequence and a baseline rollout (computed once, reused for every feature). Averaging
    # over diverse states is what makes the causal score generalize beyond one game frame.
    states = sample_states(agent, wm, tokenizer, sae, layer, norm_mean, norm_std,
                           env, args.seeds, args.warmup, args.state_stride, device)
    log(f"sampled {len(states)} diverse states (warmup {args.warmup}, stride {args.state_stride})")
    seeds_init, seeds_feats, seeds_actions, seeds_baseline = [], [], [], []
    for s, (init, feats) in enumerate(states):
        init = init.to(device)
        actions = gen_actions(agent, wm, tokenizer, init, device, args.n_steps, seed=s)
        baseline = _rollout(wm, tokenizer, init, actions, device, intervention=None, seed=s)
        seeds_init.append(init); seeds_feats.append(feats)
        seeds_actions.append(actions); seeds_baseline.append(baseline)

    # Seed-state activation per feature (the `act` field — used to *verify* that fixed-norm
    # removed the magnitude confound, NOT to scale the injection).
    mean_ref = torch.stack([f for f in seeds_feats]).mean(0)  # (d_hidden,)

    # Which features to score. --features-file (a JSON list of ids) takes precedence; else
    # the N most-active (legacy). Fixed-norm runs use the candidate file.
    if args.features_file:
        order = [int(x) for x in json.loads(Path(args.features_file).read_text())]
        log(f"scoring {len(order)} candidate features from {args.features_file}")
    else:
        order = torch.argsort(mean_ref, descending=True).tolist()
        if args.features:
            order = order[: args.features]

    tag = "fixed_norm" if args.fixed_norm else ""
    store = CausalRankingStore(str(args.out_dir), layer, tag=tag)
    data = store.load() if args.resume else {"layer": layer, "scores": {}}
    data.update({
        "layer": layer, "env_id": env_id, "n_steps": int(args.n_steps),
        "scale": float(args.scale), "seeds": int(args.seeds), "warmup": int(args.warmup),
        # injection mode: fixed_norm injects scale*unit_direction for every feature (causal
        # effect independent of activation); magnitude_relative scales by the feature's own
        # activation (the original — collapses the ranking toward magnitude).
        "injection": "fixed_norm" if args.fixed_norm else "magnitude_relative",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    data.setdefault("scores", {})

    processed = 0
    for fid in order:
        if args.resume and str(fid) in data["scores"]:
            continue
        pos_divs, neg_divs = [], []
        step_accum = np.zeros(args.n_steps); n_contrib = 0
        unit = sae.W_dec[fid] * norm_std  # raw-space feature direction (W_dec is unit-norm)
        for s in range(args.seeds):
            # fixed-norm: ref=1.0 for ALL features → injection = scale * unit_direction,
            # identical magnitude regardless of activation. (This equals what the original
            # run already did for its act==0 features.)
            ref = 1.0 if args.fixed_norm else max(float(seeds_feats[s][fid]), MAG_FLOOR)
            for scale, bucket in ((args.scale, pos_divs), (-args.scale, neg_divs)):
                direction = (scale * ref * unit).detach()
                iv = _rollout(wm, tokenizer, seeds_init[s], seeds_actions[s], device,
                              intervention=(layer, direction), seed=s)
                steps = per_step_divergence(seeds_baseline[s], iv)
                bucket.append(float(np.mean(steps)))
                step_accum += np.array(steps, dtype=float); n_contrib += 1
        pos = float(np.mean(pos_divs)); neg = float(np.mean(neg_divs))
        data["scores"][str(fid)] = {
            "id": int(fid),
            "score": round((pos + neg) / 2.0, 4),  # mean |token divergence| over signs+seeds
            "pos": round(pos, 4),
            "neg": round(neg, 4),
            "act": round(float(mean_ref[fid]), 4),
            # Per-step mean divergence (over seeds*signs) — for qualitative characterization.
            "trace": [round(x, 3) for x in (step_accum / max(n_contrib, 1)).tolist()],
        }
        processed += 1
        if processed % args.save_every == 0:
            store.save(data)
            log(f"  scored {processed}/{len(order)} (last #{fid}: {data['scores'][str(fid)]['score']})")

    store.save(data)
    log(f"Done: scored {processed} features ({data['injection']}, seeds={args.seeds}, "
        f"scale=±{args.scale}, n_steps={args.n_steps}) → {store.path}")
    try:
        env.close()
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None, help="Cache root (default: SAE dir)")
    p.add_argument("--features", type=int, default=0, help="Score only the N most-active features (0=all)")
    p.add_argument("--features-file", type=str, default=None,
                   help="JSON list of feature ids to score (candidate set; overrides --features)")
    p.add_argument("--fixed-norm", action="store_true",
                   help="Magnitude-INDEPENDENT injection: scale*unit_direction for every feature "
                        "(ref=1.0), so causal score doesn't covary with activation. Writes "
                        "causal_fixed_norm_L{layer}.json.")
    p.add_argument("--seeds", type=int, default=2, help="Number of DIVERSE seed states (time-sampled) averaged")
    p.add_argument("--state-stride", type=int, default=30, help="Steps between sampled seed states along the playthrough")
    p.add_argument("--n-steps", type=int, default=20, help="Imagined rollout length")
    p.add_argument("--scale", type=float, default=5.0, help="Magnitude-relative intervention scale (±)")
    p.add_argument("--warmup", type=int, default=30, help="Real-env steps to a representative seed state")
    p.add_argument("--state-seed", type=int, default=0, help="RNG seed for the warmup → varies WHICH seed states are sampled (robustness check)")
    p.add_argument("--resume", action="store_true", help="Skip features already scored in the cache")
    p.add_argument("--save-every", type=int, default=10, help="Save the cache every N scored features")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = args.sae.resolve().parent
    return args


if __name__ == "__main__":
    run(parse_args())
