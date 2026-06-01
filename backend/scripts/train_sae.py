#!/usr/bin/env python
"""
Offline sparse-autoencoder trainer for IRIS world-model residual activations.

Harvests the residual-stream activation at one or more transformer layers by
replaying saved episodes through the world model, then trains one ReLU/L1 SAE
per layer (a layer *sweep*, so the most interpretable layer can be chosen after
inspecting the metrics).  Each SAE is saved as ``sae_L{layer}.pt`` for the
inference engine to load.

This is an offline tool — it is intentionally outside the served FastAPI app and
imports IRIS the same way the engine does (via sys.path).

Example
-------
    python backend/scripts/train_sae.py \
        --episodes-dir /path/to/iris/outputs/.../media/episodes/test \
        --env-id BreakoutNoFrameskip-v4 \
        --checkpoint /path/to/iris/checkpoints/Breakout.pt \
        --layers 3 5 6 7 --expansion 8 --l1 2e-3 --epochs 2 \
        --out-dir /path/to/iris/checkpoints

Harvesting uses the same WM forward the live engine uses (16 obs tokens + 1
action token, no KV cache).  By default it stores the *last* token's residual
(the action position, index -1) — the position whose output predicts the next
observation and the target of Phase-3 interventions.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

# ---------------------------------------------------------------------------
# Path setup: backend/ (for sae, inference) and IRIS src/ (for episode, models)
# ---------------------------------------------------------------------------

_BACKEND = Path(__file__).resolve().parents[1]
_IRIS_ROOT = Path(os.environ.get("IRIS_ROOT", Path(__file__).resolve().parents[3] / "iris"))
_IRIS_SRC = Path(os.environ.get("IRIS_SRC", _IRIS_ROOT / "src"))
for _p in (str(_BACKEND), str(_IRIS_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sae import SparseAutoencoder, sae_loss, save_artifact  # noqa: E402
from inference import _load_agent  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------

def load_episodes(episodes_dirs: List[Path], max_episodes: int) -> List:
    """Glob *.pt under each dir (recursively) and load Episode objects.

    Files that are not episodes (e.g. checkpoints) are skipped.  We glob rather
    than use EpisodesDataset.load_disk_checkpoint because saved episodes are named
    ``episode_<id>_epoch_<n>.pt``, which that loader's int(stem) parsing rejects.
    """
    from episode import Episode

    files: List[Path] = []
    for d in episodes_dirs:
        files.extend(sorted(d.rglob("*.pt")))
    episodes = []
    for f in files:
        try:
            ep = Episode(**torch.load(f, map_location="cpu"))
        except Exception:
            continue  # not an episode file
        episodes.append(ep)
        if max_episodes and len(episodes) >= max_episodes:
            break
    return episodes


# ---------------------------------------------------------------------------
# Activation harvesting
# ---------------------------------------------------------------------------

def harvest_activations(
    agent,
    episodes: List,
    layers: List[int],
    token_policy: str,
    device: torch.device,
    harvest_batch: int,
    max_frames: int,
) -> Dict[int, torch.Tensor]:
    """Replay episodes through the WM and collect residual activations per layer.

    Registers a forward hook on each swept ``transformer.blocks[L]`` and runs the
    same 17-token no-cache forward the live engine uses.  Returns ``{layer: (M, 256)}``
    CPU tensors, where M is the number of harvested vectors (one per frame for
    token_policy="last", or 17 per frame for "all").
    """
    wm = agent.world_model
    tokenizer = agent.tokenizer

    captured: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            captured[layer_idx] = out.detach()
        return hook

    handles = [wm.transformer.blocks[L].register_forward_hook(make_hook(L)) for L in layers]
    store: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}
    total_frames = 0
    t0 = time.perf_counter()

    try:
        done = False
        for ei, ep in enumerate(episodes):
            if done:
                break
            obs_all = ep.observations          # (T, 3, 64, 64) uint8
            actions_all = ep.actions           # (T,)
            T = obs_all.shape[0]
            for i in range(0, T, harvest_batch):
                obs = obs_all[i:i + harvest_batch].float().div(255).to(device)   # (N,3,64,64)
                acts = actions_all[i:i + harvest_batch].to(device).long().view(-1, 1)  # (N,1)
                with torch.no_grad():
                    tokens_obs = tokenizer.encode(obs, should_preprocess=True).tokens   # (N,16)
                    tokens = torch.cat([tokens_obs, acts], dim=1)                        # (N,17)
                    captured.clear()
                    wm(tokens, past_keys_values=None)
                for L in layers:
                    resid = captured[L]         # (N, 17, 256)
                    if token_policy == "last":
                        vecs = resid[:, -1, :]                       # (N, 256)
                    else:  # "all"
                        vecs = resid.reshape(-1, resid.shape[-1])    # (N*17, 256)
                    store[L].append(vecs.cpu())
                total_frames += obs.shape[0]
                if max_frames and total_frames >= max_frames:
                    done = True
                    break
            if (ei + 1) % 25 == 0:
                log(f"  harvested {total_frames} frames from {ei + 1} episodes "
                    f"({time.perf_counter() - t0:.0f}s)")
    finally:
        for h in handles:
            h.remove()

    acts_by_layer = {L: torch.cat(store[L], dim=0) for L in layers}
    log(f"Harvest complete: {total_frames} frames, "
        f"{acts_by_layer[layers[0]].shape[0]} vectors/layer "
        f"({time.perf_counter() - t0:.0f}s)")
    return acts_by_layer


# ---------------------------------------------------------------------------
# Per-layer SAE training
# ---------------------------------------------------------------------------

def train_one_layer(acts: torch.Tensor, layer: int, args, device: torch.device):
    """Train a single SAE on (M, d_in) activations. Returns (sae, mean, std, steps, metrics)."""
    mean = acts.mean(0)
    std = acts.std(0).clamp_min(1e-6)
    acts_norm = (acts - mean) / std                         # normalise on CPU
    M, d_in = acts_norm.shape
    d_hidden = d_in * args.expansion

    sae = SparseAutoencoder(d_in, d_hidden).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)

    steps = 0
    last = {"recon": torch.tensor(0.0), "l0": torch.tensor(0.0)}
    for epoch in range(args.epochs):
        perm = torch.randperm(M)
        for i in range(0, M, args.batch):
            idx = perm[i:i + args.batch]
            x = acts_norm[idx].to(device)
            loss, m = sae_loss(sae, x, args.l1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            steps += 1
            last = m
            if steps % args.log_every == 0:
                log(f"  L{layer} epoch {epoch} step {steps}: "
                    f"recon={float(m['recon']):.3f} l1={float(m['l1']):.2f} "
                    f"l0={float(m['l0']):.1f}")

    metrics = {
        "recon": round(float(last["recon"]), 4),
        "l0": round(float(last["l0"]), 2),
        "frames": int(M),
    }
    return sae, mean, std, steps, metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes-dir", type=Path, required=True, nargs="+",
                   help="One or more directories of saved episode .pt files (searched "
                        "recursively). Use real env episodes (e.g. media/episodes/test), "
                        "not imagination. All dirs must be the SAME game as --env-id.")
    p.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="IRIS checkpoint .pt (default: <iris_root>/checkpoints/last.pt)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write sae_L{layer}.pt (default: <iris_root>/checkpoints)")
    p.add_argument("--layers", type=int, nargs="+", default=[3, 5, 6, 7],
                   help="Transformer layers to sweep (residual after each block)")
    p.add_argument("--expansion", type=int, default=8, help="Dictionary = expansion * d_in")
    p.add_argument("--l1", type=float, default=2e-3, help="L1 sparsity coefficient")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--token-policy", choices=["last", "all"], default="last",
                   help="'last' = action-token residual (matches the live panel/intervention)")
    p.add_argument("--harvest-batch", type=int, default=128, help="Frames per WM forward during harvest")
    p.add_argument("--max-episodes", type=int, default=0, help="Cap episodes loaded (0 = all)")
    p.add_argument("--max-frames", type=int, default=0, help="Cap frames harvested (0 = all)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = args.checkpoint or (_IRIS_ROOT / "checkpoints" / "last.pt")
    out_dir = args.out_dir or (_IRIS_ROOT / "checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading agent: {checkpoint} (env={args.env_id}, device={args.device})")
    agent, _env, _cfg = _load_agent(_IRIS_ROOT, checkpoint, args.env_id, args.device)

    num_layers = len(agent.world_model.transformer.blocks)
    bad = [L for L in args.layers if not (0 <= L < num_layers)]
    if bad:
        raise SystemExit(f"Layers {bad} out of range (model has {num_layers} layers 0..{num_layers - 1})")

    log(f"Loading episodes from {args.episodes_dir}")
    episodes = load_episodes(args.episodes_dir, args.max_episodes)
    if not episodes:
        raise SystemExit(f"No episode .pt files found under {args.episodes_dir}")
    log(f"Loaded {len(episodes)} episodes")

    acts_by_layer = harvest_activations(
        agent, episodes, args.layers, args.token_policy, device,
        args.harvest_batch, args.max_frames,
    )

    summary = []
    for L in args.layers:
        log(f"Training SAE for layer {L} "
            f"({acts_by_layer[L].shape[0]} vectors, d_hidden={args.expansion * acts_by_layer[L].shape[1]})")
        sae, mean, std, steps, metrics = train_one_layer(acts_by_layer[L], L, args, device)
        out_path = out_dir / f"sae_L{L}.pt"
        save_artifact(
            out_path, sae, layer=L, norm_mean=mean, norm_std=std, env_id=args.env_id,
            l1_coeff=args.l1, trained_steps=steps, metrics=metrics,
            token_policy=args.token_policy, expansion_factor=args.expansion,
        )
        log(f"  saved {out_path}  (recon={metrics['recon']} l0={metrics['l0']})")
        summary.append((L, metrics["recon"], metrics["l0"]))

    log("\n=== Sweep summary (pick the layer with low recon AND interpretable l0 ~10-50) ===")
    log(f"{'layer':>6} {'recon':>10} {'l0':>8}")
    for L, recon, l0 in summary:
        log(f"{L:>6} {recon:>10.4f} {l0:>8.2f}")


if __name__ == "__main__":
    main()
