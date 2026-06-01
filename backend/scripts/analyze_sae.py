#!/usr/bin/env python
"""
Offline analysis of trained SAE(s): firing structure, co-firing, cross-layer compare.

Reuses train_sae.py's harvest path to collect layer-L residuals over a frame sample,
encodes them through each SAE, and reports (markdown + JSON):
  - per-feature firing frequency, dead-feature count, mean L0
  - top co-firing feature pairs: P(i active | j active) among frequently-firing features
  - if multiple sae_L*.pt are given, a cross-layer comparison table

Report only — no UI. Output written next to the SAEs or to --out.

Example
-------
    python backend/scripts/analyze_sae.py \
        --checkpoint /path/to/iris/checkpoints/Breakout.pt \
        --env-id BreakoutNoFrameskip-v4 \
        --episodes-dir /path/to/iris/episodes_breakout \
        --sae /path/to/iris/checkpoints/sae_L5.pt /path/to/.../sae_L6.pt \
        --frames 4000 --device mps --out /path/to/sae_runs
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_BACKEND = Path(__file__).resolve().parents[1]
_IRIS_ROOT = Path(os.environ.get("IRIS_ROOT", Path(__file__).resolve().parents[3] / "iris"))
_IRIS_SRC = Path(os.environ.get("IRIS_SRC", _IRIS_ROOT / "src"))
for _p in (str(_BACKEND), str(_IRIS_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference import _load_agent  # noqa: E402
from sae import load_artifact  # noqa: E402
from scripts.train_sae import harvest_activations, load_episodes  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


@torch.no_grad()
def analyze_one(sae, meta, acts: torch.Tensor, device, top_cofire: int):
    """Encode activations through one SAE; return a stats dict."""
    norm_mean = meta["norm"]["mean"].to(device)
    norm_std = meta["norm"]["std"].to(device).clamp_min(1e-6)
    d_hidden = sae.d_hidden

    # Encode in batches → binary firing matrix (M, d_hidden) and activation sums.
    fired_counts = torch.zeros(d_hidden)
    l0_sum = 0.0
    M = acts.shape[0]
    bs = 8192
    fire_cols = []  # store boolean firing per batch for co-firing (kept on CPU)
    for i in range(0, M, bs):
        x = ((acts[i:i + bs].to(device) - norm_mean) / norm_std)
        f = sae.encode(x)                          # (b, d_hidden)
        fired = (f > 0)
        fired_counts += fired.float().sum(0).cpu()
        l0_sum += float(fired.float().sum(1).sum().cpu())
        fire_cols.append(fired.cpu())
    fired_all = torch.cat(fire_cols, dim=0)        # (M, d_hidden) bool

    freq = (fired_counts / M)                       # firing frequency per feature
    dead = int((fired_counts == 0).sum())
    mean_l0 = l0_sum / M

    # Co-firing among the most-frequently-firing features (cap for an O(k^2) matrix).
    k = min(top_cofire, d_hidden)
    top_ids = torch.topk(fired_counts, k).indices
    sub = fired_all[:, top_ids].float()            # (M, k)
    counts = sub.sum(0)                            # (k,)
    co = sub.t() @ sub                             # (k, k) joint counts
    # P(i active | j active) = co[i,j] / counts[j]
    cond = co / counts.clamp_min(1.0).unsqueeze(0)
    pairs = []
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            p = float(cond[a, b])
            if p >= 0.5 and float(counts[b]) >= 0.02 * M:  # meaningful + not rare
                pairs.append((int(top_ids[a]), int(top_ids[b]), p))
    pairs.sort(key=lambda t: t[2], reverse=True)

    return {
        "layer": int(meta["layer"]),
        "d_hidden": int(d_hidden),
        "n_vectors": int(M),
        "recon_metric": meta.get("metrics", {}).get("recon"),
        "l0_trained": meta.get("metrics", {}).get("l0"),
        "mean_l0_measured": round(mean_l0, 2),
        "dead_features": dead,
        "active_features": int(d_hidden - dead),
        "top_firing": [(int(i), round(float(freq[i]), 3))
                       for i in torch.topk(freq, 10).indices.tolist()],
        "top_cofiring_pairs": [
            {"a": a, "b": b, "p_a_given_b": round(p, 3)} for a, b, p in pairs[:15]
        ],
    }


def write_report(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_sae.json").write_text(json.dumps(results, indent=2))

    lines = ["# SAE analysis", ""]
    lines.append("## Cross-layer comparison")
    lines.append("")
    lines.append("| layer | d_hidden | dead | active | mean L0 (measured) | recon (trained) | l0 (trained) |")
    lines.append("|------:|---------:|-----:|-------:|-------------------:|----------------:|-------------:|")
    for r in results:
        lines.append(
            f"| {r['layer']} | {r['d_hidden']} | {r['dead_features']} | {r['active_features']} | "
            f"{r['mean_l0_measured']} | {r['recon_metric']} | {r['l0_trained']} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"## Layer {r['layer']}")
        lines.append(f"- vectors analyzed: {r['n_vectors']}")
        lines.append(f"- dead features: {r['dead_features']} / {r['d_hidden']}")
        lines.append(f"- top firing features (id, freq): {r['top_firing']}")
        lines.append("- top co-firing pairs P(a|b):")
        if r["top_cofiring_pairs"]:
            for pr in r["top_cofiring_pairs"]:
                lines.append(f"    - #{pr['a']} | #{pr['b']}  →  {pr['p_a_given_b']}")
        else:
            lines.append("    - (none above threshold)")
        lines.append("")
    (out_dir / "analysis_sae.md").write_text("\n".join(lines))
    log(f"Wrote {out_dir/'analysis_sae.md'} and analysis_sae.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--episodes-dir", type=Path, nargs="+", required=True)
    p.add_argument("--sae", type=Path, nargs="+", required=True, help="One or more sae_L*.pt")
    p.add_argument("--frames", type=int, default=4000, help="Frame sample size for harvest")
    p.add_argument("--harvest-batch", type=int, default=128)
    p.add_argument("--top-cofire", type=int, default=64, help="Features in the co-firing matrix")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=Path, default=None, help="Report dir (default: first SAE's dir)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    log(f"Loading agent {args.checkpoint} (device={args.device})")
    agent, _env, _cfg = _load_agent(_IRIS_ROOT, args.checkpoint, args.env_id, args.device)
    agent.eval()

    saes = []
    for sp in args.sae:
        sae, meta = load_artifact(str(sp), device=args.device)
        saes.append((sae, meta))
    layers = sorted({int(m["layer"]) for _s, m in saes})

    episodes = load_episodes(args.episodes_dir, max_episodes=0)
    log(f"Loaded {len(episodes)} episodes; harvesting layers {layers} (token_policy=last)")
    acts_by_layer = harvest_activations(
        agent, episodes, layers, "last", device, args.harvest_batch, args.frames
    )

    results = []
    for sae, meta in saes:
        L = int(meta["layer"])
        log(f"Analyzing SAE layer {L} …")
        results.append(analyze_one(sae, meta, acts_by_layer[L], device, args.top_cofire))

    out_dir = args.out or args.sae[0].parent
    write_report(results, out_dir)


if __name__ == "__main__":
    main()
