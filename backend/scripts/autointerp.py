#!/usr/bin/env python
"""
Autointerp: label SAE features with a vision LLM from their top-activating frames.

For each SAE feature this finds the frames where it fires hardest, renders them into a
4x4 grid of 84x84 game frames, and asks a vision-capable Claude model what they have in
common (<=10 words). The label — or null when there's no clear pattern — is cached
alongside the feature's firing stats and example frames (see ``autointerp_store.py``).

This is a one-time, money-spending pipeline (~$1-5 for ~2K features). It is built to be
re-runnable cheaply: ``--resume`` skips already-labeled features, the index is saved
incrementally, and ``--no-llm`` builds every grid/stat/example WITHOUT calling the API
(free) — useful to sanity-check the grids before spending, and to spot-check ~20 labels.

Set ANTHROPIC_API_KEY in the environment before a real (LLM) run.

Examples
--------
    # Dry run: build grids + stats for the 20 most-firing features, no API calls.
    python backend/scripts/autointerp.py \
        --checkpoint /path/iris/checkpoints/Breakout.pt \
        --sae /path/iris/checkpoints/sae_L5.pt \
        --frames 2000 --limit 20 --no-llm --device mps

    # Full labeling run (after a key is set), resumable.
    python backend/scripts/autointerp.py \
        --checkpoint /path/iris/checkpoints/Breakout.pt \
        --sae /path/iris/checkpoints/sae_L5.pt \
        --frames 3000 --device mps --resume
"""

import argparse
import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_BACKEND = Path(__file__).resolve().parents[1]
_IRIS_ROOT = Path(os.environ.get("IRIS_ROOT", Path(__file__).resolve().parents[3] / "iris"))
_IRIS_SRC = Path(os.environ.get("IRIS_SRC", _IRIS_ROOT / "src"))
for _p in (str(_BACKEND), str(_IRIS_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference import _load_agent  # noqa: E402
from sae import load_artifact  # noqa: E402
from autointerp_store import AutoInterpStore, png_to_b64  # noqa: E402

# The Breakout-specific prompt. UNCLEAR is the model's escape hatch so we never force a
# label onto a polysemantic/uncharacterizable feature (a false label is worse than none).
PROMPT = (
    "These are the 16 Atari Breakout game frames where one feature of a sparse "
    "autoencoder fired most strongly, arranged in a 4x4 grid (read left-to-right, "
    "top-to-bottom in decreasing activation order). What visual pattern do these frames "
    "share? Describe the common pattern in 10 words or fewer, as a short noun phrase. "
    "If there is no clear shared pattern, reply with exactly: UNCLEAR"
)
_REFUSAL_MARKERS = ("unclear", "no clear", "cannot", "can't", "i'm unable", "no common")


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Frame collection + per-feature stats
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_frames(args, agent, env, sae, layer, norm_mean, norm_std, device):
    """Run the policy in the env; return (obs_uint8 [F,H,W,3], feats [F,d_hidden] float32).

    Mirrors label_features.py: encode each real frame, run the WM forward once, read the
    last-token (action position) residual, normalise, and SAE-encode it to the feature
    activations. The raw RGB observation is what we later show the LLM.
    """
    from einops import rearrange

    captured = {}
    h = wm_block_hook(agent, layer, captured)

    obs = env.reset()
    agent.actor_critic.reset(n=1)
    obs_list, feat_list = [], []
    for i in range(args.frames):
        obs_t = rearrange(torch.FloatTensor(obs).div(255), "n h w c -> n c h w").to(device)
        act = agent.act(obs_t, should_sample=True).cpu().numpy()
        enc = agent.tokenizer.encode(obs_t, should_preprocess=True)
        act_tensor = torch.tensor([[int(act[0])]], dtype=torch.long, device=device)
        tokens = torch.cat([enc.tokens, act_tensor], dim=1)  # (1, 17)
        captured.clear()
        agent.world_model(tokens, past_keys_values=None)
        resid = captured["resid"]                            # (1, 17, E)
        x = (resid[0, -1] - norm_mean) / norm_std
        feats = sae.encode(x.unsqueeze(0)).squeeze(0)        # (d_hidden,)
        obs_list.append(obs[0].astype(np.uint8))             # (H, W, C)
        feat_list.append(feats.cpu().numpy().astype(np.float32))
        obs, _r, done, _ = env.step(act)
        d = bool(done[0]) if hasattr(done, "__len__") else bool(done)
        if d:
            obs = env.reset()
            agent.actor_critic.reset(n=1)
        if (i + 1) % 500 == 0:
            log(f"  collected {i + 1}/{args.frames} frames")
    h.remove()
    return np.stack(obs_list), np.stack(feat_list)


def wm_block_hook(agent, layer, captured):
    return agent.world_model.transformer.blocks[layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("resid", o.detach())
    )


def make_grid(frames_rgb, grid=4, tile=84, pad=2):
    """Tile up to grid*grid RGB frames into one PNG (bytes). Missing cells are black."""
    cell = tile + pad
    canvas = Image.new("RGB", (grid * cell - pad, grid * cell - pad), (0, 0, 0))
    for idx, fr in enumerate(frames_rgb[: grid * grid]):
        img = Image.fromarray(fr).resize((tile, tile), Image.NEAREST)
        r, c = divmod(idx, grid)
        canvas.paste(img, (c * cell, r * cell))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Vision LLM labeling
# ---------------------------------------------------------------------------

def label_with_claude(client, model, grid_png):
    """Ask Claude for a label; return a clean phrase or None (UNCLEAR/refusal/empty)."""
    b64 = png_to_b64(grid_png)
    resp = client.messages.create(
        model=model,
        max_tokens=64,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = text.strip().strip('"').strip("'").rstrip(".").strip()
    if not text or any(m in text.lower() for m in _REFUSAL_MARKERS):
        return None
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(args) -> None:
    device = torch.device(args.device)
    log(f"Loading agent {args.checkpoint} + SAE {args.sae} (device={args.device})")
    agent, env, _cfg = _load_agent(_IRIS_ROOT, args.checkpoint, args.env_id, args.device)
    agent.eval()

    sae, meta = load_artifact(str(args.sae), device=args.device)
    layer = int(meta["layer"])
    env_id = meta.get("env_id") or args.env_id
    norm_mean = meta["norm"]["mean"].to(device)
    norm_std = meta["norm"]["std"].to(device).clamp_min(1e-6)
    log(f"SAE layer={layer} d_hidden={sae.d_hidden} env={env_id}")

    obs_all, feats = collect_frames(args, agent, env, sae, layer, norm_mean, norm_std, device)
    F = feats.shape[0]
    log(f"Collected {F} frames; computing per-feature stats")

    firing_rate = (feats > 0).mean(axis=0)        # (d_hidden,)
    mean_act = feats.mean(axis=0)                  # post-ReLU mean over all frames
    max_act = feats.max(axis=0)

    store = AutoInterpStore(str(args.out_dir), layer)
    index = store.load_index() if args.resume else {"layer": layer, "features": {}}
    index.update({
        "layer": layer, "env_id": env_id, "d_hidden": int(sae.d_hidden),
        "n_frames": int(F), "grid": int(args.grid), "prompt": PROMPT,
        "model": args.model if not args.no_llm else None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    index.setdefault("features", {})

    client = None
    if not args.no_llm:
        import anthropic  # imported lazily so --no-llm needs no SDK/key
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    # Process features in descending firing-rate order so a --limit run hits the most
    # active (most interpretable) features first.
    order = np.argsort(-firing_rate)
    n_examples = args.grid * args.grid
    processed = labeled = 0
    for fid in order.tolist():
        if args.limit and processed >= args.limit:
            break
        if args.resume and str(fid) in index["features"] and "label" in index["features"][str(fid)]:
            continue
        # Dead feature: never fires → no examples, null label, stats only.
        if max_act[fid] <= 0.0:
            index["features"][str(fid)] = _feat_meta(fid, None, firing_rate, mean_act, max_act)
            processed += 1
            continue

        top_idx = np.argsort(-feats[:, fid])[:n_examples]
        top_idx = [int(i) for i in top_idx if feats[i, fid] > 0.0]
        grid_png = make_grid(obs_all[top_idx], grid=args.grid)
        store.write_grid(fid, grid_png)
        store.write_examples(
            fid,
            examples_b64=[png_to_b64(make_grid([obs_all[i]], grid=1)) for i in top_idx],
            top_activations=[float(feats[i, fid]) for i in top_idx],
            frame_indices=top_idx,
        )

        label = None
        if client is not None:
            try:
                label = label_with_claude(client, args.model, grid_png)
            except Exception as exc:
                log(f"  #{fid}: LLM error ({exc}) — leaving unlabeled")
        index["features"][str(fid)] = _feat_meta(fid, label, firing_rate, mean_act, max_act)
        processed += 1
        labeled += int(label is not None)
        if label is not None:
            log(f"  #{fid}: {label}  (fire={firing_rate[fid]:.3f})")
        if processed % args.save_every == 0:
            store.save_index(index)
            log(f"  ...saved index ({processed} processed, {labeled} labeled)")

    store.save_index(index)
    log(f"Done: processed {processed} features, {labeled} labeled "
        f"({'NO-LLM dry run' if args.no_llm else args.model}) → {store.index_path}")
    try:
        env.close()
    except Exception:
        pass


def _feat_meta(fid, label, firing_rate, mean_act, max_act) -> dict:
    return {
        "id": int(fid),
        "label": label,
        "firing_rate": round(float(firing_rate[fid]), 4),
        "mean_activation": round(float(mean_act[fid]), 4),
        "max_activation": round(float(max_act[fid]), 4),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--sae", type=Path, required=True, help="Trained sae_L*.pt artifact")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Cache root (default: the SAE artifact's directory)")
    p.add_argument("--frames", type=int, default=3000, help="Frames to sample for stats/examples")
    p.add_argument("--grid", type=int, default=4, help="Grid side: grid*grid example frames per feature")
    p.add_argument("--limit", type=int, default=0, help="Process only the N most-firing features (0=all)")
    p.add_argument("--model", type=str, default="claude-sonnet-4-6", help="Vision-capable Claude model")
    p.add_argument("--no-llm", action="store_true", help="Build grids/stats/examples without calling the API")
    p.add_argument("--resume", action="store_true", help="Skip features already labeled in the index")
    p.add_argument("--save-every", type=int, default=25, help="Save the index every N processed features")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = args.sae.resolve().parent
    return args


if __name__ == "__main__":
    run(parse_args())
