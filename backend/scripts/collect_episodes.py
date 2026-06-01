#!/usr/bin/env python
"""
Collect fresh Atari episodes by running a trained IRIS agent in the real env.

Produces saved ``Episode`` .pt files (the same format IRIS training uses) to build
a larger dataset for SAE training.  This runs ONLY the actor-critic policy + the
Atari env (no world model), so it is cheap; the SAE harvest/train is a separate step.

Offline tool — imports IRIS via sys.path like train_sae.py, reuses the engine's
``_load_agent``.  Modeled on iris/src/collector.py Collector.collect, but writes
plain Episode files directly (no EpisodesDataset / RAM monitoring / epochs).

Example
-------
    python backend/scripts/collect_episodes.py \
        --checkpoint /path/to/iris/checkpoints/Breakout.pt \
        --env-id BreakoutNoFrameskip-v4 \
        --out-dir /path/to/iris/episodes_breakout \
        --num-episodes 150 --epsilon 0.01 --device mps
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: backend/ (for inference) and IRIS src/ (for episode, models)
# ---------------------------------------------------------------------------

_BACKEND = Path(__file__).resolve().parents[1]
_IRIS_ROOT = Path(os.environ.get("IRIS_ROOT", Path(__file__).resolve().parents[3] / "iris"))
_IRIS_SRC = Path(os.environ.get("IRIS_SRC", _IRIS_ROOT / "src"))
for _p in (str(_BACKEND), str(_IRIS_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference import _load_agent  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


@torch.no_grad()
def collect_one_episode(agent, env, device, epsilon: float, max_steps: int):
    """Run the policy in the env for one episode; return an IRIS Episode.

    Mirrors iris/src/collector.py:51-93 for a single SingleProcessEnv:
      - env.reset() → (1, H, W, C) uint8
      - rearrange to (1, C, H, W) float in [0,1] for agent.act
      - agent.act(should_sample=True), with prob epsilon a uniform random action
      - accumulate channel-last obs; convert to (T, C, H, W) uint8 at the end
    """
    from einops import rearrange

    num_actions = env.num_actions
    agent.actor_critic.reset(n=1)
    obs = env.reset()                                   # (1, H, W, C) uint8

    observations, actions, rewards, dones = [], [], [], []
    ep_return = 0.0
    for _ in range(max_steps):
        observations.append(obs[0])                    # (H, W, C) uint8
        obs_t = rearrange(torch.FloatTensor(obs).div(255), "n h w c -> n c h w").to(device)
        act = agent.act(obs_t, should_sample=True).cpu().numpy()   # (1,)
        if epsilon > 0.0 and np.random.random() < epsilon:
            act = np.array([np.random.randint(num_actions)])

        obs, reward, done, _ = env.step(act)
        r = float(reward[0]) if hasattr(reward, "__len__") else float(reward)
        d = bool(done[0]) if hasattr(done, "__len__") else bool(done)
        actions.append(int(act[0]))
        rewards.append(r)
        dones.append(1 if d else 0)
        ep_return += r
        if d:
            break

    from episode import Episode

    obs_np = np.stack(observations, axis=0)            # (T, H, W, C) uint8
    episode = Episode(
        observations=torch.ByteTensor(obs_np).permute(0, 3, 1, 2).contiguous(),  # (T, C, H, W)
        actions=torch.LongTensor(actions),
        rewards=torch.FloatTensor(rewards),
        ends=torch.LongTensor(dones),
        mask_padding=torch.ones(len(actions), dtype=torch.bool),
    )
    return episode, ep_return


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="IRIS checkpoint .pt (default: <iris_root>/checkpoints/last.pt)")
    p.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory to write episode_*.pt files (created if missing)")
    p.add_argument("--num-episodes", type=int, default=150)
    p.add_argument("--epsilon", type=float, default=0.01,
                   help="Per-step probability of a uniform random action (diversity)")
    p.add_argument("--max-steps", type=int, default=10000, help="Safety cap on steps per episode")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--start-index", type=int, default=0,
                   help="First episode file index (to append to an existing dir)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = args.checkpoint or (_IRIS_ROOT / "checkpoints" / "last.pt")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading agent: {checkpoint} (env={args.env_id}, device={args.device})")
    agent, env, _cfg = _load_agent(_IRIS_ROOT, checkpoint, args.env_id, args.device)
    agent.eval()

    log(f"Collecting {args.num_episodes} episodes → {args.out_dir} (epsilon={args.epsilon})")
    t0 = time.perf_counter()
    total_frames = 0
    returns = []
    for i in range(args.num_episodes):
        idx = args.start_index + i
        episode, ep_return = collect_one_episode(
            agent, env, device, args.epsilon, args.max_steps
        )
        episode.save(args.out_dir / f"episode_{idx}.pt")
        total_frames += len(episode)
        returns.append(ep_return)
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - t0
            log(f"  [{i + 1}/{args.num_episodes}] last_len={len(episode)} "
                f"last_return={ep_return:.0f} total_frames={total_frames} "
                f"mean_return={np.mean(returns):.1f} ({elapsed:.0f}s)")

    elapsed = time.perf_counter() - t0
    log(f"Done: {args.num_episodes} episodes, {total_frames} frames, "
        f"mean_return={np.mean(returns):.1f}, mean_len={total_frames / args.num_episodes:.0f} "
        f"({elapsed:.0f}s)")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
