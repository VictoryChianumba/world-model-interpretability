"""
Breakout-SPECIFIC state extraction from decoded world-model frames.

>>> THIS IS A BREAKOUT-ONLY HACK. IT DOES NOT GENERALIZE. <<<

It reads paddle / ball / brick state out of a 64×64 RGB frame by crude pixel
inspection (brightness blobs in fixed screen bands). Two layers of approximation:
  1. The frames are world-model *imagined* reconstructions decoded from 16 SAE/
     tokenizer tokens — blurry and lossy, not real Atari pixels.
  2. The band/threshold heuristics are tuned by eye for Breakout's layout only.

It exists so the rollout trajectory charts have *something* to plot (paddle x,
ball x/y) so divergence between baseline and intervened rollouts is legible. Treat
the numbers as soft signals, not ground truth. For any other game, this returns
garbage — do not extend it; write a game-specific extractor instead.

All coordinates are normalized to [0, 1] (x: left→right, y: top→bottom). A field is
``None`` when no confident blob is found in its band.
"""

from typing import Dict, List, Optional

import numpy as np

# Vertical bands as fractions of frame height (Breakout layout on 64×64):
#   bricks occupy the upper-middle, the ball roams the play area, the paddle sits
#   near the bottom. These are deliberately loose.
_BRICK_BAND = (0.20, 0.45)
_PLAY_BAND = (0.30, 0.88)     # where the ball can be
_PADDLE_BAND = (0.85, 1.00)
_N_BRICK_ROWS = 6
_MIN_BLOB_PIXELS = 2          # below this, treat as "not found" (noise on lossy frames)


def _gray(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 → (H, W) float brightness in [0, 1]."""
    return rgb.astype(np.float32).mean(axis=2) / 255.0


def _band_rows(h: int, lo: float, hi: float) -> slice:
    return slice(int(round(lo * h)), max(int(round(hi * h)), int(round(lo * h)) + 1))


def _brightest_centroid(region: np.ndarray, thresh: float):
    """Return (cx, cy, n) centroid of bright pixels in a region (region-local coords),
    or None if too few bright pixels."""
    mask = region > thresh
    n = int(mask.sum())
    if n < _MIN_BLOB_PIXELS:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean()), n


def extract_breakout_state(rgb: np.ndarray) -> Dict[str, Optional[object]]:
    """Extract {paddle_x, ball_x, ball_y, bricks} from a 64×64 RGB frame (Breakout only).

    paddle_x / ball_x / ball_y are normalized [0,1] or None. ``bricks`` is a length-6
    list of per-row mean brightness (coarse "how many bricks remain" proxy), always present.
    """
    if rgb is None or rgb.ndim != 3:
        return {"paddle_x": None, "ball_x": None, "ball_y": None, "bricks": [0.0] * _N_BRICK_ROWS}
    g = _gray(rgb)
    H, W = g.shape

    # --- Paddle: brightest horizontal blob in the bottom band → its x centroid ---
    paddle_rows = _band_rows(H, *_PADDLE_BAND)
    pr = g[paddle_rows, :]
    pthr = max(0.4, float(pr.mean() + 2 * pr.std()))
    pc = _brightest_centroid(pr, pthr)
    paddle_x = (pc[0] / max(W - 1, 1)) if pc is not None else None

    # --- Ball: brightest small blob in the play band → x, y centroid ---
    play_rows = _band_rows(H, *_PLAY_BAND)
    play_lo = int(round(_PLAY_BAND[0] * H))
    pl = g[play_rows, :]
    bthr = max(0.5, float(pl.mean() + 3 * pl.std()))
    bc = _brightest_centroid(pl, bthr)
    if bc is not None:
        ball_x = bc[0] / max(W - 1, 1)
        ball_y = (play_lo + bc[1]) / max(H - 1, 1)
    else:
        ball_x = ball_y = None

    # --- Bricks: mean brightness per row-band (coarse remaining-bricks proxy) ---
    b_lo = int(round(_BRICK_BAND[0] * H))
    b_hi = max(int(round(_BRICK_BAND[1] * H)), b_lo + _N_BRICK_ROWS)
    brick_region = g[b_lo:b_hi, :]
    rows = np.array_split(brick_region, _N_BRICK_ROWS, axis=0)
    bricks: List[float] = [round(float(r.mean()), 4) for r in rows]

    return {
        "paddle_x": None if paddle_x is None else round(paddle_x, 4),
        "ball_x": None if ball_x is None else round(ball_x, 4),
        "ball_y": None if ball_y is None else round(ball_y, 4),
        "bricks": bricks,
    }
