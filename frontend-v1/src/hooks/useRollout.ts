"use client";

import { useCallback, useState } from "react";

const API_BASE =
  typeof window !== "undefined"
    ? `http://${window.location.hostname}:8000`
    : "http://localhost:8000";

/** Per-frame extracted Breakout state (normalized [0,1]; null when not found). */
export interface FrameState {
  paddle_x: number | null;
  ball_x: number | null;
  ball_y: number | null;
  bricks: number[];
}

export interface RolloutFrame {
  frame: string; // base64 PNG (64x64)
  state: FrameState;
}

/** Result of POST /rollout: paired baseline vs intervened, one entry per seed per step. */
export interface RolloutResult {
  feature_id: number;
  scale: number;
  layer: number;
  n_steps: number;
  n_seeds: number;
  n_obs_tokens: number;
  actions: number[];
  baseline: RolloutFrame[][]; // [seed][step]
  intervened: RolloutFrame[][]; // [seed][step]
  // Per-step, seed-averaged divergence between baseline and intervened — robust
  // signal even when pixel state-extraction fails on lossy imagined frames.
  token_divergence: number[]; // 0..n_obs_tokens obs tokens changed
  pixel_divergence: number[]; // 0..255 mean abs pixel diff
}

/**
 * useRollout — runs the paused-only N-step intervention experiment.
 *
 * Calls POST /rollout (heavy: ~80s CPU / less on MPS), exposing loading + error so the
 * UI can show a "Run rollout" button with a clear pending state. The result is NOT part
 * of the live socket stream — it lives here as on-demand experiment output.
 */
export function useRollout() {
  const [result, setResult] = useState<RolloutResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    async (featureId: number, scale: number, nSteps = 20, nSeeds = 2) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/rollout`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            feature_id: featureId,
            scale,
            n_steps: nSteps,
            n_seeds: nSeeds,
          }),
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || `rollout failed (${res.status})`);
        }
        setResult(await res.json());
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setResult(null);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const clear = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, run, clear };
}
