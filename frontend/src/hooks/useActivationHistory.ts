"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { SAEFeature } from "@/hooks/useVisualizerSocket";

/**
 * useActivationHistory — rolling per-feature firing history for sparklines.
 *
 * The socket streams only the top-K firing features each frame (`sae_features`). For a
 * pinned feature absent from that list this frame, its activation is ~0 (not firing). We
 * snapshot each new frame (detected by `step` changing, so re-renders don't double-count)
 * into a bounded ring of sparse {id: mag} maps; `historyFor(id)` reconstructs a dense
 * series with zeros where the feature wasn't firing. Stable features look steady; flickery
 * ones look spiky.
 */
export function useActivationHistory(
  saeFeatures: SAEFeature[] | null,
  step: number | undefined,
  length = 40,
) {
  const [frames, setFrames] = useState<Array<Record<number, number>>>([]);
  const lastStep = useRef<number | null>(null);

  useEffect(() => {
    if (step == null || step === lastStep.current) return;
    lastStep.current = step;
    const snapshot: Record<number, number> = {};
    for (const f of saeFeatures ?? []) snapshot[f.id] = f.mag;
    setFrames((prev) => [...prev, snapshot].slice(-length));
  }, [step, saeFeatures, length]);

  const historyFor = useCallback(
    (featureId: number): number[] => frames.map((f) => f[featureId] ?? 0),
    [frames],
  );

  const currentFor = useCallback(
    (featureId: number): number => {
      const last = frames[frames.length - 1];
      return last ? last[featureId] ?? 0 : 0;
    },
    [frames],
  );

  return { historyFor, currentFor, depth: frames.length };
}
