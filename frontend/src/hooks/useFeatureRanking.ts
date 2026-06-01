"use client";

import { useEffect, useMemo, useState } from "react";
import { API_BASE } from "@/lib/config";
import type { SAEFeature } from "@/hooks/useVisualizerSocket";

export type RankingMetric = "firing" | "stability" | "causal";

/** A feature row in the discovery panel, normalized across ranking metrics. */
export interface RankedFeature {
  id: number;
  score: number; // higher = ranked higher, whatever the metric's underlying quantity
  detail?: string; // short metric-specific annotation shown under the score
}

export interface RankingState {
  items: RankedFeature[];
  available: boolean;
  note?: string; // shown when unavailable (e.g. pipeline not run, window warming up)
}

/**
 * useFeatureRanking — the discovery panel's data, switchable across ranking metrics.
 *
 * - "firing":   top-K by current activation magnitude, straight off the live socket
 *               (the LLM-SAE convention; churns frame-to-frame).
 * - "stability": features that fire consistently (low CV) over the recent window;
 *                polled live from /ranking/stability while selected.
 * - "causal":   offline causal-importance scores from /ranking/causal (one-shot);
 *                unavailable until scripts/causal_importance.py has been run.
 */
export function useFeatureRanking(
  metric: RankingMetric,
  saeFeatures: SAEFeature[] | null,
  layer: number | null,
): RankingState {
  const [remote, setRemote] = useState<RankingState>({ items: [], available: false });

  // Live top-firing is derived directly; no fetch.
  const firing: RankingState = useMemo(
    () => ({
      items: (saeFeatures ?? []).map((f) => ({ id: f.id, score: f.mag })),
      available: saeFeatures != null,
    }),
    [saeFeatures],
  );

  useEffect(() => {
    if (metric === "firing") return; // derived, nothing to fetch
    let cancelled = false;
    const url =
      metric === "stability"
        ? `${API_BASE}/ranking/stability?top=20`
        : `${API_BASE}/ranking/causal?top=20`;

    const load = async () => {
      try {
        const res = await fetch(url);
        if (!res.ok) {
          if (!cancelled)
            setRemote({
              items: [],
              available: false,
              note:
                metric === "causal"
                  ? "Not computed — run scripts/causal_importance.py"
                  : "unavailable",
            });
          return;
        }
        const data = await res.json();
        if (cancelled) return;
        const items: RankedFeature[] = (data.features ?? []).map((f: Record<string, number>) =>
          metric === "stability"
            ? {
                id: f.id,
                score: f.score,
                detail: `${Math.round((f.firing_rate ?? 0) * 100)}% fire · cv ${(f.cv ?? 0).toFixed(2)}`,
              }
            : { id: f.id, score: f.score, detail: `Δ${(f.score ?? 0).toFixed(2)} tok` },
        );
        setRemote({
          items,
          available: data.available ?? items.length > 0,
          note:
            metric === "stability" && !data.available
              ? "Window warming up — resume the loop to stream frames"
              : metric === "causal" && items.length === 0
                ? "Not computed — run scripts/causal_importance.py"
                : undefined,
        });
      } catch {
        if (!cancelled) setRemote({ items: [], available: false, note: "backend offline" });
      }
    };

    load();
    // Stability reflects the live window, so poll; causal is static once computed.
    const timer = metric === "stability" ? setInterval(load, 1500) : undefined;
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [metric, layer]);

  return metric === "firing" ? firing : remote;
}
