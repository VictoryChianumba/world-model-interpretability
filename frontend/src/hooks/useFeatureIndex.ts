"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { API_BASE } from "@/lib/config";

/** One feature's autointerp record (label may be null = unlabeled / dead / polysemantic). */
export interface FeatureRecord {
  id: number;
  label: string | null;
  firing_rate?: number;
  mean_activation?: number;
  max_activation?: number;
}

export interface SearchHit {
  id: number;
  label: string | null;
  firing_rate?: number;
}

/**
 * useFeatureIndex — the autointerp label index for one SAE layer.
 *
 * Loads the cache written by scripts/autointerp.py (`GET /features`). When the pipeline
 * hasn't been run the index is empty and every label resolves to null — the UI shows
 * "unlabeled". Powers the search box: a numeric query jumps straight to that feature id;
 * a text query matches against labels.
 */
export function useFeatureIndex(layer: number | null) {
  const [features, setFeatures] = useState<Record<number, FeatureRecord>>({});
  const [loaded, setLoaded] = useState(false);

  const refresh = useCallback(async () => {
    if (layer == null) {
      setFeatures({});
      setLoaded(false);
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/features?layer=${layer}`);
      const data = await res.json();
      const map: Record<number, FeatureRecord> = {};
      for (const [id, rec] of Object.entries(data.features ?? {})) {
        map[Number(id)] = rec as FeatureRecord;
      }
      setFeatures(map);
    } catch {
      setFeatures({});
    } finally {
      setLoaded(true);
    }
  }, [layer]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const labelFor = useCallback(
    (featureId: number): string | null => features[featureId]?.label ?? null,
    [features],
  );

  // Pre-sort labeled features by firing rate once, for fast keyword search.
  const labeled = useMemo(
    () =>
      Object.values(features)
        .filter((f) => f.label)
        .sort((a, b) => (b.firing_rate ?? 0) - (a.firing_rate ?? 0)),
    [features],
  );

  const search = useCallback(
    (query: string, limit = 20): SearchHit[] => {
      const q = query.trim().toLowerCase();
      if (!q) return [];
      // Pure number → jump to that feature id directly (works even with no labels).
      if (/^\d+$/.test(q)) {
        const id = Number(q);
        return [{ id, label: features[id]?.label ?? null, firing_rate: features[id]?.firing_rate }];
      }
      return labeled
        .filter((f) => f.label!.toLowerCase().includes(q))
        .slice(0, limit)
        .map((f) => ({ id: f.id, label: f.label, firing_rate: f.firing_rate }));
    },
    [features, labeled],
  );

  return { features, loaded, refresh, labelFor, search };
}
