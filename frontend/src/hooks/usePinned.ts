"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE } from "@/lib/config";

/** One pinned feature card on the canvas (mirrors the backend `/pinned` record). */
export interface PinnedFeature {
  env_id: string;
  layer: number;
  feature_id: number;
  custom_label: string | null;
  intervention_scale: number; // 0 = observation-only, non-zero = steer in rollout
  x: number;
  y: number;
  updated_at?: string;
}

/** Fields a partial update may change. Omitted fields are left untouched server-side. */
export type PinUpdate = Partial<
  Pick<PinnedFeature, "custom_label" | "intervention_scale" | "x" | "y">
>;

/**
 * usePinned — the canvas's pinned-feature cards for one (envId, layer).
 *
 * The backend `/pinned` store is the source of truth (survives reloads). Reads happen on
 * mount / when env+layer change; writes are optimistic — local state updates immediately
 * so dragging and slider changes feel instant, and the POST syncs in the background. Only
 * the fields you pass are sent, matching the store's merge semantics (a drag persists just
 * x/y, a slider just intervention_scale).
 */
export function usePinned(envId: string | undefined, layer: number | null) {
  const [pins, setPins] = useState<PinnedFeature[]>([]);
  const ready = !!envId && layer != null;
  // Keep the latest count without retriggering callbacks, for cascade placement.
  const countRef = useRef(0);
  countRef.current = pins.length;

  const refresh = useCallback(async () => {
    if (!envId || layer == null) {
      setPins([]);
      return;
    }
    try {
      const params = new URLSearchParams({ env_id: envId, layer: String(layer) });
      const res = await fetch(`${API_BASE}/pinned?${params}`);
      setPins(await res.json());
    } catch {
      // backend not ready — keep whatever we have
    }
  }, [envId, layer]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const post = useCallback(
    async (feature_id: number, fields: PinUpdate) => {
      if (!envId || layer == null) return;
      try {
        await fetch(`${API_BASE}/pinned`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ env_id: envId, layer, feature_id, ...fields }),
        });
      } catch {
        // best-effort; local state already reflects the change
      }
    },
    [envId, layer],
  );

  const isPinned = useCallback(
    (featureId: number) => pins.some((p) => p.feature_id === featureId),
    [pins],
  );

  /** Pin a feature (no-op if already pinned). New cards cascade so they don't overlap. */
  const pin = useCallback(
    async (featureId: number, fields: PinUpdate = {}) => {
      if (!envId || layer == null) return;
      if (pins.some((p) => p.feature_id === featureId)) return;
      const n = countRef.current;
      const x = fields.x ?? 24 + (n % 4) * 188;
      const y = fields.y ?? 24 + Math.floor(n / 4) * 132;
      const optimistic: PinnedFeature = {
        env_id: envId,
        layer,
        feature_id: featureId,
        custom_label: fields.custom_label ?? null,
        intervention_scale: fields.intervention_scale ?? 0,
        x,
        y,
      };
      setPins((prev) => [...prev, optimistic]);
      await post(featureId, { ...fields, x, y });
    },
    [envId, layer, pins, post],
  );

  const unpin = useCallback(
    async (featureId: number) => {
      if (!envId || layer == null) return;
      setPins((prev) => prev.filter((p) => p.feature_id !== featureId));
      const params = new URLSearchParams({
        env_id: envId,
        layer: String(layer),
        feature_id: String(featureId),
      });
      try {
        await fetch(`${API_BASE}/pinned?${params}`, { method: "DELETE" });
      } catch {
        /* best-effort */
      }
    },
    [envId, layer],
  );

  /**
   * Merge fields into a pinned card. `persist` controls whether the change is written
   * through to the backend — pass false during a continuous gesture (drag/slider move) and
   * true once to commit on release, to avoid a POST per pixel.
   */
  const update = useCallback(
    (featureId: number, fields: PinUpdate, persist = true) => {
      setPins((prev) =>
        prev.map((p) => (p.feature_id === featureId ? { ...p, ...fields } : p)),
      );
      if (persist) void post(featureId, fields);
    },
    [post],
  );

  return { pins, ready, refresh, pin, unpin, update, isPinned };
}
