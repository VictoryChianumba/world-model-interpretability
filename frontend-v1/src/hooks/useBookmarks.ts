"use client";

import { useCallback, useEffect, useState } from "react";

const API_BASE =
  typeof window !== "undefined"
    ? `http://${window.location.hostname}:8000`
    : "http://localhost:8000";

export interface Bookmark {
  env_id: string;
  layer: number;
  feature_id: number;
  label: string;
  notes: string;
  source: string;
  updated_at: string;
}

/**
 * useBookmarks — load/save/delete SAE feature labels for one (env_id, layer).
 *
 * Bookmarks persist on the backend (bookmarks.json), so labels survive across
 * sessions. Keyed by feature_id within the current env+layer; `labelFor` is the
 * fast lookup the feature panel uses to show a label next to a feature.
 */
export function useBookmarks(envId: string | undefined, layer: number | null) {
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);

  const refresh = useCallback(async () => {
    if (!envId || layer == null) {
      setBookmarks([]);
      return;
    }
    try {
      const params = new URLSearchParams({ env_id: envId, layer: String(layer) });
      const res = await fetch(`${API_BASE}/bookmarks?${params}`);
      setBookmarks(await res.json());
    } catch {
      // backend not ready — leave existing bookmarks
    }
  }, [envId, layer]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const save = useCallback(
    async (featureId: number, label: string, notes = "") => {
      if (!envId || layer == null) return;
      await fetch(`${API_BASE}/bookmarks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          env_id: envId,
          layer,
          feature_id: featureId,
          label,
          notes,
          source: "user",
        }),
      });
      await refresh();
    },
    [envId, layer, refresh],
  );

  const remove = useCallback(
    async (featureId: number) => {
      if (!envId || layer == null) return;
      const params = new URLSearchParams({
        env_id: envId,
        layer: String(layer),
        feature_id: String(featureId),
      });
      await fetch(`${API_BASE}/bookmarks?${params}`, { method: "DELETE" });
      await refresh();
    },
    [envId, layer, refresh],
  );

  const labelFor = useCallback(
    (featureId: number): string | undefined =>
      bookmarks.find((b) => b.feature_id === featureId)?.label,
    [bookmarks],
  );

  return { bookmarks, refresh, save, remove, labelFor };
}
