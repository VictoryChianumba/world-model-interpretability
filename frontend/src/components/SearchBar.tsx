"use client";

import { useEffect, useRef, useState } from "react";
import type { SearchHit } from "@/hooks/useFeatureIndex";

interface Props {
  search: (query: string) => SearchHit[];
  onPin: (featureId: number) => void;
  isPinned: (featureId: number) => boolean;
  indexLoaded: boolean;
}

/**
 * SearchBar — the primary discovery surface (Neuronpedia-style).
 *
 * Type a feature id to jump straight to it, or a label keyword to get a dropdown of
 * matching features. Selecting one pins it onto the canvas. Label search only works once
 * the autointerp pipeline has produced labels; id search works regardless.
 */
export default function SearchBar({ search, onPin, isPinned, indexLoaded }: Props) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  const hits = query.trim() ? search(query) : [];

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);

  const pin = (id: number) => {
    onPin(id);
    setQuery("");
    setOpen(false);
  };

  const isNumeric = /^\d+$/.test(query.trim());

  return (
    <div ref={wrapRef} className="relative w-full max-w-md">
      <input
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && isNumeric) pin(Number(query.trim()));
          if (e.key === "Escape") setOpen(false);
        }}
        placeholder="Search feature id or label…"
        className="w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-indigo-400"
      />
      {open && query.trim() && (
        <div className="absolute z-20 mt-1 max-h-72 w-full overflow-auto rounded-md border border-slate-700 bg-slate-900 shadow-xl">
          {hits.length === 0 && (
            <div className="px-3 py-2 text-xs text-slate-500">
              {isNumeric
                ? "Press Enter to pin this feature id"
                : indexLoaded
                  ? "No labels match. Run the autointerp pipeline to label features, or search by id."
                  : "Loading feature labels…"}
            </div>
          )}
          {hits.map((h) => {
            const pinned = isPinned(h.id);
            return (
              <button
                key={h.id}
                disabled={pinned}
                onClick={() => pin(h.id)}
                className={`flex w-full items-center justify-between px-3 py-1.5 text-left text-sm ${
                  pinned ? "cursor-default text-slate-500" : "text-slate-100 hover:bg-slate-800"
                }`}
              >
                <span className="truncate">
                  <span className="font-mono text-indigo-300">#{h.id}</span>{" "}
                  <span className={h.label ? "" : "italic text-slate-500"}>
                    {h.label ?? "unlabeled"}
                  </span>
                </span>
                <span className="ml-2 flex-shrink-0 text-[10px] text-slate-500">
                  {pinned ? "pinned" : h.firing_rate != null ? `${(h.firing_rate * 100).toFixed(0)}%` : ""}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
