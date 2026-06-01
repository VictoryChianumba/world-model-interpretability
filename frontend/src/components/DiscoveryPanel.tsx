"use client";

import type { SAEFeature } from "@/hooks/useVisualizerSocket";

interface Props {
  features: SAEFeature[] | null;
  labelFor: (featureId: number) => string | null;
  onPin: (featureId: number) => void;
  isPinned: (featureId: number) => boolean;
  paused: boolean;
}

/**
 * DiscoveryPanel — the top-firing features for the current frame.
 *
 * This is where you find candidates, not the main interaction surface: it refreshes as the
 * paused frame changes and a click pins a feature onto the canvas (the real workspace). The
 * list is the live top-K from the socket, annotated with autointerp labels where available.
 */
export default function DiscoveryPanel({ features, labelFor, onPin, isPinned, paused }: Props) {
  const max = features && features.length ? Math.max(...features.map((f) => f.mag)) : 1;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-slate-800 px-3 py-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Top firing
        </span>
        <span className="text-[10px] text-slate-500">{paused ? "paused" : "live"}</span>
      </div>
      <div className="flex-1 overflow-auto">
        {!features?.length && (
          <div className="px-3 py-4 text-xs text-slate-500">
            No SAE features (load an agent with a trained SAE).
          </div>
        )}
        {features?.map((f) => {
          const label = labelFor(f.id);
          const pinned = isPinned(f.id);
          return (
            <button
              key={f.id}
              disabled={pinned}
              onClick={() => onPin(f.id)}
              title={pinned ? "already on canvas" : "pin to canvas"}
              className={`group flex w-full flex-col gap-1 border-b border-slate-800/60 px-3 py-1.5 text-left ${
                pinned ? "cursor-default opacity-50" : "hover:bg-slate-800/60"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="truncate text-xs">
                  <span className="font-mono text-indigo-300">#{f.id}</span>{" "}
                  <span className={label ? "text-slate-200" : "italic text-slate-500"}>
                    {label ?? "unlabeled"}
                  </span>
                </span>
                <span className="ml-2 flex-shrink-0 font-mono text-[10px] text-slate-400">
                  {f.mag.toFixed(2)}
                  <span className="ml-1 text-indigo-400 opacity-0 group-hover:opacity-100">
                    {pinned ? "" : "+pin"}
                  </span>
                </span>
              </div>
              <div className="h-1 w-full rounded bg-slate-800">
                <div
                  className="h-1 rounded bg-indigo-500"
                  style={{ width: `${Math.min(100, (f.mag / max) * 100)}%` }}
                />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
