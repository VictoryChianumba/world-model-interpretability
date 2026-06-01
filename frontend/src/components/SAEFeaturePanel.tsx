"use client";

import type { SAEFeature } from "@/hooks/useVisualizerSocket";

interface Props {
  features: SAEFeature[] | null;
  layer: number | null;
  selectedId?: number | null;
  onSelect?: (id: number) => void;
  /** Lookup a saved label for a feature id (from the bookmark store), if any. */
  labelFor?: (id: number) => string | undefined;
}

/**
 * SAEFeaturePanel — the top-K sparse-autoencoder features firing this frame.
 *
 * Each row is one dictionary feature (by id) with a bar scaled to its activation
 * magnitude. The features are read from the world-model residual stream at the
 * SAE's layer (the action-token position), so this is "what the world model is
 * representing right now", not anything about the agent's policy.
 *
 * `features === null` means no SAE is loaded for the current agent — the panel
 * shows how to enable it rather than an empty chart.
 */
export default function SAEFeaturePanel({ features, layer, selectedId, onSelect, labelFor }: Props) {
  if (!Array.isArray(features)) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center bg-[#0d0d20] px-4 text-center gap-1">
        <span className="text-xs text-gray-600">No SAE loaded</span>
        <span className="text-[9px] text-gray-700">
          Train one with scripts/train_sae.py and drop sae_L*.pt in the SAE dir
        </span>
      </div>
    );
  }

  const maxMag = Math.max(...features.map((f) => f.mag), 1e-6);

  return (
    <div className="w-full h-full flex flex-col bg-[#0d0d20] p-1.5 overflow-hidden">
      <div className="text-[10px] text-gray-400 px-1 pb-1 flex-shrink-0 flex items-center justify-between">
        <span>SAE Features</span>
        <span className="text-gray-600">
          {layer != null ? `layer ${layer}` : ""} · top {features.length}
        </span>
      </div>
      <div className="flex-1 flex flex-col gap-0.5 overflow-y-auto pr-1">
        {features.length === 0 && (
          <div className="flex-1 flex items-center justify-center">
            <span className="text-[10px] text-gray-600">No features firing</span>
          </div>
        )}
        {features.map((f) => {
          const pct = (f.mag / maxMag) * 100;
          const selected = selectedId === f.id;
          const label = labelFor?.(f.id);
          return (
            <button
              key={f.id}
              type="button"
              onClick={() => onSelect?.(f.id)}
              title={label ?? (onSelect ? "Set as intervention target" : undefined)}
              className={`flex items-center gap-1.5 text-[9px] w-full text-left rounded-sm px-0.5 ${
                onSelect ? "hover:bg-gray-800/50 cursor-pointer" : "cursor-default"
              } ${selected ? "bg-pink-500/15 ring-1 ring-pink-500/50" : ""}`}
            >
              <span
                className={`font-mono w-10 text-right flex-shrink-0 ${
                  selected ? "text-pink-300" : "text-gray-500"
                }`}
              >
                #{f.id}
              </span>
              <div className="flex-1 h-3 bg-gray-800/60 rounded-sm overflow-hidden relative">
                <div
                  className={`h-full rounded-sm ${selected ? "bg-pink-500/80" : "bg-indigo-500/80"}`}
                  style={{ width: `${pct}%` }}
                />
                {label && (
                  <span className="absolute inset-0 flex items-center px-1 text-[8px] text-gray-200 truncate pointer-events-none">
                    {label}
                  </span>
                )}
              </div>
              <span className="font-mono text-gray-400 w-10 flex-shrink-0">
                {f.mag.toFixed(2)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
