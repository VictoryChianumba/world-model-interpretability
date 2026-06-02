"use client";

import type { RankedFeature, RankingMetric } from "@/hooks/useFeatureRanking";

interface Props {
  metric: RankingMetric;
  onMetricChange: (m: RankingMetric) => void;
  items: RankedFeature[];
  available: boolean;
  note?: string;
  labelFor: (featureId: number) => string | null;
  onPin: (featureId: number) => void;
  isPinned: (featureId: number) => boolean;
  paused: boolean;
}

// Two discovery axes. A "causal" axis was investigated and deliberately NOT shipped here:
// fixed-norm causal importance only reached cross-set Spearman ~0.49 (< 0.6 bar), so it is
// not reproducible enough to rank features in the UI. It remains a characterization tool
// (scripts/causal_importance.py + /feature). See WRITEUP Part VI "Causal importance, revisited".
const METRICS: { key: RankingMetric; label: string; title: string }[] = [
  { key: "firing", label: "firing", title: "Top-K by current activation magnitude (LLM-SAE convention; churns frame-to-frame)" },
  { key: "stability", label: "stable", title: "Fires consistently across the recent window (low coefficient of variation)" },
];

/**
 * DiscoveryPanel — find candidate features to pin, under a switchable ranking.
 *
 * The metric toggle is the substrate-adaptation experiment made visible: "firing" is the
 * borrowed LLM convention (and visibly churns); "stable" is the world-model alternative that
 * survived validation. Clicking a row pins it to the canvas. Score bars are normalized within
 * the current ranking.
 */
export default function DiscoveryPanel({
  metric,
  onMetricChange,
  items,
  available,
  note,
  labelFor,
  onPin,
  isPinned,
  paused,
}: Props) {
  const max = items.length ? Math.max(...items.map((i) => i.score), 1e-6) : 1;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-slate-800 px-3 py-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Discover
        </span>
        <span className="text-[10px] text-slate-500">
          {metric === "firing" ? (paused ? "paused" : "live") : metric === "stability" ? "window" : "offline"}
        </span>
      </div>

      {/* Ranking toggle */}
      <div className="flex gap-0.5 border-b border-slate-800 px-2 py-1.5">
        {METRICS.map((m) => (
          <button
            key={m.key}
            title={m.title}
            onClick={() => onMetricChange(m.key)}
            className={`flex-1 rounded px-1.5 py-0.5 text-[10px] capitalize ${
              metric === m.key ? "bg-indigo-600 text-white" : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto">
        {!available && (
          <div className="px-3 py-4 text-xs text-slate-500">
            {note ?? "No data for this ranking yet."}
          </div>
        )}
        {available && items.length === 0 && (
          <div className="px-3 py-4 text-xs text-slate-500">No features pass this ranking.</div>
        )}
        {items.map((f) => {
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
                  {f.score.toFixed(2)}
                  <span className="ml-1 text-indigo-400 opacity-0 group-hover:opacity-100">
                    {pinned ? "" : "+pin"}
                  </span>
                </span>
              </div>
              <div className="h-1 w-full rounded bg-slate-800">
                <div
                  className="h-1 rounded bg-indigo-500"
                  style={{ width: `${Math.min(100, (f.score / max) * 100)}%` }}
                />
              </div>
              {f.detail && <span className="text-[9px] text-slate-500">{f.detail}</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}
