"use client";

import { useState } from "react";
import RolloutPanel from "@/components/RolloutPanel";
import TrajectoryCharts from "@/components/TrajectoryCharts";
import type { Intervention, RolloutResult } from "@/hooks/useRollout";

interface Props {
  interventions: Intervention[]; // canvas cards with non-zero scale
  paused: boolean;
  result: RolloutResult | null;
  loading: boolean;
  error: string | null;
  onRun: (nSteps: number, nSeeds: number) => void;
}

/**
 * RolloutComparison — the primary experiment surface (beneath the canvas).
 *
 * The intervention is read straight off the canvas: every pinned card with a non-zero
 * scale contributes. "Run rollout" fires the paused-only paired baseline-vs-intervened
 * imagination and shows scrubbable frames + divergence/state trajectory charts. Cards with
 * scale 0 are observation-only and don't steer.
 */
export default function RolloutComparison({
  interventions,
  paused,
  result,
  loading,
  error,
  onRun,
}: Props) {
  const [frameIdx, setFrameIdx] = useState(0);
  const [nSteps, setNSteps] = useState(20);
  const [nSeeds, setNSeeds] = useState(2);

  const canRun = paused && interventions.length > 0 && !loading;

  return (
    <div className="flex h-full flex-col">
      {/* Control row: steered features + run */}
      <div className="flex flex-shrink-0 flex-wrap items-center gap-2 border-b border-slate-800 px-3 py-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Rollout
        </span>
        <div className="flex flex-1 flex-wrap items-center gap-1">
          {interventions.length === 0 ? (
            <span className="text-[11px] text-slate-500">
              Set a non-zero scale on a card to steer the rollout.
            </span>
          ) : (
            interventions.map((iv) => (
              <span
                key={iv.feature_id}
                className="rounded bg-pink-500/15 px-1.5 py-0.5 font-mono text-[10px] text-pink-300"
              >
                #{iv.feature_id} ×{iv.scale.toFixed(1)}
              </span>
            ))
          )}
        </div>

        <label className="flex items-center gap-1 text-[10px] text-slate-400">
          steps
          <select
            value={nSteps}
            onChange={(e) => setNSteps(Number(e.target.value))}
            className="rounded border border-slate-700 bg-slate-900 px-1 py-0.5 text-slate-200"
          >
            {[10, 20, 30, 40].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1 text-[10px] text-slate-400">
          seeds
          <select
            value={nSeeds}
            onChange={(e) => setNSeeds(Number(e.target.value))}
            className="rounded border border-slate-700 bg-slate-900 px-1 py-0.5 text-slate-200"
          >
            {[1, 2, 3, 4].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>

        <button
          onClick={() => {
            setFrameIdx(0);
            onRun(nSteps, nSeeds);
          }}
          disabled={!canRun}
          title={!paused ? "pause the loop first (rollout is paused-only)" : undefined}
          className={`rounded px-3 py-1 text-xs font-semibold ${
            canRun
              ? "bg-indigo-600 text-white hover:bg-indigo-500"
              : "cursor-not-allowed bg-slate-800 text-slate-500"
          }`}
        >
          {loading ? "Running…" : "Run rollout"}
        </button>
      </div>

      {/* Body: scrubbable frames + trajectory charts share one playhead */}
      <div className="flex min-h-0 flex-1">
        <div className="w-1/2 border-r border-slate-800">
          <RolloutPanel
            result={result}
            loading={loading}
            error={error}
            frameIdx={frameIdx}
            onFrameIdx={setFrameIdx}
          />
        </div>
        <div className="w-1/2">
          <TrajectoryCharts result={result} frameIdx={frameIdx} />
        </div>
      </div>
    </div>
  );
}
