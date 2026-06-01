"use client";

import { useMemo } from "react";
import { Group } from "@visx/group";
import { LinePath, Line } from "@visx/shape";
import { scaleLinear } from "@visx/scale";
import type { RolloutResult, RolloutFrame } from "@/hooks/useRollout";

interface Props {
  result: RolloutResult | null;
  /** Current scrub index, drawn as a vertical playhead on each chart. */
  frameIdx: number;
}

type VarKey = "paddle_x" | "ball_x" | "ball_y";
const VARS: { key: VarKey; label: string }[] = [
  { key: "paddle_x", label: "paddle x" },
  { key: "ball_x", label: "ball x" },
  { key: "ball_y", label: "ball y" },
];

const BASE_COLOR = "#818cf8"; // indigo — baseline
const IV_COLOR = "#f472b6"; // pink — intervened

/**
 * TrajectoryCharts — small multiples of extracted state over rollout time.
 *
 * One chart per state variable (paddle x, ball x, ball y). Each overlays baseline
 * (indigo) vs intervened (pink): every seed as a faint line plus the across-seed mean
 * bold. Divergence between the indigo and pink means — sustained across seeds — is the
 * readable evidence that the feature changed the imagined dynamics. A single-seed gap
 * is NOT evidence (stochastic sampling); that's why all seeds are shown.
 */
export default function TrajectoryCharts({ result, frameIdx }: Props) {
  if (!result) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-[#0d0d20]">
        <span className="text-[10px] text-gray-600">Trajectory charts appear after a rollout</span>
      </div>
    );
  }
  return (
    <div className="w-full h-full flex bg-[#0d0d20] divide-x divide-gray-800 overflow-hidden">
      {/* Divergence: the robust "dynamics diverged" signal (always present, even when
          pixel state-extraction can't track objects on lossy imagined frames). */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="px-1 py-0.5 text-[9px] text-pink-400 uppercase tracking-wider flex-shrink-0">
          divergence (tokens Δ)
        </div>
        <div className="flex-1 min-h-0">
          <DivergenceChart
            values={result.token_divergence}
            yMax={result.n_obs_tokens}
            frameIdx={frameIdx}
          />
        </div>
      </div>
      {/* Pixel-state variables (Breakout-specific; may be sparse on lossy frames). */}
      {VARS.map((v) => (
        <div key={v.key} className="flex-1 flex flex-col min-w-0">
          <div className="px-1 py-0.5 text-[9px] text-gray-500 uppercase tracking-wider flex-shrink-0">
            {v.label}
          </div>
          <div className="flex-1 min-h-0">
            <OneChart result={result} varKey={v.key} frameIdx={frameIdx} />
          </div>
        </div>
      ))}
    </div>
  );
}

/** Single divergence line (token or pixel diff over rollout time) on a 0..yMax scale. */
function DivergenceChart({
  values,
  yMax,
  frameIdx,
}: {
  values: number[];
  yMax: number;
  frameIdx: number;
}) {
  const W = 140;
  const H = 90;
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;
  const n = values.length;
  const x = scaleLinear({ domain: [0, Math.max(n - 1, 1)], range: [0, iw] });
  const y = scaleLinear({ domain: [0, Math.max(yMax, 1)], range: [ih, 0] });
  const pts = values.map((v, t) => ({ t, v }));

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <Group left={M.left} top={M.top}>
        {pts.length >= 2 && (
          <LinePath data={pts} x={(p) => x(p.t)} y={(p) => y(p.v)} stroke={IV_COLOR} strokeWidth={1.8} />
        )}
        <Line
          from={{ x: x(Math.min(frameIdx, n - 1)), y: 0 }}
          to={{ x: x(Math.min(frameIdx, n - 1)), y: ih }}
          stroke="#4b5563"
          strokeWidth={0.5}
          strokeDasharray="2,2"
        />
      </Group>
    </svg>
  );
}

/** Extract one variable's per-seed series; null values become gaps (filtered per segment). */
function seriesFor(seeds: RolloutFrame[][], key: VarKey): (number | null)[][] {
  return seeds.map((steps) => steps.map((f) => f.state[key]));
}

/** Across-seed mean at each step, ignoring nulls (null if all seeds null at that step). */
function meanSeries(series: (number | null)[][]): (number | null)[] {
  const n = series[0]?.length ?? 0;
  const out: (number | null)[] = [];
  for (let t = 0; t < n; t++) {
    const vals = series.map((s) => s[t]).filter((x): x is number => x != null);
    out.push(vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null);
  }
  return out;
}

const M = { top: 4, right: 4, bottom: 4, left: 4 };

function OneChart({ result, varKey, frameIdx }: { result: RolloutResult; varKey: VarKey; frameIdx: number }) {
  const W = 140;
  const H = 90;
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;

  const base = useMemo(() => seriesFor(result.baseline, varKey), [result, varKey]);
  const iv = useMemo(() => seriesFor(result.intervened, varKey), [result, varKey]);
  const baseMean = useMemo(() => meanSeries(base), [base]);
  const ivMean = useMemo(() => meanSeries(iv), [iv]);

  const n = result.n_steps;
  const x = scaleLinear({ domain: [0, Math.max(n - 1, 1)], range: [0, iw] });
  // State variables are normalized [0,1]; fix the y domain so charts are comparable.
  const y = scaleLinear({ domain: [0, 1], range: [ih, 0] });

  // Render a polyline, splitting on nulls so gaps don't draw through missing data.
  const draw = (s: (number | null)[], color: string, width: number, opacity: number, key: string) => {
    const pts = s.map((v, t) => ({ t, v })).filter((p) => p.v != null) as { t: number; v: number }[];
    if (pts.length < 2) return null;
    return (
      <LinePath
        key={key}
        data={pts}
        x={(p) => x(p.t)}
        y={(p) => y(p.v)}
        stroke={color}
        strokeWidth={width}
        strokeOpacity={opacity}
      />
    );
  };

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <Group left={M.left} top={M.top}>
        {/* faint per-seed lines */}
        {base.map((s, i) => draw(s, BASE_COLOR, 0.6, 0.3, `b${i}`))}
        {iv.map((s, i) => draw(s, IV_COLOR, 0.6, 0.3, `i${i}`))}
        {/* bold means */}
        {draw(baseMean, BASE_COLOR, 1.8, 1, "bm")}
        {draw(ivMean, IV_COLOR, 1.8, 1, "im")}
        {/* playhead */}
        <Line
          from={{ x: x(Math.min(frameIdx, n - 1)), y: 0 }}
          to={{ x: x(Math.min(frameIdx, n - 1)), y: ih }}
          stroke="#4b5563"
          strokeWidth={0.5}
          strokeDasharray="2,2"
        />
      </Group>
    </svg>
  );
}
