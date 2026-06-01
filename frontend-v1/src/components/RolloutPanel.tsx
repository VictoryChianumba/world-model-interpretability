"use client";

import { useEffect, useRef, useState } from "react";
import type { RolloutResult } from "@/hooks/useRollout";

interface Props {
  result: RolloutResult | null;
  loading: boolean;
  error: string | null;
  /** Controlled scrub index (shared with the trajectory charts' playhead). */
  frameIdx: number;
  onFrameIdx: (i: number) => void;
}

/**
 * RolloutPanel — baseline vs intervened imagined rollouts, side by side at one frame.
 *
 * Scrubbable is the primary model: a slider moves both panes to the same step so you
 * can compare the two conditions frame-for-frame; a play button autoplays the short
 * sequence. Shows seed 0 in the viewer (representative); the trajectory charts below
 * use all seeds. Divergence over the rollout is the signal that the feature affected
 * the world model's imagined dynamics.
 */
export default function RolloutPanel({
  result,
  loading,
  error,
  frameIdx,
  onFrameIdx,
}: Props) {
  const [playing, setPlaying] = useState(false);
  const nSteps = result?.n_steps ?? 0;

  // Autoplay: advance the shared playhead ~4 fps, stop at the end.
  useEffect(() => {
    if (!playing || nSteps === 0) return;
    const id = setInterval(() => {
      onFrameIdx((frameIdxRef.current + 1) % nSteps);
    }, 250);
    return () => clearInterval(id);
  }, [playing, nSteps, onFrameIdx]);

  // Keep a ref so the interval reads the latest index without re-subscribing.
  const frameIdxRef = useRef(frameIdx);
  frameIdxRef.current = frameIdx;

  if (loading) {
    return (
      <Centered>
        <div className="flex flex-col items-center gap-2">
          <div className="w-6 h-6 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
          <span className="text-xs text-gray-400">Running rollout…</span>
          <span className="text-[9px] text-gray-600">~80s on CPU, less on MPS</span>
        </div>
      </Centered>
    );
  }
  if (error) {
    return <Centered><span className="text-xs text-red-400 px-4 text-center">{error}</span></Centered>;
  }
  if (!result) {
    return (
      <Centered>
        <span className="text-xs text-gray-600 px-4 text-center">
          Pause, pick a feature + scale, then Run rollout
        </span>
      </Centered>
    );
  }

  const i = Math.min(frameIdx, nSteps - 1);
  const base = result.baseline[0]?.[i]?.frame ?? null;
  const iv = result.intervened[0]?.[i]?.frame ?? null;

  return (
    <div className="w-full h-full flex flex-col bg-[#0d0d20] overflow-hidden">
      <div className="flex-1 flex divide-x divide-gray-800 overflow-hidden min-h-0">
        <Pane label="Baseline" frame={base} />
        <Pane label={`Intervened (×${result.scale.toFixed(0)})`} frame={iv} pink />
      </div>
      {/* Scrub + play */}
      <div className="flex items-center gap-2 px-2 py-1 border-t border-gray-800 flex-shrink-0 text-[10px]">
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          className="px-1.5 py-0.5 rounded border border-gray-700 hover:border-indigo-500 text-gray-300"
        >
          {playing ? "❚❚" : "▶"}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(nSteps - 1, 0)}
          value={i}
          onChange={(e) => {
            setPlaying(false);
            onFrameIdx(Number(e.target.value));
          }}
          className="flex-1 accent-indigo-500"
        />
        <span className="font-mono text-gray-400 w-12 text-right">
          {i + 1}/{nSteps}
        </span>
      </div>
    </div>
  );
}

function Pane({ label, frame, pink = false }: { label: string; frame: string | null; pink?: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/png;base64,${frame}`;
  }, [frame]);

  return (
    <div className="flex-1 flex flex-col min-w-0">
      <div className={`px-1 py-0.5 text-[9px] uppercase tracking-wider flex-shrink-0 ${pink ? "text-pink-400" : "text-gray-500"}`}>
        {label}
      </div>
      <div className="flex-1 flex items-center justify-center bg-black overflow-hidden min-h-0">
        {frame ? (
          <canvas ref={canvasRef} className="max-w-full max-h-full" style={{ imageRendering: "pixelated" }} />
        ) : (
          <span className="text-[9px] text-gray-700">—</span>
        )}
      </div>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="w-full h-full flex items-center justify-center bg-[#0d0d20]">{children}</div>
  );
}
