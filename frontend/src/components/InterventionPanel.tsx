"use client";

import { useEffect, useRef, useState } from "react";

interface Props {
  /** base64 PNG: WM-imagined next frame, no intervention. */
  baseline: string | null;
  /** base64 PNG: imagined next frame with the feature intervention applied. */
  intervened: string | null;
  /** base64 grayscale PNG: |baseline - intervened| per pixel. */
  diff: string | null;
  /** Currently targeted SAE feature id (null = none selected). */
  featureId: number | null;
  /** Intervention scale — a magnitude-relative multiplier of the feature's own activation. */
  scale: number;
  onScaleChange: (scale: number) => void;
  onClear: () => void;
  /** Whether a trained SAE is loaded (needed to intervene at all). */
  saeLoaded: boolean;
  /** How many of the 16 generated obs tokens the intervention flipped (or null). */
  nChanged?: number | null;
  /** Saved label for the targeted feature (from the bookmark store), if any. */
  label?: string;
  /** Persist a label for the targeted feature. */
  onSaveLabel?: (label: string) => void;
}

/**
 * InterventionPanel — the causal experiment over world-model dynamics.
 *
 * Pick a feature in the SAE panel, set a scale, and Step: the backend rolls the
 * world model forward twice from the current observation — once normally, once
 * with `scale * feature_direction` added to the residual — and we show baseline |
 * intervened | diff. This visualises how that feature steers the model's PREDICTED
 * next frame; it says nothing about the agent's behaviour (the policy never touches
 * the world model — see project notes).
 */
export default function InterventionPanel({
  baseline,
  intervened,
  diff,
  featureId,
  scale,
  onScaleChange,
  onClear,
  saeLoaded,
  nChanged,
  label,
  onSaveLabel,
}: Props) {
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState(false);

  // Reset the draft whenever the targeted feature or its saved label changes.
  useEffect(() => {
    setDraft(label ?? "");
    setEditing(false);
  }, [featureId, label]);

  return (
    <div className="w-full h-full flex flex-col bg-[#0d0d20] overflow-hidden">
      {/* Controls */}
      <div className="flex items-center gap-2 px-2 py-1 border-b border-gray-800 flex-shrink-0 text-[10px]">
        {!saeLoaded ? (
          <span className="text-gray-600">No SAE loaded — intervention unavailable</span>
        ) : featureId == null ? (
          <span className="text-gray-600">
            Click a feature in the SAE panel to target it, then Step
          </span>
        ) : (
          <>
            <span className="text-pink-300 font-mono">feature #{featureId}</span>
            <span className="text-gray-500">×</span>
            <input
              type="range"
              min={-20}
              max={20}
              step={1}
              value={scale}
              onChange={(e) => onScaleChange(Number(e.target.value))}
              className="flex-1 accent-pink-500 min-w-20"
            />
            <span className="font-mono text-gray-300 w-8 text-right">
              {scale.toFixed(0)}×
            </span>
            {nChanged != null && (
              <span
                className="font-mono text-gray-400 whitespace-nowrap"
                title="generated obs tokens changed vs baseline (of 16)"
              >
                Δ{nChanged}/16
              </span>
            )}
            <button
              type="button"
              onClick={onClear}
              className="text-gray-500 hover:text-gray-300 px-1"
              title="Clear intervention"
            >
              ✕
            </button>
          </>
        )}
      </div>

      {/* Label editor for the targeted feature (persists to bookmark store) */}
      {saeLoaded && featureId != null && onSaveLabel && (
        <div className="flex items-center gap-2 px-2 py-1 border-b border-gray-800 flex-shrink-0 text-[10px]">
          <span className="text-gray-500">label</span>
          {editing ? (
            <>
              <input
                type="text"
                value={draft}
                autoFocus
                placeholder="describe this feature…"
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    onSaveLabel(draft.trim());
                    setEditing(false);
                  } else if (e.key === "Escape") {
                    setDraft(label ?? "");
                    setEditing(false);
                  }
                }}
                className="flex-1 bg-gray-800 text-gray-100 rounded px-1.5 py-0.5 border border-gray-700 focus:outline-none focus:border-pink-500"
              />
              <button
                type="button"
                onClick={() => {
                  onSaveLabel(draft.trim());
                  setEditing(false);
                }}
                className="text-pink-300 hover:text-pink-200 px-1"
              >
                save
              </button>
            </>
          ) : (
            <>
              <span className={`flex-1 truncate ${label ? "text-gray-200" : "text-gray-600 italic"}`}>
                {label || "unlabeled"}
              </span>
              <button
                type="button"
                onClick={() => setEditing(true)}
                className="text-gray-400 hover:text-gray-200 px-1"
              >
                {label ? "edit" : "add label"}
              </button>
            </>
          )}
        </div>
      )}

      {/* Three sub-panes */}
      <div className="flex-1 flex divide-x divide-gray-800 overflow-hidden">
        <FramePane label="Baseline" frame={baseline} />
        <FramePane label="Intervened" frame={intervened} grayscale={false} />
        <FramePane label="Diff" frame={diff} grayscale />
      </div>
    </div>
  );
}

function FramePane({
  label,
  frame,
  grayscale = false,
}: {
  label: string;
  frame: string | null;
  grayscale?: boolean;
}) {
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
      <div className="px-1 py-0.5 text-[9px] text-gray-500 uppercase tracking-wider flex-shrink-0">
        {label}
      </div>
      <div className="flex-1 flex items-center justify-center bg-black overflow-hidden">
        {frame ? (
          <canvas
            ref={canvasRef}
            className="max-w-full max-h-full"
            style={{ imageRendering: "pixelated" }}
          />
        ) : (
          <span className="text-[9px] text-gray-700 px-2 text-center">
            {grayscale ? "—" : "Step to compute"}
          </span>
        )}
      </div>
    </div>
  );
}
