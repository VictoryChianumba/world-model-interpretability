"use client";

import { useEffect, useRef } from "react";

interface Props {
  /** base64 PNG of the world-model-predicted next frame, or null. */
  imaginedNext: string | null;
}

/**
 * ImaginedFramePanel — renders the world model's predicted *next* frame.
 *
 * Unlike GameFrame (the live observation) and the tokenizer reconstruction (an
 * autoencoder round-trip of the CURRENT frame), this is a genuine WM prediction:
 * the autoregressive rollout from the current observation + chosen action. It is
 * only produced while single-stepping (paused), so the empty state nudges the
 * user toward the Step control.
 */
export default function ImaginedFramePanel({ imaginedNext }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imaginedNext) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.imageSmoothingEnabled = false; // preserve pixel-art clarity
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/png;base64,${imaginedNext}`;
  }, [imaginedNext]);

  return (
    <div className="relative flex items-center justify-center w-full h-full bg-black">
      <canvas
        ref={canvasRef}
        className="max-w-full max-h-full"
        style={{ imageRendering: "pixelated" }}
      />
      {!imaginedNext && (
        <div className="absolute inset-0 flex items-center justify-center px-4 text-center">
          <span className="text-xs text-gray-600">
            Pause, then Step ▶| to predict the next frame
          </span>
        </div>
      )}
    </div>
  );
}
