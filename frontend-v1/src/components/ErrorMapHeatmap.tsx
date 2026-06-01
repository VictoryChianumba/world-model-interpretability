"use client";

/**
 * ErrorMapHeatmap — renders a per-pixel reconstruction error map.
 *
 * The backend sends a grayscale PNG (values 0-255) via base64.
 * We decode it with a canvas, then hand the 2-D array to visx's HeatmapRect
 * with a diverging blue→yellow→red colour scale so low errors appear cool
 * and high errors appear warm.
 */

import { useEffect, useState } from "react";
import { Group } from "@visx/group";
import { HeatmapRect } from "@visx/heatmap";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type BinDatum = { count: number };
type ColumnDatum = { bins: BinDatum[] };

// ---------------------------------------------------------------------------
// Colour scale  (diverging: blue → light-yellow → red)
// ---------------------------------------------------------------------------

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * Math.max(0, Math.min(1, t));
}

/**
 * Diverging colour: #4575b4 (blue, 0) → #ffffbf (light-yellow, 0.5) → #d73027 (red, 1).
 * This is the classic RdBu_r ramp used in scientific visualisation.
 */
// Accepts visx's NumberLike (number | { valueOf(): number }) so it satisfies the
// HeatmapRect `colorScale` prop type, then coerces to a plain number.
function errorColor(count: number | { valueOf(): number }): string {
  const t = Number(count) / 255;
  if (t <= 0.5) {
    const s = t * 2;
    return `rgb(${Math.round(lerp(69, 255, s))},${Math.round(lerp(117, 255, s))},${Math.round(lerp(180, 191, s))})`;
  }
  const s = (t - 0.5) * 2;
  return `rgb(${Math.round(lerp(255, 215, s))},${Math.round(lerp(255, 48, s))},${Math.round(lerp(191, 39, s))})`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  errorMapB64: string | null;
}

interface DecodedMap {
  data: ColumnDatum[];
  W: number;
  H: number;
}

export default function ErrorMapHeatmap({ errorMapB64 }: Props) {
  const [decoded, setDecoded] = useState<DecodedMap | null>(null);

  useEffect(() => {
    if (!errorMapB64) {
      setDecoded(null);
      return;
    }

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      const W = img.naturalWidth;
      const H = img.naturalHeight;
      canvas.width = W;
      canvas.height = H;
      ctx.drawImage(img, 0, 0);
      const { data: px } = ctx.getImageData(0, 0, W, H);

      // Build column-major data for HeatmapRect: data[col].bins[row].count
      const columns: ColumnDatum[] = Array.from({ length: W }, (_, x) => ({
        bins: Array.from({ length: H }, (_, y) => ({
          count: px[(y * W + x) * 4], // R channel = error value (grayscale)
        })),
      }));

      setDecoded({ data: columns, W, H });
    };
    img.src = `data:image/png;base64,${errorMapB64}`;
  }, [errorMapB64]);

  if (!decoded) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-[#0d0d20]">
        <span className="text-xs text-gray-600">Waiting for data…</span>
      </div>
    );
  }

  const { data, W, H } = decoded;

  // Fill the container — use a fixed SVG size that the CSS scales down/up
  const svgW = W * 4; // 4px per pixel for small Atari frames (e.g. 64×64 → 256px)
  const svgH = H * 4;
  const binW = svgW / W;
  const binH = svgH / H;

  // xScale called with column index, yScale called with row index
  const xScale = (col: number) => col * binW;
  const yScale = (row: number) => row * binH;

  return (
    <div className="w-full h-full flex items-center justify-center bg-[#0d0d20] overflow-hidden">
      {/* Legend */}
      <div className="absolute top-1 left-2 flex items-center gap-1.5 z-10 pointer-events-none">
        <span className="text-[9px] text-gray-600">Recon error</span>
        <span
          className="inline-block w-12 h-2 rounded-sm"
          style={{
            background:
              "linear-gradient(to right, #4575b4, #ffffbf, #d73027)",
          }}
        />
        <span className="text-[9px] text-gray-600">high</span>
      </div>
      <svg
        width={svgW}
        height={svgH}
        className="max-w-full max-h-full"
        style={{ imageRendering: "pixelated" }}
      >
        <Group>
          <HeatmapRect
            data={data}
            xScale={xScale}
            yScale={yScale}
            colorScale={errorColor}
            binWidth={binW}
            binHeight={binH}
            gap={0}
          />
        </Group>
      </svg>
    </div>
  );
}
