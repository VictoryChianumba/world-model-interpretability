"use client";

/**
 * /latent — Latent Space Reconstruction page.
 *
 * Shows three panes side by side:
 *   Left   – raw game frame from the Atari environment
 *   Middle – tokenizer reconstruction (decoded from predicted obs tokens)
 *   Right  – per-pixel error map, rendered as a diverging heatmap
 *
 * Below: metrics bar (reconstruction error, episode, step, fps).
 * Uses the same useVisualizerSocket hook as the main page — no duplicate
 * connection logic.
 */

import Link from "next/link";
import GameFrame from "@/components/GameFrame";
import ErrorMapHeatmap from "@/components/ErrorMapHeatmap";
import { useVisualizerSocket } from "@/hooks/useVisualizerSocket";

export default function LatentPage() {
  // Connect to /ws/latent so the backend knows to compute reconstruction decode.
  const { state } = useVisualizerSocket(undefined, undefined, undefined, "/ws/latent");

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* ── Header ───────────────────────────────────────────────── */}
      <header className="flex items-center gap-4 px-4 py-2 bg-gray-900 border-b border-gray-800 flex-shrink-0">
        <span className="text-xs font-semibold text-gray-200 tracking-wide uppercase">
          Latent Space Reconstruction
        </span>
        <div className="flex-1" />
        {/* Connection indicator */}
        <div className="flex items-center gap-1.5">
          <div
            className={`w-2 h-2 rounded-full ${
              state.connected ? "bg-green-400 animate-pulse" : "bg-red-500"
            }`}
          />
          <span className="text-xs text-gray-400">
            {state.loading ? "Loading…" : state.connected ? "Connected" : "Disconnected"}
          </span>
        </div>
        <Link
          href="/"
          className="text-xs text-gray-400 hover:text-indigo-300 transition-colors"
        >
          ← Main View
        </Link>
      </header>

      {/* ── Three-pane layout ────────────────────────────────────── */}
      <main
        data-testid="latent-panes"
        className="flex flex-1 overflow-hidden divide-x divide-gray-800"
      >
        {/* Left: raw game frame */}
        <div
          data-testid="pane-raw"
          className="flex-1 flex flex-col min-w-0"
        >
          <PaneTitle>Raw Frame</PaneTitle>
          <div className="flex-1 overflow-hidden">
            <GameFrame frame={state.frame} loading={state.loading} />
          </div>
        </div>

        {/* Middle: reconstruction */}
        <div
          data-testid="pane-reconstruction"
          className="flex-1 flex flex-col min-w-0"
        >
          <PaneTitle>Reconstruction</PaneTitle>
          <div className="flex-1 overflow-hidden relative">
            {state.reconstruction != null ? (
              <GameFrame frame={state.reconstruction} loading={false} />
            ) : (
              <Placeholder text="Waiting for data…" />
            )}
          </div>
        </div>

        {/* Right: error map */}
        <div
          data-testid="pane-error-map"
          className="flex-1 flex flex-col min-w-0"
        >
          <PaneTitle>Error Map</PaneTitle>
          <div className="flex-1 overflow-hidden relative">
            <ErrorMapHeatmap errorMapB64={state.error_map} />
          </div>
        </div>
      </main>

      {/* ── Metrics bar ──────────────────────────────────────────── */}
      <footer className="flex items-center gap-6 px-4 py-1.5 bg-gray-900 border-t border-gray-800 flex-shrink-0 text-xs text-gray-400">
        <MetricItem
          label="Recon MAE"
          value={
            state.reconstruction_error != null
              ? state.reconstruction_error.toFixed(1)
              : "—"
          }
        />
        <MetricItem
          label="Episode"
          value={state.metrics?.episode != null ? String(state.metrics.episode) : "—"}
        />
        <MetricItem
          label="Step"
          value={state.metrics?.step != null ? String(state.metrics.step) : "—"}
        />
        <MetricItem
          label="FPS"
          value={state.metrics?.infer_fps != null ? String(state.metrics.infer_fps) : "—"}
        />
      </footer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function PaneTitle({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-2 py-0.5 text-[10px] text-gray-500 bg-gray-900 border-b border-gray-800 flex-shrink-0 uppercase tracking-wider">
      {children}
    </div>
  );
}

function Placeholder({ text }: { text: string }) {
  return (
    <div className="w-full h-full flex items-center justify-center bg-black">
      <span className="text-xs text-gray-600">{text}</span>
    </div>
  );
}

function MetricItem({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className="text-gray-600">{label}</span>
      <span className="font-mono text-gray-300">{value}</span>
    </span>
  );
}
