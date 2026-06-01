"use client";

import { useState } from "react";
import AttentionHeatmap from "@/components/AttentionHeatmap";
import MorphingGraph from "@/components/MorphingGraph";
import ActivationNorms from "@/components/ActivationNorms";
import LogPane from "@/components/LogPane";
import type { VisualizerState } from "@/hooks/useVisualizerSocket";

interface Props {
  state: VisualizerState;
  selectedLayer: number;
}

/**
 * ModelInternals — the demoted attention / norms / log views.
 *
 * These were the centre of v1 but are not part of the v2 redesign's main flow, so they
 * live behind this default-hidden tab as reference instrumentation rather than the primary
 * surface.
 */
export default function ModelInternals({ state, selectedLayer }: Props) {
  const [attnView, setAttnView] = useState<"heatmap" | "graph">("heatmap");
  const numLayers = state.config?.num_layers ?? 0;
  const numHeads = state.config?.num_heads ?? 0;

  return (
    <div className="flex h-full min-h-0 gap-3 p-3">
      <div className="flex min-h-0 flex-1 flex-col rounded-lg border border-slate-800 bg-[#0d0d20]">
        <div className="flex items-center gap-2 border-b border-slate-800 px-3 py-1.5">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Attention · layer {selectedLayer}
          </span>
          <div className="ml-auto flex gap-1">
            {(["heatmap", "graph"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setAttnView(v)}
                className={`rounded px-2 py-0.5 text-[10px] ${
                  attnView === v ? "bg-indigo-600 text-white" : "bg-slate-800 text-slate-400"
                }`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
        <div className="min-h-0 flex-1 overflow-auto">
          {attnView === "heatmap" ? (
            <AttentionHeatmap
              attention={state.attention}
              selectedLayer={selectedLayer}
              tokenLayout={state.token_layout}
              numHeads={numHeads}
            />
          ) : (
            <MorphingGraph
              attention={state.attention}
              selectedLayer={selectedLayer}
              tokenLayout={state.token_layout}
            />
          )}
        </div>
      </div>

      <div className="flex w-80 flex-shrink-0 flex-col gap-3">
        <div className="rounded-lg border border-slate-800 bg-[#0d0d20] p-2">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Residual norms
          </div>
          <ActivationNorms norms={state.norms} numLayers={numLayers} />
        </div>
        <div className="min-h-0 flex-1 overflow-hidden rounded-lg border border-slate-800 bg-[#0d0d20]">
          <LogPane metrics={state.metrics} events={state.events} />
        </div>
      </div>
    </div>
  );
}
