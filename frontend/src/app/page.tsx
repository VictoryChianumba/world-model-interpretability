"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import ControlBar from "@/components/ControlBar";
import GameFrame from "@/components/GameFrame";
import SearchBar from "@/components/SearchBar";
import DiscoveryPanel from "@/components/DiscoveryPanel";
import RolloutComparison from "@/components/RolloutComparison";
import ModelInternals from "@/components/ModelInternals";
import type { CardVM } from "@/components/FeatureCanvas";
import type { AgentInfo, ControlCommand } from "@/hooks/useVisualizerSocket";
import { useVisualizerSocket } from "@/hooks/useVisualizerSocket";
import { useBookmarks } from "@/hooks/useBookmarks";
import { useRollout, type Intervention } from "@/hooks/useRollout";
import { usePinned } from "@/hooks/usePinned";
import { useFeatureIndex } from "@/hooks/useFeatureIndex";
import { useActivationHistory } from "@/hooks/useActivationHistory";
import { useFeatureRanking, type RankingMetric } from "@/hooks/useFeatureRanking";
import { API_BASE } from "@/lib/config";

// react-konva touches `window` at import → load the canvas client-side only.
const FeatureCanvas = dynamic(() => import("@/components/FeatureCanvas"), {
  ssr: false,
  loading: () => <div className="h-full w-full rounded-lg bg-[#161629]" />,
});

export default function Home() {
  const [selectedAgent, setSelectedAgent] = useState<string>("");
  const [selectedLayer, setSelectedLayer] = useState<number>(5); // attention layer (internals)
  const [selectedDevice, setSelectedDevice] = useState<string>("cpu");
  const [availableDevices, setAvailableDevices] = useState<string[]>(["cpu"]);
  const [view, setView] = useState<"canvas" | "internals">("canvas");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [paused, setPaused] = useState(false);
  const [rankingMetric, setRankingMetric] = useState<RankingMetric>("firing");

  useEffect(() => {
    fetch(`${API_BASE}/devices`)
      .then((r) => r.json())
      .then((data: { available: string[]; default: string }) => {
        setAvailableDevices(data.available);
        setSelectedDevice(data.default);
      })
      .catch(() => {});
  }, []);

  const { state, sendControl: rawSendControl } = useVisualizerSocket(
    selectedAgent || undefined,
    undefined,
    selectedDevice,
  );

  // Track pause locally (the socket doesn't surface it) so the rollout button can gate on
  // it — the experiment is paused-only. ControlBar's buttons go through this wrapper too.
  const sendControl = useCallback(
    async (cmd: ControlCommand) => {
      if (cmd.command === "pause") setPaused(true);
      else if (cmd.command === "resume" || cmd.command === "restart") setPaused(false);
      // "step" leaves paused === true.
      await rawSendControl(cmd);
    },
    [rawSendControl],
  );

  const { result: rollout, loading: rolloutLoading, error: rolloutError, runMulti } = useRollout();

  useEffect(() => {
    if (state.config?.agents && state.config.agents.length > 0 && !selectedAgent) {
      setSelectedAgent(state.config.agents[0].id);
    }
  }, [state.config, selectedAgent]);

  useEffect(() => {
    const numLayers = state.config?.num_layers;
    if (numLayers != null && numLayers > 0) {
      setSelectedLayer((l) => Math.max(0, Math.min(isNaN(l) ? 5 : l, numLayers - 1)));
    }
  }, [state.config]);

  const agents: AgentInfo[] = state.config?.agents ?? [];
  const numLayers = state.config?.num_layers ?? 10;
  const envId = agents.find((a) => a.id === selectedAgent)?.env_id;
  const saeLayer = state.sae_layer;

  // Pinned cards (canvas), autointerp labels (search/labels), bookmark labels (fallback),
  // and per-feature firing history (sparklines).
  const { pins, pin, unpin, update, isPinned } = usePinned(envId, saeLayer);
  const { features, loaded: indexLoaded, labelFor: autoLabelFor, search } = useFeatureIndex(saeLayer);
  const { labelFor: bookmarkLabelFor } = useBookmarks(envId, saeLayer);
  const { historyFor, currentFor } = useActivationHistory(state.sae_features, state.metrics?.step);
  const ranking = useFeatureRanking(rankingMetric, state.sae_features, saeLayer);

  // Resolve a card's display label: user override → autointerp → bookmark → null.
  const resolveLabel = useCallback(
    (id: number, custom: string | null): string | null =>
      custom ?? autoLabelFor(id) ?? bookmarkLabelFor(id) ?? null,
    [autoLabelFor, bookmarkLabelFor],
  );

  const cards: CardVM[] = useMemo(
    () =>
      pins.map((p) => ({
        featureId: p.feature_id,
        x: p.x,
        y: p.y,
        label: resolveLabel(p.feature_id, p.custom_label),
        customLabel: p.custom_label,
        activation: currentFor(p.feature_id),
        maxActivation: features[p.feature_id]?.max_activation ?? 4,
        history: historyFor(p.feature_id),
        interventionScale: p.intervention_scale,
      })),
    [pins, resolveLabel, currentFor, features, historyFor],
  );

  // Discovery-panel labels use the same resolution (sans per-card custom override).
  const discoveryLabelFor = useCallback(
    (id: number) => autoLabelFor(id) ?? bookmarkLabelFor(id) ?? null,
    [autoLabelFor, bookmarkLabelFor],
  );

  // The rollout intervention is read straight off the canvas: cards with non-zero scale.
  const interventions: Intervention[] = useMemo(
    () =>
      pins
        .filter((p) => p.intervention_scale !== 0)
        .map((p) => ({ feature_id: p.feature_id, scale: p.intervention_scale })),
    [pins],
  );

  const steeringCount = interventions.length;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#0b0b16] text-slate-100">
      <ControlBar
        agents={agents}
        selectedAgent={selectedAgent}
        onAgentChange={(a) => setSelectedAgent(a.id)}
        availableDevices={availableDevices}
        selectedDevice={selectedDevice}
        onDeviceChange={setSelectedDevice}
        selectedLayer={selectedLayer}
        maxLayer={numLayers - 1}
        onLayerChange={setSelectedLayer}
        connected={state.connected}
        loading={state.loading}
        sendControl={sendControl}
        navSlot={
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-0.5">
              {(["canvas", "internals"] as const).map((v) => (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  className={`rounded px-2 py-0.5 text-[10px] capitalize transition-colors ${
                    view === v ? "bg-indigo-600 text-white" : "text-slate-500 hover:text-slate-300"
                  }`}
                >
                  {v === "internals" ? "Model internals" : "Canvas"}
                </button>
              ))}
            </div>
            <Link href="/latent" className="text-xs text-slate-400 transition-colors hover:text-indigo-300">
              Latent View
            </Link>
          </div>
        }
      />

      {view === "canvas" ? (
        <main className="flex min-h-0 flex-1">
          {/* Sidebar: live frame (context) + discovery panel (find candidates) */}
          {sidebarOpen && (
            <aside className="flex w-64 flex-shrink-0 flex-col border-r border-slate-800">
              <div className="flex-shrink-0 border-b border-slate-800">
                <div className="px-3 py-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Live {paused && <span className="text-amber-400">· paused</span>}
                </div>
                <div className="h-40">
                  <GameFrame frame={state.frame} loading={state.loading} />
                </div>
                {state.metrics && (
                  <div className="flex justify-between px-3 py-1 font-mono text-[10px] text-slate-500">
                    <span>step {state.metrics.step}</span>
                    <span>ep {state.metrics.episode}</span>
                    <span>{state.metrics.infer_fps?.toFixed(0)} fps</span>
                  </div>
                )}
              </div>
              <div className="min-h-0 flex-1">
                <DiscoveryPanel
                  metric={rankingMetric}
                  onMetricChange={setRankingMetric}
                  items={ranking.items}
                  available={ranking.available}
                  note={ranking.note}
                  labelFor={discoveryLabelFor}
                  onPin={pin}
                  isPinned={isPinned}
                  paused={paused}
                />
              </div>
            </aside>
          )}

          {/* Main: search + canvas (primary) + rollout comparison (experiment) */}
          <section className="flex min-w-0 flex-1 flex-col">
            <div className="flex flex-shrink-0 items-center gap-3 border-b border-slate-800 px-3 py-2">
              <button
                onClick={() => setSidebarOpen((s) => !s)}
                title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
                className="rounded border border-slate-700 px-1.5 py-0.5 text-xs text-slate-400 hover:text-slate-200"
              >
                {sidebarOpen ? "⟨" : "⟩"}
              </button>
              <SearchBar search={search} onPin={pin} isPinned={isPinned} indexLoaded={indexLoaded} />
              <div className="ml-auto text-[11px] text-slate-500">
                {pins.length} card{pins.length === 1 ? "" : "s"}
                {steeringCount > 0 && <span className="text-pink-400"> · {steeringCount} steering</span>}
                {saeLayer != null && <span> · SAE L{saeLayer}</span>}
              </div>
            </div>

            <div className="min-h-0 flex-[3] p-2">
              <FeatureCanvas
                cards={cards}
                onMove={(id, x, y, persist) => update(id, { x, y }, persist)}
                onScale={(id, scale, persist) => update(id, { intervention_scale: scale }, persist)}
                onRemove={unpin}
                onRelabel={(id, label) => update(id, { custom_label: label })}
              />
            </div>

            <div className="min-h-0 flex-[2] border-t border-slate-800">
              <RolloutComparison
                interventions={interventions}
                paused={paused}
                result={rollout}
                loading={rolloutLoading}
                error={rolloutError}
                onRun={(nSteps, nSeeds) => runMulti(interventions, nSteps, nSeeds)}
              />
            </div>
          </section>
        </main>
      ) : (
        <main className="min-h-0 flex-1">
          <ModelInternals state={state} selectedLayer={selectedLayer} />
        </main>
      )}
    </div>
  );
}
