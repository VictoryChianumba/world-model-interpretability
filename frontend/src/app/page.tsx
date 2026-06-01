"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import ControlBar from "@/components/ControlBar";
import GameFrame from "@/components/GameFrame";
import ImaginedFramePanel from "@/components/ImaginedFramePanel";
import AttentionHeatmap from "@/components/AttentionHeatmap";
import MorphingGraph from "@/components/MorphingGraph";
import ActivationNorms from "@/components/ActivationNorms";
import SAEFeaturePanel from "@/components/SAEFeaturePanel";
import InterventionPanel from "@/components/InterventionPanel";
import LogPane from "@/components/LogPane";
import type { AgentInfo } from "@/hooks/useVisualizerSocket";
import { useVisualizerSocket } from "@/hooks/useVisualizerSocket";
import { useBookmarks } from "@/hooks/useBookmarks";

const API_BASE =
  typeof window !== "undefined"
    ? `http://${window.location.hostname}:8000`
    : "http://localhost:8000";

export default function Home() {
  const [selectedAgent, setSelectedAgent] = useState<string>("");
  const [selectedLayer, setSelectedLayer] = useState<number>(5);
  const [selectedDevice, setSelectedDevice] = useState<string>("cpu");
  const [availableDevices, setAvailableDevices] = useState<string[]>(["cpu"]);
  const [activeView, setActiveView] = useState<"heatmap" | "graph">("heatmap");
  const [ivFeatureId, setIvFeatureId] = useState<number | null>(null);
  // Magnitude-relative multiplier of the feature's own activation, injected on all
  // token positions. Comparable across features; default range ±20 (final range to
  // be tuned from a measured scale sweep after the SAE retrain). Default 5×.
  const [ivScale, setIvScale] = useState<number>(5);

  // Fetch available devices from the backend once on mount
  useEffect(() => {
    fetch(`${API_BASE}/devices`)
      .then((r) => r.json())
      .then((data: { available: string[]; default: string }) => {
        setAvailableDevices(data.available);
        setSelectedDevice(data.default);
      })
      .catch(() => {
        // Backend not ready yet — keep cpu default, will retry on reconnect
      });
  }, []);

  const { state, sendControl } = useVisualizerSocket(
    selectedAgent || undefined,
    undefined,
    selectedDevice,
  );

  // Once we receive the agent list from the backend, default-select the first
  useEffect(() => {
    if (state.config?.agents && state.config.agents.length > 0 && !selectedAgent) {
      setSelectedAgent(state.config.agents[0].id);
    }
  }, [state.config, selectedAgent]);

  // Clamp layer to valid range when config updates
  useEffect(() => {
    const numLayers = state.config?.num_layers;
    if (numLayers != null && numLayers > 0) {
      setSelectedLayer((l) => Math.max(0, Math.min(isNaN(l) ? 5 : l, numLayers - 1)));
    }
  }, [state.config]);

  const agents: AgentInfo[] = state.config?.agents ?? [];
  const numLayers = state.config?.num_layers ?? 10;
  const numHeads = state.config?.num_heads ?? 4;

  // Feature bookmarks for the current game + SAE layer (persist across sessions).
  const envId = agents.find((a) => a.id === selectedAgent)?.env_id;
  const { save: saveBookmark, labelFor } = useBookmarks(envId, state.sae_layer);

  function handleAgentChange(agent: AgentInfo) {
    setSelectedAgent(agent.id);
    // sendControl is called by ControlBar
  }

  // Intervention: select a feature (from the SAE panel) and set scale, then push
  // to the backend. The effect takes hold on the next Step (imagination is
  // step-mode only).
  function applyIntervention(featureId: number | null, scale: number) {
    setIvFeatureId(featureId);
    setIvScale(scale);
    sendControl({
      command: "set_intervention",
      payload: { feature_id: featureId, scale },
    });
  }

  function handleSelectFeature(id: number) {
    applyIntervention(id, ivScale);
  }

  function handleScaleChange(scale: number) {
    applyIntervention(ivFeatureId, scale);
  }

  function handleClearIntervention() {
    setIvFeatureId(null);
    sendControl({ command: "set_intervention", payload: { feature_id: null, scale: 0 } });
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* ── Top bar ─────────────────────────────────────────────────── */}
      <ControlBar
        agents={agents}
        selectedAgent={selectedAgent}
        onAgentChange={handleAgentChange}
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
          <Link
            href="/latent"
            className="text-xs text-gray-400 hover:text-indigo-300 transition-colors"
          >
            Latent View
          </Link>
        }
      />

      {/* ── Three-pane layout ────────────────────────────────────────── */}
      <main className="flex flex-1 overflow-hidden divide-x divide-gray-800">
        {/* Left: live game frame (top) + WM-imagined next frame (bottom) */}
        <div className="flex-1 flex flex-col min-w-0 divide-y divide-gray-800">
          <div className="flex-1 flex flex-col overflow-hidden">
            <PaneTitle>Game</PaneTitle>
            <div className="flex-1 overflow-hidden">
              <GameFrame frame={state.frame} loading={state.loading} />
            </div>
          </div>
          <div className="flex-1 flex flex-col overflow-hidden">
            <PaneTitle>Imagined Next Frame · Intervention</PaneTitle>
            <div className="flex-1 overflow-hidden">
              {ivFeatureId == null ? (
                <ImaginedFramePanel imaginedNext={state.imagined_next} />
              ) : (
                <InterventionPanel
                  baseline={state.imagined_next}
                  intervened={state.imagined_intervened}
                  diff={state.intervention_diff}
                  featureId={ivFeatureId}
                  scale={ivScale}
                  onScaleChange={handleScaleChange}
                  onClear={handleClearIntervention}
                  saeLoaded={state.sae_layer != null}
                  nChanged={state.intervention?.n_changed ?? null}
                  label={ivFeatureId != null ? labelFor(ivFeatureId) : undefined}
                  onSaveLabel={
                    ivFeatureId != null
                      ? (label: string) => saveBookmark(ivFeatureId, label)
                      : undefined
                  }
                />
              )}
            </div>
          </div>
        </div>

        {/* Middle: attention heatmap / graph toggle */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Custom header with view-mode toggle — layer slider in ControlBar
              controls both views; switching view never resets selectedLayer */}
          <div className="px-2 py-0.5 bg-gray-900 border-b border-gray-800 flex-shrink-0 flex items-center justify-between">
            <span className="text-[10px] text-gray-500 uppercase tracking-wider">
              Attention · Layer {selectedLayer}
            </span>
            <div className="flex items-center gap-0.5">
              <button
                data-testid="view-toggle-heatmap"
                className={`px-2 py-0.5 text-[9px] rounded transition-colors ${
                  activeView === "heatmap"
                    ? "bg-indigo-600 text-white"
                    : "text-gray-500 hover:text-gray-300"
                }`}
                onClick={() => setActiveView("heatmap")}
              >
                Heatmap
              </button>
              <button
                data-testid="view-toggle-graph"
                className={`px-2 py-0.5 text-[9px] rounded transition-colors ${
                  activeView === "graph"
                    ? "bg-indigo-600 text-white"
                    : "text-gray-500 hover:text-gray-300"
                }`}
                onClick={() => setActiveView("graph")}
              >
                Graph
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-hidden">
            {activeView === "heatmap" ? (
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

        {/* Right: norms + SAE features + log, stacked */}
        <div className="flex-1 flex flex-col min-w-0 divide-y divide-gray-800">
          <div className="flex-1 flex flex-col overflow-hidden">
            <PaneTitle>Activation Norms</PaneTitle>
            <div className="flex-1 overflow-hidden">
              <ActivationNorms norms={state.norms} numLayers={numLayers} />
            </div>
          </div>
          <div className="flex-1 flex flex-col overflow-hidden">
            <PaneTitle>SAE Features</PaneTitle>
            <div className="flex-1 overflow-hidden">
              <SAEFeaturePanel
                features={state.sae_features}
                layer={state.sae_layer}
                selectedId={ivFeatureId}
                onSelect={handleSelectFeature}
                labelFor={labelFor}
              />
            </div>
          </div>
          <div className="flex-1 flex flex-col overflow-hidden">
            <PaneTitle>Log</PaneTitle>
            <div className="flex-1 overflow-hidden">
              <LogPane metrics={state.metrics} events={state.events} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function PaneTitle({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-2 py-0.5 text-[10px] text-gray-500 bg-gray-900 border-b border-gray-800 flex-shrink-0 uppercase tracking-wider">
      {children}
    </div>
  );
}
