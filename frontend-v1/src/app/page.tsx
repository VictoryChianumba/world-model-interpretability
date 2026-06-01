"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import ControlBar from "@/components/ControlBar";
import GameFrame from "@/components/GameFrame";
import AttentionHeatmap from "@/components/AttentionHeatmap";
import MorphingGraph from "@/components/MorphingGraph";
import ActivationNorms from "@/components/ActivationNorms";
import SAEFeaturePanel from "@/components/SAEFeaturePanel";
import RolloutPanel from "@/components/RolloutPanel";
import TrajectoryCharts from "@/components/TrajectoryCharts";
import LogPane from "@/components/LogPane";
import type { AgentInfo } from "@/hooks/useVisualizerSocket";
import { useVisualizerSocket } from "@/hooks/useVisualizerSocket";
import { useBookmarks } from "@/hooks/useBookmarks";
import { useRollout } from "@/hooks/useRollout";

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
  // Top-level view: the intervention experiment is primary; attention/norms are demoted.
  const [mainView, setMainView] = useState<"experiment" | "analysis">("experiment");
  const [ivFeatureId, setIvFeatureId] = useState<number | null>(null);
  // Magnitude-relative multiplier of the feature's own activation (±20, default 5×).
  const [ivScale, setIvScale] = useState<number>(5);
  // Shared scrub index between the rollout viewer and the trajectory charts.
  const [frameIdx, setFrameIdx] = useState<number>(0);

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
  const { result: rollout, loading: rolloutLoading, error: rolloutError, run: runRollout } =
    useRollout();

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
  const { labelFor } = useBookmarks(envId, state.sae_layer);

  function handleAgentChange(agent: AgentInfo) {
    setSelectedAgent(agent.id);
    // sendControl is called by ControlBar
  }

  // Selecting a feature / changing scale just updates local experiment state; the
  // rollout endpoint reads these when "Run rollout" is clicked (paused-only).
  function handleSelectFeature(id: number) {
    setIvFeatureId(id);
  }

  function handleRunRollout() {
    if (ivFeatureId == null) return;
    setFrameIdx(0);
    runRollout(ivFeatureId, ivScale);
  }

  const saeLoaded = state.sae_layer != null;

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
          <div className="flex items-center gap-2">
            {/* Primary view switch: Experiment (intervention rollouts) vs Analysis (attention/norms) */}
            <div className="flex items-center gap-0.5">
              <button
                data-testid="main-view-experiment"
                className={`px-2 py-0.5 text-[9px] rounded transition-colors ${
                  mainView === "experiment" ? "bg-indigo-600 text-white" : "text-gray-500 hover:text-gray-300"
                }`}
                onClick={() => setMainView("experiment")}
              >
                Experiment
              </button>
              <button
                data-testid="main-view-analysis"
                className={`px-2 py-0.5 text-[9px] rounded transition-colors ${
                  mainView === "analysis" ? "bg-indigo-600 text-white" : "text-gray-500 hover:text-gray-300"
                }`}
                onClick={() => setMainView("analysis")}
              >
                Analysis
              </button>
            </div>
            <Link href="/latent" className="text-xs text-gray-400 hover:text-indigo-300 transition-colors">
              Latent View
            </Link>
          </div>
        }
      />

      {mainView === "experiment" ? (
        /* ── Experiment: the intervention-rollout column + live frame + features ── */
        <main className="flex flex-1 overflow-hidden divide-x divide-gray-800">
          {/* Left: live game frame + SAE feature list */}
          <div className="w-1/4 min-w-56 flex flex-col divide-y divide-gray-800">
            <div className="flex-1 flex flex-col overflow-hidden">
              <PaneTitle>Game</PaneTitle>
              <div className="flex-1 overflow-hidden">
                <GameFrame frame={state.frame} loading={state.loading} />
              </div>
            </div>
            <div className="flex-1 flex flex-col overflow-hidden">
              <PaneTitle>SAE Features {state.sae_layer != null ? `· layer ${state.sae_layer}` : ""}</PaneTitle>
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
          </div>

          {/* Right: the experiment column — controls → rollout → trajectories */}
          <div className="flex-1 flex flex-col min-w-0 divide-y divide-gray-800">
            {/* Intervention controls + Run */}
            <div className="flex items-center gap-3 px-3 py-2 bg-gray-900 flex-shrink-0 text-xs">
              {!saeLoaded ? (
                <span className="text-gray-600">No SAE loaded — intervention rollouts unavailable</span>
              ) : ivFeatureId == null ? (
                <span className="text-gray-600">Pick a feature on the left to target it</span>
              ) : (
                <>
                  <span className="text-pink-300 font-mono">
                    feature #{ivFeatureId}
                    {labelFor(ivFeatureId) ? ` · ${labelFor(ivFeatureId)}` : ""}
                  </span>
                  <span className="text-gray-500">scale ×</span>
                  <input
                    type="range"
                    min={-20}
                    max={20}
                    step={1}
                    value={ivScale}
                    onChange={(e) => setIvScale(Number(e.target.value))}
                    className="flex-1 max-w-48 accent-pink-500"
                  />
                  <span className="font-mono text-gray-300 w-8 text-right">{ivScale.toFixed(0)}×</span>
                  <button
                    type="button"
                    onClick={handleRunRollout}
                    disabled={rolloutLoading || !state.connected}
                    className="px-3 py-1 rounded text-xs font-medium border border-pink-700 text-pink-200 hover:border-pink-500 disabled:opacity-40"
                    title="Pause first; runs a paused-only N-step rollout (~80s CPU)"
                  >
                    {rolloutLoading ? "Running…" : "Run rollout"}
                  </button>
                  <span className="text-[9px] text-gray-600">pause first</span>
                </>
              )}
            </div>

            {/* Rollout comparison (scrubbable side-by-side) */}
            <div className="flex-[3] min-h-0 overflow-hidden">
              <RolloutPanel
                result={rollout}
                loading={rolloutLoading}
                error={rolloutError}
                frameIdx={frameIdx}
                onFrameIdx={setFrameIdx}
              />
            </div>

            {/* Trajectory charts */}
            <div className="flex-[2] min-h-0 overflow-hidden">
              <PaneTitle>Trajectories · baseline vs intervened (per-seed faint, mean bold)</PaneTitle>
              <div className="flex-1 overflow-hidden h-full">
                <TrajectoryCharts result={rollout} frameIdx={frameIdx} />
              </div>
            </div>
          </div>
        </main>
      ) : (
        /* ── Analysis: attention/norms/log (demoted, internals unchanged) ── */
        <main className="flex flex-1 overflow-hidden divide-x divide-gray-800">
          <div className="flex-1 flex flex-col min-w-0">
            <div className="px-2 py-0.5 bg-gray-900 border-b border-gray-800 flex-shrink-0 flex items-center justify-between">
              <span className="text-[10px] text-gray-500 uppercase tracking-wider">
                Attention · Layer {selectedLayer}
              </span>
              <div className="flex items-center gap-0.5">
                <button
                  data-testid="view-toggle-heatmap"
                  className={`px-2 py-0.5 text-[9px] rounded transition-colors ${
                    activeView === "heatmap" ? "bg-indigo-600 text-white" : "text-gray-500 hover:text-gray-300"
                  }`}
                  onClick={() => setActiveView("heatmap")}
                >
                  Heatmap
                </button>
                <button
                  data-testid="view-toggle-graph"
                  className={`px-2 py-0.5 text-[9px] rounded transition-colors ${
                    activeView === "graph" ? "bg-indigo-600 text-white" : "text-gray-500 hover:text-gray-300"
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

          <div className="flex-1 flex flex-col min-w-0 divide-y divide-gray-800">
            <div className="flex-1 flex flex-col overflow-hidden">
              <PaneTitle>Activation Norms</PaneTitle>
              <div className="flex-1 overflow-hidden">
                <ActivationNorms norms={state.norms} numLayers={numLayers} />
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
      )}
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
