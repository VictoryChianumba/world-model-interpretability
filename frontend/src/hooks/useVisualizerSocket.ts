"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TokenLayout {
  tokens_per_block: number;
  obs_per_block: number;
  labels: string[];
}

export interface Metrics {
  infer_fps: number;
  step: number;
  episode: number;
  queue_depth: number;
  hook_latency_ms: number;
  drop_rate: number;
  return: number;
}

export interface ModelConfig {
  num_layers: number;
  num_heads: number;
  embed_dim: number;
  tokens_per_block: number;
  max_blocks: number;
}

export interface AgentInfo {
  id: string;
  name: string;
  path: string;
  env_id: string;
}

export interface SAEFeature {
  id: number;       // feature index in the SAE dictionary
  mag: number;      // activation magnitude (post-ReLU)
}

export interface FrameData {
  frame_index: number;                            // monotonic step id; frame + activations
                                                  // in this message describe the same step
  frame: string;                                  // base64 PNG
  attention: Record<string, number[][][]>;        // layer_idx → [nh][T_q][T_k]
  norms: number[];                                // per layer
  metrics: Metrics;
  token_layout: TokenLayout;
  reconstruction: string | null;                  // base64 PNG or null
  error_map: string | null;                       // base64 grayscale PNG or null
  reconstruction_error: number | null;            // MAE in [0, 255] or null
  imagined_next: string | null;                   // base64 PNG: WM-predicted next frame (step mode only)
  sae_features: SAEFeature[] | null;              // top-K firing features this frame, or null
  sae_layer: number | null;                       // WM layer the SAE reads, or null if no SAE
  imagined_intervened: string | null;             // base64 PNG: intervened next frame (step mode)
  intervention_diff: string | null;               // base64 grayscale PNG: |baseline - intervened|
  intervention: { feature_id: number; scale: number; n_changed: number | null } | null;  // active intervention, or null
}

export interface EventMessage {
  id: number;
  timestamp: string;
  event: string;
  data: Record<string, unknown>;
}

export interface VisualizerState {
  connected: boolean;
  loading: boolean;
  frame_index: number | null;
  frame: string | null;
  attention: Record<string, number[][][]> | null;
  norms: number[] | null;
  metrics: Metrics | null;
  token_layout: TokenLayout | null;
  config: (ModelConfig & { agents: AgentInfo[] }) | null;
  events: EventMessage[];
  reconstruction: string | null;
  error_map: string | null;
  reconstruction_error: number | null;
  imagined_next: string | null;
  sae_features: SAEFeature[] | null;
  sae_layer: number | null;
  imagined_intervened: string | null;
  intervention_diff: string | null;
  intervention: { feature_id: number; scale: number; n_changed: number | null } | null;
}

export interface ControlCommand {
  command:
    | "loop"
    | "restart"
    | "pause"
    | "resume"
    | "step"
    | "set_intervention"
    | "switch_agent";
  payload?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const RECONNECT_DELAY_MS = 2_000;
const MAX_EVENTS = 200;
const WS_BASE =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8000`
    : "ws://localhost:8000";
const API_BASE =
  typeof window !== "undefined"
    ? `http://${window.location.hostname}:8000`
    : "http://localhost:8000";

let _eventIdCounter = 0;

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useVisualizerSocket(
  agentName?: string,
  envId?: string,
  device?: string,
  wsPath: string = "/ws",
): {
  state: VisualizerState;
  sendControl: (cmd: ControlCommand) => Promise<void>;
} {
  const [state, setState] = useState<VisualizerState>({
    connected: false,
    loading: false,
    frame_index: null,
    frame: null,
    attention: null,
    norms: null,
    metrics: null,
    token_layout: null,
    config: null,
    events: [],
    reconstruction: null,
    error_map: null,
    reconstruction_error: null,
    imagined_next: null,
    sae_features: null,
    sae_layer: null,
    imagined_intervened: null,
    intervention_diff: null,
    intervention: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  // Set to true before deliberately closing the socket (cleanup / agentName change)
  // so that onclose does not schedule an extra reconnect — the new effect body handles it.
  const intentionalCloseRef = useRef(false);

  const addEvent = useCallback((event: string, data: Record<string, unknown>) => {
    const msg: EventMessage = {
      id: ++_eventIdCounter,
      timestamp: new Date().toLocaleTimeString(),
      event,
      data,
    };
    setState((prev) => ({
      ...prev,
      events: [msg, ...prev.events].slice(0, MAX_EVENTS),
    }));
  }, []);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    let url = `${WS_BASE}${wsPath}`;
    if (agentName) {
      const params = new URLSearchParams({ agent: agentName });
      if (envId) params.set("env_id", envId);
      if (device) params.set("device", device);
      url = `${url}?${params}`;
    }

    setState((prev) => ({ ...prev, loading: true }));
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setState((prev) => ({ ...prev, connected: true, loading: false }));
      addEvent("connected", { url });
    };

    ws.onclose = (ev) => {
      if (!mountedRef.current) return;
      // If we closed deliberately (agentName change / unmount), the new effect body
      // will call connect() itself — don't schedule a duplicate reconnect.
      if (intentionalCloseRef.current) {
        intentionalCloseRef.current = false;
        setState((prev) => ({ ...prev, connected: false, loading: false }));
        return;
      }
      setState((prev) => ({ ...prev, connected: false, loading: false }));
      addEvent("disconnected", { code: ev.code });
      // Auto-reconnect after unexpected disconnect
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      addEvent("error", { message: "WebSocket error" });
    };

    ws.onmessage = (ev) => {
      if (!mountedRef.current) return;
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(ev.data as string) as Record<string, unknown>;
      } catch {
        return;
      }

      const type = msg.type as string;

      if (type === "frame") {
        const f = msg as unknown as FrameData & { type: string };
        // Test 1 instrumentation: the game frame and the activations live in this one
        // message, so they describe the same step by construction. Log both render ticks
        // with the same frame_index to make any drift visible (console.debug — off by
        // default in prod). Backend already aligns frame↔activations to the same timestep.
        if (process.env.NODE_ENV !== "production") {
          const t = Math.round(performance.now());
          // eslint-disable-next-line no-console
          console.debug(`[frame ${f.frame_index}] game frame rendered at ${t}`);
          // eslint-disable-next-line no-console
          console.debug(`[frame ${f.frame_index}] activations rendered at ${t}`);
        }
        setState((prev) => ({
          ...prev,
          loading: false,
          frame_index: f.frame_index,
          frame: f.frame,
          attention: f.attention,
          norms: f.norms,
          metrics: f.metrics,
          // Preserve the previous token_layout reference when labels haven't
          // changed — a new JSON-parsed object every frame would cause
          // MorphingGraph's Effect 1 (dep: [tokenLayout]) to fire every frame,
          // wiping and rebuilding the SVG each time and preventing the force
          // simulation from ever converging.
          token_layout:
            prev.token_layout !== null &&
            f.token_layout.labels.join("\0") ===
              prev.token_layout.labels.join("\0")
              ? prev.token_layout
              : f.token_layout,
          reconstruction: f.reconstruction ?? null,
          error_map: f.error_map ?? null,
          reconstruction_error: f.reconstruction_error ?? null,
          // Only populated on a single-step; preserve the last imagined frame
          // across free-running frames so the panel doesn't flicker to empty.
          imagined_next: f.imagined_next ?? prev.imagined_next,
          sae_features: f.sae_features ?? null,
          sae_layer: f.sae_layer ?? null,
          // Intervention outputs are step-mode only; preserve across free-run frames.
          imagined_intervened: f.imagined_intervened ?? prev.imagined_intervened,
          intervention_diff: f.intervention_diff ?? prev.intervention_diff,
          intervention: f.intervention ?? prev.intervention,
        }));
      } else if (type === "config") {
        const cfg = msg as unknown as Partial<ModelConfig> & {
          type: string;
          agents?: AgentInfo[];
        };
        const agents = cfg.agents ?? [];
        // Only update model config fields if the backend actually sent them
        // (the initial config message may only contain the agents list)
        const hasModelConfig = cfg.num_layers != null;
        setState((prev) => ({
          ...prev,
          config: {
            num_layers: cfg.num_layers ?? prev.config?.num_layers ?? 10,
            num_heads: cfg.num_heads ?? prev.config?.num_heads ?? 4,
            embed_dim: cfg.embed_dim ?? prev.config?.embed_dim ?? 256,
            tokens_per_block: cfg.tokens_per_block ?? prev.config?.tokens_per_block ?? 17,
            max_blocks: cfg.max_blocks ?? prev.config?.max_blocks ?? 20,
            agents: agents.length > 0 ? agents : (prev.config?.agents ?? []),
          },
        }));
        if (hasModelConfig) {
          addEvent("config_received", {
            layers: cfg.num_layers,
            heads: cfg.num_heads,
          });
        }
      } else if (type === "event") {
        const eventName = msg.event as string;
        const data = (msg.data ?? {}) as Record<string, unknown>;
        addEvent(eventName, data);
        // Show loading spinner on agent switch until first frame
        if (eventName === "agent_loaded") {
          setState((prev) => ({
            ...prev,
            loading: true,
            frame: null,
            attention: null,
            norms: null,
            reconstruction: null,
            error_map: null,
            reconstruction_error: null,
            imagined_next: null,
            sae_features: null,
            sae_layer: null,
            imagined_intervened: null,
            intervention_diff: null,
            intervention: null,
          }));
        }
      }
    };
  }, [agentName, envId, device, wsPath, addEvent]);

  // Mount / unmount / agentName change
  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      intentionalCloseRef.current = true; // suppress reconnect from onclose
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // Control command sender
  const sendControl = useCallback(async (cmd: ControlCommand): Promise<void> => {
    if (cmd.command === "switch_agent") {
      // Immediately clear visualisation state and show loading
      setState((prev) => ({
        ...prev,
        loading: true,
        frame: null,
        attention: null,
        norms: null,
      }));
    }
    await fetch(`${API_BASE}/control`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cmd),
    });
  }, []);

  return { state, sendControl };
}
