"use client";

import { useEffect, useRef, useState } from "react";
import { Stage, Layer, Group, Rect, Text, Line, Circle } from "react-konva";
import type Konva from "konva";

// ---------------------------------------------------------------------------
// Card view-model + props
// ---------------------------------------------------------------------------

export interface CardVM {
  featureId: number;
  x: number;
  y: number;
  label: string | null; // resolved label, or null → "unlabeled"
  customLabel: string | null; // user override (drives the edit field)
  activation: number; // current firing magnitude
  maxActivation: number; // for bar/sparkline scaling
  history: number[]; // recent firing, oldest→newest
  interventionScale: number; // 0 = observation-only
}

interface FeatureCanvasProps {
  cards: CardVM[];
  onMove: (featureId: number, x: number, y: number, persist: boolean) => void;
  onScale: (featureId: number, scale: number, persist: boolean) => void;
  onRemove: (featureId: number) => void;
  onRelabel: (featureId: number, label: string) => void;
}

const CARD_W = 184;
const CARD_H = 134;
const SCALE_MAX = 5; // slider range [-5, +5]
const TRACK_X = 12;
const TRACK_W = CARD_W - 24;
const TRACK_Y = 116;

const COL = {
  bg: "#161629",
  cardBg: "#1e1e38",
  cardBorder: "#33335a",
  steer: "#ec4899",
  text: "#e5e7eb",
  muted: "#8b8ba7",
  accent: "#818cf8",
  bar: "#6366f1",
  barBg: "#2a2a48",
  spark: "#34d399",
};

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

function scaleToHandleX(scale: number): number {
  const clamped = Math.max(-SCALE_MAX, Math.min(SCALE_MAX, scale));
  return TRACK_X + ((clamped + SCALE_MAX) / (2 * SCALE_MAX)) * TRACK_W;
}

function handleXToScale(x: number): number {
  const ratio = (x - TRACK_X) / TRACK_W;
  let s = ratio * 2 * SCALE_MAX - SCALE_MAX;
  if (Math.abs(s) < 0.25) s = 0; // snap to neutral
  return Math.round(s * 10) / 10;
}

function sparklinePoints(history: number[], scaleMax: number): number[] {
  if (history.length < 2) return [];
  const x0 = 12;
  const x1 = CARD_W - 12;
  const yTop = 74;
  const yBot = 104;
  const denom = Math.max(scaleMax, ...history, 1e-6);
  const n = history.length;
  const pts: number[] = [];
  history.forEach((v, i) => {
    const x = x0 + (i / (n - 1)) * (x1 - x0);
    const y = yBot - (Math.max(0, v) / denom) * (yBot - yTop);
    pts.push(x, y);
  });
  return pts;
}

function pointer(node: Konva.Node, cursor: string) {
  const stage = node.getStage();
  if (stage) stage.container().style.cursor = cursor;
}

// ---------------------------------------------------------------------------
// One card (Konva group)
// ---------------------------------------------------------------------------

function FeatureCard({
  card,
  onMove,
  onScale,
  onRemove,
  onEditLabel,
}: {
  card: CardVM;
  onMove: FeatureCanvasProps["onMove"];
  onScale: FeatureCanvasProps["onScale"];
  onRemove: FeatureCanvasProps["onRemove"];
  onEditLabel: (featureId: number, x: number, y: number) => void;
}) {
  const steering = card.interventionScale !== 0;
  const labelText = card.label ?? "unlabeled";
  const barRatio = Math.min(1, card.activation / Math.max(card.maxActivation, 1e-6));

  return (
    <Group
      x={card.x}
      y={card.y}
      draggable
      onDragMove={(e) => onMove(card.featureId, e.target.x(), e.target.y(), false)}
      onDragEnd={(e) => onMove(card.featureId, e.target.x(), e.target.y(), true)}
    >
      <Rect
        width={CARD_W}
        height={CARD_H}
        cornerRadius={8}
        fill={COL.cardBg}
        stroke={steering ? COL.steer : COL.cardBorder}
        strokeWidth={steering ? 2 : 1}
        shadowColor="#000"
        shadowBlur={8}
        shadowOpacity={0.4}
      />

      {/* Header: feature id + remove */}
      <Text x={10} y={8} text={`#${card.featureId}`} fontSize={12} fontStyle="bold" fill={COL.accent} />
      <Text
        x={CARD_W - 20}
        y={5}
        text="✕"
        fontSize={14}
        fill={COL.muted}
        onClick={() => onRemove(card.featureId)}
        onTap={() => onRemove(card.featureId)}
        onMouseEnter={(e) => {
          pointer(e.target, "pointer");
          (e.target as Konva.Text).fill(COL.steer);
          e.target.getLayer()?.batchDraw();
        }}
        onMouseLeave={(e) => {
          pointer(e.target, "default");
          (e.target as Konva.Text).fill(COL.muted);
          e.target.getLayer()?.batchDraw();
        }}
      />

      {/* Label (double-click to edit) */}
      <Text
        x={10}
        y={26}
        width={CARD_W - 20}
        text={labelText}
        fontSize={12}
        fontStyle={card.label ? "normal" : "italic"}
        fill={card.label ? COL.text : COL.muted}
        wrap="none"
        ellipsis
        onDblClick={() => onEditLabel(card.featureId, card.x, card.y)}
        onDblTap={() => onEditLabel(card.featureId, card.x, card.y)}
        onMouseEnter={(e) => pointer(e.target, "text")}
        onMouseLeave={(e) => pointer(e.target, "default")}
      />

      {/* Current activation + bar */}
      <Text x={10} y={46} text={`act ${card.activation.toFixed(2)}`} fontSize={10} fill={COL.muted} />
      <Rect x={70} y={47} width={CARD_W - 80} height={6} cornerRadius={3} fill={COL.barBg} />
      <Rect
        x={70}
        y={47}
        width={(CARD_W - 80) * barRatio}
        height={6}
        cornerRadius={3}
        fill={COL.bar}
      />

      {/* Sparkline */}
      <Line points={sparklinePoints(card.history, card.maxActivation)} stroke={COL.spark} strokeWidth={1.5} />

      {/* Intervention scale slider */}
      <Rect x={TRACK_X} y={TRACK_Y} width={TRACK_W} height={4} cornerRadius={2} fill={COL.barBg} />
      <Rect x={TRACK_X + TRACK_W / 2 - 0.5} y={TRACK_Y - 3} width={1} height={10} fill={COL.muted} />
      <Text
        x={CARD_W - 52}
        y={TRACK_Y - 18}
        width={42}
        align="right"
        text={steering ? `×${card.interventionScale.toFixed(1)}` : "obs"}
        fontSize={10}
        fill={steering ? COL.steer : COL.muted}
      />
      <Circle
        x={scaleToHandleX(card.interventionScale)}
        y={TRACK_Y + 2}
        radius={7}
        fill={steering ? COL.steer : COL.accent}
        stroke="#0d0d20"
        strokeWidth={1}
        draggable
        onMouseEnter={(e) => pointer(e.target, "ew-resize")}
        onMouseLeave={(e) => pointer(e.target, "default")}
        onDragStart={(e) => {
          e.cancelBubble = true; // drag the handle, not the card
        }}
        onDragMove={(e) => {
          e.cancelBubble = true;
          // node.x()/y() are group-relative (track-space): clamp to the track and lock y.
          const localX = Math.max(TRACK_X, Math.min(TRACK_X + TRACK_W, e.target.x()));
          e.target.x(localX);
          e.target.y(TRACK_Y + 2);
          onScale(card.featureId, handleXToScale(localX), false);
        }}
        onDragEnd={(e) => {
          e.cancelBubble = true;
          const localX = Math.max(TRACK_X, Math.min(TRACK_X + TRACK_W, e.target.x()));
          onScale(card.featureId, handleXToScale(localX), true);
        }}
      />
    </Group>
  );
}

// ---------------------------------------------------------------------------
// Canvas surface
// ---------------------------------------------------------------------------

export default function FeatureCanvas({
  cards,
  onMove,
  onScale,
  onRemove,
  onRelabel,
}: FeatureCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 480 });
  const [editing, setEditing] = useState<{ id: number; x: number; y: number } | null>(null);
  const editRef = useRef<HTMLInputElement>(null);

  // Responsive stage sizing.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0].contentRect;
      setSize({ width: Math.max(320, r.width), height: Math.max(320, r.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (editing && editRef.current) {
      editRef.current.focus();
      editRef.current.select();
    }
  }, [editing]);

  const editingCard = editing ? cards.find((c) => c.featureId === editing.id) : null;

  const commitLabel = () => {
    if (editing && editRef.current) onRelabel(editing.id, editRef.current.value);
    setEditing(null);
  };

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden rounded-lg" style={{ background: COL.bg }}>
      {cards.length === 0 && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-slate-500">
          Search a feature id or label, or pin one from the discovery panel, to place a card here.
        </div>
      )}
      <Stage width={size.width} height={size.height}>
        <Layer>
          {cards.map((card) => (
            <FeatureCard
              key={card.featureId}
              card={card}
              onMove={onMove}
              onScale={onScale}
              onRemove={onRemove}
              onEditLabel={(id, x, y) => setEditing({ id, x, y })}
            />
          ))}
        </Layer>
      </Stage>

      {/* Transient HTML overlay for editing a card's label (positioned over its card). */}
      {editing && (
        <input
          ref={editRef}
          defaultValue={editingCard?.customLabel ?? editingCard?.label ?? ""}
          placeholder="custom label"
          onBlur={commitLabel}
          onKeyDown={(e) => {
            if (e.key === "Enter") commitLabel();
            if (e.key === "Escape") setEditing(null);
          }}
          className="absolute z-10 rounded border border-indigo-400 bg-slate-900 px-1 py-0.5 text-xs text-slate-100 outline-none"
          style={{ left: editing.x + 8, top: editing.y + 24, width: CARD_W - 20 }}
        />
      )}
    </div>
  );
}
