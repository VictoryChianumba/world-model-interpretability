"""
File-backed store for *pinned* SAE features — the v2 canvas's source of truth.

A pinned feature is a card the user has placed on the canvas. Unlike a bookmark
(which is just a label), a pin carries interaction state: an optional user-edited
label, an intervention scale (0 = observation-only, non-zero = steer), and an x/y
position so the canvas layout survives reloads. Records are keyed by
``(env_id, layer, feature_id)`` — same scheme as ``BookmarkStore``.

The whole store is one JSON file written atomically under a lock, so the FastAPI
endpoints can read/write it safely from multiple requests. Writing on every change
(not just shutdown) is what makes the canvas reload-persistent.
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# Sentinel distinguishing "field not provided" (keep existing) from an explicit value.
_UNSET: Any = object()


def _key(env_id: str, layer: int, feature_id: int) -> str:
    return f"{env_id}::{layer}::{feature_id}"


class PinnedStore:
    """Thread-safe JSON-backed store of pinned canvas features."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal IO (callers hold the lock)
    # ------------------------------------------------------------------

    def _read(self) -> Dict[str, dict]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}  # corrupt/partial file → treat as empty rather than crash

    def _write(self, data: Dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file in the same dir, then rename.
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list(self, env_id: Optional[str] = None, layer: Optional[int] = None) -> List[dict]:
        """Return pinned features, optionally filtered by env_id and/or layer."""
        with self._lock:
            data = self._read()
        out = list(data.values())
        if env_id is not None:
            out = [p for p in out if p.get("env_id") == env_id]
        if layer is not None:
            out = [p for p in out if p.get("layer") == layer]
        return sorted(out, key=lambda p: (p.get("layer", 0), p.get("feature_id", 0)))

    def upsert(
        self,
        env_id: str,
        layer: int,
        feature_id: int,
        custom_label: Any = _UNSET,
        intervention_scale: Any = _UNSET,
        x: Any = _UNSET,
        y: Any = _UNSET,
        updated_at: str = "",
    ) -> dict:
        """Create or merge one pin; returns the stored record.

        Only fields passed explicitly are changed — omitted fields keep their existing
        value (or a default on first insert). This lets the canvas issue partial updates:
        a drag sends only x/y, a slider sends only intervention_scale, a rename sends only
        custom_label. Pass ``custom_label=""`` to clear a label back to null.
        """
        with self._lock:
            data = self._read()
            k = _key(env_id, layer, feature_id)
            rec = data.get(k) or {
                "env_id": env_id,
                "layer": int(layer),
                "feature_id": int(feature_id),
                "custom_label": None,
                "intervention_scale": 0.0,
                "x": 0.0,
                "y": 0.0,
            }
            if custom_label is not _UNSET:
                # Empty string clears the label; otherwise store the trimmed text.
                cl = (custom_label or "").strip()
                rec["custom_label"] = cl or None
            if intervention_scale is not _UNSET:
                rec["intervention_scale"] = float(intervention_scale)
            if x is not _UNSET:
                rec["x"] = float(x)
            if y is not _UNSET:
                rec["y"] = float(y)
            rec["updated_at"] = updated_at
            data[k] = rec
            self._write(data)
        return rec

    def delete(self, env_id: str, layer: int, feature_id: int) -> bool:
        """Unpin one feature; returns True if it existed."""
        with self._lock:
            data = self._read()
            existed = data.pop(_key(env_id, layer, feature_id), None) is not None
            if existed:
                self._write(data)
        return existed
