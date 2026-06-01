"""
File-backed store for SAE feature bookmarks (labels/notes that survive sessions).

A bookmark tags one SAE feature — keyed by ``(env_id, layer, feature_id)`` — with a
human (or auto-descriptive) label and optional notes. The whole store is a single
JSON file; writes are atomic (temp file + rename) and guarded by a lock, so the
FastAPI endpoints and the offline labeling script can both use it safely.

This is the backend's only persistence; everything else is in-memory.
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional


def _key(env_id: str, layer: int, feature_id: int) -> str:
    return f"{env_id}::{layer}::{feature_id}"


class BookmarkStore:
    """Thread-safe JSON-backed bookmark store keyed by (env_id, layer, feature_id)."""

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
        """Return bookmarks, optionally filtered by env_id and/or layer."""
        with self._lock:
            data = self._read()
        out = list(data.values())
        if env_id is not None:
            out = [b for b in out if b.get("env_id") == env_id]
        if layer is not None:
            out = [b for b in out if b.get("layer") == layer]
        return sorted(out, key=lambda b: (b.get("layer", 0), b.get("feature_id", 0)))

    def upsert(
        self,
        env_id: str,
        layer: int,
        feature_id: int,
        label: str,
        notes: str = "",
        source: str = "user",
        updated_at: str = "",
    ) -> dict:
        """Insert or update one bookmark; returns the stored record."""
        rec = {
            "env_id": env_id,
            "layer": int(layer),
            "feature_id": int(feature_id),
            "label": label,
            "notes": notes,
            "source": source,
            "updated_at": updated_at,
        }
        with self._lock:
            data = self._read()
            data[_key(env_id, layer, feature_id)] = rec
            self._write(data)
        return rec

    def delete(self, env_id: str, layer: int, feature_id: int) -> bool:
        """Remove one bookmark; returns True if it existed."""
        with self._lock:
            data = self._read()
            existed = data.pop(_key(env_id, layer, feature_id), None) is not None
            if existed:
                self._write(data)
        return existed
