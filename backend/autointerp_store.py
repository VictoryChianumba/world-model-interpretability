"""
Read/write helpers for the autointerp feature-label cache.

The autointerp pipeline (``scripts/autointerp.py``) labels SAE features by sending a
grid of their top-activating game frames to a vision LLM. This module is the single
source of truth for where that cache lives and how it is shaped, shared by the writer
(the script) and the reader (the ``/feature/{id}`` endpoint).

Layout, rooted at ``<root>`` (defaults to SAE_DIR) for one SAE ``layer``::

    autointerp_L{layer}.json          index: per-feature metadata + label (no images)
    autointerp_L{layer}/
        feat_{id}.png                 the 4x4 grid sent to the LLM (human spot-check)
        feat_{id}.json                that feature's top example frames (base64 PNGs)

Splitting the (light) index from the (heavy) per-feature example files keeps the
endpoint cheap: card labels and firing stats come from the index; the example frames
are read only when a feature detail view asks for them.
"""

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class AutoInterpStore:
    """Paths + (de)serialization for one SAE layer's autointerp cache."""

    def __init__(self, root: str, layer: int) -> None:
        self.root = Path(root)
        self.layer = int(layer)

    # ------------------------------------------------------------------ paths
    @property
    def index_path(self) -> Path:
        return self.root / f"autointerp_L{self.layer}.json"

    @property
    def feat_dir(self) -> Path:
        return self.root / f"autointerp_L{self.layer}"

    def grid_path(self, feature_id: int) -> Path:
        return self.feat_dir / f"feat_{feature_id}.png"

    def examples_path(self, feature_id: int) -> Path:
        return self.feat_dir / f"feat_{feature_id}.json"

    # ------------------------------------------------------------------ index
    def load_index(self) -> Dict[str, Any]:
        """Return the index dict, or a fresh skeleton if none exists yet."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("features", {})
                    return data
            except Exception:
                pass
        return {"layer": self.layer, "features": {}}

    def save_index(self, index: Dict[str, Any]) -> None:
        _atomic_write_text(self.index_path, json.dumps(index, indent=2))

    # ----------------------------------------------------------- writer (script)
    def write_examples(
        self,
        feature_id: int,
        *,
        examples_b64: List[str],
        top_activations: List[float],
        frame_indices: List[int],
    ) -> None:
        """Persist one feature's top example frames (used by the detail view)."""
        self.feat_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": int(feature_id),
            "top_activations": [round(float(a), 4) for a in top_activations],
            "frame_indices": [int(i) for i in frame_indices],
            "examples": examples_b64,
        }
        _atomic_write_text(self.examples_path(feature_id), json.dumps(payload))

    def write_grid(self, feature_id: int, grid_png: bytes) -> None:
        """Persist the 4x4 grid PNG that was sent to the LLM (for spot-checking)."""
        self.feat_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(self.grid_path(feature_id), grid_png)

    # ----------------------------------------------------------- reader (API)
    def read_feature(self, feature_id: int) -> Dict[str, Any]:
        """Return the API view of one feature: metadata + label + example frames.

        Always returns a dict (never raises) so the endpoint can report an
        unlabeled/unknown feature uniformly as ``label: null, top_activation_examples: []``.
        """
        index = self.load_index()
        meta = (index.get("features") or {}).get(str(feature_id)) or {}
        examples: List[str] = []
        ex_path = self.examples_path(feature_id)
        if ex_path.exists():
            try:
                with open(ex_path, "r") as f:
                    examples = (json.load(f) or {}).get("examples", []) or []
            except Exception:
                examples = []
        return {
            "id": int(feature_id),
            "layer": self.layer,
            "label": meta.get("label"),
            "firing_rate": meta.get("firing_rate"),
            "mean_activation": meta.get("mean_activation"),
            "max_activation": meta.get("max_activation"),
            "top_activation_examples": examples,
        }


def resolve_layer(root: str, layer: Optional[int] = None) -> Optional[int]:
    """Pick which autointerp layer to serve.

    Prefers an explicit ``layer``; otherwise, if exactly one ``autointerp_L*.json``
    cache exists under ``root``, uses that. Returns None if it can't decide.
    """
    if layer is not None:
        return int(layer)
    caches = sorted(Path(root).glob("autointerp_L*.json"))
    if len(caches) == 1:
        stem = caches[0].stem  # autointerp_L5
        try:
            return int(stem.split("_L")[-1])
        except ValueError:
            return None
    return None


# ----------------------------------------------------------------- small IO utils

def png_to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
