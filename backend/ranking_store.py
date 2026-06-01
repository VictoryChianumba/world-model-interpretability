"""
Read/write helpers for the offline causal-importance ranking cache.

`scripts/causal_importance.py` scores each SAE feature by the mean token divergence its
intervention induces in N-step imagined rollouts, and writes the scores here. The
`/ranking/causal` endpoint reads them back. One JSON file per SAE layer, rooted at SAE_DIR:

    causal_L{layer}.json   { layer, env_id, n_steps, scale, seeds, generated_at,
                             scores: { "<id>": {id, score, pos, neg} } }

`score` is the mean token divergence vs baseline (averaged over seeds and ±scale);
`pos`/`neg` are the per-sign means, kept so an asymmetric feature (drives one direction
only) is visible rather than averaged away.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class CausalRankingStore:
    """Paths + (de)serialization for one SAE layer's causal-importance scores."""

    def __init__(self, root: str, layer: int) -> None:
        self.root = Path(root)
        self.layer = int(layer)

    @property
    def path(self) -> Path:
        return self.root / f"causal_L{self.layer}.json"

    def load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("scores", {})
                    return data
            except Exception:
                pass
        return {"layer": self.layer, "scores": {}}

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def ranked(self, top: int = 20) -> Dict[str, Any]:
        """API view: top features by descending causal score, or empty/unavailable."""
        data = self.load()
        scores = list((data.get("scores") or {}).values())
        if not scores:
            return {"metric": "causal", "available": False, "features": []}
        scores.sort(key=lambda s: s.get("score", 0.0), reverse=True)
        return {
            "metric": "causal",
            "available": True,
            "n_steps": data.get("n_steps"),
            "scale": data.get("scale"),
            "seeds": data.get("seeds"),
            "n_features_scored": len(scores),
            "generated_at": data.get("generated_at"),
            "features": scores[:top],
        }


def resolve_causal_layer(root: str, layer: Optional[int] = None) -> Optional[int]:
    """Pick which causal-ranking layer to serve: explicit layer, else the single cache."""
    if layer is not None:
        return int(layer)
    caches = sorted(Path(root).glob("causal_L*.json"))
    if len(caches) == 1:
        try:
            return int(caches[0].stem.split("_L")[-1])
        except ValueError:
            return None
    return None
