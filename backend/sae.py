"""
Sparse autoencoder over IRIS world-model residual-stream activations.

Shared by the offline trainer (``scripts/train_sae.py``) and the inference engine
(Phase 2 feature panel + Phase 3 intervention).  A ReLU SAE with an L1 sparsity
penalty and unit-norm decoder rows, following the standard "Towards
Monosemanticity" recipe::

    f  = relu((x - b_dec) @ W_enc + b_enc)        # features (sparse, >= 0)
    x̂  = f @ W_dec + b_dec                         # reconstruction
    L  = ||x - x̂||²_sum + λ · ||f||₁_sum           # (mean over batch)

The decoder bias ``b_dec`` doubles as a tied pre-encoder bias (subtracted before
encoding), which centres the input and stabilises training.  Each row of
``W_dec`` is the dictionary direction of one feature and is held at unit norm, so
the L1 penalty cannot be gamed by shrinking features while growing the decoder.

The SAE operates on **normalised** activations.  Training records per-dimension
mean/std; the engine must apply the identical normalisation before ``encode``.
For an intervention, the raw-space direction of feature i is ``W_dec[i] * std``.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """ReLU sparse autoencoder with a tied decoder bias and unit-norm dictionary."""

    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_in))
        self._init_weights()

    def _init_weights(self) -> None:
        # Random unit-norm dictionary rows; tie the encoder to the decoder
        # transpose so encoder and decoder start mutually consistent.
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))
            self.W_enc.copy_(self.W_dec.t())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(..., d_in) normalised activations → (..., d_hidden) non-negative features."""
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """(..., d_hidden) features → (..., d_in) reconstructed activations."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Re-project each decoder row (feature dictionary direction) to unit norm.

        Call after every optimizer step so the L1 penalty stays meaningful.
        """
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))


def sae_loss(
    sae: SparseAutoencoder, x: torch.Tensor, l1_coeff: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Reconstruction MSE (summed over d_in) + λ·L1 (summed over d_hidden), mean over batch.

    Returns ``(loss, metrics)`` where metrics holds detached ``recon``, ``l1`` and
    ``l0`` (average number of features firing per example) for logging.
    """
    x_hat, f = sae(x)
    recon = (x_hat - x).pow(2).sum(-1).mean()
    l1 = f.sum(-1).mean()
    loss = recon + l1_coeff * l1
    l0 = (f > 0).float().sum(-1).mean()
    return loss, {"recon": recon.detach(), "l1": l1.detach(), "l0": l0.detach()}


# ---------------------------------------------------------------------------
# Artifact (de)serialisation — one schema shared by trainer and engine
# ---------------------------------------------------------------------------

def save_artifact(
    path: Path,
    sae: SparseAutoencoder,
    *,
    layer: int,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    env_id: str,
    l1_coeff: float,
    trained_steps: int,
    metrics: Dict[str, float],
    token_policy: str = "last",
    expansion_factor: int = 8,
) -> None:
    """Persist an SAE plus the metadata the engine needs to load and use it.

    ``norm_mean``/``norm_std`` are the per-dimension stats of the layer-L residual
    used during training; the engine must apply the same normalisation before
    ``encode``.  ``token_policy`` records which token position(s) the activations
    were harvested from ("last" = the action token, index -1).
    """
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "d_in": sae.d_in,
            "d_hidden": sae.d_hidden,
            "layer": layer,
            "norm": {"mean": norm_mean.detach().cpu(), "std": norm_std.detach().cpu()},
            "env_id": env_id,
            "expansion_factor": expansion_factor,
            "l1_coeff": l1_coeff,
            "trained_steps": trained_steps,
            "metrics": metrics,
            "token_policy": token_policy,
        },
        path,
    )


def load_artifact(
    path: Path, device: str = "cpu"
) -> Tuple[SparseAutoencoder, dict]:
    """Load an SAE artifact saved by :func:`save_artifact`.

    Returns ``(sae, meta)`` with the model in eval mode on ``device`` and ``meta``
    the full saved dict (norm stats, layer, metrics, …).
    """
    ckpt = torch.load(path, map_location=device)
    sae = SparseAutoencoder(ckpt["d_in"], ckpt["d_hidden"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device).eval()
    return sae, ckpt
