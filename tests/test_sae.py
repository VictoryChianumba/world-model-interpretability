"""
Tests for the sparse autoencoder module (backend/sae.py).

Pure-torch — no IRIS checkpoint or world model required.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sae import (  # noqa: E402
    SparseAutoencoder,
    sae_loss,
    save_artifact,
    load_artifact,
)


class TestSparseAutoencoder:

    def test_forward_shapes(self):
        """encode → (B, d_hidden), decode → (B, d_in); features are non-negative."""
        sae = SparseAutoencoder(d_in=16, d_hidden=64)
        x = torch.randn(8, 16)
        x_hat, f = sae(x)
        assert x_hat.shape == (8, 16)
        assert f.shape == (8, 64)
        assert (f >= 0).all(), "ReLU features must be non-negative"

    def test_decoder_rows_unit_norm_on_init_and_after_normalize(self):
        """Each decoder row (feature dictionary direction) has unit norm."""
        sae = SparseAutoencoder(d_in=16, d_hidden=64)
        norms = sae.W_dec.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

        # Perturb then re-normalise → unit norm again.
        with torch.no_grad():
            sae.W_dec.mul_(3.7)
        sae.normalize_decoder()
        norms = sae.W_dec.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_loss_and_metrics(self):
        """sae_loss returns a scalar loss and recon/l1/l0 metrics in valid ranges."""
        sae = SparseAutoencoder(d_in=16, d_hidden=64)
        x = torch.randn(32, 16)
        loss, m = sae_loss(sae, x, l1_coeff=1e-3)
        assert loss.ndim == 0
        assert m["recon"] >= 0
        assert m["l1"] >= 0
        assert 0 <= float(m["l0"]) <= 64  # firing count within [0, d_hidden]

    def test_save_load_roundtrip(self, tmp_path):
        """Artifact round-trips weights and metadata; loaded SAE matches the original."""
        sae = SparseAutoencoder(d_in=16, d_hidden=64)
        mean = torch.randn(16)
        std = torch.rand(16).add(0.1)
        path = tmp_path / "sae_L5.pt"
        save_artifact(
            path, sae, layer=5, norm_mean=mean, norm_std=std,
            env_id="BreakoutNoFrameskip-v4", l1_coeff=2e-3, trained_steps=123,
            metrics={"recon": 0.5, "l0": 12.0}, token_policy="last", expansion_factor=4,
        )

        sae2, meta = load_artifact(path, device="cpu")
        assert meta["layer"] == 5
        assert meta["d_in"] == 16 and meta["d_hidden"] == 64
        assert meta["env_id"] == "BreakoutNoFrameskip-v4"
        assert meta["token_policy"] == "last"
        assert meta["expansion_factor"] == 4
        assert torch.allclose(meta["norm"]["mean"], mean)
        assert torch.allclose(meta["norm"]["std"], std)

        x = torch.randn(4, 16)
        xa, fa = sae(x)
        xb, fb = sae2(x)
        assert torch.allclose(xa, xb, atol=1e-6)
        assert torch.allclose(fa, fb, atol=1e-6)

    def test_training_reduces_reconstruction(self):
        """A few optimizer steps on reconstructable synthetic data lower recon error."""
        torch.manual_seed(0)
        d_in, d_hidden = 8, 32

        # Synthetic data: sparse non-negative codes through a fixed unit-norm dictionary.
        true_dict = torch.randn(d_hidden, d_in)
        true_dict /= true_dict.norm(dim=1, keepdim=True)
        codes = (torch.rand(2048, d_hidden) < 0.1).float() * torch.rand(2048, d_hidden)
        data = codes @ true_dict                      # (2048, d_in)
        data = (data - data.mean(0)) / data.std(0).clamp_min(1e-6)

        sae = SparseAutoencoder(d_in, d_hidden)
        opt = torch.optim.Adam(sae.parameters(), lr=1e-2)

        with torch.no_grad():
            initial_recon = float(sae_loss(sae, data, l1_coeff=1e-4)[1]["recon"])

        for _ in range(300):
            loss, _ = sae_loss(sae, data, l1_coeff=1e-4)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()

        with torch.no_grad():
            final_recon = float(sae_loss(sae, data, l1_coeff=1e-4)[1]["recon"])

        assert final_recon < initial_recon * 0.9, (
            f"Reconstruction did not improve: {initial_recon:.3f} → {final_recon:.3f}"
        )
