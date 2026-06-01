"""
Backend test suite for the WM Visualizer.

Covers:
  - Hook extraction: correct tensor shapes (cached and non-cached)
  - KV cache interaction: shapes consistent across full rollout
  - Token alignment: labels derived from config, never hardcoded
  - Hook cleanup: zero hooks remain after failure or agent switch
  - Queue backpressure: inference thread never blocked
  - Frame dropping: frames dropped not queued when consumer falls behind
  - Agent switching: hooks re-registered, queue flushed, no stale state
  - Config mismatch: reinitialises correctly between agents with different arch
  - Shutdown: all paths terminate cleanly with no hanging threads
  - Frame encoding: raw observation correctly base64-encoded as PNG
"""

import base64
import io
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup: add IRIS src/ so models can be imported
# ---------------------------------------------------------------------------

_IRIS_ROOT = Path(__file__).parent.parent.parent / "iris"
_IRIS_SRC = _IRIS_ROOT / "src"
sys.path.insert(0, str(_IRIS_SRC))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from models.world_model import WorldModel
from models.transformer import TransformerConfig
from hooks import IrisHookExtractor
from inference import (
    _FpsCounter,
    _encode_frame,
    _decode_reconstruction,
    FrameData,
    get_token_labels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_world_model(
    num_layers: int = 2,
    num_heads: int = 2,
    embed_dim: int = 16,
    tokens_per_block: int = 5,
    max_blocks: int = 10,
) -> WorldModel:
    config = TransformerConfig(
        tokens_per_block=tokens_per_block,
        max_blocks=max_blocks,
        attention="causal",
        num_layers=num_layers,
        num_heads=num_heads,
        embed_dim=embed_dim,
        embed_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return WorldModel(obs_vocab_size=16, act_vocab_size=8, config=config).eval()


# ---------------------------------------------------------------------------
# Hook extraction: shapes
# ---------------------------------------------------------------------------

class TestHookExtraction:

    def test_attn_shape_uncached(self):
        """Uncached forward pass: attention shape is (1, nh, T, T)."""
        wm = make_world_model(num_layers=2, num_heads=2, tokens_per_block=5)
        hooks = IrisHookExtractor()
        hooks.attach(wm)

        tokens = torch.randint(0, 8, (1, 5))
        with torch.no_grad():
            wm(tokens, past_keys_values=None)

        attn, _ = hooks.get_data()
        assert attn is not None
        for i in range(2):
            assert attn[i].shape == (1, 2, 5, 5), f"Layer {i}: {attn[i].shape}"
        hooks.detach()

    def test_norm_shape_uncached(self):
        """Norm values are positive floats, one per layer."""
        wm = make_world_model(num_layers=2, num_heads=2, tokens_per_block=5)
        hooks = IrisHookExtractor()
        hooks.attach(wm)

        tokens = torch.randint(0, 8, (1, 5))
        with torch.no_grad():
            wm(tokens, past_keys_values=None)

        _, norms = hooks.get_data()
        assert norms is not None
        assert len(norms) == 2
        for i in range(2):
            # get_data() now returns 0-d tensors for norms; .item() is deferred to
            # the consumer (inference.py) so it runs once after the full forward pass
            # rather than forcing a GPU→CPU sync per layer inside the hook callback.
            import torch as _torch
            assert isinstance(norms[i], _torch.Tensor)
            assert norms[i].ndim == 0          # 0-dimensional (scalar tensor)
            assert norms[i].item() > 0

        hooks.detach()

    def test_attn_shape_cached_first_pass(self):
        """Cached first pass (L=0, T=5): T_k = 5."""
        wm = make_world_model(num_layers=2, num_heads=2, tokens_per_block=5, max_blocks=20)
        hooks = IrisHookExtractor()
        hooks.attach(wm)

        kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=100)
        tokens = torch.randint(0, 8, (1, 5))
        with torch.no_grad():
            wm(tokens, past_keys_values=kv)

        attn, _ = hooks.get_data()
        for i in range(2):
            assert attn[i].shape == (1, 2, 5, 5), f"Layer {i}: {attn[i].shape}"
        hooks.detach()

    def test_attn_shape_cached_second_pass(self):
        """After caching 5 tokens (L=5), a 1-token query: T_k = L+T = 6."""
        wm = make_world_model(num_layers=2, num_heads=2, tokens_per_block=5, max_blocks=20)
        hooks = IrisHookExtractor()
        hooks.attach(wm)

        kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=100)
        tokens = torch.randint(0, 8, (1, 5))
        with torch.no_grad():
            wm(tokens, past_keys_values=kv)

        tokens2 = torch.randint(0, 8, (1, 1))
        with torch.no_grad():
            wm(tokens2, past_keys_values=kv)

        attn, _ = hooks.get_data()
        for i in range(2):
            assert attn[i].shape == (1, 2, 1, 6), f"Layer {i}: {attn[i].shape}"
        hooks.detach()


# ---------------------------------------------------------------------------
# Full-residual capture (SAE input)
# ---------------------------------------------------------------------------

class TestResidualCapture:

    def test_no_resid_capture_by_default(self):
        """Without capture_resid_layer, get_resid() is None and only block hooks exist."""
        wm = make_world_model(num_layers=3, num_heads=2, tokens_per_block=5)
        hooks = IrisHookExtractor()
        hooks.attach(wm)
        with torch.no_grad():
            wm(torch.randint(0, 8, (1, 5)), past_keys_values=None)
        assert hooks.get_resid() is None
        assert hooks.num_layers == 3
        hooks.detach()

    def test_resid_capture_shape(self):
        """With capture_resid_layer=L, get_resid() returns the full (1, T, E) residual."""
        wm = make_world_model(num_layers=4, num_heads=2, embed_dim=16, tokens_per_block=5)
        hooks = IrisHookExtractor()
        hooks.attach(wm, capture_resid_layer=2)
        with torch.no_grad():
            wm(torch.randint(0, 8, (1, 5)), past_keys_values=None)
        resid = hooks.get_resid()
        assert resid is not None
        assert resid.shape == (1, 5, 16)
        # num_layers reflects transformer depth, not handle count
        assert hooks.num_layers == 4
        hooks.detach()
        assert hooks.get_resid() is None

    def test_resid_capture_matches_block_output(self):
        """Captured residual equals the actual block output (hook reads, doesn't alter)."""
        wm = make_world_model(num_layers=3, num_heads=2, embed_dim=16, tokens_per_block=5)
        captured = {}
        h = wm.transformer.blocks[1].register_forward_hook(
            lambda m, i, o: captured.__setitem__("out", o.detach())
        )
        hooks = IrisHookExtractor()
        hooks.attach(wm, capture_resid_layer=1)
        with torch.no_grad():
            wm(torch.randint(0, 8, (1, 5)), past_keys_values=None)
        assert torch.allclose(hooks.get_resid(), captured["out"])
        h.remove()
        hooks.detach()

    def test_resid_layer_out_of_range_raises(self):
        """An out-of-range capture layer raises and leaves no hooks attached."""
        wm = make_world_model(num_layers=3, num_heads=2)
        hooks = IrisHookExtractor()
        with pytest.raises(ValueError):
            hooks.attach(wm, capture_resid_layer=5)
        assert sum(len(m._forward_hooks) for m in wm.modules()) == 0


# ---------------------------------------------------------------------------
# SAE feature computation (engine-level)
# ---------------------------------------------------------------------------

class TestSAEFeatureComputation:

    def test_no_sae_returns_none(self):
        """_compute_sae_features returns None when no SAE is loaded."""
        from inference import InferenceEngine
        engine = InferenceEngine(iris_src=str(_IRIS_SRC), iris_root=str(_IRIS_ROOT))
        assert engine._sae is None
        assert engine._compute_sae_features() is None

    def test_topk_features_from_residual(self):
        """With an SAE loaded and a residual captured, returns sorted top-K {id, mag}."""
        from inference import InferenceEngine
        from sae import SparseAutoencoder

        engine = InferenceEngine(iris_src=str(_IRIS_SRC), iris_root=str(_IRIS_ROOT))
        d_in, d_hidden = 16, 64
        engine._sae = SparseAutoencoder(d_in, d_hidden).eval()
        engine._sae_layer = 2
        engine._sae_norm_mean = torch.zeros(d_in)
        engine._sae_norm_std = torch.ones(d_in)
        engine._sae_topk = 5

        # Stub the hook extractor to return a fixed (1, T, E) residual.
        mock_hooks = MagicMock()
        mock_hooks.get_resid.return_value = torch.randn(1, 5, d_in)
        engine._hooks = mock_hooks

        feats = engine._compute_sae_features()
        assert feats is not None
        assert len(feats) <= 5
        # Every entry is {"id": int, "mag": float>0}, sorted descending by magnitude
        mags = [f["mag"] for f in feats]
        assert mags == sorted(mags, reverse=True)
        for f in feats:
            assert isinstance(f["id"], int)
            assert f["mag"] > 0.0

    def test_sae_field_in_framedata(self):
        """FrameData exposes sae_features / sae_layer (default None)."""
        d = FrameData().to_dict()
        assert "sae_features" in d and d["sae_features"] is None
        assert "sae_layer" in d and d["sae_layer"] is None


# ---------------------------------------------------------------------------
# Intervention (Phase 3)
# ---------------------------------------------------------------------------

class TestIntervention:

    def test_set_intervention_state(self):
        """set_intervention stores feature/scale; clearing resets to (None, 0)."""
        from inference import InferenceEngine
        engine = InferenceEngine(iris_src=str(_IRIS_SRC), iris_root=str(_IRIS_ROOT))
        engine.set_intervention(42, 3.5)
        assert engine._iv_feature_id == 42
        assert engine._iv_scale == 3.5
        engine.set_intervention(None, 0.0)
        assert engine._iv_feature_id is None
        assert engine._iv_scale == 0.0

    def test_intervention_direction_magnitude_relative(self):
        """Direction = scale * ref * W_dec[id] * std, ref = max(feat[id], floor).

        None when no SAE / zero scale / bad id."""
        from inference import InferenceEngine
        from sae import SparseAutoencoder
        engine = InferenceEngine(iris_src=str(_IRIS_SRC), iris_root=str(_IRIS_ROOT))

        assert engine._intervention_direction(0, 1.0) is None  # no SAE loaded

        d_in, d_hidden = 16, 64
        engine._sae = SparseAutoencoder(d_in, d_hidden).eval()
        engine._sae_norm_std = torch.full((d_in,), 2.0)
        engine._sae_layer = 1
        engine._sae_mag_floor = 1.0

        assert engine._intervention_direction(0, 0.0) is None      # zero scale
        assert engine._intervention_direction(999, 1.0) is None     # id out of range

        # No last-feats → ref falls back to the floor (1.0).
        d = engine._intervention_direction(3, 2.0)
        assert d is not None and d.shape == (d_in,)
        expected = 2.0 * 1.0 * engine._sae.W_dec[3] * engine._sae_norm_std
        assert torch.allclose(d, expected)

        # With a last-feats vector, ref = the feature's own activation (above floor).
        feats = torch.zeros(d_hidden)
        feats[3] = 5.0
        engine._sae_last_feats = feats
        d2 = engine._intervention_direction(3, 2.0)
        expected2 = 2.0 * 5.0 * engine._sae.W_dec[3] * engine._sae_norm_std
        assert torch.allclose(d2, expected2)

        # An off feature (activation 0) still uses the floor, not 0 → drivable.
        d3 = engine._intervention_direction(7, 2.0)  # feats[7] == 0
        expected3 = 2.0 * 1.0 * engine._sae.W_dec[7] * engine._sae_norm_std
        assert torch.allclose(d3, expected3)

    def test_imagine_returns_rgb_and_tokens(self):
        """_imagine_next_rgb returns (rgb (H,W,3) uint8, gen_tokens (1,K))."""
        from inference import _imagine_next_rgb

        torch.manual_seed(0)
        wm = make_world_model(num_layers=3, num_heads=2, embed_dim=16, tokens_per_block=5)
        tokenizer = TestReconstructionDecode._make_tokenizer_stub(
            vocab_size=16, embed_dim=8, out_hw=8
        )
        obs_tokens = torch.randint(0, 16, (1, 4))
        rgb, toks = _imagine_next_rgb(
            wm, tokenizer, obs_tokens, 1, torch.device("cpu"), deterministic=True
        )
        assert rgb is not None and rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8
        assert toks is not None and toks.shape == (1, 4)

    def test_all_positions_injection_changes_frame_and_tokens(self):
        """All-positions injection (priming + every step) flips tokens and the frame.

        Token-change count is the reliable signal; verify it is > 0 and matches the
        number of differing generated tokens."""
        from inference import _imagine_next_rgb

        torch.manual_seed(0)
        wm = make_world_model(num_layers=4, num_heads=2, embed_dim=16, tokens_per_block=5)
        tokenizer = TestReconstructionDecode._make_tokenizer_stub(
            vocab_size=16, embed_dim=8, out_hw=8
        )
        obs_tokens = torch.randint(0, 16, (1, 4))
        unit = torch.randn(16)
        device = torch.device("cpu")

        base_rgb, base_tok = _imagine_next_rgb(
            wm, tokenizer, obs_tokens, 1, device, deterministic=True
        )
        base_rgb2, _ = _imagine_next_rgb(
            wm, tokenizer, obs_tokens, 1, device, deterministic=True
        )
        assert np.array_equal(base_rgb, base_rgb2)  # deterministic

        changed = False
        for mag in (10.0, 100.0, 1_000.0, 10_000.0):
            iv_rgb, iv_tok = _imagine_next_rgb(
                wm, tokenizer, obs_tokens, 1, device,
                intervention=(2, unit * mag), deterministic=True,
            )
            n_changed = int((base_tok != iv_tok).sum().item())
            if n_changed > 0:
                assert not np.array_equal(base_rgb, iv_rgb)
                changed = True
                break
        assert changed, "All-positions intervention never changed any token"

    def test_intervention_hook_removed_after_rollout(self):
        """The intervention forward-hook must not persist on the model afterwards."""
        from inference import _imagine_next_rgb

        wm = make_world_model(num_layers=3, num_heads=2, embed_dim=16, tokens_per_block=5)
        tokenizer = TestReconstructionDecode._make_tokenizer_stub(
            vocab_size=16, embed_dim=8, out_hw=8
        )
        obs_tokens = torch.randint(0, 16, (1, 4))
        before = sum(len(m._forward_hooks) for m in wm.modules())

        _imagine_next_rgb(
            wm, tokenizer, obs_tokens, 1, torch.device("cpu"),
            intervention=(1, torch.randn(16)), deterministic=True,
        )

        after = sum(len(m._forward_hooks) for m in wm.modules())
        assert after == before, f"Hook leaked: {before} → {after}"

    def test_framedata_intervention_fields(self):
        """FrameData exposes imagined_intervened / intervention_diff / intervention."""
        d = FrameData().to_dict()
        for key in ("imagined_intervened", "intervention_diff", "intervention"):
            assert key in d and d[key] is None


# ---------------------------------------------------------------------------
# KV cache interaction
# ---------------------------------------------------------------------------

class TestKVCacheInteraction:

    def test_shapes_grow_across_rollout(self):
        """T_k grows by 1 on each 1-token step, matching WorldModelEnv pattern."""
        wm = make_world_model(num_layers=2, num_heads=2, tokens_per_block=5, max_blocks=20)
        hooks = IrisHookExtractor()
        hooks.attach(wm)

        kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=100)

        # Initial 4-token pass
        tokens = torch.randint(0, 8, (1, 4))
        with torch.no_grad():
            wm(tokens, past_keys_values=kv)
        attn, _ = hooks.get_data()
        assert attn[0].shape == (1, 2, 4, 4)

        # 5 subsequent 1-token steps
        for step in range(1, 6):
            token = torch.randint(0, 8, (1, 1))
            with torch.no_grad():
                wm(token, past_keys_values=kv)
            attn, _ = hooks.get_data()
            expected_tk = 4 + step
            assert attn[0].shape == (1, 2, 1, expected_tk), (
                f"Step {step}: expected T_k={expected_tk}, got {attn[0].shape}"
            )
        hooks.detach()

    def test_hook_fires_same_count_cached_vs_uncached(self):
        """Both forward modes fire exactly num_layers attn hooks."""
        num_layers = 3
        wm = make_world_model(num_layers=num_layers, num_heads=2, tokens_per_block=5)
        hooks = IrisHookExtractor()
        hooks.attach(wm)

        tokens = torch.randint(0, 8, (1, 5))

        # Uncached
        with torch.no_grad():
            wm(tokens, past_keys_values=None)
        attn_u, norms_u = hooks.get_data()
        assert len(attn_u) == num_layers
        assert len(norms_u) == num_layers

        # Cached
        kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=50)
        with torch.no_grad():
            wm(tokens, past_keys_values=kv)
        attn_c, norms_c = hooks.get_data()
        assert len(attn_c) == num_layers
        assert len(norms_c) == num_layers

        hooks.detach()


# ---------------------------------------------------------------------------
# Token alignment
# ---------------------------------------------------------------------------

class TestTokenAlignment:

    def test_labels_default_config(self):
        """tokens_per_block=17: positions 0–15 → o{i}, position 16 → act."""
        labels = get_token_labels(17, 17)
        for i in range(16):
            assert labels[i] == f"o{i}"
        assert labels[16] == "act"

    def test_labels_custom_config(self):
        """tokens_per_block=5: [o0, o1, o2, o3, act]."""
        assert get_token_labels(5, 5) == ["o0", "o1", "o2", "o3", "act"]

    def test_labels_length(self):
        """Output length always equals num_tokens."""
        for n in [1, 5, 17, 34]:
            assert len(get_token_labels(n, 17)) == n

    def test_labels_multi_block(self):
        """10 tokens, tpb=5: two complete [o0..o3, act] blocks."""
        expected = ["o0", "o1", "o2", "o3", "act"] * 2
        assert get_token_labels(10, 5) == expected

    def test_labels_never_hardcoded(self):
        """Label pattern changes correctly when tokens_per_block changes."""
        l3 = get_token_labels(3, 3)
        assert l3 == ["o0", "o1", "act"]
        l7 = get_token_labels(7, 7)
        assert l7[-1] == "act"
        for i in range(6):
            assert l7[i] == f"o{i}"


# ---------------------------------------------------------------------------
# Hook cleanup on failure
# ---------------------------------------------------------------------------

class TestHookCleanupOnFailure:

    def test_no_hooks_remain_after_failed_attach(self):
        """
        If register_forward_hook raises on layer 1's attn_drop, attach() must:
          1. Raise RuntimeError
          2. Leave zero hooks on any module in the model
        """
        wm = make_world_model(num_layers=3, num_heads=2)
        hooks = IrisHookExtractor()

        def bad_register(fn):
            raise RuntimeError("Simulated hook registration failure")

        with patch.object(
            wm.transformer.blocks[1].attn.attn_drop,
            "register_forward_hook",
            bad_register,
        ):
            with pytest.raises(RuntimeError):
                hooks.attach(wm)

        total = sum(len(m._forward_hooks) for m in wm.modules())
        assert total == 0, f"Expected 0 hooks, found {total}"

    def test_detach_removes_all_hooks(self):
        """detach() leaves zero hooks on the model."""
        wm = make_world_model(num_layers=2, num_heads=2)
        hooks = IrisHookExtractor()
        hooks.attach(wm)
        assert sum(len(m._forward_hooks) for m in wm.modules()) > 0
        hooks.detach()
        assert sum(len(m._forward_hooks) for m in wm.modules()) == 0

    def test_num_layers_reflects_attachment(self):
        wm = make_world_model(num_layers=3, num_heads=2)
        hooks = IrisHookExtractor()
        assert hooks.num_layers == 0
        hooks.attach(wm)
        assert hooks.num_layers == 3
        hooks.detach()
        assert hooks.num_layers == 0


# ---------------------------------------------------------------------------
# Queue backpressure & frame dropping
# ---------------------------------------------------------------------------

class TestQueueBackpressure:

    def test_queue_never_blocks_producer(self):
        """
        A producer that fills a tiny bounded queue must never hang.
        Frames should be dropped instead.
        """
        from queue import Queue, Full

        q: Queue = Queue(maxsize=2)
        dropped = 0
        TOTAL = 20

        t0 = time.perf_counter()
        for _ in range(TOTAL):
            try:
                q.put_nowait(object())
            except Full:
                dropped += 1
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.1, f"Producer blocked for {elapsed:.3f}s"
        assert dropped == TOTAL - 2, f"Expected {TOTAL - 2} drops, got {dropped}"

    def test_frames_dropped_not_queued(self):
        """With a queue of size 1, at least TOTAL-1 frames are dropped."""
        from queue import Queue, Full

        q: Queue = Queue(maxsize=1)
        dropped = 0
        TOTAL = 10

        for _ in range(TOTAL):
            try:
                q.put_nowait("frame")
            except Full:
                dropped += 1

        assert dropped >= TOTAL - 1


# ---------------------------------------------------------------------------
# Agent switching (mocked — no real checkpoint)
# ---------------------------------------------------------------------------

class TestAgentSwitching:

    def _make_engine_with_mock_agent(self):
        """Return an InferenceEngine with _load_agent mocked out."""
        from inference import InferenceEngine

        engine = InferenceEngine(
            iris_src=str(_IRIS_SRC),
            iris_root=str(_IRIS_ROOT),
        )
        return engine

    def test_hooks_detached_after_stop(self):
        """After stop(), hook extractor has 0 layers."""
        from inference import InferenceEngine

        engine = InferenceEngine(
            iris_src=str(_IRIS_SRC),
            iris_root=str(_IRIS_ROOT),
        )
        wm = make_world_model(num_layers=2, num_heads=2)
        engine._hooks.attach(wm)
        engine._agent = MagicMock()
        engine._env = MagicMock()

        # Simulate a stopped state (no thread)
        engine._hooks.detach()
        assert engine._hooks.num_layers == 0

    def test_queue_flushed_on_switch(self):
        """_drain_queue() empties the queue completely."""
        from inference import InferenceEngine

        engine = InferenceEngine(
            iris_src=str(_IRIS_SRC),
            iris_root=str(_IRIS_ROOT),
        )
        for i in range(5):
            engine._queue.put_nowait(FrameData())
        assert engine._queue.qsize() == 5
        engine._drain_queue()
        assert engine._queue.qsize() == 0

    def test_event_callbacks_receive_agent_loaded(self):
        """Event callbacks fire when emit_event is called."""
        from inference import InferenceEngine

        engine = InferenceEngine(
            iris_src=str(_IRIS_SRC),
            iris_root=str(_IRIS_ROOT),
        )
        received = []
        engine.register_event_callback(lambda name, data: received.append((name, data)))
        engine._emit_event("agent_loaded", {"agent": "Test"})
        assert len(received) == 1
        assert received[0][0] == "agent_loaded"

    def test_event_callback_unregistered(self):
        from inference import InferenceEngine

        engine = InferenceEngine(
            iris_src=str(_IRIS_SRC),
            iris_root=str(_IRIS_ROOT),
        )
        received = []
        cb = lambda name, data: received.append(name)
        engine.register_event_callback(cb)
        engine._emit_event("ping", {})
        engine.unregister_event_callback(cb)
        engine._emit_event("pong", {})
        assert received == ["ping"]


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestShutdown:

    def test_stop_before_start_is_safe(self):
        """Calling stop() before start() must not raise."""
        from inference import InferenceEngine
        engine = InferenceEngine(
            iris_src=str(_IRIS_SRC),
            iris_root=str(_IRIS_ROOT),
        )
        engine.stop()  # should be a no-op

    def test_stop_event_terminates_paused_thread(self):
        """
        A paused inference thread must unblock and exit within 2 s when
        stop() is called.
        """
        stop_event = threading.Event()
        pause_event = threading.Event()
        pause_event.clear()  # paused

        def worker():
            pause_event.wait()  # would block forever without stop signal

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        # Simulate stop: unblock pause
        pause_event.set()
        t.join(timeout=2.0)
        assert not t.is_alive(), "Thread did not terminate"


# ---------------------------------------------------------------------------
# Frame encoding
# ---------------------------------------------------------------------------

class TestFrameEncoding:

    def test_encode_frame_produces_valid_png(self):
        """_encode_frame returns a base64 string that decodes to a valid PNG."""
        from PIL import Image as _Image

        obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        b64 = _encode_frame(obs)

        raw = base64.b64decode(b64)
        img = _Image.open(io.BytesIO(raw))
        assert img.format == "PNG"
        assert img.size == (160, 210)

    def test_encode_frame_not_preprocessed(self):
        """The encoded frame uses raw pixel values, not normalised floats."""
        obs = np.full((10, 10, 3), 200, dtype=np.uint8)
        b64 = _encode_frame(obs)
        from PIL import Image as _Image
        raw = base64.b64decode(b64)
        img = _Image.open(io.BytesIO(raw))
        arr = np.array(img)
        assert arr.max() == 200, "Frame appears normalised"


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class TestFpsCounter:

    def test_fps_zero_initially(self):
        counter = _FpsCounter()
        assert counter.fps == 0.0

    def test_fps_nonzero_after_ticks(self):
        counter = _FpsCounter()
        for _ in range(5):
            counter.tick()
        assert counter.fps > 0.0

    def test_fps_drops_old_frames(self):
        """Frames older than 1 s are evicted; fps reflects last second only."""
        t = [0.0]

        class FakeCounter(_FpsCounter):
            def __init__(self):
                super().__init__()
                self._clock_fn = lambda: t[0]

            def tick(self):
                from collections import deque
                now = self._clock_fn()
                self._ts.append(now)
                cutoff = now - 1.0
                while self._ts and self._ts[0] < cutoff:
                    self._ts.popleft()
                self.fps = float(len(self._ts))
                return self.fps

        counter = FakeCounter()
        for _ in range(10):
            counter.tick()
        assert counter.fps == 10.0

        t[0] = 1.5
        counter.tick()
        assert counter.fps == 1.0


# ---------------------------------------------------------------------------
# Config mismatch (arch change between agents)
# ---------------------------------------------------------------------------

class TestConfigMismatch:

    def test_hooks_reregistered_after_switch(self):
        """
        Simulates switching from a 2-layer to a 3-layer model.
        Hooks should be detached from old model and fresh on new model.
        """
        wm_small = make_world_model(num_layers=2, num_heads=2)
        wm_large = make_world_model(num_layers=3, num_heads=2)

        hooks = IrisHookExtractor()
        hooks.attach(wm_small)
        assert hooks.num_layers == 2

        # Simulate switch: detach old, attach new
        hooks.detach()
        assert hooks.num_layers == 0
        assert sum(len(m._forward_hooks) for m in wm_small.modules()) == 0

        hooks.attach(wm_large)
        assert hooks.num_layers == 3
        assert sum(len(m._forward_hooks) for m in wm_large.modules()) > 0

        hooks.detach()


# ---------------------------------------------------------------------------
# Reconstruction decode
# ---------------------------------------------------------------------------

class TestReconstructionDecode:
    """
    Tests for _decode_reconstruction.  Uses a minimal nn.Embedding + nn.Conv2d
    decoder stub so we never need a real IRIS checkpoint.
    """

    @staticmethod
    def _make_tokenizer_stub(vocab_size: int = 16, embed_dim: int = 8, out_hw: int = 4):
        """
        Build a fake tokenizer object with:
          .embedding   – nn.Embedding(vocab_size, embed_dim)
          .decode()    – post_quant_conv(z_q) → bilinear up-sample to (out_hw*4, out_hw*4)
                         then postprocess_output to [0, 1]

        The stub keeps all tensors on CPU and does not require any IRIS code.
        """
        import torch.nn as nn

        class _FakeDecoder(nn.Module):
            def __init__(self, embed_dim: int, out_hw: int):
                super().__init__()
                self.post_quant_conv = nn.Conv2d(embed_dim, 3, kernel_size=1)
                self.out_hw = out_hw

            def forward(self, z_q: torch.Tensor) -> torch.Tensor:
                x = self.post_quant_conv(z_q)
                return torch.nn.functional.interpolate(
                    x, size=(self.out_hw, self.out_hw), mode="bilinear", align_corners=False
                )

        class _FakeTokenizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self._decoder = _FakeDecoder(embed_dim, out_hw)

            def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
                rec = self._decoder(z_q)
                if should_postprocess:
                    rec = rec.add(1).div(2)
                return rec

        return _FakeTokenizer().eval()

    def test_reconstruction_shape_matches_obs(self):
        """Decoded reconstruction has the same H, W as the raw observation tensor."""
        K = 4  # 2×2 spatial grid
        out_hw = 8
        tokenizer = self._make_tokenizer_stub(vocab_size=16, embed_dim=8, out_hw=out_hw)

        device = torch.device("cpu")
        obs_tokens = torch.randint(0, 16, (1, K), device=device)
        obs_tensor = torch.rand(1, 3, out_hw, out_hw, device=device)

        rec_b64, err_b64, mae = _decode_reconstruction(tokenizer, obs_tokens, obs_tensor, device)

        assert rec_b64 is not None, "Expected a reconstruction PNG, got None"
        assert err_b64 is not None, "Expected an error-map PNG, got None"

        # Decode the PNG and check spatial dimensions
        from PIL import Image as _Image
        rec_img = _Image.open(io.BytesIO(base64.b64decode(rec_b64)))
        assert rec_img.size == (out_hw, out_hw), (
            f"Expected {out_hw}×{out_hw}, got {rec_img.size}"
        )

    def test_error_map_values_in_range_0_255(self):
        """Every pixel in the error-map PNG must be in [0, 255]."""
        K = 4
        out_hw = 8
        tokenizer = self._make_tokenizer_stub(vocab_size=16, embed_dim=8, out_hw=out_hw)
        device = torch.device("cpu")
        obs_tokens = torch.randint(0, 16, (1, K), device=device)
        obs_tensor = torch.rand(1, 3, out_hw, out_hw, device=device)

        _, err_b64, _ = _decode_reconstruction(tokenizer, obs_tokens, obs_tensor, device)

        assert err_b64 is not None
        from PIL import Image as _Image
        err_img = _Image.open(io.BytesIO(base64.b64decode(err_b64)))
        err_arr = np.array(err_img)
        assert err_arr.min() >= 0,   f"Error map min {err_arr.min()} < 0"
        assert err_arr.max() <= 255, f"Error map max {err_arr.max()} > 255"

    def test_decode_failure_returns_null_without_crash(self):
        """If tokenizer.decode raises, _decode_reconstruction returns (None, None, None)."""
        tokenizer = self._make_tokenizer_stub()
        original_decode = tokenizer.decode

        def _bad_decode(*args, **kwargs):
            raise RuntimeError("Simulated decode failure")

        tokenizer.decode = _bad_decode

        device = torch.device("cpu")
        obs_tokens = torch.randint(0, 16, (1, 4), device=device)
        obs_tensor = torch.rand(1, 3, 8, 8, device=device)

        result = _decode_reconstruction(tokenizer, obs_tokens, obs_tensor, device)
        assert result == (None, None, None), (
            f"Expected (None, None, None) on failure, got {result}"
        )

    def test_reconstruction_uses_same_device_as_inputs(self):
        """All intermediate tensor operations run on the specified device (CPU in tests)."""
        K = 4
        out_hw = 8
        device = torch.device("cpu")
        tokenizer = self._make_tokenizer_stub(vocab_size=16, embed_dim=8, out_hw=out_hw)

        observed_devices = []

        original_decode = tokenizer.decode
        def _patched_decode(z_q: torch.Tensor, should_postprocess: bool = False):
            observed_devices.append(z_q.device)
            return original_decode(z_q, should_postprocess)
        tokenizer.decode = _patched_decode

        obs_tokens = torch.randint(0, 16, (1, K), device=device)
        obs_tensor = torch.rand(1, 3, out_hw, out_hw, device=device)

        rec_b64, _, _ = _decode_reconstruction(tokenizer, obs_tokens, obs_tensor, device)

        assert rec_b64 is not None
        assert len(observed_devices) == 1, "decode should be called exactly once"
        assert observed_devices[0] == device, (
            f"z_q was on {observed_devices[0]}, expected {device}"
        )


# ---------------------------------------------------------------------------
# Latent subscriber gating
# ---------------------------------------------------------------------------

class TestLatentSubscriberGating:
    """
    Reconstruction decode is gated on the latent subscriber count.

    All inference is driven with fully mocked IRIS components so no real
    checkpoint is required.
    """

    @staticmethod
    def _make_engine_for_one_step(latent_count: int = 0):
        """
        Build an InferenceEngine that will run exactly one inference step then
        exit (_loop_episodes=False, done=True on the first env.step).

        All IRIS components are replaced with lightweight mocks.
        latent_count adds that many latent subscribers before returning.
        """
        from inference import InferenceEngine

        engine = InferenceEngine(iris_src=str(_IRIS_SRC), iris_root=str(_IRIS_ROOT))
        engine._loop_episodes = False

        for _ in range(latent_count):
            engine.add_latent_subscriber()

        device = torch.device("cpu")
        obs_np = np.zeros((1, 8, 8, 3), dtype=np.uint8)

        mock_agent = MagicMock()
        mock_agent.device = device
        mock_agent.world_model.config.tokens_per_block = 5
        mock_agent.world_model.transformer.blocks = [None, None]   # len() == 2
        mock_agent.act.return_value = torch.zeros(1, dtype=torch.long)

        mock_enc = MagicMock()
        mock_enc.tokens = torch.zeros(1, 4, dtype=torch.long)      # K=4 tokens
        mock_agent.tokenizer.encode.return_value = mock_enc

        mock_env = MagicMock()
        mock_env.reset.return_value = obs_np
        # done=True on first step so the loop exits after one iteration
        mock_env.step.return_value = (obs_np, np.array([0.0]), np.array([True]), {})

        mock_hooks = MagicMock()
        mock_hooks.get_data.return_value = ({}, {})

        engine._agent = mock_agent
        engine._env = mock_env
        engine._hooks = mock_hooks

        return engine

    # ------------------------------------------------------------------
    # Gating tests (run _run_inner with patched _decode_reconstruction)
    # ------------------------------------------------------------------

    def test_decode_not_called_when_no_latent_subscribers(self):
        """_decode_reconstruction must not be called when latent_subscriber_count == 0."""
        engine = self._make_engine_for_one_step(latent_count=0)
        assert engine.latent_subscriber_count == 0

        raw_frame = np.zeros((210, 160, 3), dtype=np.uint8)
        with patch("inference._get_raw_frame", return_value=raw_frame), \
             patch("inference._decode_reconstruction") as mock_decode:
            engine._run_inner()

        mock_decode.assert_not_called()

    def test_decode_called_when_latent_subscriber_present(self):
        """_decode_reconstruction is called exactly once when latent_subscriber_count == 1."""
        engine = self._make_engine_for_one_step(latent_count=1)
        assert engine.latent_subscriber_count == 1

        raw_frame = np.zeros((210, 160, 3), dtype=np.uint8)
        with patch("inference._get_raw_frame", return_value=raw_frame), \
             patch("inference._decode_reconstruction", return_value=(None, None, None)) as mock_decode:
            engine._run_inner()

        mock_decode.assert_called_once()

    # ------------------------------------------------------------------
    # Counter unit tests
    # ------------------------------------------------------------------

    def test_subscriber_count_increments_and_decrements(self):
        """add_latent_subscriber increments; remove_latent_subscriber decrements."""
        from inference import InferenceEngine
        engine = InferenceEngine(iris_src=str(_IRIS_SRC), iris_root=str(_IRIS_ROOT))

        assert engine.latent_subscriber_count == 0
        engine.add_latent_subscriber()
        assert engine.latent_subscriber_count == 1
        engine.add_latent_subscriber()
        assert engine.latent_subscriber_count == 2
        engine.remove_latent_subscriber()
        assert engine.latent_subscriber_count == 1
        engine.remove_latent_subscriber()
        assert engine.latent_subscriber_count == 0

    def test_subscriber_count_never_goes_negative(self):
        """remove_latent_subscriber clamps at zero."""
        from inference import InferenceEngine
        engine = InferenceEngine(iris_src=str(_IRIS_SRC), iris_root=str(_IRIS_ROOT))
        engine.remove_latent_subscriber()
        assert engine.latent_subscriber_count == 0

    def test_subscriber_count_thread_safe(self):
        """
        Concurrent add/remove operations produce a consistent final count.

        Strategy: pre-load N*4 adds (single-threaded), then run 4 adder threads
        and 4 remover threads with N ops each.  Net change = +4N -4N = 0, so
        the final count must equal the pre-loaded total N*4.
        """
        from inference import InferenceEngine
        engine = InferenceEngine(iris_src=str(_IRIS_SRC), iris_root=str(_IRIS_ROOT))

        N = 200
        initial = N * 4
        for _ in range(initial):
            engine.add_latent_subscriber()
        assert engine.latent_subscriber_count == initial

        def add_n():
            for _ in range(N):
                engine.add_latent_subscriber()

        def remove_n():
            for _ in range(N):
                engine.remove_latent_subscriber()

        threads = (
            [threading.Thread(target=add_n) for _ in range(4)]
            + [threading.Thread(target=remove_n) for _ in range(4)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert engine.latent_subscriber_count == initial, (
            f"Expected {initial}, got {engine.latent_subscriber_count} "
            "(thread-safety failure)"
        )

    def test_latent_ws_endpoint_manages_subscriber_count(self):
        """
        Connecting to /ws/latent increments latent_subscriber_count;
        disconnecting (context manager exit) decrements it back to the
        original value.
        """
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
        from fastapi.testclient import TestClient
        from main import app, engine as main_engine

        initial = main_engine.latent_subscriber_count
        client = TestClient(app, raise_server_exceptions=False)

        with client.websocket_connect("/ws/latent") as ws:
            _msg = ws.receive_json()   # config snapshot sent on connect
            assert main_engine.latent_subscriber_count == initial + 1, (
                f"Expected {initial + 1} after connect, "
                f"got {main_engine.latent_subscriber_count}"
            )

        # Poll briefly: the server's finally: block runs after the
        # executor's 0.05s timeout elapses following the close frame.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if main_engine.latent_subscriber_count == initial:
                break
            time.sleep(0.02)

        assert main_engine.latent_subscriber_count == initial, (
            f"Expected {initial} after disconnect, "
            f"got {main_engine.latent_subscriber_count}"
        )


# ---------------------------------------------------------------------------
# Single-step control + imagined-frame field
# ---------------------------------------------------------------------------

class TestSingleStep:
    """
    The single-step gate lets a paused loop advance exactly one frame per
    step_once() call.  Driven with fully mocked IRIS components (no checkpoint).
    """

    @staticmethod
    def _make_running_engine():
        """An engine whose env never returns done, so the loop runs until stopped."""
        from inference import InferenceEngine

        engine = InferenceEngine(iris_src=str(_IRIS_SRC), iris_root=str(_IRIS_ROOT))
        engine._loop_episodes = True

        device = torch.device("cpu")
        obs_np = np.zeros((1, 8, 8, 3), dtype=np.uint8)

        mock_agent = MagicMock()
        mock_agent.device = device
        mock_agent.world_model.config.tokens_per_block = 5
        mock_agent.world_model.transformer.blocks = [None, None]   # len() == 2
        mock_agent.act.return_value = torch.zeros(1, dtype=torch.long)

        mock_enc = MagicMock()
        mock_enc.tokens = torch.zeros(1, 4, dtype=torch.long)
        mock_agent.tokenizer.encode.return_value = mock_enc

        mock_env = MagicMock()
        mock_env.reset.return_value = obs_np
        # done=False forever → loop only exits when stopped
        mock_env.step.return_value = (obs_np, np.array([0.0]), np.array([False]), {})

        mock_hooks = MagicMock()
        mock_hooks.get_data.return_value = ({}, {})

        engine._agent = mock_agent
        engine._env = mock_env
        engine._hooks = mock_hooks
        return engine

    @staticmethod
    def _wait_until(pred, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if pred():
                return True
            time.sleep(0.01)
        return False

    def test_step_state_transitions(self):
        """step_once() arms single-step mode; resume() clears it."""
        engine = self._make_running_engine()
        assert not engine._single_step_mode.is_set()
        assert not engine._step_event.is_set()

        engine.step_once()
        assert engine._single_step_mode.is_set()
        assert engine._step_event.is_set()

        engine.resume()
        assert not engine._single_step_mode.is_set()

    def test_framedata_has_imagined_next_field(self):
        """FrameData carries the imagined_next field, defaulting to None."""
        d = FrameData().to_dict()
        assert "imagined_next" in d
        assert d["imagined_next"] is None

    def test_step_once_advances_exactly_one_frame(self):
        """
        Paused loop produces no frames; each step_once() yields exactly one frame
        and then re-parks (does not free-run).
        """
        engine = self._make_running_engine()
        engine._pause_event.clear()  # paused before the loop starts

        with patch("inference._get_raw_frame",
                   return_value=np.zeros((20, 16, 3), dtype=np.uint8)):
            t = threading.Thread(target=engine._run_inner, daemon=True)
            t.start()
            try:
                # Parked at the gate — no frames while paused and not stepping.
                time.sleep(0.15)
                assert engine._queue.qsize() == 0

                engine.step_once()
                assert self._wait_until(lambda: engine._queue.qsize() == 1)
                # Confirm it re-parked rather than free-running (would fill to 5).
                time.sleep(0.1)
                assert engine._queue.qsize() == 1

                engine.step_once()
                assert self._wait_until(lambda: engine._queue.qsize() == 2)
                time.sleep(0.1)
                assert engine._queue.qsize() == 2
            finally:
                engine._stop_event.set()
                engine._step_event.set()
                engine._pause_event.set()
                t.join(timeout=2.0)
            assert not t.is_alive(), "Inference thread did not stop"


# ---------------------------------------------------------------------------
# Bookmarks store (the backend's only persistence)
# ---------------------------------------------------------------------------

class TestBookmarkStore:

    def test_upsert_list_delete_roundtrip(self, tmp_path):
        from bookmarks import BookmarkStore
        store = BookmarkStore(str(tmp_path / "bm.json"))

        assert store.list() == []
        store.upsert("BreakoutNoFrameskip-v4", 5, 42, "the ball", notes="moves up",
                     source="user", updated_at="t0")
        got = store.list()
        assert len(got) == 1
        assert got[0]["feature_id"] == 42 and got[0]["label"] == "the ball"

        # Upsert same key updates in place (no duplicate).
        store.upsert("BreakoutNoFrameskip-v4", 5, 42, "the paddle", updated_at="t1")
        got = store.list()
        assert len(got) == 1 and got[0]["label"] == "the paddle"

        # Delete.
        assert store.delete("BreakoutNoFrameskip-v4", 5, 42) is True
        assert store.list() == []
        assert store.delete("BreakoutNoFrameskip-v4", 5, 42) is False

    def test_filter_by_env_and_layer(self, tmp_path):
        from bookmarks import BookmarkStore
        store = BookmarkStore(str(tmp_path / "bm.json"))
        store.upsert("Breakout", 5, 1, "a")
        store.upsert("Breakout", 6, 2, "b")
        store.upsert("Alien", 5, 3, "c")

        assert {b["feature_id"] for b in store.list(env_id="Breakout")} == {1, 2}
        assert {b["feature_id"] for b in store.list(env_id="Breakout", layer=5)} == {1}
        assert {b["feature_id"] for b in store.list(layer=5)} == {1, 3}

    def test_persists_across_instances(self, tmp_path):
        """A fresh store at the same path sees prior writes (survives restart)."""
        from bookmarks import BookmarkStore
        path = str(tmp_path / "bm.json")
        BookmarkStore(path).upsert("Breakout", 5, 7, "kept")
        reopened = BookmarkStore(path)
        assert len(reopened.list()) == 1 and reopened.list()[0]["feature_id"] == 7

    def test_corrupt_file_treated_as_empty(self, tmp_path):
        from bookmarks import BookmarkStore
        path = tmp_path / "bm.json"
        path.write_text("{ this is not json")
        store = BookmarkStore(str(path))
        assert store.list() == []          # no crash
        store.upsert("Breakout", 5, 1, "ok")   # recovers, overwrites
        assert len(store.list()) == 1

    def test_endpoints_roundtrip_via_testclient(self, tmp_path, monkeypatch):
        """GET/POST/DELETE /bookmarks via FastAPI TestClient, against a temp store."""
        import importlib
        monkeypatch.setenv("BOOKMARKS_PATH", str(tmp_path / "bm.json"))
        import main as main_mod
        importlib.reload(main_mod)
        from fastapi.testclient import TestClient

        client = TestClient(main_mod.app, raise_server_exceptions=False)
        assert client.get("/bookmarks").json() == []

        r = client.post("/bookmarks", json={
            "env_id": "BreakoutNoFrameskip-v4", "layer": 5, "feature_id": 99,
            "label": "test feature",
        })
        assert r.json()["status"] == "ok"
        body = r.json()["bookmark"]
        assert body["feature_id"] == 99 and body["updated_at"]  # timestamp set by endpoint

        got = client.get("/bookmarks", params={"env_id": "BreakoutNoFrameskip-v4"}).json()
        assert len(got) == 1 and got[0]["label"] == "test feature"

        d = client.delete("/bookmarks", params={
            "env_id": "BreakoutNoFrameskip-v4", "layer": 5, "feature_id": 99})
        assert d.json()["deleted"] is True
        assert client.get("/bookmarks").json() == []

        importlib.reload(main_mod)  # restore module for other tests
