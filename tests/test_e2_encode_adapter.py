"""Unit tests for E2 MiniCPM5+Adapter prompt encoding.

Pure CPU / small-tensor tests.  No real MiniCPM5 model, no MPS, no checkpoint
weights are loaded.  The save/load round-trip, metadata contract, adapter
config parsing, and sanity-metric primitives are exercised with tiny synthetic
tensors.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from wan.utils.prompt_embedding import (  # noqa: E402
    EXPECTED_HIDDEN_DIM,
    FORMAT_VERSION,
    load_prompt_embedding,
)
from scripts.e1_encode_with_adapter import (  # noqa: E402
    MODEL_ID,
    compute_sanity_metrics_from_frames,
    load_adapter_config,
    save_minicpm_embedding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_fake_adapter_config(tmpdir: str, **overrides) -> str:
    """Write a minimal adapter config.json to a temp dir."""
    cfg = {
        "hidden_dim": 2048,
        "output_dim": 4096,
        "num_queries": 64,
        "num_resampler_layers": 2,
        "num_heads": 8,
        "ffn_mult": 4,
        "dtype": "float32",
    }
    cfg.update(overrides)
    path = os.path.join(tmpdir, "config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return path


def _write_fake_weights(tmpdir: str) -> str:
    """Write a tiny dummy .safetensors (sha256 only needs a real file)."""
    from safetensors.torch import save_file
    path = os.path.join(tmpdir, "adapter.safetensors")
    save_file({"dummy": torch.zeros(4)}, path)
    return path


# ---------------------------------------------------------------------------
# Tests 1-5: save/load round-trip + metadata contract
# ---------------------------------------------------------------------------

class TestOutputFormatCompatible(unittest.TestCase):
    """Mock context [64,4096] bf16 -> save -> load_prompt_embedding round-trip."""

    def test_output_format_compatible(self):
        tmpdir = tempfile.mkdtemp()
        out = os.path.join(tmpdir, "emb.safetensors")
        w = _write_fake_weights(tmpdir)
        c = _write_fake_adapter_config(tmpdir)
        ctx = torch.randn(64, 4096, dtype=torch.bfloat16)
        prompt = "Move the camera slowly forward."

        save_minicpm_embedding(out, ctx, prompt, token_count=12,
                               adapter_weights_path=w, adapter_config_path=c)
        loaded, meta = load_prompt_embedding(
            out, expected_prompt=prompt, expected_hidden_dim=4096,
            max_text_len=512,
        )
        self.assertEqual(tuple(loaded.shape), (64, 4096))
        self.assertEqual(loaded.dtype, torch.bfloat16)
        self.assertEqual(meta["hidden_dim"], 4096)
        self.assertEqual(meta["format_version"], FORMAT_VERSION)
        self.assertEqual(EXPECTED_HIDDEN_DIM, 4096)


class TestMetadataComplete(unittest.TestCase):
    """Sidecar JSON must contain all required keys."""

    REQUIRED_KEYS = {
        "prompt_sha256", "dtype", "shape", "hidden_dim",
        "text_len", "model_id", "format_version",
    }

    def test_metadata_complete(self):
        tmpdir = tempfile.mkdtemp()
        out = os.path.join(tmpdir, "emb.safetensors")
        w = _write_fake_weights(tmpdir)
        c = _write_fake_adapter_config(tmpdir)
        ctx = torch.randn(64, 4096, dtype=torch.bfloat16)
        prompt = "Pan the camera to the right."

        save_minicpm_embedding(out, ctx, prompt, token_count=15,
                               adapter_weights_path=w, adapter_config_path=c)
        json_path = out.replace(".safetensors", ".json")
        with open(json_path) as fh:
            meta = json.load(fh)

        for key in self.REQUIRED_KEYS:
            self.assertIn(key, meta, f"missing metadata key: {key}")
        # Extra E2 fields
        self.assertIn("token_count", meta)
        self.assertIn("adapter_weights", meta)
        self.assertIn("adapter_weights_sha256", meta)


class TestPromptSha256Correct(unittest.TestCase):
    """prompt_sha256 must match hashlib.sha256(prompt).hexdigest()."""

    def test_prompt_sha256_correct(self):
        tmpdir = tempfile.mkdtemp()
        out = os.path.join(tmpdir, "emb.safetensors")
        w = _write_fake_weights(tmpdir)
        c = _write_fake_adapter_config(tmpdir)
        ctx = torch.randn(64, 4096, dtype=torch.bfloat16)
        prompt = "Glide forward along the Great Wall."

        save_minicpm_embedding(out, ctx, prompt, token_count=11,
                               adapter_weights_path=w, adapter_config_path=c)
        json_path = out.replace(".safetensors", ".json")
        with open(json_path) as fh:
            meta = json.load(fh)

        expected = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.assertEqual(meta["prompt_sha256"], expected)


class TestModelIdMinicpm(unittest.TestCase):
    """model_id must be 'minicpm5-2b+adapter', NOT 'umt5-xxl'."""

    def test_model_id_minicpm(self):
        tmpdir = tempfile.mkdtemp()
        out = os.path.join(tmpdir, "emb.safetensors")
        w = _write_fake_weights(tmpdir)
        c = _write_fake_adapter_config(tmpdir)
        ctx = torch.randn(64, 4096, dtype=torch.bfloat16)
        prompt = "Fly forward toward the castle."

        save_minicpm_embedding(out, ctx, prompt, token_count=10,
                               adapter_weights_path=w, adapter_config_path=c)
        json_path = out.replace(".safetensors", ".json")
        with open(json_path) as fh:
            meta = json.load(fh)

        self.assertEqual(meta["model_id"], "minicpm5-2b+adapter")
        self.assertNotEqual(meta["model_id"], "umt5-xxl")
        self.assertEqual(MODEL_ID, "minicpm5-2b+adapter")


class TestContextShape64x4096(unittest.TestCase):
    """Saved context must be [64, 4096], not [1, 64, 4096]."""

    def test_context_shape_64x4096(self):
        tmpdir = tempfile.mkdtemp()
        out = os.path.join(tmpdir, "emb.safetensors")
        w = _write_fake_weights(tmpdir)
        c = _write_fake_adapter_config(tmpdir)
        # Simulate adapter output [1, 64, 4096] -> squeeze -> [64, 4096]
        adapter_out = torch.randn(1, 64, 4096, dtype=torch.bfloat16)
        ctx_squeezed = adapter_out[0].contiguous()
        self.assertEqual(tuple(ctx_squeezed.shape), (64, 4096))
        self.assertNotEqual(ctx_squeezed.dim(), 3)

        save_minicpm_embedding(out, ctx_squeezed, "test", token_count=5,
                               adapter_weights_path=w, adapter_config_path=c)
        loaded, _ = load_prompt_embedding(out, expected_prompt="test")
        self.assertEqual(tuple(loaded.shape), (64, 4096))
        self.assertEqual(loaded.dim(), 2)


# ---------------------------------------------------------------------------
# Test 6: adapter config loading
# ---------------------------------------------------------------------------

class TestAdapterConfigLoad(unittest.TestCase):
    """load_adapter_config must read architecture params correctly."""

    def test_adapter_config_load(self):
        tmpdir = tempfile.mkdtemp()
        cfg_path = _write_fake_adapter_config(
            tmpdir, hidden_dim=2048, output_dim=4096,
            num_queries=64, num_resampler_layers=2,
            num_heads=8, ffn_mult=4,
        )
        cfg = load_adapter_config(cfg_path)
        self.assertEqual(cfg["hidden_dim"], 2048)
        self.assertEqual(cfg["output_dim"], 4096)
        self.assertEqual(cfg["num_queries"], 64)
        self.assertEqual(cfg["num_resampler_layers"], 2)
        self.assertEqual(cfg["num_heads"], 8)
        self.assertEqual(cfg["ffn_mult"], 4)

    def test_adapter_config_load_defaults(self):
        """Missing optional fields should fall back to defaults."""
        tmpdir = tempfile.mkdtemp()
        minimal = {
            "hidden_dim": 2048, "output_dim": 4096,
            "num_queries": 64, "num_resampler_layers": 2,
        }
        path = os.path.join(tmpdir, "cfg.json")
        with open(path, "w") as fh:
            json.dump(minimal, fh)
        cfg = load_adapter_config(path)
        self.assertEqual(cfg["num_heads"], 8)
        self.assertEqual(cfg["ffn_mult"], 4)


# ---------------------------------------------------------------------------
# Test 7: sanity metrics calculation with mock frames
# ---------------------------------------------------------------------------

class TestSanityMetricsCalculation(unittest.TestCase):
    """compute_sanity_metrics_from_frames must compute core metrics correctly."""

    def test_frame_count_and_resolution(self):
        # 5 frames, 4x4 pixels, 3 channels
        frames = np.random.rand(5, 4, 4, 3).astype(np.float32)
        m = compute_sanity_metrics_from_frames(frames)
        self.assertEqual(m["frame_count"], 5)
        self.assertEqual(m["resolution"], "4x4")

    def test_all_black(self):
        frames = np.zeros((4, 8, 8, 3), dtype=np.float32)  # all zeros
        m = compute_sanity_metrics_from_frames(frames)
        self.assertTrue(m["all_black"])
        self.assertFalse(m["all_white"])

    def test_all_white(self):
        frames = np.ones((4, 8, 8, 3), dtype=np.float32)  # all ones
        m = compute_sanity_metrics_from_frames(frames)
        self.assertTrue(m["all_white"])
        self.assertFalse(m["all_black"])

    def test_frozen_video(self):
        # Identical frames -> zero temporal diff -> frozen
        frames = np.random.rand(8, 8, 8, 3).astype(np.float32)
        stacked = np.stack([frames] * 5, axis=0)  # 5 identical frames
        m = compute_sanity_metrics_from_frames(stacked)
        self.assertTrue(m["frozen_video"])
        self.assertAlmostEqual(m["temporal_mad_mean"], 0.0, places=6)

    def test_temporal_mad_nonzero(self):
        # Random frames should have non-zero temporal MAD
        frames = np.random.rand(5, 8, 8, 3).astype(np.float32)
        m = compute_sanity_metrics_from_frames(frames)
        self.assertFalse(m["frozen_video"])
        self.assertGreater(m["temporal_mad_mean"], 0.0)
        self.assertGreaterEqual(m["temporal_mad_max"], m["temporal_mad_mean"])

    def test_nan_inf_corruption(self):
        frames = np.random.rand(4, 4, 4, 3).astype(np.float32)
        frames[1, 2, 2, 0] = float("nan")
        m = compute_sanity_metrics_from_frames(frames)
        self.assertTrue(m["nan_inf_corruption"])

    def test_per_frame_mean_std(self):
        # Uniform 0.5 frames -> per-frame mean ~0.5, std ~0
        frames = np.full((4, 4, 4, 3), 0.5, dtype=np.float32)
        m = compute_sanity_metrics_from_frames(frames)
        self.assertEqual(len(m["per_frame_mean"]), 4)
        self.assertEqual(len(m["per_frame_std"]), 4)
        for mean_val in m["per_frame_mean"]:
            self.assertAlmostEqual(mean_val, 0.5, places=5)
        for std_val in m["per_frame_std"]:
            self.assertAlmostEqual(std_val, 0.0, places=5)

    def test_uint8_input(self):
        # uint8 frames should be normalized to [0,1]
        frames = np.full((3, 4, 4, 3), 128, dtype=np.uint8)
        m = compute_sanity_metrics_from_frames(frames)
        self.assertEqual(m["frame_count"], 3)
        self.assertAlmostEqual(m["per_frame_mean"][0], 128.0 / 255.0, places=3)


if __name__ == "__main__":
    unittest.main()
