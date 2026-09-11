"""Unit tests for the low-memory streaming safetensors checkpoint loader."""

import json
import os
import tempfile
import unittest

import torch
import torch.nn as nn
from safetensors.torch import save_file

from wan.utils.streaming_loader import (
    StreamingLoadStats,
    create_meta_model,
    load_sharded_safetensors_streaming,
)


class _SmallModel(nn.Module):
    """Small model with parameters and a buffer for testing."""

    def __init__(self, dim=8):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.register_buffer("buf", torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)


def _make_sharded_checkpoint(tmpdir, tensors, num_shards=2):
    """Create a fake sharded safetensors checkpoint in tmpdir."""
    keys = list(tensors.keys())
    shard_names = []
    weight_map = {}

    for i in range(num_shards):
        shard_name = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
        shard_names.append(shard_name)
        shard_tensors = {}
        # Distribute keys across shards
        for j, key in enumerate(keys):
            if j % num_shards == i:
                shard_tensors[key] = tensors[key]
                weight_map[key] = shard_name
        save_file(shard_tensors, os.path.join(tmpdir, shard_name))

    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": weight_map,
    }
    with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)

    return shard_names, weight_map


def _make_single_checkpoint(tmpdir, tensors):
    """Create a fake single-file safetensors checkpoint in tmpdir."""
    save_file(tensors, os.path.join(tmpdir, "model.safetensors"))


class TestStreamingLoader(unittest.TestCase):
    """Tests for load_sharded_safetensors_streaming."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        torch.manual_seed(42)
        self.dim = 8
        self.tensors = {
            "linear1.weight": torch.randn(self.dim, self.dim),
            "linear1.bias": torch.randn(self.dim),
            "linear2.weight": torch.randn(self.dim, self.dim),
            "linear2.bias": torch.randn(self.dim),
            "buf": torch.ones(self.dim),
            "norm.weight": torch.randn(self.dim),
            "norm.bias": torch.randn(self.dim),
        }

    def test_create_meta_model(self):
        """create_meta_model should produce a model with all meta tensors."""
        model = create_meta_model(_SmallModel, dim=self.dim)
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        self.assertTrue(all(b.is_meta for b in model.buffers()))
        # linear1: 8*8+8=72, linear2: 8*8+8=72, norm: 8+8=16 => total 160
        self.assertEqual(sum(p.numel() for p in model.parameters()), 160)

    def test_sharded_load_fp16(self):
        """2+ shards should load correctly with F32 -> FP16 casting."""
        _make_sharded_checkpoint(self.tmpdir, self.tensors, num_shards=2)
        model = create_meta_model(_SmallModel, dim=self.dim)

        stats = load_sharded_safetensors_streaming(
            model, self.tmpdir, target_dtype=torch.float16
        )

        self.assertTrue(stats.success, stats.summary())
        self.assertEqual(stats.shard_count, 2)
        self.assertEqual(stats.tensor_count, len(self.tensors))
        self.assertEqual(len(stats.loaded_keys), len(self.tensors))
        self.assertEqual(len(stats.missing_keys), 0)
        self.assertEqual(len(stats.unexpected_keys), 0)
        self.assertEqual(stats.meta_params_remaining, 0)
        self.assertEqual(stats.meta_buffers_remaining, 0)

        # Verify dtype
        for p in model.parameters():
            self.assertEqual(p.dtype, torch.float16)
        for b in model.buffers():
            self.assertEqual(b.dtype, torch.float16)

    def test_sharded_load_values_correct(self):
        """Loaded parameter values should match source (within FP16 precision)."""
        _make_sharded_checkpoint(self.tmpdir, self.tensors, num_shards=2)
        model = create_meta_model(_SmallModel, dim=self.dim)

        stats = load_sharded_safetensors_streaming(
            model, self.tmpdir, target_dtype=torch.float32
        )
        self.assertTrue(stats.success)

        # Check each tensor value
        for key, source in self.tensors.items():
            if key in dict(model.named_parameters()):
                loaded = dict(model.named_parameters())[key]
            else:
                loaded = dict(model.named_buffers())[key]
            self.assertTrue(
                torch.allclose(loaded, source, atol=1e-6),
                f"Value mismatch for {key}",
            )

    def test_buffer_loaded_correctly(self):
        """Buffers should be loaded alongside parameters."""
        _make_sharded_checkpoint(self.tmpdir, self.tensors, num_shards=2)
        model = create_meta_model(_SmallModel, dim=self.dim)

        stats = load_sharded_safetensors_streaming(
            model, self.tmpdir, target_dtype=torch.float32
        )
        self.assertTrue(stats.success)
        self.assertTrue(torch.allclose(model.buf, torch.ones(self.dim)))

    def test_single_file_load(self):
        """Single (non-sharded) safetensors file should load correctly."""
        _make_single_checkpoint(self.tmpdir, self.tensors)
        model = create_meta_model(_SmallModel, dim=self.dim)

        stats = load_sharded_safetensors_streaming(
            model, self.tmpdir, target_dtype=torch.float32
        )
        self.assertTrue(stats.success)
        self.assertEqual(stats.shard_count, 1)
        self.assertEqual(stats.tensor_count, len(self.tensors))

    def test_missing_key_detected(self):
        """Missing keys should be reported in stats.missing_keys."""
        # Remove one tensor from checkpoint
        incomplete = {k: v for k, v in self.tensors.items() if k != "linear2.bias"}
        _make_sharded_checkpoint(self.tmpdir, incomplete, num_shards=2)
        model = create_meta_model(_SmallModel, dim=self.dim)

        stats = load_sharded_safetensors_streaming(
            model, self.tmpdir, target_dtype=torch.float32
        )
        self.assertFalse(stats.success)
        self.assertIn("linear2.bias", stats.missing_keys)
        self.assertEqual(stats.meta_params_remaining, 1)  # linear2.bias still meta

    def test_unexpected_key_detected(self):
        """Unexpected keys should be reported in stats.unexpected_keys."""
        extra = dict(self.tensors)
        extra["nonexistent.layer.weight"] = torch.randn(4, 4)
        _make_sharded_checkpoint(self.tmpdir, extra, num_shards=2)
        model = create_meta_model(_SmallModel, dim=self.dim)

        stats = load_sharded_safetensors_streaming(
            model, self.tmpdir, target_dtype=torch.float32
        )
        self.assertFalse(stats.success)
        self.assertIn("nonexistent.layer.weight", stats.unexpected_keys)

    def test_shape_mismatch_detected(self):
        """Shape mismatches should be reported and not assigned."""
        bad = dict(self.tensors)
        bad["linear1.weight"] = torch.randn(16, 16)  # wrong shape
        _make_sharded_checkpoint(self.tmpdir, bad, num_shards=2)
        model = create_meta_model(_SmallModel, dim=self.dim)

        stats = load_sharded_safetensors_streaming(
            model, self.tmpdir, target_dtype=torch.float32
        )
        self.assertFalse(stats.success)
        self.assertIn("linear1.weight", stats.shape_mismatch)
        expected_shape, actual_shape = stats.shape_mismatch["linear1.weight"]
        self.assertEqual(tuple(expected_shape), (self.dim, self.dim))
        self.assertEqual(tuple(actual_shape), (16, 16))

    def test_no_meta_tensors_after_load(self):
        """After successful load, no meta parameters or buffers should remain."""
        _make_sharded_checkpoint(self.tmpdir, self.tensors, num_shards=3)
        model = create_meta_model(_SmallModel, dim=self.dim)

        stats = load_sharded_safetensors_streaming(
            model, self.tmpdir, target_dtype=torch.float32
        )
        self.assertTrue(stats.success)
        self.assertFalse(any(p.is_meta for p in model.parameters()))
        self.assertFalse(any(b.is_meta for b in model.buffers()))

    def test_no_full_state_dict_constructed(self):
        """Loader should not construct a full state dict (verified by API usage)."""
        _make_sharded_checkpoint(self.tmpdir, self.tensors, num_shards=2)
        model = create_meta_model(_SmallModel, dim=self.dim)

        # The streaming loader uses safe_open + get_tensor, not load_file
        # We verify this by checking that the loader doesn't call _load_safetensors_state_dict
        # (which uses load_file and constructs a full state dict)
        import wan.image2video as i2v
        original = i2v._load_safetensors_state_dict
        called = [False]

        def spy(*args, **kwargs):
            called[0] = True
            return original(*args, **kwargs)

        i2v._load_safetensors_state_dict = spy
        try:
            stats = load_sharded_safetensors_streaming(
                model, self.tmpdir, target_dtype=torch.float32
            )
            self.assertTrue(stats.success)
            self.assertFalse(called[0], "Streaming loader should not call _load_safetensors_state_dict")
        finally:
            i2v._load_safetensors_state_dict = original

    def test_target_device_cpu(self):
        """Loader should support target_device='cpu'."""
        _make_sharded_checkpoint(self.tmpdir, self.tensors, num_shards=2)
        model = create_meta_model(_SmallModel, dim=self.dim)

        stats = load_sharded_safetensors_streaming(
            model, self.tmpdir, target_dtype=torch.float32, target_device=torch.device("cpu")
        )
        self.assertTrue(stats.success)
        self.assertEqual(stats.target_device, torch.device("cpu"))
        for p in model.parameters():
            self.assertEqual(p.device, torch.device("cpu"))

    def test_stats_summary(self):
        """StreamingLoadStats.summary() should return a readable string."""
        stats = StreamingLoadStats()
        stats.tensor_count = 5
        stats.shard_count = 2
        summary = stats.summary()
        self.assertIn("FAILED", summary)  # no keys loaded = not success
        self.assertIn("tensors loaded: 5", summary)

    def test_file_not_found_raises(self):
        """Missing checkpoint directory should raise FileNotFoundError."""
        model = create_meta_model(_SmallModel, dim=self.dim)
        empty_dir = tempfile.mkdtemp()
        with self.assertRaises(FileNotFoundError):
            load_sharded_safetensors_streaming(
                model, empty_dir, target_dtype=torch.float32
            )


if __name__ == "__main__":
    unittest.main()
