"""M2.1 integration tests: streaming loader target_device, MPS fail-fast, no double copy."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from safetensors.torch import save_file

from wan.image2video import load_dit_model
from wan.utils.streaming_loader import (
    StreamingLoadStats,
    create_meta_model,
    load_sharded_safetensors_streaming,
)


class _SmallModel(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.register_buffer("buf", torch.zeros(dim))


def _make_checkpoint(tmpdir, dim=4):
    tensors = {
        "linear.weight": torch.randn(dim, dim),
        "linear.bias": torch.randn(dim),
        "buf": torch.ones(dim),
    }
    save_file(tensors, os.path.join(tmpdir, "model.safetensors"))
    return tensors


def _patch_dit_kwargs():
    """Patch _dit_kwargs_from_config to return empty dict for _SmallModel."""
    return patch("wan.image2video._dit_kwargs_from_config", return_value={})


class TestTargetDevicePropagation(unittest.TestCase):
    """Test that target_device is correctly passed to streaming loader."""

    def test_load_dit_model_passes_target_device(self):
        """load_dit_model should pass target_device to streaming loader."""
        tmpdir = tempfile.mkdtemp()
        _make_checkpoint(tmpdir)

        with patch("wan.image2video.load_sharded_safetensors_streaming") as mock_loader, \
             patch("wan.image2video.create_meta_model", return_value=_SmallModel()), \
             _patch_dit_kwargs():
            mock_stats = StreamingLoadStats(
                tensor_count=3, loaded_keys=["a", "b", "c"],
                expected_keys=["a", "b", "c"],
            )
            mock_loader.return_value = mock_stats

            load_dit_model(
                _SmallModel, tmpdir, None, MagicMock(),
                torch.float16,
                target_device=torch.device("mps"),
                allow_legacy_fallback=False,
            )

            call_kwargs = mock_loader.call_args[1]
            self.assertEqual(call_kwargs["target_device"], torch.device("mps"))
            self.assertEqual(call_kwargs["target_dtype"], torch.float16)

    def test_load_dit_model_default_target_device_none(self):
        """Default target_device should be None (CPU load, caller moves)."""
        tmpdir = tempfile.mkdtemp()
        _make_checkpoint(tmpdir)

        with patch("wan.image2video.load_sharded_safetensors_streaming") as mock_loader, \
             patch("wan.image2video.create_meta_model", return_value=_SmallModel()), \
             _patch_dit_kwargs():
            mock_stats = StreamingLoadStats(
                tensor_count=3, loaded_keys=["a", "b", "c"],
                expected_keys=["a", "b", "c"],
            )
            mock_loader.return_value = mock_stats

            load_dit_model(
                _SmallModel, tmpdir, None, MagicMock(), torch.float16,
            )

            call_kwargs = mock_loader.call_args[1]
            self.assertIsNone(call_kwargs["target_device"])


class TestMPSFailFast(unittest.TestCase):
    """Test that MPS streaming failure does not fall back to legacy loader."""

    def test_mps_streaming_failure_raises(self):
        """When allow_legacy_fallback=False, streaming failure should raise."""
        tmpdir = tempfile.mkdtemp()
        # Create checkpoint with missing key
        save_file({"linear.weight": torch.randn(4, 4)},
                  os.path.join(tmpdir, "model.safetensors"))

        with _patch_dit_kwargs():
            with self.assertRaises(RuntimeError):
                load_dit_model(
                    _SmallModel, tmpdir, None, MagicMock(),
                    torch.float16,
                    target_device=torch.device("mps"),
                    allow_legacy_fallback=False,
                )

    def test_cpu_streaming_failure_falls_back(self):
        """When allow_legacy_fallback=True (default), streaming failure falls back."""
        tmpdir = tempfile.mkdtemp()
        # Create incomplete checkpoint
        save_file({"linear.weight": torch.randn(4, 4)},
                  os.path.join(tmpdir, "model.safetensors"))

        with _patch_dit_kwargs():
            result = load_dit_model(
                _SmallModel, tmpdir, None, MagicMock(),
                torch.float16, allow_legacy_fallback=True,
            )
            self.assertIsNotNone(result)


class TestDeviceConsistency(unittest.TestCase):
    """Test that loaded tensors are on the correct device."""

    def test_cpu_load_all_tensors_on_cpu(self):
        """Streaming load to CPU should place all tensors on CPU."""
        tmpdir = tempfile.mkdtemp()
        _make_checkpoint(tmpdir)

        model = create_meta_model(_SmallModel)
        stats = load_sharded_safetensors_streaming(
            model, tmpdir, target_dtype=torch.float32,
            target_device=torch.device("cpu"),
        )
        self.assertTrue(stats.success)
        self.assertTrue(all(p.device == torch.device("cpu") for p in model.parameters()))
        self.assertTrue(all(b.device == torch.device("cpu") for b in model.buffers()))

    def test_no_meta_residual_after_load(self):
        """Successful load should leave no meta tensors."""
        tmpdir = tempfile.mkdtemp()
        _make_checkpoint(tmpdir)

        model = create_meta_model(_SmallModel)
        stats = load_sharded_safetensors_streaming(
            model, tmpdir, target_dtype=torch.float32,
        )
        self.assertTrue(stats.success)
        self.assertFalse(any(p.is_meta for p in model.parameters()))
        self.assertFalse(any(b.is_meta for b in model.buffers()))
        self.assertEqual(stats.meta_params_remaining, 0)
        self.assertEqual(stats.meta_buffers_remaining, 0)

    def test_fp16_cast_correct(self):
        """F32 source should be cast to FP16 target."""
        tmpdir = tempfile.mkdtemp()
        _make_checkpoint(tmpdir)

        model = create_meta_model(_SmallModel)
        stats = load_sharded_safetensors_streaming(
            model, tmpdir, target_dtype=torch.float16,
        )
        self.assertTrue(stats.success)
        self.assertTrue(all(p.dtype == torch.float16 for p in model.parameters()))
        self.assertTrue(all(b.dtype == torch.float16 for b in model.buffers()))


if __name__ == "__main__":
    unittest.main()
