"""Unit tests for wan.utils.device abstraction (M0)."""

import unittest
from unittest.mock import patch

import torch

from wan.utils.device import (
    device_autocast,
    device_empty_cache,
    device_synchronize,
    get_device_memory,
    get_device_name,
    is_cpu,
    is_cuda,
    is_mps,
    resolve_device,
)


class DeviceResolutionTests(unittest.TestCase):
    """Test resolve_device for auto/cuda/mps/cpu."""

    def test_cpu_always_available(self):
        dev = resolve_device("cpu")
        self.assertEqual(dev.type, "cpu")

    def test_explicit_torch_device_passthrough(self):
        dev = resolve_device(torch.device("cpu"))
        self.assertEqual(dev.type, "cpu")

    def test_unknown_device_raises(self):
        with self.assertRaises(ValueError):
            resolve_device("tpu")

    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_prefers_cuda(self, _mock):
        dev = resolve_device("auto")
        self.assertEqual(dev.type, "cuda")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_auto_falls_back_to_mps(self, _mock_mps, _mock_cuda):
        dev = resolve_device("auto")
        self.assertEqual(dev.type, "mps")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, _mock_mps, _mock_cuda):
        dev = resolve_device("auto")
        self.assertEqual(dev.type, "cpu")

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_unavailable_raises(self, _mock):
        with self.assertRaises(ValueError):
            resolve_device("cuda")

    @patch("torch.backends.mps.is_available", return_value=False)
    def test_mps_unavailable_raises(self, _mock):
        with self.assertRaises(ValueError):
            resolve_device("mps")


class DeviceTypeCheckTests(unittest.TestCase):
    """Test is_cuda / is_mps / is_cpu."""

    def test_is_cuda(self):
        self.assertTrue(is_cuda(torch.device("cuda")))
        self.assertTrue(is_cuda("cuda:0"))
        self.assertFalse(is_cuda(torch.device("cpu")))
        self.assertFalse(is_cuda(torch.device("mps")))

    def test_is_mps(self):
        self.assertTrue(is_mps(torch.device("mps")))
        self.assertTrue(is_mps("mps"))
        self.assertFalse(is_mps(torch.device("cuda")))
        self.assertFalse(is_mps(torch.device("cpu")))

    def test_is_cpu(self):
        self.assertTrue(is_cpu(torch.device("cpu")))
        self.assertTrue(is_cpu("cpu"))
        self.assertFalse(is_cpu(torch.device("cuda")))
        self.assertFalse(is_cpu(torch.device("mps")))


class DeviceHelperTests(unittest.TestCase):
    """Test device_synchronize / device_empty_cache / device_autocast."""

    def test_cpu_synchronize_noop(self):
        # Should not raise
        device_synchronize(torch.device("cpu"))

    def test_cpu_empty_cache_noop(self):
        # Should not raise
        device_empty_cache(torch.device("cpu"))

    def test_cpu_autocast_context(self):
        with device_autocast(torch.device("cpu"), dtype=torch.bfloat16):
            x = torch.randn(4, 4)
            self.assertEqual(x.dtype, torch.float32)  # autocast doesn't change input dtype

    def test_mps_autocast_context(self):
        # MPS autocast may not be fully supported, but context should enter/exit
        try:
            with device_autocast(torch.device("mps"), dtype=torch.float16):
                pass
        except Exception:
            self.skipTest("MPS autocast not available in this PyTorch build")

    def test_get_device_name_cpu(self):
        name = get_device_name(torch.device("cpu"))
        self.assertEqual(name, "CPU")

    def test_get_device_memory_cpu_returns_none(self):
        mem = get_device_memory(torch.device("cpu"))
        self.assertIsNone(mem)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="Test GPU")
    def test_get_device_name_cuda(self, _mock_name, _mock_avail):
        name = get_device_name(torch.device("cuda"))
        self.assertEqual(name, "Test GPU")


if __name__ == "__main__":
    unittest.main()
