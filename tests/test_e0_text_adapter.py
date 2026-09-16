"""Unit tests for E0 Text Adapter.

Pure CPU / small-tensor tests.  No real MiniCPM5, no MPS, no training,
no video generation.
"""
from __future__ import annotations

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from wan.adapters.text_adapter import TextAdapter, ResamplerBlock  # noqa: E402


class TestTextAdapter(unittest.TestCase):

    def test_poc_module_import(self):
        # The class under test must be importable from the public package path.
        from wan.adapters.text_adapter import TextAdapter as TA
        self.assertIs(TA, TextAdapter)

    def test_adapter_output_shape(self):
        adapter = TextAdapter()  # defaults: 64 queries, 2 layers
        x = torch.randn(2, 50, 2048)
        y = adapter(x)
        self.assertEqual(tuple(y.shape), (2, 64, 4096))
        self.assertEqual(y.dtype, torch.float32)

    def test_adapter_custom_config(self):
        adapter = TextAdapter(num_queries=128, num_resampler_layers=4)
        x = torch.randn(2, 50, 2048)
        y = adapter(x)
        self.assertEqual(tuple(y.shape), (2, 128, 4096))
        # Custom depth is honored.
        self.assertEqual(len(adapter.layers), 4)
        self.assertEqual(adapter.num_queries, 128)

    def test_adapter_attention_mask(self):
        adapter = TextAdapter()
        torch.manual_seed(0)
        x_prefix = torch.randn(2, 30, 2048)
        # Tail is a different random realization; with the mask it is ignored,
        # so the masked output must match the prefix-only reference.
        x_tail = torch.cat([x_prefix, torch.randn(2, 20, 2048)], dim=1)
        mask = torch.ones(2, 50)
        mask[:, 30:] = 0.0

        with torch.no_grad():
            y_ref = adapter(x_prefix)
            y_masked = adapter(x_tail, mask)
            y_unmasked = adapter(x_tail)

        # Masked tracks reference; unmasked diverges.
        self.assertLess((y_masked - y_ref).abs().max().item(),
                        (y_unmasked - y_ref).abs().max().item())

    def test_adapter_gradient_flow(self):
        adapter = TextAdapter()
        x = torch.randn(2, 50, 2048, requires_grad=True)
        y = adapter(x)
        loss = y.sum()
        loss.backward()
        # Learned query parameter must receive gradients.
        self.assertIsNotNone(adapter.queries.grad)
        self.assertGreater(adapter.queries.grad.abs().sum().item(), 0.0)
        # Input gradient flows back.
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_adapter_dtype_bf16(self):
        adapter = TextAdapter()  # default float32 params
        x = torch.randn(2, 50, 2048, dtype=torch.bfloat16)
        y = adapter(x)
        # Adapter runs in param dtype (float32); must not crash on bf16 input.
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(tuple(y.shape), (2, 64, 4096))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_adapter_raises_on_bad_shape(self):
        adapter = TextAdapter()
        with self.assertRaises(ValueError):
            adapter(torch.randn(2, 50))  # missing feature dim
        with self.assertRaises(ValueError):
            adapter(torch.randn(2, 50, 1024))  # wrong feature dim

    def test_benchmark_script_import(self):
        # scripts/e0_benchmark.py must be importable at function level.
        import importlib
        mod = importlib.import_module("scripts.e0_benchmark")
        self.assertTrue(hasattr(mod, "minicpm5_available"))
        self.assertTrue(hasattr(mod, "run_umt5_worker"))
        self.assertTrue(hasattr(mod, "run_minicpm5_worker"))
        self.assertTrue(hasattr(mod, "main"))
        # Availability check returns a (bool, str) tuple for the real path.
        ok, reason = mod.minicpm5_available(mod.DEFAULT_MINICPM5_DIR)
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(reason, str)


if __name__ == "__main__":
    unittest.main()
