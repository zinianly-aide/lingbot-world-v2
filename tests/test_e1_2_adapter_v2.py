from __future__ import annotations

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from wan.adapters.text_adapter_v2 import VariableLengthTextAdapter
from scripts.e1_2_train_adapter import pad_raw, masked_cosine_loss, masked_mse


class TestVariableLengthAdapter(unittest.TestCase):
    def _small(self):
        return VariableLengthTextAdapter(
            hidden_dim=16,
            bottleneck_dim=8,
            output_dim=24,
            max_queries=12,
            num_resampler_layers=1,
            num_heads=2,
            ffn_mult=2,
        )

    def test_variable_target_lengths_and_zero_padding(self):
        torch.manual_seed(0)
        m = self._small()
        hidden = torch.randn(2, 7, 16)
        input_mask = torch.tensor([[1,1,1,1,1,0,0], [1,1,1,1,1,1,1]])
        lengths = torch.tensor([3, 6])
        out, mask = m(hidden, input_mask, lengths)
        self.assertEqual(tuple(out.shape), (2, 6, 24))
        self.assertEqual(mask.tolist(), [[True,True,True,False,False,False], [True]*6])
        self.assertTrue(torch.equal(out[0, 3:], torch.zeros_like(out[0, 3:])))

    def test_no_output_layernorm(self):
        m = self._small()
        self.assertFalse(hasattr(m, "output_norm"))
        self.assertTrue(hasattr(m, "output_proj"))

    def test_target_limit(self):
        m = self._small()
        hidden = torch.randn(1, 4, 16)
        with self.assertRaises(ValueError):
            m(hidden, torch.ones(1, 4), torch.tensor([13]))

    def test_default_parameter_budget(self):
        m = VariableLengthTextAdapter()
        params = sum(p.numel() for p in m.parameters())
        self.assertLess(params, 30_000_000)
        self.assertGreater(params, 10_000_000)


class TestExactWanGeometryHelpers(unittest.TestCase):
    def test_pad_raw_keeps_active_tokens_and_zero_pads(self):
        x = torch.randn(2, 3, 4)
        mask = torch.tensor([[1,1,0], [1,1,1]], dtype=torch.bool)
        out, out_mask = pad_raw(x, mask, text_len=7)
        self.assertEqual(tuple(out.shape), (2, 7, 4))
        self.assertTrue(torch.equal(out[:, :3], x))
        self.assertTrue(torch.equal(out[:, 3:], torch.zeros_like(out[:, 3:])))
        self.assertEqual(out_mask[:, :3].tolist(), mask.tolist())
        self.assertFalse(out_mask[:, 3:].any().item())

    def test_masked_losses_ignore_padding(self):
        teacher = torch.randn(1, 4, 8)
        student = teacher.clone()
        student[:, 2:] = 999.0
        mask = torch.tensor([[1,1,0,0]], dtype=torch.bool)
        self.assertAlmostEqual(float(masked_cosine_loss(student, teacher, mask)), 0.0, places=5)
        self.assertAlmostEqual(float(masked_mse(student, teacher, mask)), 0.0, places=6)

    def test_same_length_contract_supports_tokenwise_distillation(self):
        torch.manual_seed(1)
        m = self._small_adapter()
        hidden = torch.randn(1, 5, 16)
        out, mask = m(hidden, torch.ones(1, 5), torch.tensor([4]))
        teacher = torch.randn(1, 4, 24)
        self.assertEqual(tuple(out.shape), tuple(teacher.shape))
        self.assertTrue(mask.all().item())

    @staticmethod
    def _small_adapter():
        return VariableLengthTextAdapter(
            hidden_dim=16, bottleneck_dim=8, output_dim=24,
            max_queries=8, num_resampler_layers=1, num_heads=2, ffn_mult=2,
        )


if __name__ == "__main__":
    unittest.main()
