"""Unit tests for device-aware attention backend (M1)."""

import unittest

import torch

from wan.modules.attention import (
    _build_padding_mask,
    attention,
    sdpa_attention,
)


class PaddingMaskTests(unittest.TestCase):
    """Test _build_padding_mask for SDPA."""

    def test_no_lens_returns_none(self):
        mask = _build_padding_mask(None, None, 8, 8, torch.device("cpu"))
        self.assertIsNone(mask)

    def test_full_length_all_true(self):
        b, lq, lk = 2, 4, 4
        q_lens = torch.tensor([4, 4], dtype=torch.int32)
        k_lens = torch.tensor([4, 4], dtype=torch.int32)
        mask = _build_padding_mask(q_lens, k_lens, lq, lk, torch.device("cpu"))
        self.assertEqual(mask.shape, (b, 1, lq, lk))
        self.assertTrue(mask.all())

    def test_partial_length_masks_padding(self):
        b, lq, lk = 1, 4, 4
        q_lens = torch.tensor([2], dtype=torch.int32)
        k_lens = torch.tensor([3], dtype=torch.int32)
        mask = _build_padding_mask(q_lens, k_lens, lq, lk, torch.device("cpu"))
        # q positions 0,1 valid; 2,3 invalid
        self.assertTrue(mask[0, 0, 0, :3].all())
        self.assertTrue(mask[0, 0, 1, :3].all())
        self.assertFalse(mask[0, 0, 2, 0].item())
        self.assertFalse(mask[0, 0, 3, 0].item())
        # k position 3 invalid
        self.assertFalse(mask[0, 0, 0, 3].item())

    def test_q_lens_only(self):
        b, lq, lk = 1, 4, 6
        q_lens = torch.tensor([3], dtype=torch.int32)
        mask = _build_padding_mask(q_lens, None, lq, lk, torch.device("cpu"))
        self.assertEqual(mask.shape, (b, 1, lq, lk))
        self.assertTrue(mask[0, 0, :3, :].all())
        self.assertFalse(mask[0, 0, 3, 0].item())

    def test_k_lens_only(self):
        b, lq, lk = 1, 6, 4
        k_lens = torch.tensor([2], dtype=torch.int32)
        mask = _build_padding_mask(None, k_lens, lq, lk, torch.device("cpu"))
        self.assertEqual(mask.shape, (b, 1, lq, lk))
        self.assertTrue(mask[0, 0, :, :2].all())
        self.assertFalse(mask[0, 0, 0, 2].item())


class SdpaAttentionTests(unittest.TestCase):
    """Test sdpa_attention on CPU."""

    def test_basic_attention_shape(self):
        b, lq, lk, nq, nk, c = 2, 8, 8, 4, 4, 32
        q = torch.randn(b, lq, nq, c)
        k = torch.randn(b, lk, nk, c)
        v = torch.randn(b, lk, nk, c)
        out = sdpa_attention(q, k, v, dtype=torch.float32)
        self.assertEqual(out.shape, (b, lq, nq, c))

    def test_gqa_head_repeat(self):
        """Nq != Nk (GQA) should repeat k/v heads."""
        b, lq, lk, nq, nk, c = 1, 4, 4, 8, 2, 16
        q = torch.randn(b, lq, nq, c)
        k = torch.randn(b, lk, nk, c)
        v = torch.randn(b, lk, nk, c)
        out = sdpa_attention(q, k, v, dtype=torch.float32)
        self.assertEqual(out.shape, (b, lq, nq, c))

    def test_causal_attention(self):
        b, lq, lk, n, c = 1, 4, 4, 2, 16
        q = torch.randn(b, lq, n, c)
        k = torch.randn(b, lk, n, c)
        v = torch.randn(b, lk, n, c)
        out = sdpa_attention(q, k, v, causal=True, dtype=torch.float32)
        self.assertEqual(out.shape, (b, lq, n, c))

    def test_padding_mask_produces_finite_output(self):
        b, lq, lk, n, c = 1, 8, 8, 2, 16
        q = torch.randn(b, lq, n, c)
        k = torch.randn(b, lk, n, c)
        v = torch.randn(b, lk, n, c)
        q_lens = torch.tensor([5], dtype=torch.int32)
        k_lens = torch.tensor([6], dtype=torch.int32)
        out = sdpa_attention(q, k, v, q_lens=q_lens, k_lens=k_lens, dtype=torch.float32)
        self.assertEqual(out.shape, (b, lq, n, c))
        self.assertTrue(torch.isfinite(out).all())

    def test_q_scale(self):
        b, lq, lk, n, c = 1, 4, 4, 2, 16
        q = torch.randn(b, lq, n, c)
        k = torch.randn(b, lk, n, c)
        v = torch.randn(b, lk, n, c)
        out = sdpa_attention(q, k, v, q_scale=0.5, dtype=torch.float32)
        self.assertEqual(out.shape, (b, lq, n, c))
        self.assertTrue(torch.isfinite(out).all())


class AttentionDispatchTests(unittest.TestCase):
    """Test attention() dispatch to SDPA on CPU."""

    def test_cpu_dispatches_to_sdpa(self):
        b, lq, lk, n, c = 1, 4, 4, 2, 16
        q = torch.randn(b, lq, n, c)
        k = torch.randn(b, lk, n, c)
        v = torch.randn(b, lk, n, c)
        # CPU should use SDPA (FlashAttention is CUDA-only)
        out = attention(q, k, v, dtype=torch.float32)
        self.assertEqual(out.shape, (b, lq, n, c))
        self.assertTrue(torch.isfinite(out).all())

    def test_cpu_with_padding_lens(self):
        b, lq, lk, n, c = 1, 6, 6, 2, 16
        q = torch.randn(b, lq, n, c)
        k = torch.randn(b, lk, n, c)
        v = torch.randn(b, lk, n, c)
        q_lens = torch.tensor([4], dtype=torch.int32)
        k_lens = torch.tensor([5], dtype=torch.int32)
        out = attention(q, k, v, q_lens=q_lens, k_lens=k_lens, dtype=torch.float32)
        self.assertEqual(out.shape, (b, lq, n, c))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
