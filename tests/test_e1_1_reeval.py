"""Unit tests for E1.1 corrected downstream-metric re-evaluation.

Pure CPU / small-tensor tests.  No real MiniCPM5, no LingBot DiT, no MPS, no
real checkpoint loading.  The metric primitives, the ProbeAttention
non-uniformity contract, the per-token-vs-pool projection distinction, and the
freeze contracts are exercised with tiny synthetic tensors.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from safetensors.torch import save_file  # noqa: E402

from wan.adapters.text_adapter import TextAdapter  # noqa: E402
from wan.modules.model import WanRMSNorm  # noqa: E402
from scripts.e1_1_reeval import (  # noqa: E402
    CONTEXT_DIM,
    TE_DIM,
    NUM_PROBES,
    NUM_HEADS,
    ProbeAttention,
    load_text_embedding,
    per_token_project,
    pooled_cosine_1536,
    pooled_mse_1536,
    probe_cosine,
    probe_mse,
    compute_prompt_metrics,
)


def _make_mock_text_embedding(seed: int = 0) -> nn.Sequential:
    """A tiny text_embedding MLP (4096->1536->1536) with fixed random weights.

    Has the same non-linear GELU structure as the real LingBot
    text_embedding, so per-token-projection is NOT equivalent to
    pool-then-project.
    """
    g = torch.Generator().manual_seed(seed)
    te = nn.Sequential(OrderedDict([
        ("0", nn.Linear(CONTEXT_DIM, TE_DIM)),
        ("1", nn.GELU(approximate="tanh")),
        ("2", nn.Linear(TE_DIM, TE_DIM)),
    ]))
    with torch.no_grad():
        for layer in (te[0], te[2]):
            layer.weight.normal_(0.0, 0.1, generator=g)
            layer.bias.zero_()
    te.eval()
    return te


class TestPerTokenProjection(unittest.TestCase):

    def test_per_token_projection_then_pool(self):
        """Per-token projection THEN mean-pool must differ from pool THEN project.

        This guards the corrected E1.1 metric: because text_embedding contains
        a non-linear GELU, mean_pool(te(x)) != te(mean_pool(x)).
        """
        torch.manual_seed(123)
        te = _make_mock_text_embedding(seed=7)
        teacher = torch.randn(4, CONTEXT_DIM)
        student = torch.randn(8, CONTEXT_DIM)

        # Correct (implemented) path: project per token, then mean-pool.
        t_correct = per_token_project(teacher, te).mean(dim=0)   # [1536]
        s_correct = per_token_project(student, te).mean(dim=0)

        # Wrong (banned) path: mean-pool first, then project.
        t_wrong = te(teacher.mean(dim=0, keepdim=True).float())[0]  # [1536]
        s_wrong = te(student.mean(dim=0, keepdim=True).float())[0]

        # The two paths must NOT be numerically identical (GELU nonlinearity).
        self.assertGreater(
            (t_correct - t_wrong).abs().max().item(), 1e-4,
            "per-token-then-pool should differ from pool-then-project (GELU non-linearity)",
        )
        self.assertGreater(
            (s_correct - s_wrong).abs().max().item(), 1e-4,
            "per-token-then-pool should differ from pool-then-project (GELU non-linearity)",
        )
        # And the cosines must differ.
        cos_correct = F.cosine_similarity(t_correct, s_correct, dim=0).item()
        cos_wrong = F.cosine_similarity(t_wrong, s_wrong, dim=0).item()
        self.assertNotAlmostEqual(cos_correct, cos_wrong, places=3,
                                  msg="correct vs wrong metric cosines should differ")


class TestProbeAttention(unittest.TestCase):

    def test_probe_attention_non_uniform(self):
        """Non-uniform input must yield non-uniform attention weights (std>0)."""
        torch.manual_seed(PROBE_SEED := 42)
        probe = ProbeAttention(seed=42).float()
        # Non-uniform context: first half all 1, second half all 0.
        context = torch.zeros(1, 16, TE_DIM)
        context[0, :8, :] = 1.0
        attn_w = probe.attention_weights(context)  # [1,H,P,L]
        # Uniform distribution would be 1/L everywhere -> std ~ 0.
        self.assertGreater(attn_w.std().item(), 1e-4,
                           "attention weights must be non-uniform")
        # And they must sum to 1 over the key axis.
        sums = attn_w.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_probe_attention_shape(self):
        """context [1,10,1536] -> output [1,32,1536]."""
        probe = ProbeAttention(seed=42).float()
        context = torch.randn(1, 10, TE_DIM)
        out = probe(context)
        self.assertEqual(tuple(out.shape), (1, NUM_PROBES, TE_DIM))

    def test_probe_attention_deterministic(self):
        """Same seed -> identical outputs (reproducible probe metric)."""
        p1 = ProbeAttention(seed=42).float()
        p2 = ProbeAttention(seed=42).float()
        ctx = torch.randn(1, 12, TE_DIM)
        o1 = p1(ctx)
        o2 = p2(ctx)
        self.assertTrue(torch.allclose(o1, o2, atol=1e-6))


class TestAdapterForwardShape(unittest.TestCase):

    def test_epoch5_epoch20_forward_shape_consistent(self):
        """Two adapters with different weights both emit [1,64,4096]."""
        torch.manual_seed(0)
        a5 = TextAdapter(hidden_dim=2048, output_dim=4096, num_queries=64,
                         num_resampler_layers=2, num_heads=8, ffn_mult=4).float()
        torch.manual_seed(1)
        a20 = TextAdapter(hidden_dim=2048, output_dim=4096, num_queries=64,
                          num_resampler_layers=2, num_heads=8, ffn_mult=4).float()
        x = torch.randn(1, 50, 2048)
        mask = torch.ones(1, 50)
        out5 = a5(x, mask)
        out20 = a20(x, mask)
        self.assertEqual(tuple(out5.shape), (1, 64, 4096))
        self.assertEqual(tuple(out20.shape), (1, 64, 4096))
        # Different seeds -> different weights -> different outputs.
        self.assertFalse(torch.allclose(out5, out20, atol=1e-4))


class TestIdenticalInputs(unittest.TestCase):

    def test_pooled_cosine_identical(self):
        """teacher == student -> pooled_cosine == 1.0."""
        ctx = torch.randn(20, TE_DIM)
        self.assertAlmostEqual(pooled_cosine_1536(ctx, ctx), 1.0, places=5)

    def test_probe_cosine_identical(self):
        """teacher == student -> probe_cosine == 1.0."""
        probe = ProbeAttention(seed=42).float()
        ctx = torch.randn(1, 20, TE_DIM)
        p_out = probe(ctx)[0]
        self.assertAlmostEqual(probe_cosine(p_out, p_out), 1.0, places=5)

    def test_pooled_mse_zero_on_identical(self):
        ctx = torch.randn(20, TE_DIM)
        self.assertAlmostEqual(pooled_mse_1536(ctx, ctx), 0.0, places=6)

    def test_probe_mse_zero_on_identical(self):
        probe = ProbeAttention(seed=42).float()
        ctx = torch.randn(1, 20, TE_DIM)
        p_out = probe(ctx)[0]
        self.assertAlmostEqual(probe_mse(p_out, p_out), 0.0, places=6)


class TestTextEmbeddingFrozen(unittest.TestCase):

    def test_text_embedding_frozen(self):
        """Load a mock text_embedding from a fake shard -> all params frozen."""
        # Build a fake shard-1 safetensors with the exact key names.
        g = torch.Generator().manual_seed(0)
        sd = {
            "text_embedding.0.weight": torch.randn(TE_DIM, CONTEXT_DIM, generator=g),
            "text_embedding.0.bias": torch.randn(TE_DIM, generator=g),
            "text_embedding.2.weight": torch.randn(TE_DIM, TE_DIM, generator=g),
            "text_embedding.2.bias": torch.randn(TE_DIM, generator=g),
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "shard1.safetensors")
            save_file(sd, path)
            te = load_text_embedding(path)
        # All parameters must be frozen.
        for p in te.parameters():
            self.assertFalse(p.requires_grad)
        # And forward still works per-token.
        x = torch.randn(4, CONTEXT_DIM)
        out = te(x)
        self.assertEqual(tuple(out.shape), (4, TE_DIM))


class TestWanRMSNormBehavior(unittest.TestCase):

    def test_wan_rmsnorm_matches_formula(self):
        """WanRMSNorm output must equal x * rsqrt(mean(x^2)+eps) * weight."""
        dim = 8
        eps = 1e-6
        norm = WanRMSNorm(dim, eps=eps).float()
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, steps=dim))
        x = torch.randn(2, 5, dim)
        y = norm(x)
        # Manual formula.
        expected = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        expected = expected.type_as(x) * norm.weight
        self.assertTrue(torch.allclose(y, expected, atol=1e-6),
                        "WanRMSNorm does not match x*rsqrt(mean(x^2)+eps)*weight")

    def test_wan_rmsnorm_output_dtype_preserved(self):
        """RMSNorm computes in fp32 but returns the input dtype (when weight
        matches input dtype, as in the real DiT where everything runs in one
        autocast dtype)."""
        norm = WanRMSNorm(8, eps=1e-6).bfloat16()
        x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        y = norm(x)
        self.assertEqual(y.dtype, torch.bfloat16)


class TestComputePromptMetrics(unittest.TestCase):

    def test_compute_prompt_metrics_shapes_and_identical(self):
        """teacher == student (in 4096 space) -> high pooled_cosine."""
        te = _make_mock_text_embedding(seed=3)
        probe = ProbeAttention(seed=42).float()
        teacher = torch.randn(17, CONTEXT_DIM)
        m = compute_prompt_metrics(teacher, teacher, te, probe)
        self.assertAlmostEqual(m["pooled_cosine_1536"], 1.0, places=5)
        self.assertAlmostEqual(m["probe_cosine"], 1.0, places=4)
        self.assertAlmostEqual(m["pooled_mse_1536"], 0.0, places=5)
        self.assertAlmostEqual(m["probe_mse"], 0.0, places=5)
        # Non-uniformity diagnostics present and positive.
        self.assertGreater(m["teacher_attn_weight_std"], 0.0)
        self.assertGreater(m["student_attn_weight_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
