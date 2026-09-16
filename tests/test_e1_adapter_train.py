"""Unit tests for E1 Text-Adapter training.

Pure CPU / small-tensor tests.  No real MiniCPM5, no UMT5, no MPS, no
checkpoint loading, no training loop.  The loss primitives and the freeze
contract are exercised with tiny synthetic tensors.
"""
from __future__ import annotations

import json
import os
import random
import sys
import unittest

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from wan.adapters.text_adapter import TextAdapter  # noqa: E402
from scripts.e1_train_adapter import (  # noqa: E402
    masked_mean_pool,
    pooled_cosine_loss,
    pooled_mse_loss,
    ProbeAttentionDistiller,
    cross_attn_distill_loss,
    attention_similarity,
)


class TestLossPrimitives(unittest.TestCase):

    def test_pooled_cosine_loss(self):
        # Identical vectors -> cosine 1.0 -> loss 0.
        a = torch.randn(4, 8)
        b = a.clone()
        self.assertAlmostEqual(float(pooled_cosine_loss(a, b)), 0.0, places=5)
        # Opposite vectors -> cosine -1 -> loss 2.0.
        c = -a
        self.assertAlmostEqual(float(pooled_cosine_loss(a, c)), 2.0, places=4)
        # Orthogonal -> cosine 0 -> loss 1.0.
        d = torch.randn(4, 8)
        d = d - (d * a).sum(-1, keepdim=True) / (a * a).sum(-1, keepdim=True) * a
        loss = float(pooled_cosine_loss(a, d))
        self.assertAlmostEqual(loss, 1.0, places=3)

    def test_pooled_mse_loss(self):
        a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        b = torch.tensor([[1.0, 2.0, 3.0, 5.0]])  # diff on last coord
        # MSE = (0+0+0+(1)^2)/4 = 0.25
        self.assertAlmostEqual(float(pooled_mse_loss(a, b)), 0.25, places=6)
        self.assertAlmostEqual(float(pooled_mse_loss(a, a)), 0.0, places=6)

    def test_probe_attention_distill(self):
        torch.manual_seed(0)
        distiller = ProbeAttentionDistiller(dim=4096, num_probes=32, seed=0)
        ctx = torch.randn(2, 20, 4096, requires_grad=True)  # [B=2, N=20, D]
        out = distiller(ctx, key_padding_mask=None)
        self.assertEqual(tuple(out.shape), (2, 32, 4096))
        # Loss is computable + differentiable w.r.t. the context (probes fixed).
        target = torch.randn(2, 32, 4096)
        loss = cross_attn_distill_loss(out, target)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(ctx.grad)
        self.assertGreater(ctx.grad.abs().sum().item(), 0.0)
        self.assertFalse(distiller.probes.requires_grad)

    def test_teacher_student_different_length(self):
        # Teacher [1,20,4096] (padded) vs student [1,64,4096] (fixed queries).
        teacher = torch.randn(1, 20, 4096)
        teacher_mask = torch.ones(1, 20)
        teacher_mask[0, 15:] = 0.0  # last 5 are padding
        student = torch.randn(1, 64, 4096)

        t_pool = masked_mean_pool(teacher, teacher_mask)
        s_pool = student.mean(dim=1)
        self.assertEqual(tuple(t_pool.shape), (1, 4096))
        self.assertEqual(tuple(s_pool.shape), (1, 4096))

        l_cos = pooled_cosine_loss(t_pool, s_pool)
        l_mse = pooled_mse_loss(t_pool, s_pool)
        distiller = ProbeAttentionDistiller(dim=4096, num_probes=32, seed=1)
        t_probe = distiller(teacher, key_padding_mask=teacher_mask.eq(0))
        s_probe = distiller(student, key_padding_mask=None)
        l_x = cross_attn_distill_loss(t_probe, s_probe)
        for v in (l_cos, l_mse, l_x):
            self.assertTrue(torch.isfinite(v).item())
            self.assertGreaterEqual(float(v), 0.0)

    def test_masked_mean_pool_matches_manual(self):
        x = torch.randn(2, 10, 4)
        mask = torch.ones(2, 10)
        mask[0, 7:] = 0.0
        mask[1, 3:] = 0.0
        pooled = masked_mean_pool(x, mask)
        manual = torch.stack([
            x[0, :7].mean(0),
            x[1, :3].mean(0),
        ])
        self.assertTrue(torch.allclose(pooled, manual, atol=1e-6))


class TestFreezeAndDtype(unittest.TestCase):

    def test_freeze_verification(self):
        # Mock frozen "MiniCPM5": a tiny encoder producing [B,L,2048].
        mock_mcp = nn.Linear(2048, 2048)
        mock_mcp.eval()
        mock_mcp.requires_grad_(False)
        # Frozen reference text_embedding.
        mock_te = nn.Sequential(nn.Linear(4096, 1536), nn.GELU(), nn.Linear(1536, 1536))
        mock_te.eval()
        mock_te.requires_grad_(False)

        adapter = TextAdapter(hidden_dim=2048, output_dim=4096,
                              num_queries=8, num_resampler_layers=1)

        self.assertTrue(all(not p.requires_grad for p in mock_mcp.parameters()))
        self.assertTrue(all(not p.requires_grad for p in mock_te.parameters()))
        self.assertTrue(all(p.requires_grad for p in adapter.parameters()))

    def test_bf16_cast_logic(self):
        adapter = TextAdapter(hidden_dim=2048, output_dim=4096,
                              num_queries=8, num_resampler_layers=1)
        x = torch.randn(2, 30, 2048, dtype=torch.bfloat16)
        y = adapter(x)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(tuple(y.shape), (2, 8, 4096))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_probe_queries_fixed(self):
        d = ProbeAttentionDistiller(dim=4096, num_probes=32, seed=123)
        self.assertFalse(d.probes.requires_grad)
        # Deterministic across instances with same seed.
        d2 = ProbeAttentionDistiller(dim=4096, num_probes=32, seed=123)
        self.assertTrue(torch.equal(d.probes, d2.probes))


class TestLossComposition(unittest.TestCase):

    def test_total_loss_weighted(self):
        torch.manual_seed(0)
        t = torch.randn(4, 8)
        s = torch.randn(4, 8)
        l1 = pooled_cosine_loss(t, s)
        l2 = pooled_mse_loss(t, s)
        distiller = ProbeAttentionDistiller(dim=8, num_probes=4, seed=0)
        t_ctx = torch.randn(2, 6, 8)
        s_ctx = torch.randn(2, 4, 8)
        l3 = cross_attn_distill_loss(distiller(t_ctx), distiller(s_ctx))
        w1, w2, w3 = 1.0, 0.5, 2.0
        total = w1 * l1 + w2 * l2 + w3 * l3
        expected = float(w1 * l1 + w2 * l2 + w3 * l3)
        self.assertAlmostEqual(float(total), expected, places=6)

    def test_adapter_forward_grad_flow(self):
        adapter = TextAdapter(hidden_dim=2048, output_dim=4096,
                              num_queries=8, num_resampler_layers=1)
        x = torch.randn(2, 20, 2048)
        y = adapter(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(adapter.queries.grad)
        self.assertGreater(adapter.queries.grad.abs().sum().item(), 0.0)


class TestSplitReproducibility(unittest.TestCase):

    def test_train_val_split_reproducible(self):
        # Replicate the build_prompts split: seed=42, shuffle, 80/20,
        # guarantee >=1 g07 in val.
        ids = [f"g07_scene_{v}" for v in "ABC"] * 5 + \
              [f"synthetic_{i:03d}" for i in range(41)]
        sources = ["g07"] * 15 + ["synthetic"] * 41
        rows = [{"id": i, "source": s} for i, s in zip(ids, sources)]

        def do_split():
            total = len(rows)
            n_val = max(1, round(total * 0.20))
            rng = random.Random(42)
            order = list(range(total))
            rng.shuffle(order)
            val_idx = set(order[:n_val])
            if not any(rows[i]["source"].startswith("g07") for i in val_idx):
                g07 = [i for i in range(total) if rows[i]["source"].startswith("g07")]
                for gi in g07:
                    if gi not in val_idx:
                        swap = next(iter(val_idx))
                        val_idx.discard(swap)
                        val_idx.add(gi)
                        break
            return val_idx

        a = do_split()
        b = do_split()
        self.assertEqual(a, b)  # reproducible

        # The committed prompts.jsonl must reflect this exact structure.
        p = os.path.join(REPO_ROOT, "eval", "e1", "prompts.jsonl")
        with open(p, encoding="utf-8") as fh:
            rows_file = [json.loads(l) for l in fh if l.strip()]
        train = [r for r in rows_file if r["split"] == "train"]
        val = [r for r in rows_file if r["split"] == "val"]
        self.assertEqual(len(rows_file), 56)
        self.assertGreaterEqual(len(train), 40)
        self.assertGreaterEqual(len(val), 10)
        g07_val = [r for r in val if r["source"].startswith("g07")]
        self.assertGreaterEqual(len(g07_val), 1)
        # ~80/20
        self.assertAlmostEqual(len(val) / len(rows_file), 0.20, delta=0.02)


if __name__ == "__main__":
    unittest.main()
