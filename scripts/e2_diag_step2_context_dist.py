#!/usr/bin/env python3
"""E2 root-cause Step 2: context distribution comparison (teacher vs student).

For the single_subject_A prompt:
  teacher (UMT5):   eval/e1/teacher_contexts/9706a197....safetensors  [L,4096] bf16
  student (MiniCPM+Adapter epoch5): eval/e2/embeddings/single_subject_minicpm.safetensors [64,4096] bf16

Compares distributions before AND after the frozen LingBot text_embedding
(4096->1536), plus an analysis of the adapter's final LayerNorm(4096) effect.

Pure CPU, no VAE/DiT/MiniCPM5.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.e1_1_reeval import load_text_embedding  # noqa: E402

TE_SHARD = (
    "/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/"
    "snapshots/7e36a5f919f86cb4255cc9bfc30adb44963fbde1/"
    "model-00001-of-00006.safetensors"
)
TEACHER = REPO_ROOT / "eval/e1/teacher_contexts/9706a197a01da940f3e828a0dac73026990e8add07092d3148047b081e4e08ea.safetensors"
STUDENT = REPO_ROOT / "eval/e2/embeddings/single_subject_minicpm.safetensors"
OUT = REPO_ROOT / "eval/e2_diag/step2_context_dist/report.json"


def dist_stats(x: torch.Tensor) -> dict:
    """x [..., D] float. Summary over all elements + per-last-dim norms."""
    x = x.float()
    flat = x.reshape(-1)
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "absmax": float(x.abs().max().item()),
        "l2_norm_total": float(flat.norm().item()),
        "per_token_l2_mean": float(x.norm(dim=-1).mean().item()),
        "per_token_l2_std": float(x.norm(dim=-1).std().item()),
    }


def per_token_stats(x: torch.Tensor) -> dict:
    """x [N, D] -> per-token mean/std arrays + their summary."""
    x = x.float()
    pm = x.mean(dim=-1)   # [N]
    ps = x.std(dim=-1)     # [N]
    return {
        "token_count": int(x.shape[0]),
        "per_token_mean_range": [float(pm.min().item()), float(pm.max().item())],
        "per_token_mean_std": float(pm.std().item()),
        "per_token_std_mean": float(ps.mean().item()),
        "per_token_std_min": float(ps.min().item()),
        "per_token_std_max": float(ps.max().item()),
        "token_l2": [round(float(v), 3) for v in x.norm(dim=-1).tolist()],
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    teacher = load_file(str(TEACHER))["context"].float()    # [L,4096]
    student = load_file(str(STUDENT))["context"].float()   # [64,4096]
    print(f"teacher {tuple(teacher.shape)}  student {tuple(student.shape)}")

    te = load_text_embedding(TE_SHARD).float()

    # ---- before projection (4096 space) ----
    rep = {
        "teacher_4096": dist_stats(teacher),
        "student_4096": dist_stats(student),
        "teacher_4096_per_token": per_token_stats(teacher),
        "student_4096_per_token": per_token_stats(student),
    }

    # ---- after per-token projection (1536 space) ----
    t1536 = te(teacher)    # [L,1536]
    s1536 = te(student)    # [64,1536]
    rep["teacher_1536"] = dist_stats(t1536)
    rep["student_1536"] = dist_stats(s1536)
    rep["teacher_1536_per_token"] = per_token_stats(t1536)
    rep["student_1536_per_token"] = per_token_stats(s1536)

    # ---- mean-pool comparison in both spaces ----
    # In 4096 space
    t_pool_4096 = teacher.mean(dim=0)
    s_pool_4096 = student.mean(dim=0)
    cos_4096 = float(torch.nn.functional.cosine_similarity(
        t_pool_4096.unsqueeze(0), s_pool_4096.unsqueeze(0)).item())
    # In 1536 space (correct E1.1 path)
    t_pool_1536 = t1536.mean(dim=0)
    s_pool_1536 = s1536.mean(dim=0)
    cos_1536 = float(torch.nn.functional.cosine_similarity(
        t_pool_1536.unsqueeze(0), s_pool_1536.unsqueeze(0)).item())
    # Wrong path: project the pooled vector
    wrong_1536 = te(teacher.mean(dim=0, keepdim=True))[0]
    rep["pool_comparison"] = {
        "pooled_cosine_4096": cos_4096,
        "pooled_cosine_1536_correct_per_token_then_pool": cos_1536,
        "pooled_cosine_1536_wrong_pool_then_project": float(
            torch.nn.functional.cosine_similarity(wrong_1536.unsqueeze(0),
                                                  s_pool_1536.unsqueeze(0)).item()),
    }

    # ---- LayerNorm(4096) effect analysis ----
    # The adapter ends with nn.LayerNorm(4096).  LayerNorm normalizes each token
    # to zero-mean unit-var then affine.  Check whether the student's per-token
    # stats are artificially flattened by LayerNorm (vs teacher's natural spread).
    # Compare: student per-token std (after LN) vs teacher per-token std.
    rep["layernorm_diagnosis"] = {
        "teacher_per_token_std_mean_4096": rep["teacher_4096_per_token"]["per_token_std_mean"],
        "student_per_token_std_mean_4096": rep["student_4096_per_token"]["per_token_std_mean"],
        "teacher_token_l2_mean_4096": rep["teacher_4096"]["per_token_l2_mean"],
        "student_token_l2_mean_4096": rep["student_4096"]["per_token_l2_mean"],
        "note": ("Adapter output goes through LayerNorm(4096) which forces each of "
                 "the 64 student tokens to have unit variance. Teacher UMT5 tokens "
                 "have naturally varying norms. This norm mismatch may reduce the "
                 "margins the downstream cross-attention RMSNorm expects."),
    }

    # ---- L2 norm ratio student/teacher ----
    rep["norm_ratios"] = {
        "student_per_token_l2_over_teacher": (
            rep["student_4096"]["per_token_l2_mean"] /
            max(rep["teacher_4096"]["per_token_l2_mean"], 1e-8)),
    }

    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
