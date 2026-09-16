#!/usr/bin/env python3
"""E1.1: Corrected downstream-metric re-evaluation of two Adapter checkpoints.

Compares the *epoch5 best* adapter against the *epoch20 final* adapter using the
CORRECT downstream metric pipeline that matches how the LingBot 1.3B DiT actually
consumes the [L, 4096] teacher / [64, 4096] student context:

    correct:  [L,4096] --per-token--> text_embedding --> [L,1536]  --> metrics
    wrong:    mean-pool [L,4096] -> [4096] -> text_embedding -> [1536]
              (the wrong path was used in the E1 report; it is NOT equivalent
               because text_embedding contains a non-linear GELU).

On the 1536-dim projected space we compute, per prompt:
  * pooled_cosine : cosine(mean_pool(teacher_1536), mean_pool(student_1536))
  * pooled_mse    : MSE(mean_pool(teacher_1536), mean_pool(student_1536))
  * probe_cosine  : cosine(flatten(probe(teacher_1536)),
                           flatten(probe(student_1536)))
  * probe_mse     : MSE(probe(teacher_1536), probe(student_1536))

The probe is a frozen, random-init module ISOMORPHIC to WanCrossAttention
(q/k/v/o Linear(1536,1536) + WanRMSNorm(1536, eps=1e-6), 12 heads, head_dim=128)
plus 32 fixed random probe queries.  Because it has learned (random) q/k/o
projections, it produces NON-UNIFORM attention weights (it is not a degenerate
mean pool).

This script ONLY:
  * loads frozen MiniCPM5 + a frozen TextAdapter to re-encode prompts,
  * loads the frozen LingBot text_embedding (shard 1 weights),
  * builds the frozen ProbeAttention,
  * reads cached teacher contexts.
It does NOT modify the DiT, does NOT retrain the adapter, does NOT touch UMT5,
and does NOT start video generation.

Usage:
    python scripts/e1_1_reeval.py \
        --prompts eval/e1/prompts.jsonl \
        --teacher-dir eval/e1/teacher_contexts \
        --adapter-epoch5 eval/e1/best_adapter.safetensors \
        --adapter-epoch20 eval/e1/adapter_final/adapter.safetensors \
        --adapter-config eval/e1/adapter_final/config.json \
        --text-embedding-ckpt <lingbot shard1 path> \
        --output-dir eval/e1.1 \
        --device mps
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from collections import OrderedDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from wan.adapters.text_adapter import TextAdapter  # noqa: E402
from wan.modules.model import WanRMSNorm  # noqa: E402

DEFAULT_MINICPM5_DIR = (
    "/Volumes/ssd/huggingface/hub/models--openbmb--MiniCPM5-2B/snapshots/"
    "12a3808a956f869c767195e9266b59c4d21d92e2"
)
DEFAULT_LINGBOT_SHARD1 = (
    "/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/"
    "snapshots/7e36a5f919f86cb4255cc9bfc30adb44963fbde1/"
    "model-00001-of-00006.safetensors"
)
DEFAULT_MINICPM5_ID = "openbmb/MiniCPM5-2B"
DEFAULT_MINICPM5_SNAPSHOT = "12a3808a956f869c767195e9266b59c4d21d92e2"

CONTEXT_DIM = 4096   # adapter / UMT5 context width
TE_DIM = 1536        # LingBot DiT inner dim (after text_embedding)
NUM_PROBES = 32
NUM_HEADS = 12       # 1.3B DiT: dim=1536, num_heads=12, head_dim=128
HEAD_DIM = TE_DIM // NUM_HEADS  # 128
PROBE_SEED = 42
TIE_EPS = 1e-6       # |delta| below this counts as tie for win/tie/loss


# ---------------------------------------------------------------------------
# Frozen LingBot text_embedding (per-token projection, NOT mean-pool first)
# ---------------------------------------------------------------------------

def load_text_embedding(ckpt_path: str) -> nn.Sequential:
    """Build the frozen LingBot text_embedding MLP from shard-1 weights.

    Structure (must match WanModel.text_embedding):
        nn.Linear(4096, 1536) -> nn.GELU(approximate='tanh') -> nn.Linear(1536, 1536)

    All parameters are fp32 and requires_grad=False.
    """
    te = nn.Sequential(OrderedDict([
        ("0", nn.Linear(CONTEXT_DIM, TE_DIM)),
        ("1", nn.GELU(approximate="tanh")),
        ("2", nn.Linear(TE_DIM, TE_DIM)),
    ]))
    sd = load_file(ckpt_path)
    missing = []
    for name, module in [("0", te[0]), ("2", te[2])]:
        w_key = f"text_embedding.{name}.weight"
        b_key = f"text_embedding.{name}.bias"
        if w_key not in sd or b_key not in sd:
            missing.append(w_key)
            continue
        module.weight.data.copy_(sd[w_key].float())
        module.bias.data.copy_(sd[b_key].float())
    if missing:
        raise RuntimeError(f"text_embedding weights missing: {missing}")
    te.eval()
    te.requires_grad_(False)
    return te


# ---------------------------------------------------------------------------
# ProbeAttention: frozen, random-init, isomorphic to WanCrossAttention @ 1536
# ---------------------------------------------------------------------------

class ProbeAttention(nn.Module):
    """Frozen probe attention over the 1536-dim projected space.

    Isomorphic to ``WanCrossAttention`` (1.3B DiT, dim=1536, num_heads=12):
      * q, k, v, o : nn.Linear(1536, 1536)
      * norm_q, norm_k : WanRMSNorm(1536, eps=1e-6)
      * multi-head scaled dot-product attention (12 heads, head_dim=128)
    plus 32 fixed random probe queries acting as the cross-attention "x".

    All weights are randomly initialized (seed=42) and frozen (requires_grad=False).
    Because q/k/o are random linear projections, attention weights over a
    non-uniform context are NON-uniform (it is not a degenerate mean pool).
    """

    def __init__(self, dim: int = TE_DIM, num_heads: int = NUM_HEADS,
                 num_probes: int = NUM_PROBES, eps: float = 1e-6, seed: int = PROBE_SEED):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        # Dedicated generator so init is fully reproducible and independent of
        # the global RNG state.
        g = torch.Generator().manual_seed(seed)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

        # Fixed probe queries [P, dim].
        probes = torch.randn(num_probes, dim, generator=g) * 0.02
        self.register_buffer("probes", probes)

        # Deterministic init with the dedicated generator.
        for lin in (self.q, self.k, self.v, self.o):
            nn.init.xavier_uniform_(lin.weight, generator=g)
            nn.init.zeros_(lin.bias)
        # WanRMSNorm weight stays at ones (default).

        self.eval()
        self.requires_grad_(False)

    def _project_qkv(self, context: torch.Tensor):
        """Return (q, k, v) as [B, H, L, head_dim].

        context: [B, L, dim].  Probes are used as the query source.
        """
        b = context.shape[0]
        probes = self.probes.unsqueeze(0).expand(b, -1, -1).to(context.dtype)
        q = self.norm_q(self.q(probes))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: [B, L, dim] -> [B, num_probes, dim]."""
        b = context.shape[0]
        q, k, v = self._project_qkv(context)
        out = F.scaled_dot_product_attention(q, k, v)  # [B,H,P,head_dim]
        out = out.transpose(1, 2).contiguous().view(b, -1, self.dim)
        out = self.o(out)
        return out

    def attention_weights(self, context: torch.Tensor) -> torch.Tensor:
        """Return softmax attention weights [B, num_heads, P, L] (verification only)."""
        q, k, _ = self._project_qkv(context)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B,H,P,L]
        return torch.softmax(scores, dim=-1)


# ---------------------------------------------------------------------------
# Metric primitives (imported by unit tests)
# ---------------------------------------------------------------------------

def per_token_project(context: torch.Tensor, te: nn.Sequential) -> torch.Tensor:
    """Apply frozen text_embedding PER TOKEN: [L,4096] -> [L,1536] (fp32).

    This is the CORRECT path.  It is NOT equivalent to
    ``te(context.mean(dim=0))`` because te contains a non-linear GELU.
    """
    return te(context.float())


def pooled_cosine_1536(teacher_1536: torch.Tensor,
                       student_1536: torch.Tensor) -> float:
    """cosine(mean(teacher_1536,0), mean(student_1536,0)) on the 1536 space."""
    tp = teacher_1536.mean(dim=0)
    sp = student_1536.mean(dim=0)
    return float(F.cosine_similarity(tp, sp, dim=0).item())


def pooled_mse_1536(teacher_1536: torch.Tensor,
                    student_1536: torch.Tensor) -> float:
    return float(F.mse_loss(student_1536.mean(dim=0), teacher_1536.mean(dim=0)).item())


def probe_cosine(probe_teacher: torch.Tensor, probe_student: torch.Tensor) -> float:
    """cosine over flattened [P,1536] probe outputs."""
    return float(F.cosine_similarity(
        probe_teacher.flatten().unsqueeze(0),
        probe_student.flatten().unsqueeze(0),
        dim=-1,
    ).item())


def probe_mse(probe_teacher: torch.Tensor, probe_student: torch.Tensor) -> float:
    return float(F.mse_loss(probe_student, probe_teacher).item())


@torch.no_grad()
def compute_prompt_metrics(teacher_4096: torch.Tensor,
                           student_4096: torch.Tensor,
                           te: nn.Sequential,
                           probe: ProbeAttention) -> dict:
    """Compute all downstream metrics for one prompt.

    Args:
        teacher_4096: [L, 4096] (any dtype; cast to fp32 internally).
        student_4096: [64, 4096] (any dtype; cast to fp32 internally).
        te: frozen text_embedding.
        probe: frozen ProbeAttention.

    Returns:
        dict with pooled_cosine, pooled_mse, probe_cosine, probe_mse,
        plus attention weight std (non-uniformity check) for teacher & student.
    """
    t1536 = per_token_project(teacher_4096, te)    # [L,1536]
    s1536 = per_token_project(student_4096, te)   # [64,1536]

    p_cos = pooled_cosine_1536(t1536, s1536)
    p_mse = pooled_mse_1536(t1536, s1536)

    t_probe = probe(t1536.unsqueeze(0))[0]       # [32,1536]
    s_probe = probe(s1536.unsqueeze(0))[0]        # [32,1536]
    pr_cos = probe_cosine(t_probe, s_probe)
    pr_mse = probe_mse(t_probe, s_probe)

    # Non-uniformity diagnostics (attention weight std over the key axis).
    t_attn_w = probe.attention_weights(t1536.unsqueeze(0))  # [1,H,32,L]
    s_attn_w = probe.attention_weights(s1536.unsqueeze(0))  # [1,H,32,64]
    t_attn_std = float(t_attn_w.std().item())
    s_attn_std = float(s_attn_w.std().item())

    return {
        "pooled_cosine_1536": p_cos,
        "pooled_mse_1536": p_mse,
        "probe_cosine": pr_cos,
        "probe_mse": pr_mse,
        "teacher_attn_weight_std": t_attn_std,
        "student_attn_weight_std": s_attn_std,
    }


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_adapter_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_adapter(config: dict, weights_path: str, device: torch.device) -> TextAdapter:
    adapter = TextAdapter(
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_queries=config["num_queries"],
        num_resampler_layers=config["num_resampler_layers"],
        num_heads=config.get("num_heads", 8),
        ffn_mult=config.get("ffn_mult", 4),
    )
    state = load_file(weights_path)
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if missing:
        logging.warning("Adapter load: %d missing keys (first 3): %s", len(missing), missing[:3])
    if unexpected:
        logging.warning("Adapter load: %d unexpected keys (first 3): %s", len(unexpected), unexpected[:3])
    adapter = adapter.float().to(device)
    adapter.eval()
    adapter.requires_grad_(False)
    return adapter


# ---------------------------------------------------------------------------
# Encoding: MiniCPM5 + Adapter for one checkpoint, all prompts
# ---------------------------------------------------------------------------

def encode_all_prompts(
    prompts: list[dict],
    adapter_weights: str,
    adapter_cfg: dict,
    minicpm5_dir: str,
    out_dir: str,
    device: torch.device,
    log: logging.Logger,
) -> int:
    """Load MiniCPM5 + adapter once, encode all prompts, save [64,4096] bf16.

    Returns the number of prompts encoded.  MiniCPM5 + adapter are unloaded on
    return (del + gc + mps cache).
    """
    from transformers import AutoModel, AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_dir) or ".", exist_ok=True)

    log.info("Loading MiniCPM5 tokenizer (use_fast=False) from %s ...", minicpm5_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        minicpm5_dir, local_files_only=True, use_fast=False)
    log.info("Loading MiniCPM5 (bf16, frozen) ...")
    minicpm = AutoModel.from_pretrained(
        minicpm5_dir, torch_dtype=torch.bfloat16, local_files_only=True)
    minicpm = minicpm.to(device)
    minicpm.eval()
    minicpm.requires_grad_(False)

    adapter = build_adapter(adapter_cfg, adapter_weights, device)
    log.info("Adapter loaded: %s", adapter_weights)

    n = 0
    try:
        for i, pr in enumerate(prompts):
            pid = pr["id"]
            text = pr["text"]
            out_path = os.path.join(out_dir, f"{pid}.safetensors")
            if os.path.isfile(out_path):
                log.info("  [%d/%d] %s already exists, skip.", i + 1, len(prompts), pid)
                n += 1
                continue
            enc = tokenizer(
                [text], return_tensors="pt", padding=True, truncation=True,
                max_length=512,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = minicpm(**enc).last_hidden_state       # [1,L,2048] bf16
                ctx = adapter(hidden, enc["attention_mask"])    # [1,64,4096] fp32
            ctx_bf16 = ctx[0].to(torch.bfloat16).contiguous()   # [64,4096]
            assert ctx_bf16.shape == (64, 4096), f"unexpected shape {ctx_bf16.shape}"
            save_file({"context": ctx_bf16}, out_path)
            n += 1
            if (i + 1) % 10 == 0 or i == 0:
                log.info("  [%d/%d] encoded %s (L=%d)",
                         i + 1, len(prompts), pid, int(enc["attention_mask"].sum().item()))
    finally:
        del minicpm, adapter, tokenizer
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    log.info("Encoded %d prompts -> %s", n, out_dir)
    return n


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(t.mean().item()),
        "median": float(t.median().item()),
        "std": float(t.std().item()) if len(t) > 1 else 0.0,
        "count": len(t),
    }


def summarize(per_prompt: list[dict], split: str | None = None) -> dict:
    """Aggregate metrics.  If split given, filter to that split first."""
    rows = per_prompt
    if split is not None:
        rows = [r for r in per_prompt if r.get("split") == split]
    out = {}
    for metric in ("pooled_cosine_1536", "pooled_mse_1536",
                   "probe_cosine", "probe_mse"):
        out[metric] = _stats([r[metric] for r in rows])
    return out


def classify_win_tie_loss(delta: float) -> str:
    if abs(delta) < TIE_EPS:
        return "tie"
    return "win" if delta > 0 else "loss"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1.1: corrected downstream-metric re-eval.")
    p.add_argument("--prompts", default="eval/e1/prompts.jsonl")
    p.add_argument("--teacher-dir", default="eval/e1/teacher_contexts")
    p.add_argument("--adapter-epoch5", default="eval/e1/best_adapter.safetensors")
    p.add_argument("--adapter-epoch20", default="eval/e1/adapter_final/adapter.safetensors")
    p.add_argument("--adapter-config", default="eval/e1/adapter_final/config.json")
    p.add_argument("--text-embedding-ckpt", default=DEFAULT_LINGBOT_SHARD1)
    p.add_argument("--minicpm5-dir", default=DEFAULT_MINICPM5_DIR)
    p.add_argument("--minicpm5-model-id", default=DEFAULT_MINICPM5_ID)
    p.add_argument("--minicpm5-snapshot", default=DEFAULT_MINICPM5_SNAPSHOT)
    p.add_argument("--output-dir", default="eval/e1.1")
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    p.add_argument("--skip-encoding", action="store_true",
                   help="Skip MiniCPM5 encoding (student contexts already on disk).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("e1.1_reeval")
    wall_start = time.time()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    stu5_dir = os.path.join(out_dir, "student_contexts_epoch5")
    stu20_dir = os.path.join(out_dir, "student_contexts_epoch20")

    device = torch.device(args.device)
    if device.type == "mps":
        torch.mps.empty_cache()

    # ---- load prompts + teacher manifest ----
    prompts = load_jsonl(args.prompts)
    manifest = {m["id"]: m for m in load_jsonl(
        os.path.join(args.teacher_dir, "manifest.jsonl"))}
    log.info("Loaded %d prompts, %d teacher manifest entries.",
             len(prompts), len(manifest))
    missing_teacher = [p["id"] for p in prompts if p["id"] not in manifest]
    if missing_teacher:
        log.error("Prompts missing teacher context: %s", missing_teacher)
        return 2

    adapter_cfg = load_adapter_config(args.adapter_config)

    # ---- ENCODING (MiniCPM5 loaded twice, once per checkpoint) ----
    if not args.skip_encoding:
        log.info("=== Encoding epoch5 best adapter ===")
        encode_all_prompts(prompts, args.adapter_epoch5, adapter_cfg,
                           args.minicpm5_dir, stu5_dir, device, log)
        log.info("=== Encoding epoch20 final adapter ===")
        encode_all_prompts(prompts, args.adapter_epoch20, adapter_cfg,
                           args.minicpm5_dir, stu20_dir, device, log)
    else:
        log.info("--skip-encoding: reusing existing student contexts on disk.")

    # ---- frozen text_embedding + probe (CPU, small) ----
    log.info("Loading frozen text_embedding from %s ...", args.text_embedding_ckpt)
    te = load_text_embedding(args.text_embedding_ckpt).float()
    assert all(not p.requires_grad for p in te.parameters()), "text_embedding must be frozen"

    log.info("Building frozen ProbeAttention (seed=%d) ...", PROBE_SEED)
    probe = ProbeAttention(seed=PROBE_SEED).float()
    assert all(not p.requires_grad for p in probe.parameters()), "probe must be frozen"

    # ---- per-prompt metrics for both checkpoints ----
    results5: list[dict] = []
    results20: list[dict] = []
    for pr in prompts:
        pid = pr["id"]
        split = pr.get("split", "unknown")
        # teacher [L,4096] bf16
        t_path = os.path.join(args.teacher_dir, manifest[pid]["filename"])
        teacher = load_file(t_path)["context"]
        # student epoch5 / epoch20 [64,4096] bf16
        s5 = load_file(os.path.join(stu5_dir, f"{pid}.safetensors"))["context"]
        s20 = load_file(os.path.join(stu20_dir, f"{pid}.safetensors"))["context"]

        m5 = compute_prompt_metrics(teacher, s5, te, probe)
        m20 = compute_prompt_metrics(teacher, s20, te, probe)
        results5.append({"id": pid, "split": split, **m5})
        results20.append({"id": pid, "split": split, **m20})

    # ---- write per-prompt results ----
    with open(os.path.join(out_dir, "results_epoch5.jsonl"), "w", encoding="utf-8") as fh:
        for r in results5:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "results_epoch20.jsonl"), "w", encoding="utf-8") as fh:
        for r in results20:
            fh.write(json.dumps(r) + "\n")

    # ---- summaries by split & checkpoint ----
    summary = {
        "by_checkpoint": {
            "epoch5": {sp: summarize(results5, sp) for sp in ("train", "val")},
            "epoch20": {sp: summarize(results20, sp) for sp in ("train", "val")},
        },
        "delta_epoch20_minus_epoch5": {},
        "win_tie_loss": {},
        "recommendation": None,
        "best_epoch": None,
    }

    # ---- per-prompt delta + win/tie/loss ----
    by_id5 = {r["id"]: r for r in results5}
    by_id20 = {r["id"]: r for r in results20}
    deltas = {sp: {} for sp in ("train", "val")}
    wtl = {metric: {"train": {"win": 0, "tie": 0, "loss": 0},
                    "val": {"win": 0, "tie": 0, "loss": 0}}
           for metric in ("pooled_cosine_1536", "probe_cosine")}
    for pid in by_id5:
        r5, r20 = by_id5[pid], by_id20[pid]
        split = r5["split"]
        for metric in ("pooled_cosine_1536", "pooled_mse_1536",
                       "probe_cosine", "probe_mse"):
            d = r20[metric] - r5[metric]
            deltas.setdefault(split, {}).setdefault(metric, []).append(d)
        # win/tie/loss: for cosine, higher=win; for mse, lower=win, but the task
        # defines win = epoch20 higher on *cosine* metrics.
        for metric in ("pooled_cosine_1536", "probe_cosine"):
            d = r20[metric] - r5[metric]
            wtl[metric][split][classify_win_tie_loss(d)] += 1

    for sp in ("train", "val"):
        summary["delta_epoch20_minus_epoch5"][sp] = {
            metric: _stats(vals) for metric, vals in deltas[sp].items()
        }
    summary["win_tie_loss"] = wtl

    # ---- recommendation (val split priority) ----
    val5 = summary["by_checkpoint"]["epoch5"]["val"]
    val20 = summary["by_checkpoint"]["epoch20"]["val"]
    d_pooled_cos = val20["pooled_cosine_1536"]["mean"] - val5["pooled_cosine_1536"]["mean"]
    d_probe_cos = val20["probe_cosine"]["mean"] - val5["probe_cosine"]["mean"]

    # Rule: epoch5 wins if it is >= epoch20 on either primary metric; epoch20
    # wins only if it is clearly (>0.02) better on val.
    epoch20_better = (d_pooled_cos > 0.02) and (d_probe_cos > 0.0)
    if epoch20_better:
        best_epoch = "epoch20"
        rec = (f"Recommend epoch20: val pooled_cosine_1536 improves by "
               f"{d_pooled_cos:+.4f} (>0.02) and probe_cosine by {d_probe_cos:+.4f}.")
    else:
        best_epoch = "epoch5"
        rec = (f"Recommend epoch5: val pooled_cosine_1536 delta "
               f"{d_pooled_cos:+.4f}, probe_cosine delta {d_probe_cos:+.4f}; "
               f"epoch5 matches or beats epoch20 on held-out val (early stopping "
               f"generalizes better, and is equally lightweight).")
    summary["recommendation"] = rec
    summary["best_epoch"] = best_epoch

    # ---- manifest ----
    manifest_out = {
        "best_epoch": best_epoch,
        "epoch5_adapter_sha256": sha256_file(args.adapter_epoch5),
        "epoch20_adapter_sha256": sha256_file(args.adapter_epoch20),
        "adapter_config": adapter_cfg,
        "minicpm5_model_id": args.minicpm5_model_id,
        "minicpm5_snapshot": args.minicpm5_snapshot,
        "text_embedding_source": os.path.abspath(args.text_embedding_ckpt),
        "prompts_file_sha256": sha256_file(args.prompts),
        "teacher_context_count": len(manifest),
        "student_context_epoch5_count": len(results5),
        "student_context_epoch20_count": len(results20),
        "probe_seed": PROBE_SEED,
        "probe_num_heads": NUM_HEADS,
        "probe_head_dim": HEAD_DIM,
        "probe_num_queries": NUM_PROBES,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_time_seconds": round(time.time() - wall_start, 1),
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest_out, fh, indent=2)

    # ---- stdout report ----
    log.info("=== E1.1 Re-eval done in %.1fs ===", time.time() - wall_start)
    log.info("VAL epoch5: %s", json.dumps(val5, indent=2))
    log.info("VAL epoch20: %s", json.dumps(val20, indent=2))
    log.info("VAL delta (e20-e5): pooled_cos=%.4f probe_cos=%.4f",
             d_pooled_cos, d_probe_cos)
    log.info("Win/tie/loss (val): %s",
             json.dumps({m: wtl[m]["val"] for m in wtl}, indent=2))
    log.info("RECOMMENDATION: %s", rec)
    print(json.dumps({"status": "PASS", "best_epoch": best_epoch,
                      "recommendation": rec}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
