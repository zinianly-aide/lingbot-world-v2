#!/usr/bin/env python3
"""E1: Train the TextAdapter to distill UMT5-XXL teacher context from MiniCPM5-2B.

Frozen backbone:
  * MiniCPM5-2B  (bf16, frozen)  -> last_hidden_state [B, L, 2048]
  * text_embedding (LingBot 1.3B shard 1, fp32, frozen) -> reference metric only

Trainable:
  * TextAdapter (fp32)  [B, L, 2048] -> [B, 64, 4096]

Losses (all in the 4096-dim context space):
  1. pooled_cosine: 1 - cosine(teacher_mean_pool, student_mean_pool)
  2. pooled_mse:     MSE(teacher_mean_pool, student_mean_pool)
  3. cross_attn_distill: MSE over fixed probe-queries attention outputs.

Reference metrics (NOT in loss):
  * attention_similarity: cosine of teacher vs student probe outputs
  * post_projection_cosine / post_projection_mse: after LingBot text_embedding

G1 / video / diffusion are NOT touched.  No MiniCPM5 or LingBot parameters are
trained.  The UMT5 teacher is consumed only from pre-cached .safetensors files
produced by scripts/e1_prepare_teacher.py; UMT5 is never loaded here.

Usage:
    python scripts/e1_train_adapter.py \
        --prompts eval/e1/prompts.jsonl \
        --teacher-dir eval/e1/teacher_contexts \
        --text-embedding-ckpt <lingbot shard1> \
        --minicpm5-dir <minicpm5 snapshot> \
        --output-dir eval/e1
"""
from __future__ import annotations

import argparse
import gc
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

DEFAULT_MINICPM5_DIR = (
    "/Volumes/ssd/huggingface/hub/models--openbmb--MiniCPM5-2B/snapshots/"
    "12a3808a956f869c767195e9266b59c4d21d92e2"
)
DEFAULT_LINGBOT_SHARD1 = (
    "/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/"
    "snapshots/7e36a5f919f86cb4255cc9bfc30adb44963fbde1/"
    "model-00001-of-00006.safetensors"
)

CONTEXT_DIM = 4096
TE_DIM = 1536
NUM_PROBES = 32


# ---------------------------------------------------------------------------
# Loss / metric primitives (imported by unit tests)
# ---------------------------------------------------------------------------

def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean-pool over the token dim, honoring a 1=valid / 0=pad mask.

    Args:
        x: [B, L, D]
        mask: [B, L] (1 = keep).  If None, simple mean over L.
    Returns:
        [B, D]
    """
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(x.dtype).unsqueeze(-1)  # [B,L,1]
    summed = (x * m).sum(dim=1)
    n = m.sum(dim=1).clamp(min=1.0)
    return summed / n


def pooled_cosine_loss(teacher_pooled: torch.Tensor,
                       student_pooled: torch.Tensor) -> torch.Tensor:
    """1 - mean(cosine_sim(teacher, student)).  Lower is better."""
    cos = F.cosine_similarity(teacher_pooled, student_pooled, dim=-1)
    return (1.0 - cos).mean()


def pooled_mse_loss(teacher_pooled: torch.Tensor,
                     student_pooled: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_pooled, teacher_pooled)


class ProbeAttentionDistiller(nn.Module):
    """Fixed probe queries -> scaled dot-product attention over a context.

    probes: [P, D], created once with a fixed seed, requires_grad=False.
    forward(context, key_padding_mask=None) -> [B, P, D].
    """

    def __init__(self, dim: int = CONTEXT_DIM, num_probes: int = NUM_PROBES,
                 seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        probes = torch.randn(num_probes, dim, generator=g) * 0.02
        self.register_buffer("probes", probes)

    def forward(self, context: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # context: [B, N, D]; key_padding_mask: [B, N] with True = ignore.
        b, n, d = context.shape
        q = self.probes.unsqueeze(0).expand(b, -1, -1).to(context.dtype)
        scale = d ** -0.5
        scores = torch.matmul(q, context.transpose(1, 2)) * scale  # [B,P,N]
        if key_padding_mask is not None:
            kpm = key_padding_mask.to(torch.bool).unsqueeze(1)  # [B,1,N]
            scores = scores.masked_fill(kpm, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        # If a whole row was masked (shouldn't happen), softmax gives NaN; guard.
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.matmul(attn, context)


def cross_attn_distill_loss(teacher_out: torch.Tensor,
                            student_out: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_out, teacher_out)


def attention_similarity(teacher_out: torch.Tensor,
                        student_out: torch.Tensor) -> float:
    """Cosine sim averaged over probes and batch (higher better)."""
    cos = F.cosine_similarity(teacher_out, student_out, dim=-1)  # [B,P]
    return float(cos.mean().item())


# ---------------------------------------------------------------------------
# LingBot text_embedding loader (frozen, reference metric only)
# ---------------------------------------------------------------------------

def load_text_embedding(ckpt_path: str) -> nn.Sequential:
    """Build the frozen LingBot text_embedding MLP from shard-1 weights."""
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
# Prompt / teacher loading
# ---------------------------------------------------------------------------

def load_prompts(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_teacher_context(path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a cached [L,4096] bf16 teacher context -> (fp32 ctx [1,L,4096], mask [1,L])."""
    sd = load_file(path)
    ctx = sd["context"].float()  # [L,4096]
    L = ctx.shape[0]
    mask = torch.ones(1, L, dtype=torch.float32)
    return ctx.unsqueeze(0).to(device), mask.to(device)


def collate_teacher(batch_ctx: list[torch.Tensor],
                    batch_mask: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad list of [1,L_i,4096] -> [B, maxL, 4096] and [B, maxL]."""
    b = len(batch_ctx)
    max_l = max(c.shape[1] for c in batch_ctx)
    out = torch.zeros(b, max_l, CONTEXT_DIM, dtype=torch.float32)
    out_mask = torch.zeros(b, max_l, dtype=torch.float32)
    for i, (c, m) in enumerate(zip(batch_ctx, batch_mask)):
        L = c.shape[1]
        out[i, :L] = c[0]
        out_mask[i, :L] = m[0]
    return out, out_mask


# ---------------------------------------------------------------------------
# Metrics bundle for an eval step
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(student_ctx: torch.Tensor,
                    teacher_ctx: torch.Tensor,
                    teacher_mask: torch.Tensor,
                    distiller: ProbeAttentionDistiller,
                    text_embedding: nn.Sequential | None) -> dict:
    """Return all train/val metrics.  student_ctx [B,64,4096] fp32."""
    t_pool = masked_mean_pool(teacher_ctx, teacher_mask)       # [B,4096]
    s_pool = student_ctx.mean(dim=1)                            # [B,4096]

    m_pool_cosine = float((1.0 - F.cosine_similarity(t_pool, s_pool, dim=-1)).mean().item())
    m_pool_mse = float(F.mse_loss(s_pool, t_pool).item())

    t_probe = distiller(teacher_ctx, key_padding_mask=teacher_mask.eq(0))
    s_probe = distiller(student_ctx, key_padding_mask=None)
    m_xattn = float(F.mse_loss(s_probe, t_probe).item())
    m_attn_sim = float(F.cosine_similarity(t_probe, s_probe, dim=-1).mean().item())

    m_pp_cos = m_pp_mse = float("nan")
    if text_embedding is not None:
        t_pp = text_embedding(t_pool)
        s_pp = text_embedding(s_pool)
        m_pp_cos = float(F.cosine_similarity(t_pp, s_pp, dim=-1).mean().item())
        m_pp_mse = float(F.mse_loss(s_pp, t_pp).item())

    return {
        "pooled_cosine": m_pool_cosine,   # 1-cosine, lower better
        "pooled_cosine_sim": 1.0 - m_pool_cosine,
        "pooled_mse": m_pool_mse,
        "cross_attn": m_xattn,
        "attention_similarity": m_attn_sim,
        "post_projection_cosine": m_pp_cos,
        "post_projection_mse": m_pp_mse,
    }


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1: train TextAdapter (UMT5 distillation).")
    p.add_argument("--prompts", default="eval/e1/prompts.jsonl")
    p.add_argument("--teacher-dir", default="eval/e1/teacher_contexts")
    p.add_argument("--text-embedding-ckpt", default=DEFAULT_LINGBOT_SHARD1)
    p.add_argument("--minicpm5-dir", default=DEFAULT_MINICPM5_DIR)
    p.add_argument("--output-dir", default="eval/e1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    p.add_argument("--loss-weights", default="1.0,1.0,1.0",
                   help="w_cosine,w_mse,w_cross_attn")
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_minicpm5_dir(arg: str) -> str:
    if os.path.isfile(os.path.join(arg, "config.json")):
        return arg
    import glob
    snaps = glob.glob(os.path.join(arg, "snapshots", "*")) if os.path.isdir(arg) else []
    for s in sorted(snaps, reverse=True):
        if os.path.isfile(os.path.join(s, "config.json")):
            return s
    return arg


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("e1_train")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    w_cos, w_mse, w_xattn = [float(x) for x in args.loss_weights.split(",")]
    log.info("Loss weights: cosine=%.3f mse=%.3f xattn=%.3f", w_cos, w_mse, w_xattn)

    out_dir = os.path.join(REPO_ROOT, args.output_dir)
    final_dir = os.path.join(out_dir, "adapter_final")
    os.makedirs(final_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "training_log.jsonl")

    # ---- prompts + teacher paths ----
    rows = load_prompts(os.path.join(REPO_ROOT, args.prompts))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    log.info("Prompts: total=%d train=%d val=%d", len(rows), len(train_rows), len(val_rows))

    # Map prompt id -> teacher context filename.  Prefer the prepare-step
    # manifest; fall back to sha256(text).safetensors naming.
    from wan.utils.prompt_embedding import prompt_sha256
    teacher_map: dict[str, str] = {}
    manifest_path = os.path.join(REPO_ROOT, args.teacher_dir, "manifest.jsonl")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fh:
            for line in fh:
                m = json.loads(line)
                teacher_map[m["id"]] = os.path.join(
                    REPO_ROOT, args.teacher_dir, m["filename"])
    else:
        # Fallback: filename = sha256(text).safetensors
        from wan.utils.prompt_embedding import prompt_sha256
        for r in rows:
            sha = prompt_sha256(r["text"])
            teacher_map[r["id"]] = os.path.join(REPO_ROOT, args.teacher_dir, f"{sha}.safetensors")

    # ---- frozen MiniCPM5 ----
    from transformers import AutoModel, AutoTokenizer
    mcp_dir = resolve_minicpm5_dir(args.minicpm5_dir)
    log.info("Loading MiniCPM5 from %s (frozen, bf16)...", mcp_dir)
    mcp_tok = AutoTokenizer.from_pretrained(mcp_dir, local_files_only=True, use_fast=False)
    minicpm = AutoModel.from_pretrained(mcp_dir, torch_dtype=torch.bfloat16,
                                        local_files_only=True)
    minicpm = minicpm.to(device)
    minicpm.eval()
    minicpm.requires_grad_(False)

    # ---- trainable adapter ----
    adapter = TextAdapter(hidden_dim=2048, output_dim=4096, num_queries=64,
                         num_resampler_layers=2, num_heads=8, ffn_mult=4)
    adapter = adapter.float().to(device)

    # ---- frozen text_embedding (reference metric) ----
    text_embedding = load_text_embedding(os.path.join(REPO_ROOT, args.text_embedding_ckpt))
    text_embedding = text_embedding.to(device)

    # ---- fixed probe distiller ----
    distiller = ProbeAttentionDistiller(dim=CONTEXT_DIM, num_probes=NUM_PROBES,
                                        seed=args.seed).to(device)
    # probes is a buffer (requires_grad=False by construction).

    # ---- freeze verification ----
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    total = sum(p.numel() for p in adapter.parameters())
    mcp_total = sum(p.numel() for p in minicpm.parameters())
    te_total = sum(p.numel() for p in text_embedding.parameters())
    log.info("TextAdapter: trainable=%d  total=%d", trainable, total)
    log.info("MiniCPM5 params (frozen): %d", mcp_total)
    log.info("text_embedding params (frozen): %d", te_total)
    assert all(not p.requires_grad for p in minicpm.parameters()), "MiniCPM5 must be frozen"
    assert all(not p.requires_grad for p in text_embedding.parameters()), "text_embedding must be frozen"
    assert all(p.requires_grad for p in adapter.parameters()), "Adapter must be trainable"
    assert not distiller.probes.requires_grad, "probe queries must not train"

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    def make_batch(recs: list[dict]):
        texts = [r["text"] for r in recs]
        enc = mcp_tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_prompt_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        t_ctx, t_mask = [], []
        for r in recs:
            c, m = load_teacher_context(teacher_map[r["id"]], device)
            t_ctx.append(c)
            t_mask.append(m)
        t_ctx, t_mask = collate_teacher(t_ctx, t_mask)
        t_ctx = t_ctx.to(device)
        t_mask = t_mask.to(device)
        return enc, t_ctx, t_mask

    best_val_cos = -1.0
    best_path = os.path.join(out_dir, "best_adapter.safetensors")
    log_fp = open(log_path, "w", encoding="utf-8")
    first_backward_checked = False

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        order = list(range(len(train_rows)))
        g = torch.Generator().manual_seed(args.seed + epoch)
        perm = torch.randperm(len(order), generator=g).tolist()
        ep_losses = []
        t0 = time.time()
        for si in range(0, len(train_rows), args.batch_size):
            idx = perm[si:si + args.batch_size]
            recs = [train_rows[i] for i in idx]
            enc, t_ctx, t_mask = make_batch(recs)

            with torch.no_grad():
                hidden = minicpm(**enc).last_hidden_state  # [B,L,2048] bf16
            attn_mask = enc["attention_mask"]  # [B,L]
            student_ctx = adapter(hidden, attn_mask)  # [B,64,4096] fp32

            t_pool = masked_mean_pool(t_ctx, t_mask)
            s_pool = student_ctx.mean(dim=1)
            l_cos = pooled_cosine_loss(t_pool.detach(), s_pool)
            l_mse = pooled_mse_loss(t_pool.detach(), s_pool)
            t_probe = distiller(t_ctx, key_padding_mask=t_mask.eq(0))
            s_probe = distiller(student_ctx, key_padding_mask=None)
            l_x = cross_attn_distill_loss(t_probe.detach(), s_probe)
            loss = w_cos * l_cos + w_mse * l_mse + w_xattn * l_x

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if not first_backward_checked:
                # Verify MiniCPM5 grads are still None after backward.
                mcp_grad_none = all(p.grad is None for p in minicpm.parameters())
                te_grad_none = all(p.grad is None for p in text_embedding.parameters())
                log.info("After first backward: MiniCPM5 all-grad-None=%s  text_embedding all-grad-None=%s",
                         mcp_grad_none, te_grad_none)
                first_backward_checked = True

            ep_losses.append({
                "loss": float(loss.item()),
                "pooled_cosine": float(l_cos.item()),
                "pooled_mse": float(l_mse.item()),
                "cross_attn": float(l_x.item()),
            })

            del hidden, student_ctx, t_ctx, t_mask, enc
            if device.type == "mps":
                torch.mps.empty_cache()

        # ---- validation ----
        adapter.eval()
        val_metrics_acc = []
        with torch.no_grad():
            for si in range(0, len(val_rows), args.batch_size):
                recs = val_rows[si:si + args.batch_size]
                enc, t_ctx, t_mask = make_batch(recs)
                hidden = minicpm(**enc).last_hidden_state
                attn_mask = enc["attention_mask"]
                student_ctx = adapter(hidden, attn_mask)
                m = compute_metrics(student_ctx, t_ctx, t_mask, distiller, text_embedding)
                val_metrics_acc.append(m)
                del hidden, student_ctx, t_ctx, t_mask, enc
                if device.type == "mps":
                    torch.mps.empty_cache()

        def avg(metric_list, key):
            return sum(d[key] for d in metric_list) / max(1, len(metric_list))

        train_avg = {
            "pooled_cosine": sum(d["pooled_cosine"] for d in ep_losses) / len(ep_losses),
            "pooled_mse": sum(d["pooled_mse"] for d in ep_losses) / len(ep_losses),
            "cross_attn": sum(d["cross_attn"] for d in ep_losses) / len(ep_losses),
            "total_loss": sum(d["loss"] for d in ep_losses) / len(ep_losses),
        }
        val_avg = {k: avg(val_metrics_acc, k) for k in val_metrics_acc[0].keys()}

        rec = {
            "epoch": epoch,
            "epoch_wall_s": round(time.time() - t0, 2),
            "train": {k: round(v, 6) for k, v in train_avg.items()},
            "val": {k: round(v, 6) for k, v in val_avg.items()},
        }
        log_fp.write(json.dumps(rec) + "\n")
        log_fp.flush()
        log.info("epoch %d/%d  train[cos=%.4f mse=%.5f xattn=%.5f loss=%.5f]  "
                 "val[cos_sim=%.4f mse=%.5f xattn=%.5f attn_sim=%.4f pp_cos=%.4f pp_mse=%.5f]",
                 epoch, args.epochs,
                 train_avg["pooled_cosine"], train_avg["pooled_mse"],
                 train_avg["cross_attn"], train_avg["total_loss"],
                 val_avg["pooled_cosine_sim"], val_avg["pooled_mse"],
                 val_avg["cross_attn"], val_avg["attention_similarity"],
                 val_avg["post_projection_cosine"], val_avg["post_projection_mse"])

        # Best checkpoint by val pooled_cosine_sim (higher better).
        if val_avg["pooled_cosine_sim"] > best_val_cos:
            best_val_cos = val_avg["pooled_cosine_sim"]
            save_file(adapter.state_dict(), best_path)
            log.info("  >> new best val pooled_cosine_sim=%.4f -> %s",
                     best_val_cos, best_path)

    # ---- final save ----
    final_path = os.path.join(final_dir, "adapter.safetensors")
    save_file(adapter.state_dict(), final_path)
    config = {
        "hidden_dim": 2048, "output_dim": 4096, "num_queries": 64,
        "num_resampler_layers": 2, "num_heads": 8, "ffn_mult": 4,
        "dtype": "float32",
        "training_config": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "optimizer": "AdamW", "device": args.device,
            "loss_weights": {"cosine": w_cos, "mse": w_mse, "cross_attn": w_xattn},
            "num_probes": NUM_PROBES, "seed": args.seed,
            "trainable_params": trainable, "adapter_total_params": total,
        },
        "best_val_pooled_cosine_sim": round(best_val_cos, 6),
    }
    with open(os.path.join(final_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    log_fp.close()
    log.info("Final adapter saved: %s", final_path)
    log.info("Best adapter: %s (val cos_sim=%.4f)", best_path, best_val_cos)
    log.info("Training log: %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
