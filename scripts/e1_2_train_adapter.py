#!/usr/bin/env python3
"""E1.2: train TextAdapter with REAL downstream activation distillation.

Root cause found in E2 diagnosis: the E1 adapter's output LayerNorm(4096) leaves
each student token at std=1.0 (L2~64), while UMT5 teacher tokens are std~0.06
(L2~3.9).  Feeding this 16x-too-large input into the FROZEN LingBot
text_embedding (Linear+GELU+Linear) saturates the GELU and produces 25x-too-large
1536-dim activations, which destabilizes the DiT cross-attention -> flickering,
unrecognizable videos.

E1.2 fixes the LOSS (architecture unchanged, output_norm stays so the checkpoint
still loads via scripts/e1_encode_with_adapter.py; the loss will drive
output_norm.gamma down to match teacher scale):

  Teacher: UMT5 [L,4096] -> frozen text_embedding -> [L,1536]
  Student: MiniCPM5 hidden [Lm,2048] -> Adapter -> [64,4096] -> frozen text_embedding -> [64,1536]

Frozen probes of REAL LingBot 1.3B cross-attention layers (blocks 0/14/29):
  * K/V pool loss (weight 2.0): mean-pool norm_k(k(context)) and v(context)
    over tokens, align teacher vs student.  Length-invariant (L vs 64).
  * attn-output loss: fixed frozen probes [P,1536] as queries, full SDPA through
    block-0's REAL q/k/v/o/norm_q/norm_k; align [P,1536] outputs.
  * aux: mean-pooled 1536 cosine + MSE.

Best checkpoint chosen by held-out (val) K/V loss.  Only the adapter trains.
Does NOT modify DiT / text_embedding / UMT5 / VAE.
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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from wan.adapters.text_adapter import TextAdapter  # noqa: E402
from wan.modules.model import WanRMSNorm  # noqa: E402

TE_SHARD = (
    "/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/"
    "snapshots/7e36a5f919f86cb4255cc9bfc30adb44963fbde1/"
    "model-00001-of-00006.safetensors"
)
CKPT_DIR = (
    "/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/"
    "snapshots/7e36a5f919f86cb4255cc9bfc30adb44963fbde1"
)
CONTEXT_DIM = 4096
TE_DIM = 1536
NUM_HEADS = 12
HEAD_DIM = TE_DIM // NUM_HEADS
NUM_PROBES = 32
PROBE_BLOCKS = [0, 14, 29]   # spread across the 30-block DiT
EPS = 1e-6


# ---------------------------------------------------------------------------
# Frozen LingBot text_embedding
# ---------------------------------------------------------------------------

def load_text_embedding(ckpt_path: str) -> nn.Sequential:
    te = nn.Sequential(OrderedDict([
        ("0", nn.Linear(CONTEXT_DIM, TE_DIM)),
        ("1", nn.GELU(approximate="tanh")),
        ("2", nn.Linear(TE_DIM, TE_DIM)),
    ]))
    sd = load_file(ckpt_path)
    for name, mod in [("0", te[0]), ("2", te[2])]:
        mod.weight.data.copy_(sd[f"text_embedding.{name}.weight"].float())
        mod.bias.data.copy_(sd[f"text_embedding.{name}.bias"].float())
    te.eval()
    te.requires_grad_(False)
    return te


class _ShardKeyLoader:
    """Lazily load named tensors from the correct checkpoint shard (per index.json)."""

    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        idx = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json")))
        self.wm = idx["weight_map"]
        self._cache: dict[str, str] = {}  # key -> shard path
        self._shards: dict[str, dict] = {}  # shard path -> loaded tensors

    def _shard_for(self, key: str) -> str:
        shard = self.wm[key]
        return os.path.join(self.ckpt_dir, shard)

    def get(self, key: str) -> torch.Tensor:
        if key in self._cache:
            return self._cache[key]
        shard_path = self._shard_for(key)
        if shard_path not in self._shards:
            self._shards[shard_path] = load_file(shard_path)
        val = self._shards[shard_path][key].float()
        self._cache[key] = val
        return val


# ---------------------------------------------------------------------------
# Real (checkpoint-loaded) cross-attention K/V + attn-output probe
# ---------------------------------------------------------------------------

class RealCrossAttnDistiller(nn.Module):
    """Frozen, real-weight cross-attention K/V pool + attn-output probe.

    Holds, for PROBE_BLOCKS:
        k_b, norm_k_b, v_b  (used for the K/V pool loss)
    and for block 0 also:
        q_0, norm_q_0, o_0  (used for the full attn-output loss)
    plus fixed frozen probe queries [P, TE_DIM].
    """

    def __init__(self, ckpt_dir: str, blocks=(0, 14, 29), num_probes=NUM_PROBES,
                 seed=42):
        super().__init__()
        loader = _ShardKeyLoader(ckpt_dir)
        self.blocks = list(blocks)

        # K/V for every block in `blocks`.
        for b in self.blocks:
            pre = f"blocks.{b}.cross_attn."
            k = nn.Linear(TE_DIM, TE_DIM)
            k.weight.data.copy_(loader.get(f"{pre}k.weight"))
            k.bias.data.copy_(loader.get(f"{pre}k.bias"))
            v = nn.Linear(TE_DIM, TE_DIM)
            v.weight.data.copy_(loader.get(f"{pre}v.weight"))
            v.bias.data.copy_(loader.get(f"{pre}v.bias"))
            nk = WanRMSNorm(TE_DIM, eps=EPS)
            nk.weight.data.copy_(loader.get(f"{pre}norm_k.weight"))
            self.add_module(f"k_{b}", k)
            self.add_module(f"v_{b}", v)
            self.add_module(f"nk_{b}", nk)

        # Full attn-output path uses block 0.
        b0 = self.blocks[0]
        pre = f"blocks.{b0}.cross_attn."
        q = nn.Linear(TE_DIM, TE_DIM)
        q.weight.data.copy_(loader.get(f"{pre}q.weight"))
        q.bias.data.copy_(loader.get(f"{pre}q.bias"))
        o = nn.Linear(TE_DIM, TE_DIM)
        o.weight.data.copy_(loader.get(f"{pre}o.weight"))
        o.bias.data.copy_(loader.get(f"{pre}o.bias"))
        nq = WanRMSNorm(TE_DIM, eps=EPS)
        nq.weight.data.copy_(loader.get(f"{pre}norm_q.weight"))
        self.q_0, self.o_0, self.nq_0 = q, o, nq

        g = torch.Generator().manual_seed(seed)
        probes = torch.randn(num_probes, TE_DIM, generator=g) * 0.02
        self.register_buffer("probes", probes)

        self.eval()
        self.requires_grad_(False)

    def kv_pool(self, ctx1536: torch.Tensor, block: int):
        """ctx1536 [L,1536] -> (K_pool [1536], V_pool [1536]).

        K = norm_k(k(ctx)) then mean over tokens; V = v(ctx) then mean.
        Matches WanCrossAttention: k is RMSNormed, v is not.
        """
        k = getattr(self, f"k_{block}")
        v = getattr(self, f"v_{block}")
        nk = getattr(self, f"nk_{block}")
        K = nk(k(ctx1536)).mean(dim=0)
        V = v(ctx1536).mean(dim=0)
        return K, V

    def attn_output(self, ctx1536: torch.Tensor) -> torch.Tensor:
        """ctx1536 [L,1536] -> [P,1536] using block-0 real cross-attn."""
        b0 = self.blocks[0]
        q = self.nq_0(self.q_0(self.probes))          # [P,1536]
        k = getattr(self, f"nk_{b0}")(getattr(self, f"k_{b0}")(ctx1536))  # [L,1536]
        v = getattr(self, f"v_{b0}")(ctx1536)          # [L,1536]
        P, D = q.shape
        L = k.shape[0]
        q = q.view(1, P, NUM_HEADS, HEAD_DIM).transpose(1, 2)   # [1,H,P,hd]
        k = k.view(1, L, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(1, L, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)           # [1,H,P,hd]
        out = out.transpose(1, 2).contiguous().view(1, P, D)
        out = self.o_0(out)[0]                                  # [P,1536]
        return out


# ---------------------------------------------------------------------------
# Loss bundle for one (teacher1536, student1536) pair
# ---------------------------------------------------------------------------

def compute_e12_losses(teacher1536: torch.Tensor, student1536: torch.Tensor,
                       distiller: RealCrossAttnDistiller,
                       kv_weight: float) -> dict:
    """Return raw loss terms (all differentiable w.r.t. student1536).

    teacher1536: [L,1536] (detached). student1536: [64,1536] (requires grad).
    """
    teacher1536 = teacher1536.detach()

    # K/V pool loss over all probe blocks.
    kv_loss = 0.0
    for b in distiller.blocks:
        tK, tV = distiller.kv_pool(teacher1536, b)
        sK, sV = distiller.kv_pool(student1536, b)
        kv_loss = kv_loss + F.mse_loss(sK, tK) + F.mse_loss(sV, tV)
    kv_loss = kv_loss / len(distiller.blocks)

    # Attn-output loss (block 0 real cross-attn).
    t_out = distiller.attn_output(teacher1536)
    s_out = distiller.attn_output(student1536)
    attn_loss = F.mse_loss(s_out, t_out)
    attn_cos = F.cosine_similarity(s_out.flatten().unsqueeze(0),
                                   t_out.flatten().unsqueeze(0)).mean()

    # Aux: mean-pooled 1536 cosine + MSE.
    t_pool = teacher1536.mean(dim=0)
    s_pool = student1536.mean(dim=0)
    pool_cos = F.cosine_similarity(s_pool.unsqueeze(0), t_pool.unsqueeze(0)).mean()
    pool_mse = F.mse_loss(s_pool, t_pool)

    total = attn_loss + kv_weight * kv_loss + 0.5 * pool_mse
    return {
        "total": total,
        "attn_mse": attn_loss.detach(),
        "kv_mse": kv_loss.detach(),
        "pool_mse": pool_mse.detach(),
        "pool_cos": pool_cos.detach(),
        "attn_cos": attn_cos.detach(),
    }


# ---------------------------------------------------------------------------
# Data: cached MiniCPM5 hidden + teacher context
# ---------------------------------------------------------------------------

def load_prompts(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="eval/e1/prompts.jsonl")
    ap.add_argument("--teacher-dir", default="eval/e1/teacher_contexts")
    ap.add_argument("--cache-dir", default="eval/e1.2/cache")
    ap.add_argument("--out-dir", default="eval/e1.2")
    ap.add_argument("--text-embedding-shard", default=TE_SHARD)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--kv-weight", type=float, default=2.0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("e1.2_train")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_prompts(REPO_ROOT / args.prompts)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    log.info("Prompts: total=%d train=%d val=%d", len(rows), len(train_rows), len(val_rows))

    # teacher filename map
    manifest = {json.loads(l)["id"]: json.loads(l)["filename"]
                for l in open(REPO_ROOT / args.teacher_dir / "manifest.jsonl") if l.strip()}

    # frozen text_embedding + real cross-attn distiller
    te = load_text_embedding(args.text_embedding_shard).to(device)
    distiller = RealCrossAttnDistiller(args.ckpt_dir).to(device)
    assert all(not p.requires_grad for p in te.parameters())
    assert all(not p.requires_grad for p in distiller.parameters())

    # trainable adapter (same arch as E1; output_norm kept for checkpoint compat)
    adapter = TextAdapter(hidden_dim=2048, output_dim=4096, num_queries=64,
                         num_resampler_layers=2, num_heads=8, ffn_mult=4)
    adapter = adapter.float().to(device)
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    log.info("Adapter trainable params: %d", trainable)

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    cache_dir = REPO_ROOT / args.cache_dir

    def load_example(r):
        """Return (hidden[Lm,2048], mask[Lm], teacher[L,4096]) on device."""
        c = load_file(str(cache_dir / f"{r['id']}.safetensors"))
        hidden = c["hidden"].to(device)          # [Lm,2048] fp32
        mask = c["mask"].to(device).bool()       # [Lm]
        t = load_file(str(REPO_ROOT / args.teacher_dir / manifest[r["id"]]))["context"]
        teacher = t.float().to(device)          # [L,4096]
        return hidden, mask, teacher

    def run_epoch(rows, train: bool):
        adapter.train() if train else adapter.eval()
        agg = []
        g = torch.Generator().manual_seed(args.seed + (int(time.time()) if train else 0))
        order = torch.randperm(len(rows), generator=g).tolist() if train else list(range(len(rows)))
        for i in order:
            r = rows[i]
            hidden, mask, teacher4096 = load_example(r)
            with torch.no_grad():
                t1536 = te(teacher4096)                       # [L,1536]
            h = hidden.unsqueeze(0)                            # [1,Lm,2048]
            m = mask.unsqueeze(0)                              # [1,Lm]
            s4096 = adapter(h, m)[0]                          # [64,4096]
            s1536 = te(s4096)                                 # [64,1536]
            losses = compute_e12_losses(t1536, s1536, distiller, args.kv_weight)
            if train:
                opt.zero_grad(set_to_none=True)
                losses["total"].backward()
                opt.step()
            agg.append({k: float(v) for k, v in losses.items() if k != "total"})
            agg[-1]["total"] = float(losses["total"].item())
            del hidden, mask, teacher4096, t1536, h, m, s4096, s1536
            if device.type == "mps":
                torch.mps.empty_cache()
        keys = agg[0].keys()
        return {k: sum(a[k] for a in agg) / len(agg) for k in keys}

    log.info("Training for %d epochs ...", args.epochs)
    log_path = out_dir / "training_log.jsonl"
    log_fp = open(log_path, "w")
    best_kv = float("inf")
    best_path = out_dir / "adapter_best.safetensors"
    wall0 = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(train_rows, train=True)
        with torch.no_grad():
            va = run_epoch(val_rows, train=False)
        # best by held-out kv loss (lower better)
        is_best = va["kv_mse"] < best_kv
        if is_best:
            best_kv = va["kv_mse"]
            save_file(adapter.state_dict(), str(best_path))
        rec = {
            "epoch": epoch, "epoch_wall_s": round(time.time() - t0, 2),
            "train": {k: round(v, 6) for k, v in tr.items()},
            "val": {k: round(v, 6) for k, v in va.items()},
            "is_best": is_best,
        }
        log_fp.write(json.dumps(rec) + "\n"); log_fp.flush()
        log.info("epoch %d/%d  train[tot=%.4f attn=%.4f kv=%.4f pool_cos=%.4f]  "
                 "val[tot=%.4f attn=%.4f kv=%.4f pool_cos=%.4f]%s",
                 epoch, args.epochs, tr["total"], tr["attn_mse"], tr["kv_mse"],
                 tr["pool_cos"], va["total"], va["attn_mse"], va["kv_mse"],
                 va["pool_cos"], "  *BEST*" if is_best else "")

    # final save + config
    final_path = out_dir / "adapter_final" / "adapter.safetensors"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(adapter.state_dict(), str(final_path))
    cfg = {
        "hidden_dim": 2048, "output_dim": 4096, "num_queries": 64,
        "num_resampler_layers": 2, "num_heads": 8, "ffn_mult": 4, "dtype": "float32",
        "training_config": {
            "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
            "optimizer": "AdamW", "device": args.device, "kv_weight": args.kv_weight,
            "probe_blocks": PROBE_BLOCKS, "num_probes": NUM_PROBES, "seed": args.seed,
            "trainable_params": trainable,
        },
        "best_val_kv_mse": round(best_kv, 6),
    }
    with open(out_dir / "adapter_final" / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    log_fp.close()
    log.info("Done in %.1fs. best adapter: %s (val kv_mse=%.5f)",
             time.time() - wall0, best_path, best_kv)
    print(json.dumps({"status": "PASS", "best_val_kv_mse": round(best_kv, 6),
                      "best_path": str(best_path)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
