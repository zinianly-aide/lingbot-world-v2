#!/usr/bin/env python3
"""Train E1.2 VariableLengthTextAdapter against Wan's real conditioning path.

No MiniCPM or UMT5 model is loaded here. Inputs are:
- cached MiniCPM5 hidden states from scripts/e1_2_prepare_cache.py
- cached UMT5 teacher contexts from E1
- frozen pretrained Wan text_embedding and selected cross-attention K/V weights

The student is forced to use the teacher token length, then both teacher and
student are padded with raw zeros to Wan's exact text_len=512 before the frozen
text_embedding. Best checkpoint is selected by held-out real Wan K/V loss.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import OrderedDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from wan.adapters.text_adapter_v2 import VariableLengthTextAdapter
from wan.modules.model import WanRMSNorm

CONTEXT_DIM = 4096
WAN_DIM = 1536
TEXT_LEN = 512
DEFAULT_DIT_DIR = "/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/snapshots/7e36a5f919f86cb4255cc9bfc30adb44963fbde1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="eval/e1/prompts.jsonl")
    p.add_argument("--teacher-dir", default="eval/e1/teacher_contexts")
    p.add_argument("--student-cache", default="eval/e1.2/minicpm_cache")
    p.add_argument("--dit-dir", default=DEFAULT_DIT_DIR)
    p.add_argument("--output-dir", default="eval/e1.2")
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--blocks", default="0,7,15,22,29")
    p.add_argument("--bottleneck-dim", type=int, default=768)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw-weight", type=float, default=0.5)
    p.add_argument("--projected-weight", type=float, default=1.0)
    p.add_argument("--kv-weight", type=float, default=2.0)
    p.add_argument("--moment-weight", type=float, default=0.25)
    return p.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def build_manifest_map(directory):
    return {
        r["id"]: r
        for r in read_jsonl(os.path.join(directory, "manifest.jsonl"))
    }


def _index_for_dir(dit_dir):
    for name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        p = os.path.join(dit_dir, name)
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)["weight_map"]
    return None


def load_checkpoint_tensors(dit_dir: str, keys: list[str]) -> dict[str, torch.Tensor]:
    """Load only requested tensors from sharded/single safetensors."""
    index = _index_for_dir(dit_dir)
    out = {}
    if index is not None:
        by_shard = {}
        for key in keys:
            if key not in index:
                raise KeyError(f"checkpoint key not found: {key}")
            by_shard.setdefault(index[key], []).append(key)
        for shard, shard_keys in by_shard.items():
            with safe_open(os.path.join(dit_dir, shard), framework="pt", device="cpu") as f:
                for key in shard_keys:
                    out[key] = f.get_tensor(key).float()
        return out

    singles = [
        os.path.join(dit_dir, "model.safetensors"),
        os.path.join(dit_dir, "diffusion_pytorch_model.safetensors"),
    ]
    single = next((p for p in singles if os.path.isfile(p)), None)
    if single is None:
        raise FileNotFoundError(f"no safetensors checkpoint/index under {dit_dir}")
    with safe_open(single, framework="pt", device="cpu") as f:
        available = set(f.keys())
        for key in keys:
            if key not in available:
                raise KeyError(f"checkpoint key not found: {key}")
            out[key] = f.get_tensor(key).float()
    return out


def build_text_embedding(dit_dir, device):
    keys = [
        "text_embedding.0.weight", "text_embedding.0.bias",
        "text_embedding.2.weight", "text_embedding.2.bias",
    ]
    sd = load_checkpoint_tensors(dit_dir, keys)
    te = nn.Sequential(OrderedDict([
        ("0", nn.Linear(CONTEXT_DIM, WAN_DIM)),
        ("1", nn.GELU(approximate="tanh")),
        ("2", nn.Linear(WAN_DIM, WAN_DIM)),
    ]))
    for idx in ("0", "2"):
        te[int(idx)].weight.data.copy_(sd[f"text_embedding.{idx}.weight"])
        te[int(idx)].bias.data.copy_(sd[f"text_embedding.{idx}.bias"])
    return te.to(device).eval().requires_grad_(False)


class FrozenKVProjector(nn.Module):
    def __init__(self, k_w, k_b, v_w, v_b, norm_w, device):
        super().__init__()
        self.k = nn.Linear(WAN_DIM, WAN_DIM)
        self.v = nn.Linear(WAN_DIM, WAN_DIM)
        self.norm_k = WanRMSNorm(WAN_DIM, eps=1e-6)
        self.k.weight.data.copy_(k_w); self.k.bias.data.copy_(k_b)
        self.v.weight.data.copy_(v_w); self.v.bias.data.copy_(v_b)
        self.norm_k.weight.data.copy_(norm_w)
        self.to(device).eval().requires_grad_(False)

    def forward(self, context):
        return self.norm_k(self.k(context)), self.v(context)


def build_kv_projectors(dit_dir, blocks, device):
    keys = []
    for b in blocks:
        p = f"blocks.{b}.cross_attn"
        keys += [f"{p}.k.weight", f"{p}.k.bias", f"{p}.v.weight", f"{p}.v.bias", f"{p}.norm_k.weight"]
    sd = load_checkpoint_tensors(dit_dir, keys)
    out = nn.ModuleDict()
    for b in blocks:
        p = f"blocks.{b}.cross_attn"
        out[str(b)] = FrozenKVProjector(
            sd[f"{p}.k.weight"], sd[f"{p}.k.bias"],
            sd[f"{p}.v.weight"], sd[f"{p}.v.bias"],
            sd[f"{p}.norm_k.weight"], device,
        )
    return out


def masked_cosine_loss(student, teacher, mask):
    valid = mask.bool()
    s = student[valid]
    t = teacher[valid]
    return (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()


def masked_mse(student, teacher, mask):
    valid = mask.bool().unsqueeze(-1).expand_as(student)
    return F.mse_loss(student[valid], teacher[valid])


def moment_loss(student, teacher, mask):
    valid = mask.bool()
    s = student[valid]
    t = teacher[valid]
    return F.mse_loss(s.mean(0), t.mean(0)) + F.mse_loss(s.std(0), t.std(0))


def pad_raw(ctx, mask, text_len=TEXT_LEN):
    b, n, d = ctx.shape
    if n > text_len:
        raise ValueError(f"context length {n} > text_len {text_len}")
    out = ctx.new_zeros(b, text_len, d)
    out[:, :n] = ctx
    out_mask = torch.zeros(b, text_len, dtype=torch.bool, device=ctx.device)
    out_mask[:, :n] = mask.bool()
    return out, out_mask


def load_batch(recs, teacher_dir, teacher_manifest, cache_dir, cache_manifest, device):
    student_hidden, student_masks, teacher_ctx, target_lengths = [], [], [], []
    max_m = max_t = 0
    loaded = []
    for r in recs:
        cm = cache_manifest[r["id"]]
        tm = teacher_manifest[r["id"]]
        c = load_file(os.path.join(cache_dir, cm["filename"]))
        t = load_file(os.path.join(teacher_dir, tm["filename"]))["context"].float()
        h = c["hidden"].float()
        m = c["attention_mask"].bool()
        if int(t.shape[0]) != int(cm["target_length"]):
            raise ValueError(f"target length mismatch for {r['id']}")
        loaded.append((h, m, t))
        max_m = max(max_m, h.shape[0]); max_t = max(max_t, t.shape[0])
        target_lengths.append(t.shape[0])

    b = len(recs)
    sh = torch.zeros(b, max_m, loaded[0][0].shape[-1])
    sm = torch.zeros(b, max_m, dtype=torch.bool)
    tc = torch.zeros(b, max_t, CONTEXT_DIM)
    tm = torch.zeros(b, max_t, dtype=torch.bool)
    for i, (h, m, t) in enumerate(loaded):
        sh[i, :h.shape[0]] = h; sm[i, :m.shape[0]] = m
        tc[i, :t.shape[0]] = t; tm[i, :t.shape[0]] = True
    return sh.to(device), sm.to(device), tc.to(device), tm.to(device), torch.tensor(target_lengths, device=device)


def compute_losses(adapter, sh, sm, teacher_raw, teacher_mask, target_lengths, te, kvs, args):
    student_raw, student_mask = adapter(sh, sm, target_lengths)
    if not torch.equal(student_mask, teacher_mask):
        raise RuntimeError("student/teacher token geometry mismatch")

    l_raw = masked_cosine_loss(student_raw, teacher_raw, teacher_mask) + 0.10 * masked_mse(student_raw, teacher_raw, teacher_mask)
    l_moment = moment_loss(student_raw, teacher_raw, teacher_mask)

    s_pad, pad_mask = pad_raw(student_raw, student_mask)
    t_pad, _ = pad_raw(teacher_raw, teacher_mask)
    s_proj = te(s_pad)
    with torch.no_grad():
        t_proj = te(t_pad)

    l_proj = masked_cosine_loss(s_proj, t_proj, pad_mask) + 0.10 * masked_mse(s_proj, t_proj, pad_mask)

    kv_losses = []
    kv_cos = []
    for projector in kvs.values():
        sk, sv = projector(s_proj)
        with torch.no_grad():
            tk, tv = projector(t_proj)
        block_loss = (
            masked_cosine_loss(sk, tk, pad_mask) +
            masked_cosine_loss(sv, tv, pad_mask) +
            0.10 * masked_mse(sk, tk, pad_mask) +
            0.10 * masked_mse(sv, tv, pad_mask)
        )
        kv_losses.append(block_loss)
        with torch.no_grad():
            kv_cos.append(0.5 * ((1 - masked_cosine_loss(sk, tk, pad_mask)) + (1 - masked_cosine_loss(sv, tv, pad_mask))))
    l_kv = torch.stack(kv_losses).mean()
    kv_cosine = torch.stack(kv_cos).mean()

    total = args.raw_weight*l_raw + args.projected_weight*l_proj + args.kv_weight*l_kv + args.moment_weight*l_moment
    return total, {
        "total": float(total.detach()), "raw": float(l_raw.detach()),
        "projected": float(l_proj.detach()), "kv": float(l_kv.detach()),
        "kv_cosine": float(kv_cosine.detach()), "moment": float(l_moment.detach()),
    }


def mean_metrics(rows):
    return {k: sum(r[k] for r in rows)/len(rows) for k in rows[0]}


def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    blocks = [int(x) for x in args.blocks.split(",") if x.strip()]

    prompts = read_jsonl(os.path.join(REPO_ROOT, args.prompts))
    train_rows = [r for r in prompts if r["split"] == "train"]
    val_rows = [r for r in prompts if r["split"] == "val"]
    teacher_dir = os.path.join(REPO_ROOT, args.teacher_dir)
    cache_dir = os.path.join(REPO_ROOT, args.student_cache)
    output_dir = os.path.join(REPO_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    teacher_manifest = build_manifest_map(teacher_dir)
    cache_manifest = build_manifest_map(cache_dir)

    te = build_text_embedding(args.dit_dir, device)
    kvs = build_kv_projectors(args.dit_dir, blocks, device)
    adapter = VariableLengthTextAdapter(
        bottleneck_dim=args.bottleneck_dim,
        num_resampler_layers=args.layers,
        num_heads=args.heads,
    ).float().to(device)

    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_rows)/args.batch_size)
    total_steps = max(1, args.epochs*steps_per_epoch)
    warmup = max(1, int(total_steps*args.warmup_ratio))
    def lr_lambda(step):
        if step < warmup:
            return max(1e-3, step/max(1, warmup))
        progress = (step-warmup)/max(1, total_steps-warmup)
        return 0.5*(1.0 + math.cos(math.pi*progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_kv = float("inf")
    best_path = os.path.join(output_dir, "best_adapter_v2.safetensors")
    log_path = os.path.join(output_dir, "training_log_v2.jsonl")
    with open(log_path, "w", encoding="utf-8") as logf:
        for epoch in range(1, args.epochs+1):
            adapter.train()
            rng = random.Random(args.seed + epoch)
            order = train_rows[:]; rng.shuffle(order)
            train_m = []
            for i in range(0, len(order), args.batch_size):
                batch = order[i:i+args.batch_size]
                sh, sm, tc, tm, tl = load_batch(batch, teacher_dir, teacher_manifest, cache_dir, cache_manifest, device)
                loss, metrics = compute_losses(adapter, sh, sm, tc, tm, tl, te, kvs, args)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
                opt.step(); sched.step()
                train_m.append(metrics)

            adapter.eval(); val_m = []
            with torch.no_grad():
                for i in range(0, len(val_rows), args.batch_size):
                    batch = val_rows[i:i+args.batch_size]
                    sh, sm, tc, tm, tl = load_batch(batch, teacher_dir, teacher_manifest, cache_dir, cache_manifest, device)
                    _, metrics = compute_losses(adapter, sh, sm, tc, tm, tl, te, kvs, args)
                    val_m.append(metrics)
            tr = mean_metrics(train_m); va = mean_metrics(val_m)
            record = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train": tr, "val": va}
            logf.write(json.dumps(record) + "\n"); logf.flush()
            print(json.dumps(record))
            if va["kv"] < best_kv:
                best_kv = va["kv"]
                save_file(adapter.state_dict(), best_path)

    final_path = os.path.join(output_dir, "adapter_v2_final.safetensors")
    save_file(adapter.state_dict(), final_path)
    config = {
        "architecture": "VariableLengthTextAdapter",
        "hidden_dim": 2048, "bottleneck_dim": args.bottleneck_dim,
        "output_dim": 4096, "max_queries": 512,
        "num_resampler_layers": args.layers, "num_heads": args.heads,
        "trainable_params": trainable,
        "blocks": blocks, "best_val_real_kv_loss": best_kv,
        "selection_metric": "val.real_kv_loss",
        "loss_weights": {"raw": args.raw_weight, "projected": args.projected_weight, "kv": args.kv_weight, "moment": args.moment_weight},
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "warmup_ratio": args.warmup_ratio, "grad_clip": args.grad_clip, "seed": args.seed},
        "contract": "teacher token length -> raw zero pad to 512 -> pretrained text_embedding -> pretrained cross-attn K/V",
        "output_norm": False,
    }
    with open(os.path.join(output_dir, "adapter_v2_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(json.dumps({"status": "PASS", "best": best_path, "best_val_kv": best_kv, "trainable_params": trainable}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
