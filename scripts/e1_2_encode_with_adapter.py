#!/usr/bin/env python3
"""Encode one prompt with MiniCPM5 + E1.2 adapter (v1 trained or v2 skeleton).

- arch=v1 (default): loads the VALIDATED E1.2 training product
  (wan.adapters.text_adapter.TextAdapter, fixed 64 queries,
  eval/e1.2/adapter_best.safetensors) and emits [64,4096] bf16, compatible with
  load_prompt_embedding.
- arch=v2: VariableLengthTextAdapter (variable teacher-length geometry, no
  output norm); expects eval/e1.2/best_adapter_v2.safetensors (v2 training not
  yet performed in this repo).

UMT5 encoder is NOT loaded in either path.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
from safetensors.torch import load_file

from wan.utils.prompt_embedding import save_prompt_embedding, prompt_sha256

DEFAULT_MINICPM5_DIR = "/Volumes/ssd/huggingface/hub/models--openbmb--MiniCPM5-2B/snapshots/12a3808a956f869c767195e9266b59c4d21d92e2"
DEFAULT_UMT5_TOKENIZER = "/Volumes/ssd/lingbot-assets/google/umt5-xxl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--arch", default="v1", choices=["v1", "v2"])
    p.add_argument("--adapter-weights", default=None)  # defaults per arch
    p.add_argument("--adapter-config", default=None)   # defaults per arch
    p.add_argument("--minicpm5-dir", default=DEFAULT_MINICPM5_DIR)
    p.add_argument("--umt5-tokenizer", default=DEFAULT_UMT5_TOKENIZER)
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    arch = args.arch

    if arch == "v1":
        weights = args.adapter_weights or "eval/e1.2/adapter_best.safetensors"
        config = args.adapter_config or "eval/e1.2/adapter_final/config.json"
    else:
        weights = args.adapter_weights or "eval/e1.2/best_adapter_v2.safetensors"
        config = args.adapter_config or "eval/e1.2/adapter_v2_config.json"

    with open(os.path.join(REPO_ROOT, config), encoding="utf-8") as f:
        cfg = json.load(f)

    from transformers import AutoModel, AutoTokenizer
    mcp_tok = AutoTokenizer.from_pretrained(args.minicpm5_dir, local_files_only=True, use_fast=False)
    minicpm = AutoModel.from_pretrained(args.minicpm5_dir, dtype=torch.bfloat16, local_files_only=True).to(device)
    minicpm.eval().requires_grad_(False)

    enc = mcp_tok([args.prompt], return_tensors="pt", padding=False, truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        hidden = minicpm(**enc).last_hidden_state
        if arch == "v1":
            from wan.adapters.text_adapter import TextAdapter
            adapter = TextAdapter(
                hidden_dim=cfg.get("hidden_dim", 2048),
                output_dim=cfg.get("output_dim", 4096),
                num_queries=cfg.get("num_queries", 64),
                num_resampler_layers=cfg.get("num_resampler_layers", 2),
                num_heads=cfg.get("num_heads", 8),
                ffn_mult=cfg.get("ffn_mult", 4),
            ).float().to(device)
            adapter.load_state_dict(load_file(os.path.join(REPO_ROOT, weights)), strict=True)
            adapter.eval().requires_grad_(False)
            ctx = adapter(hidden, enc["attention_mask"])
            ctx = ctx[0].to(torch.bfloat16).contiguous()
            if ctx.shape != (cfg.get("num_queries", 64), cfg.get("output_dim", 4096)):
                raise RuntimeError(f"unexpected v1 adapter output shape: {tuple(ctx.shape)}")
            contract = "fixed 64 queries; trained with real Wan K/V distillation (validated)"
        else:
            from wan.adapters.text_adapter_v2 import VariableLengthTextAdapter
            from wan.modules.tokenizers import HuggingfaceTokenizer
            t5_tok = HuggingfaceTokenizer(args.umt5_tokenizer, seq_len=512, clean="whitespace")
            _, t5_mask = t5_tok([args.prompt], return_mask=True, add_special_tokens=True)
            target_len = int(t5_mask[0].gt(0).sum().item())
            adapter = VariableLengthTextAdapter(
                hidden_dim=cfg.get("hidden_dim", 2048),
                bottleneck_dim=cfg["bottleneck_dim"],
                output_dim=cfg.get("output_dim", 4096),
                max_queries=cfg.get("max_queries", 512),
                num_resampler_layers=cfg["num_resampler_layers"],
                num_heads=cfg["num_heads"],
            ).float().to(device)
            adapter.load_state_dict(load_file(os.path.join(REPO_ROOT, weights)), strict=True)
            adapter.eval().requires_grad_(False)
            ctx, out_mask = adapter(
                hidden,
                enc["attention_mask"],
                torch.tensor([target_len], device=device),
            )
            ctx = ctx[0, out_mask[0]].to(torch.bfloat16).contiguous()
            if ctx.shape != (target_len, 4096):
                raise RuntimeError(f"unexpected v2 adapter output shape: {tuple(ctx.shape)}")
            contract = "variable-length; no output norm; exact Wan 512-pad geometry"

    output = os.path.abspath(args.output)
    save_prompt_embedding(output, ctx.cpu(), args.prompt, text_len=512, dtype=torch.bfloat16)
    sidecar = output.replace(".safetensors", ".json")
    with open(sidecar, encoding="utf-8") as f:
        meta = json.load(f)
    meta.update({
        "model_id": f"minicpm5-2b+adapter-{arch}",
        "encoder": f"minicpm5-2b+adapter-{arch}",
        "adapter_arch": arch,
        "adapter_weights": weights,
        "prompt_sha256": prompt_sha256(args.prompt),
        "adapter_contract": contract,
    })
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    out_shape = list(ctx.shape)
    out_dtype = str(ctx.dtype).replace("torch.", "")

    del adapter, minicpm, hidden, ctx, enc
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    print(json.dumps({"status": "PASS", "output": output,
                      "shape": out_shape, "dtype": out_dtype, "arch": arch}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
