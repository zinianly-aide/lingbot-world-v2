#!/usr/bin/env python3
"""Encode one prompt with MiniCPM5 + E1.2 VariableLengthTextAdapter.

UMT5 encoder is NOT loaded. The lightweight UMT5 tokenizer is used only to
recover the teacher token length so the student emits identical token geometry.
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

from wan.adapters.text_adapter_v2 import VariableLengthTextAdapter
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.utils.prompt_embedding import save_prompt_embedding, prompt_sha256

DEFAULT_MINICPM5_DIR = "/Volumes/ssd/huggingface/hub/models--openbmb--MiniCPM5-2B/snapshots/12a3808a956f869c767195e9266b59c4d21d92e2"
DEFAULT_UMT5_TOKENIZER = "/Volumes/ssd/lingbot-assets/google/umt5-xxl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--adapter-weights", default="eval/e1.2/best_adapter_v2.safetensors")
    p.add_argument("--adapter-config", default="eval/e1.2/adapter_v2_config.json")
    p.add_argument("--minicpm5-dir", default=DEFAULT_MINICPM5_DIR)
    p.add_argument("--umt5-tokenizer", default=DEFAULT_UMT5_TOKENIZER)
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    with open(os.path.join(REPO_ROOT, args.adapter_config), encoding="utf-8") as f:
        cfg = json.load(f)

    # UMT5 tokenizer only: establishes the exact target token count used by the
    # teacher while avoiding the ~11GB UMT5 encoder entirely.
    t5_tok = HuggingfaceTokenizer(args.umt5_tokenizer, seq_len=512, clean="whitespace")
    _, t5_mask = t5_tok([args.prompt], return_mask=True, add_special_tokens=True)
    target_len = int(t5_mask[0].gt(0).sum().item())

    from transformers import AutoModel, AutoTokenizer
    mcp_tok = AutoTokenizer.from_pretrained(args.minicpm5_dir, local_files_only=True, use_fast=False)
    minicpm = AutoModel.from_pretrained(args.minicpm5_dir, dtype=torch.bfloat16, local_files_only=True).to(device)
    minicpm.eval().requires_grad_(False)

    adapter = VariableLengthTextAdapter(
        hidden_dim=cfg.get("hidden_dim", 2048),
        bottleneck_dim=cfg["bottleneck_dim"],
        output_dim=cfg.get("output_dim", 4096),
        max_queries=cfg.get("max_queries", 512),
        num_resampler_layers=cfg["num_resampler_layers"],
        num_heads=cfg["num_heads"],
    ).float().to(device)
    state = load_file(os.path.join(REPO_ROOT, args.adapter_weights))
    adapter.load_state_dict(state, strict=True)
    adapter.eval().requires_grad_(False)

    enc = mcp_tok([args.prompt], return_tensors="pt", padding=False, truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        hidden = minicpm(**enc).last_hidden_state
        ctx, out_mask = adapter(
            hidden,
            enc["attention_mask"],
            torch.tensor([target_len], device=device),
        )
    ctx = ctx[0, out_mask[0]].to(torch.bfloat16).contiguous()
    if ctx.shape != (target_len, 4096):
        raise RuntimeError(f"unexpected adapter output shape: {tuple(ctx.shape)}")

    output = os.path.abspath(args.output)
    save_prompt_embedding(output, ctx.cpu(), args.prompt, text_len=512, dtype=torch.bfloat16)
    sidecar = output.replace(".safetensors", ".json")
    with open(sidecar, encoding="utf-8") as f:
        meta = json.load(f)
    meta.update({
        "model_id": "minicpm5-2b+adapter-v2",
        "encoder": "minicpm5-2b+adapter-v2",
        "target_tokenizer": "umt5",
        "target_token_count": target_len,
        "prompt_sha256": prompt_sha256(args.prompt),
        "adapter_contract": "variable-length; no output norm; exact Wan 512-pad geometry",
    })
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    del adapter, minicpm, hidden, ctx, enc
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    print(json.dumps({"status": "PASS", "output": output, "shape": [target_len, 4096], "dtype": "bfloat16"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
