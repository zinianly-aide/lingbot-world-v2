#!/usr/bin/env python3
"""E1.2 Step A: cache MiniCPM5 frozen hidden states for all E1 prompts.

Runs MiniCPM5-2B (bf16, frozen) once over all 56 prompts and saves:
    eval/e1.2/cache/<id>.safetensors  = {hidden: [L,2048] fp32, mask: [L] uint8}

This lets the E1.2 training loop avoid loading/running MiniCPM5 every epoch
(MiniCPM5 is ~4GB; caching hidden states makes training fast and light).

Reuses eval/e1/prompts.jsonl. Does NOT touch UMT5 / DiT / text_embedding.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

DEFAULT_MINICPM5_DIR = (
    "/Volumes/ssd/huggingface/hub/models--openbmb--MiniCPM5-2B/snapshots/"
    "12a3808a956f869c767195e9266b59c4d21d92e2"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="eval/e1/prompts.jsonl")
    ap.add_argument("--minicpm5-dir", default=DEFAULT_MINICPM5_DIR)
    ap.add_argument("--out-dir", default="eval/e1.2/cache")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-prompt-len", type=int, default=512)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [json.loads(l) for l in open(REPO_ROOT / args.prompts) if l.strip()]
    print(f"Encoding {len(prompts)} prompts with frozen MiniCPM5 ...")

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.minicpm5_dir, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(args.minicpm5_dir, torch_dtype=torch.bfloat16,
                                      local_files_only=True).to(device)
    model.eval()
    model.requires_grad_(False)

    manifest = []
    for i, pr in enumerate(prompts):
        pid = pr["id"]
        out_path = out_dir / f"{pid}.safetensors"
        if out_path.exists():
            print(f"  [{i+1}/{len(prompts)}] {pid} cached, skip")
            manifest.append({"id": pid, "cached": True})
            continue
        enc = tok([pr["text"]], return_tensors="pt", truncation=True,
                  max_length=args.max_prompt_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state[0]  # [L,2048] bf16
        mask = enc["attention_mask"][0].to(torch.uint8).cpu()  # [L]
        save_file({"hidden": hidden.float().cpu().contiguous(),
                   "mask": mask}, str(out_path))
        manifest.append({"id": pid, "L": int(hidden.shape[0])})
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(prompts)}] {pid} L={int(hidden.shape[0])}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. {len(prompts)} cached under {out_dir}")
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
