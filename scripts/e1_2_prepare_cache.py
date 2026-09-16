#!/usr/bin/env python3
"""Cache frozen MiniCPM5 hidden states for E1.2 Adapter experiments.

Teacher UMT5 contexts remain the source of target token length. This script
runs MiniCPM5 once for all prompts and stores bf16 hidden states + masks so
subsequent Adapter ablations never reload/re-run MiniCPM5.
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
from safetensors import safe_open
from safetensors.torch import save_file

from wan.utils.prompt_embedding import prompt_sha256

DEFAULT_MINICPM5_DIR = "/Volumes/ssd/huggingface/hub/models--openbmb--MiniCPM5-2B/snapshots/12a3808a956f869c767195e9266b59c4d21d92e2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="eval/e1/prompts.jsonl")
    p.add_argument("--teacher-dir", default="eval/e1/teacher_contexts")
    p.add_argument("--output-dir", default="eval/e1.2/minicpm_cache")
    p.add_argument("--minicpm5-dir", default=DEFAULT_MINICPM5_DIR)
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    p.add_argument("--max-prompt-len", type=int, default=512)
    return p.parse_args()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def teacher_map(teacher_dir):
    result = {}
    manifest = os.path.join(teacher_dir, "manifest.jsonl")
    for row in load_jsonl(manifest):
        result[row["id"]] = os.path.join(teacher_dir, row["filename"])
    return result


def teacher_len(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        return int(f.get_tensor("context").shape[0])


def main():
    args = parse_args()
    prompts_path = os.path.join(REPO_ROOT, args.prompts)
    teacher_dir = os.path.join(REPO_ROOT, args.teacher_dir)
    output_dir = os.path.join(REPO_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rows = load_jsonl(prompts_path)
    tmap = teacher_map(teacher_dir)
    device = torch.device(args.device)

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.minicpm5_dir, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.minicpm5_dir, dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    model.eval().requires_grad_(False)

    manifest_rows = []
    for row in rows:
        sha = prompt_sha256(row["text"])
        out = os.path.join(output_dir, f"{sha}.safetensors")
        tgt_len = teacher_len(tmap[row["id"]])
        if tgt_len > 512:
            raise ValueError(f"teacher token length >512 for {row['id']}: {tgt_len}")

        enc = tok(
            [row["text"]], return_tensors="pt", padding=False, truncation=True,
            max_length=args.max_prompt_len,
        )
        token_count = int(enc["attention_mask"].sum().item())
        enc_dev = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc_dev).last_hidden_state[0, :token_count]
        mask = torch.ones(token_count, dtype=torch.uint8)
        save_file(
            {
                "hidden": hidden.detach().cpu().to(torch.bfloat16).contiguous(),
                "attention_mask": mask,
            },
            out,
            metadata={
                "id": row["id"],
                "prompt_sha256": sha,
                "target_length": str(tgt_len),
                "minicpm_token_count": str(token_count),
                "model_id": "openbmb/MiniCPM5-2B",
            },
        )
        manifest_rows.append({
            "id": row["id"], "source": row["source"], "split": row["split"],
            "prompt_sha256": sha, "filename": os.path.basename(out),
            "target_length": tgt_len, "minicpm_token_count": token_count,
        })
        del hidden, enc_dev
        if device.type == "mps":
            torch.mps.empty_cache()

    with open(os.path.join(output_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in manifest_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    print(json.dumps({"status": "PASS", "count": len(rows), "output_dir": output_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
