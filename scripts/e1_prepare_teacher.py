#!/usr/bin/env python3
"""E1 Teacher Context pre-generation.

Encodes every prompt in eval/e1/prompts.jsonl with the UMT5 low-memory path
(reused verbatim from scripts/encode_prompt.py) and caches the resulting
[L, 4096] bf16 context tensor to eval/e1/teacher_contexts/<prompt_sha256>.safetensors.

Design constraints:
  * UMT5 is loaded ONCE (mmap + meta + tensor assign) and never lives
    alongside MiniCPM5 in this process.  This script does NOT import or load
    MiniCPM5.
  * Batched encoding: prompts are tokenized as a list (padded to text_len=512
    by the Wan tokenizer) and forwarded in chunks, so UMT5 stays resident.
  * Resume: an existing .safetensors whose metadata prompt_sha256 matches the
    current prompt text is SKIPPED (no recompute).
  * Per-record manifest written to teacher_contexts/manifest.jsonl.

Output format (matches M3.3 contract):
    <sha256>.safetensors  key="context", [L, 4096] bf16
    metadata: prompt_id, prompt_text, prompt_sha256, model_id="umt5-xxl",
              hidden_dim=4096, format_version="1.0"

Usage:
    python scripts/e1_prepare_teacher.py \
        --prompts eval/e1/prompts.jsonl \
        --output-dir eval/e1/teacher_contexts \
        --device mps
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import resource
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from scripts.encode_prompt import load_umt5_low_memory, UMT5_XXL_ENCODER_CFG  # noqa: E402
from wan.modules.tokenizers import HuggingfaceTokenizer  # noqa: E402
from wan.configs.wan_i2v_1_3B import i2v_1_3B as config  # noqa: E402
from wan.utils.prompt_embedding import prompt_sha256  # noqa: E402

DEFAULT_UMT5_CHECKPOINT = "/Volumes/ssd/lingbot-assets/models_t5_umt5-xxl-enc-bf16.pth"
DEFAULT_UMT5_TOKENIZER = "/Volumes/ssd/lingbot-assets/google/umt5-xxl"
MODEL_ID = "umt5-xxl"
HIDDEN_DIM = 4096
FORMAT_VERSION = "1.0"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1: pre-generate UMT5 teacher contexts.")
    p.add_argument("--prompts", default="eval/e1/prompts.jsonl")
    p.add_argument("--output-dir", default="eval/e1/teacher_contexts")
    p.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    p.add_argument("--batch-size", type=int, default=8,
                   help="prompts per UMT5 forward chunk (keep small on MPS)")
    p.add_argument("--t5-checkpoint", default=DEFAULT_UMT5_CHECKPOINT)
    p.add_argument("--tokenizer-path", default=DEFAULT_UMT5_TOKENIZER)
    return p.parse_args()


def load_prompts(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def existing_sha_ok(path: str, expected_sha: str) -> bool:
    """True if a safetensors already exists with matching prompt_sha256."""
    if not os.path.isfile(path):
        return False
    from safetensors import safe_open
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            stored = f.metadata() or {}
        return stored.get("prompt_sha256") == expected_sha
    except Exception:
        return False


def save_context(path: str, context: torch.Tensor, record: dict) -> None:
    """Save [L, 4096] bf16 context + metadata (all values str)."""
    metadata = {
        "prompt_id": str(record["id"]),
        "prompt_text": record["text"],
        "prompt_sha256": str(record["prompt_sha256"]),
        "source": str(record["source"]),
        "split": str(record["split"]),
        "model_id": MODEL_ID,
        "hidden_dim": str(HIDDEN_DIM),
        "format_version": FORMAT_VERSION,
        "token_count": str(record["token_count"]),
        "output_seq_len": str(context.shape[0]),
    }
    tensors = {"context": context.detach().cpu().to(torch.bfloat16).contiguous()}
    save_file(tensors, path, metadata={k: str(v) for k, v in metadata.items()})


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("e1_prepare_teacher")

    prompts_path = os.path.join(REPO_ROOT, args.prompts)
    out_dir = os.path.join(REPO_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")

    rows = load_prompts(prompts_path)
    log.info("Loaded %d prompts from %s", len(rows), prompts_path)

    # Annotate each row with sha256 and decide which need encoding.
    todo: list[dict] = []
    skipped = 0
    for r in rows:
        sha = prompt_sha256(r["text"])
        r["prompt_sha256"] = sha
        r["path"] = os.path.join(out_dir, f"{sha}.safetensors")
        if existing_sha_ok(r["path"], sha):
            skipped += 1
        else:
            todo.append(r)
    log.info("Already cached: %d, to encode: %d", skipped, len(todo))

    manifest_rows: list[dict] = []

    if todo:
        device = torch.device(args.device)
        text_len = config.text_len

        log.info("Loading UMT5 low-memory (once)...")
        t0 = time.time()
        model = load_umt5_low_memory(args.t5_checkpoint, torch.bfloat16)
        model = model.to(device)
        log.info("UMT5 loaded in %.1fs, RSS %.1f MB", time.time() - t0, rss_mb())

        tokenizer = HuggingfaceTokenizer(
            name=args.tokenizer_path, seq_len=text_len, clean="whitespace")

        # Process in chunks.
        bs = args.batch_size
        for start in range(0, len(todo), bs):
            chunk = todo[start:start + bs]
            texts = [c["text"] for c in chunk]
            ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device)

            t_chunk = time.time()
            with torch.no_grad():
                context = model(ids, mask)  # [B, text_len, 4096] bf16
            chunk_time = time.time() - t_chunk

            seq_lens = mask.gt(0).sum(dim=1).long().cpu().tolist()
            # Per-row: trim to real seq len and save.
            for i, rec in enumerate(chunk):
                L = seq_lens[i]
                ctx_i = context[i][:L]  # [L, 4096] bf16
                rec["token_count"] = int(L)
                rec["encode_wall_s"] = round(chunk_time / len(chunk), 4)
                rec["output_shape"] = [int(L), HIDDEN_DIM]
                save_context(rec["path"], ctx_i, rec)
                log.info("  [%s] L=%d shape=%s saved",
                         rec["id"], L, rec["output_shape"])

            # Free GPU tensors between chunks.
            del context, ids, mask
            if device.type == "mps":
                torch.mps.empty_cache()

        # Unload UMT5 explicitly.
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        log.info("UMT5 unloaded. Final RSS %.1f MB", rss_mb())

    # Build manifest for ALL rows (including skipped), resolved from disk where needed.
    for r in rows:
        path = r["path"]
        from safetensors import safe_open
        token_count = r.get("token_count")
        shape = r.get("output_shape")
        with safe_open(path, framework="pt", device="cpu") as f:
            ctx = f.get_tensor("context")
            shape = list(ctx.shape)
            token_count = int(shape[0])
        manifest_rows.append({
            "id": r["id"],
            "source": r["source"],
            "split": r["split"],
            "prompt_sha256": r["prompt_sha256"],
            "filename": os.path.basename(path),
            "token_count": token_count,
            "output_shape": shape,
            "cached": (r not in todo),
        })

    with open(manifest_path, "w", encoding="utf-8") as fh:
        for m in manifest_rows:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    log.info("Wrote manifest: %s (%d records)", manifest_path, len(manifest_rows))
    log.info("DONE. total=%d encoded_this_run=%d skipped=%d",
             len(rows), len(todo), skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
