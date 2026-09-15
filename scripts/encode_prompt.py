#!/usr/bin/env python3
"""Encode a text prompt using UMT5 and save the embedding to a safetensors file.

Uses the M3.3-verified low-memory path:
  1. torch.load(mmap=True) — lazy memory-mapped checkpoint, no full RAM copy
  2. Construct T5Encoder on meta device — zero RAM model skeleton
  3. Materialize weights tensor-by-tensor with assign=True — peak RSS stays low
  4. Run forward pass on CPU
  5. Save embedding, process exits (OS reclaims all memory)

This script runs as an independent subprocess. The parent process must NOT
hold UMT5 concurrently. After this process exits, all mmap pages and model
weights are released by the OS.
"""

import argparse
import gc
import json
import logging
import os
import resource
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.modules.t5 import T5Encoder  # noqa: E402
from wan.modules.tokenizers import HuggingfaceTokenizer  # noqa: E402
from wan.utils.prompt_embedding import save_prompt_embedding, prompt_sha256  # noqa: E402
from wan.configs.wan_i2v_1_3B import i2v_1_3B as config  # noqa: E402

UMT5_XXL_ENCODER_CFG = dict(
    vocab=256384, dim=4096, dim_attn=4096, dim_ffn=10240,
    num_heads=64, num_layers=24, num_buckets=32, shared_pos=False, dropout=0.1,
)


def get_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def parse_args():
    parser = argparse.ArgumentParser(description="Encode a text prompt with UMT5 (M3.3 low-memory path).")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--assets_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--text_len", type=int, default=None)
    parser.add_argument("--t5_checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    return parser.parse_args()


def resolve_t5_paths(args):
    t5_checkpoint = args.t5_checkpoint
    tokenizer_path = args.tokenizer_path
    if t5_checkpoint is None:
        candidate = os.path.join(args.assets_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        if os.path.isfile(candidate):
            t5_checkpoint = candidate
        else:
            raise FileNotFoundError(f"T5 checkpoint not found at {candidate}")
    if tokenizer_path is None:
        candidate = os.path.join(args.assets_dir, "google", "umt5-xxl")
        tokenizer_path = candidate if os.path.isdir(candidate) else args.assets_dir
    return t5_checkpoint, tokenizer_path


def load_umt5_low_memory(checkpoint_path: str, dtype: torch.dtype) -> T5Encoder:
    rss_start = get_rss_mb()
    logging.info("Loading UMT5 via M3.3 low-memory path (mmap + meta + tensor assign)...")
    logging.info("  RSS before load: %.1f MB", rss_start)

    load_start = time.time()
    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    logging.info("  torch.load(mmap=True) done in %.1fs, RSS: %.1f MB (delta %+.1f)",
                 time.time() - load_start, get_rss_mb(), get_rss_mb() - rss_start)

    with torch.device("meta"):
        model = T5Encoder(**UMT5_XXL_ENCODER_CFG)
    logging.info("  Meta T5Encoder constructed (params=%d), RSS: %.1f MB",
                 sum(p.numel() for p in model.parameters()), get_rss_mb())

    matching_keys = set(model.state_dict().keys()) & set(state_dict.keys())
    missing = set(model.state_dict().keys()) - set(state_dict.keys())
    if missing:
        logging.warning("  %d missing keys (first 5): %s", len(missing), sorted(missing)[:5])

    mat_start = time.time()
    peak_rss = get_rss_mb()
    for i, key in enumerate(sorted(matching_keys)):
        source_tensor = state_dict[key]
        if dtype is not None and source_tensor.dtype != dtype:
            source_tensor = source_tensor.to(dtype)
        model.load_state_dict({key: source_tensor}, strict=False, assign=True)
        del source_tensor
        current_rss = get_rss_mb()
        if current_rss > peak_rss:
            peak_rss = current_rss
        if (i + 1) % 50 == 0 or i == len(matching_keys) - 1:
            logging.info("  Materialized %d/%d tensors, RSS: %.1f MB",
                         i + 1, len(matching_keys), current_rss)

    logging.info("  Weight materialization done in %.1fs, peak RSS: %.1f MB",
                 time.time() - mat_start, peak_rss)

    meta_params = sum(1 for p in model.parameters() if p.is_meta)
    if meta_params > 0:
        raise RuntimeError(f"{meta_params} meta parameters remain after materialization!")

    del state_dict
    gc.collect()

    model.eval()
    model.requires_grad_(False)
    logging.info("  UMT5 low-memory load complete. Final RSS: %.1f MB", get_rss_mb())
    return model


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    rss_start = get_rss_mb()
    wall_start = time.time()

    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    text_len = args.text_len or config.text_len

    logging.info(f"Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    logging.info(f"Prompt sha256: {prompt_sha256(args.prompt)}")
    logging.info(f"Prompt chars: {len(args.prompt)}")
    logging.info(f"Device: {device}, dtype: {dtype}, text_len: {text_len}")
    logging.info(f"RSS at start: {rss_start:.1f} MB")

    t5_checkpoint, tokenizer_path = resolve_t5_paths(args)
    logging.info(f"T5 checkpoint: {t5_checkpoint}")
    logging.info(f"Tokenizer path: {tokenizer_path}")

    model = load_umt5_low_memory(t5_checkpoint, dtype)
    model = model.to(device)

    logging.info("Loading tokenizer...")
    tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=text_len, clean='whitespace')

    ids, mask = tokenizer([args.prompt], return_mask=True, add_special_tokens=True)
    token_count = int(mask.gt(0).sum().item())
    logging.info(f"Token count: {token_count}, ids shape: {ids.shape}")
    truncated = token_count >= text_len
    if truncated:
        logging.warning(f"Prompt truncated! token_count={token_count} >= text_len={text_len}")

    ids = ids.to(device)
    mask = mask.to(device)

    logging.info("Running UMT5 forward pass...")
    forward_start = time.time()
    with torch.no_grad():
        context = model(ids, mask)
    forward_time = time.time() - forward_start

    seq_lens = mask.gt(0).sum(dim=1).long()
    context_trimmed = context[0][:seq_lens[0]]

    logging.info(f"Forward done in {forward_time:.1f}s")
    logging.info(f"Context shape: {list(context_trimmed.shape)}, dtype: {context_trimmed.dtype}")
    logging.info(f"All finite: {torch.isfinite(context_trimmed).all().item()}")

    output_path = args.output
    extra_metadata = {}
    if args.variant:
        extra_metadata["variant"] = args.variant
    if args.scene:
        extra_metadata["scene"] = args.scene
    extra_metadata["token_count"] = str(token_count)
    extra_metadata["truncated"] = str(truncated)
    extra_metadata["encode_wall_time_seconds"] = f"{time.time() - wall_start:.1f}"
    extra_metadata["rss_start_mb"] = f"{rss_start:.1f}"
    extra_metadata["rss_peak_mb"] = f"{get_rss_mb():.1f}"

    save_prompt_embedding(output_path=output_path, context=context_trimmed,
                           prompt=args.prompt, text_len=text_len, dtype=dtype)

    json_path = output_path.replace(".safetensors", ".json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            meta = json.load(f)
        meta.update(extra_metadata)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

    rss_end = get_rss_mb()
    wall_total = time.time() - wall_start
    logging.info(f"Done! Encoding wall time: {wall_total:.1f}s")
    logging.info(f"RSS: start={rss_start:.1f}MB, end={rss_end:.1f}MB")
    logging.info(f"Prompt embedding saved to {output_path}")
    logging.info(f"Token count: {token_count}, truncated: {truncated}")

    sys.exit(0)


if __name__ == "__main__":
    main()
