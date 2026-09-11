#!/usr/bin/env python3
"""Encode a text prompt using UMT5 and save the embedding to a safetensors file.

This script pre-computes the T5 text embedding so that generation can skip
the UMT5 encoder entirely (via --prompt_embeds_file in generate.py). This is
critical for low-memory environments (e.g. M4 16GB) where UMT5 + DiT + VAE
cannot coexist.

Usage:
    python scripts/encode_prompt.py \
        --prompt "A cat walking on the beach" \
        --assets_dir /path/to/14b-checkpoint \
        --output output/prompt_embeds.safetensors \
        --device cpu

The output file contains:
    - context: [seq_len, 4096] tensor (the truncated UMT5 encoder output)
    - metadata: prompt_sha256, dtype, shape, hidden_dim, text_len, model_id, format_version
"""

import argparse
import logging
import os
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.modules.t5 import T5EncoderModel
from wan.utils.prompt_embedding import save_prompt_embedding, prompt_sha256
from wan.configs.wan_i2v_1_3B import i2v_1_3B as config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode a text prompt with UMT5 and save embedding to safetensors."
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The text prompt to encode."
    )
    parser.add_argument(
        "--assets_dir", type=str, required=True,
        help="Directory containing T5 checkpoint (models_t5_umt5-xxl-enc-bf16.pth) "
             "and tokenizer (google/umt5-xxl)."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the .safetensors embedding file."
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device to run T5 encoder on (default: cpu). "
             "Use cpu for lowest memory; MPS/CUDA for speed."
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype for T5 model and output embedding (default: bfloat16)."
    )
    parser.add_argument(
        "--text_len", type=int, default=None,
        help="Max sequence length for T5 (default: from config, usually 512)."
    )
    parser.add_argument(
        "--t5_checkpoint", type=str, default=None,
        help="Explicit path to T5 .pth checkpoint. Overrides --assets_dir lookup."
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="Explicit path to tokenizer. Overrides --assets_dir lookup."
    )
    return parser.parse_args()


def resolve_t5_paths(args):
    """Resolve T5 checkpoint and tokenizer paths from assets_dir or explicit args."""
    t5_checkpoint = args.t5_checkpoint
    tokenizer_path = args.tokenizer_path

    if t5_checkpoint is None:
        candidate = os.path.join(args.assets_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        if os.path.isfile(candidate):
            t5_checkpoint = candidate
        else:
            raise FileNotFoundError(
                f"T5 checkpoint not found at {candidate}. "
                f"Use --t5_checkpoint to specify explicitly."
            )

    if tokenizer_path is None:
        candidate = os.path.join(args.assets_dir, "google", "umt5-xxl")
        if os.path.isdir(candidate):
            tokenizer_path = candidate
        else:
            # Try just assets_dir as tokenizer path
            tokenizer_path = args.assets_dir

    return t5_checkpoint, tokenizer_path


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    device = torch.device(args.device)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    text_len = args.text_len or config.text_len

    logging.info(f"Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    logging.info(f"Prompt sha256: {prompt_sha256(args.prompt)}")
    logging.info(f"Device: {device}, dtype: {dtype}, text_len: {text_len}")

    t5_checkpoint, tokenizer_path = resolve_t5_paths(args)
    logging.info(f"T5 checkpoint: {t5_checkpoint}")
    logging.info(f"Tokenizer path: {tokenizer_path}")

    # Load T5 encoder
    logging.info("Loading T5 encoder (this may take a while and use ~11GB RAM)...")
    text_encoder = T5EncoderModel(
        text_len=text_len,
        dtype=dtype,
        device=device,
        checkpoint_path=t5_checkpoint,
        tokenizer_path=tokenizer_path,
    )

    # Encode prompt
    logging.info("Encoding prompt...")
    context_list = text_encoder([args.prompt], device)
    context = context_list[0]  # shape: [seq_len, hidden_dim]
    logging.info(f"Encoded context shape: {list(context.shape)}, dtype: {context.dtype}")

    # Save embedding
    output_path = args.output
    save_prompt_embedding(
        output_path=output_path,
        context=context,
        prompt=args.prompt,
        text_len=text_len,
        dtype=dtype,
    )

    # Clean up T5 to free memory
    logging.info("Releasing T5 encoder...")
    del text_encoder
    import gc
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    logging.info(f"Done! Prompt embedding saved to {output_path}")
    logging.info(f"Use with: python generate.py ... --prompt_embeds_file {output_path}")


if __name__ == "__main__":
    main()
