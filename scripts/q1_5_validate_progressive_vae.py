#!/usr/bin/env python3
"""Q1.5 gate: compare full Wan VAE decode with stateful chunked decode.

Example:
  LINGBOT_VAE_DTYPE=bf16 python scripts/q1_5_validate_progressive_vae.py \
    --latents eval/e1.2/smoke/latents.safetensors \
    --vae-pth /path/to/Wan2.1_VAE.pth \
    --device mps --chunk-size 3 \
    --output eval/quest-streaming/q1_5_vae_equivalence.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from wan.modules.vae2_1 import Wan2_1_VAE
from wan.streaming.vae_progressive import ProgressiveWanVaeDecoder
from wan.utils.staged_cache import load_generated_latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", required=True)
    parser.add_argument("--vae-pth", required=True)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--dtype", default=os.environ.get("LINGBOT_VAE_DTYPE", "bf16"), choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--max-abs-tol", type=float, default=5e-4)
    parser.add_argument("--mean-abs-tol", type=float, default=5e-5)
    parser.add_argument("--output", default="eval/quest-streaming/q1_5_vae_equivalence.json")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_stats(device: torch.device) -> dict[str, float]:
    if device.type == "mps":
        return {
            "mpsAllocatedGB": torch.mps.current_allocated_memory() / 1e9,
            "mpsDriverGB": torch.mps.driver_allocated_memory() / 1e9,
        }
    if device.type == "cuda":
        return {
            "cudaAllocatedGB": torch.cuda.memory_allocated(device) / 1e9,
            "cudaReservedGB": torch.cuda.memory_reserved(device) / 1e9,
        }
    return {}


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    latents, metadata = load_generated_latents(args.latents)
    latents = latents.to(device=device)

    vae = Wan2_1_VAE(
        vae_pth=args.vae_pth,
        dtype=dtype,
        device=device,
    )
    if dtype != torch.float32:
        vae.model = vae.model.to(dtype)
        vae.mean = vae.mean.to(dtype)
        vae.std = vae.std.to(dtype)
        vae.scale = [vae.mean, 1.0 / vae.std]

    synchronize(device)
    t0 = time.perf_counter()
    full = vae.decode([latents])[0]
    synchronize(device)
    full_sec = time.perf_counter() - t0

    chunks = list(latents.split(args.chunk_size, dim=1))
    chunk_outputs = []
    synchronize(device)
    t1 = time.perf_counter()
    with ProgressiveWanVaeDecoder(vae) as decoder:
        for chunk in chunks:
            chunk_outputs.append(decoder.decode_chunk(chunk))
        progressive_stats = decoder.stats()
    progressive = torch.cat(chunk_outputs, dim=1)
    synchronize(device)
    progressive_sec = time.perf_counter() - t1

    if full.shape != progressive.shape:
        raise RuntimeError(
            f"shape mismatch: full={tuple(full.shape)} progressive={tuple(progressive.shape)}"
        )

    diff = (full.float() - progressive.float()).abs()
    mse = torch.mean((full.float() - progressive.float()) ** 2).item()
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(2.0 / math.sqrt(mse))

    boundaries = []
    cursor = 0
    per_frame_max = diff.flatten(0, 0).flatten(1).amax(dim=1) if diff.ndim == 4 else None
    # RGB video layout is [C,T,H,W]; evaluate errors around chunk output seams.
    frame_max = diff.permute(1, 0, 2, 3).reshape(diff.shape[1], -1).amax(dim=1)
    for output in chunk_outputs[:-1]:
        cursor += int(output.shape[1])
        lo = max(0, cursor - 1)
        hi = min(int(frame_max.shape[0]), cursor + 1)
        boundaries.append({
            "outputFrame": cursor,
            "neighborMaxAbs": float(frame_max[lo:hi].max().item()) if hi > lo else 0.0,
        })

    report = {
        "gate": "Q1.5_PROGRESSIVE_VAE_EQUIVALENCE",
        "pass": bool(
            diff.max().item() <= args.max_abs_tol
            and diff.mean().item() <= args.mean_abs_tol
        ),
        "latents": str(args.latents),
        "metadata": metadata.to_dict(),
        "device": str(device),
        "dtype": args.dtype,
        "chunkSize": args.chunk_size,
        "latentShape": list(latents.shape),
        "videoShape": list(full.shape),
        "numChunks": len(chunks),
        "fullDecodeSec": full_sec,
        "progressiveDecodeSec": progressive_sec,
        "maxAbs": float(diff.max().item()),
        "meanAbs": float(diff.mean().item()),
        "mse": float(mse),
        "psnrDb": psnr,
        "maxAbsTolerance": args.max_abs_tol,
        "meanAbsTolerance": args.mean_abs_tol,
        "boundaries": boundaries,
        "progressiveStats": {
            "latentFrames": progressive_stats.latent_frames,
            "outputFrames": progressive_stats.output_frames,
            "chunks": progressive_stats.chunks,
        },
        "memory": memory_stats(device),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
