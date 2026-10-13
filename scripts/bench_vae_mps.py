#!/usr/bin/env python3
"""MPS BF16 VAE decode benchmark worker (one latent, one subprocess).

Loads a cached DiT latent, decodes it with the production Wan2_1_VAE in BF16
on MPS, runs warmup + 3 timed decodes, records median wall time and peak
driver memory, and saves the decoded [-1,1] float32 video [C,T,H,W] to an .npy
(used as the reference for the MLX error comparison).

Usage:
  python scripts/bench_vae_mps.py --latent <...safetensors> \
      --out-video <...npy> --result <...json>
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: E402

VAE_CKPT = "/Volumes/ssd/lingbot-assets/Wan2.1_VAE.pth"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True)
    ap.add_argument("--out-video", required=True)
    ap.add_argument("--result", required=True)
    args = ap.parse_args()

    device = torch.device("mps")
    torch.mps.empty_cache()

    z = load_file(args.latent)["latents"].float().contiguous()  # [C,T,H,W]
    print(f"[mps-bench] latent {tuple(z.shape)} {z.dtype}", flush=True)

    gc.collect()
    torch.mps.empty_cache()
    try:
        torch.mps.reset_peak_memory_stats()
    except Exception:
        pass

    t0 = time.perf_counter()
    vae = Wan2_1_VAE(vae_pth=VAE_CKPT, dtype=torch.bfloat16, device=device)
    vae.model = vae.model.to(torch.bfloat16)
    vae.mean = vae.mean.to(torch.bfloat16)
    vae.std = vae.std.to(torch.bfloat16)
    torch.mps.synchronize()
    load_wall = time.perf_counter() - t0

    z_dev = z.to(device=device, dtype=torch.bfloat16)

    # warmup
    torch.mps.synchronize()
    t0 = time.perf_counter()
    _ = vae.decode([z_dev])
    torch.mps.synchronize()
    warmup_wall = time.perf_counter() - t0

    times = []
    out_np = None
    for i in range(3):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        videos = vae.decode([z_dev])
        torch.mps.synchronize()
        dt = time.perf_counter() - t0
        times.append(round(dt, 3))
        out_np = videos[0].float().cpu().numpy()  # [C,T,H,W]
        print(f"[mps-bench] run {i}: {dt:.3f}s", flush=True)

    try:
        peak_mps = torch.mps.current_allocated_memory() / (1024 ** 2)
    except Exception:
        peak_mps = 0.0
    peak_driver = torch.mps.driver_allocated_memory() / (1024 ** 2)

    np.save(args.out_video, out_np)
    result = {
        "backend": "mps",
        "dtype": "bf16",
        "latent": args.latent,
        "latent_shape": list(z.shape),
        "output_shape": list(out_np.shape),
        "load_wall_s": round(load_wall, 3),
        "warmup_wall_s": round(warmup_wall, 3),
        "infer_runs_s": times,
        "infer_median_s": float(np.median(times)),
        "peak_mps_allocated_mb": round(peak_mps, 1),
        "peak_driver_mb": round(peak_driver, 1),
        "range": [float(out_np.min()), float(out_np.max())],
        "temporal_mad": float(np.abs(np.diff(out_np, axis=1)).mean()),
    }
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[mps-bench] DONE median={np.median(times):.3f}s peak_driver={peak_driver:.0f}MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
