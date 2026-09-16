#!/usr/bin/env python3
"""MLX VAE decode benchmark for the e2 single_subject seed=42 latent.

Runs in its own process (one dtype per invocation) so peak unified memory
(process RSS) is not contaminated by other workloads.

Per dtype:
  - build AutoencoderKLWan from the converted weights (scripts/convert_vae_to_mlx.py)
  - measure model load time
  - warmup 1 decode (pays graph-compilation cost)
  - 3 timed decodes, report median
  - peak unified memory = process RSS max (ru_maxrss)
  - convert output to [C,T,H,W] [-1,1] float32
  - numerical error vs the MPS BF16 reference (CPU float32)
  - sanity: non-black/white/frozen + temporal_mad

Usage:
  python scripts/bench_vae_mlx.py --dtype bf16 --result eval/bench_vae_mlx/result_bf16.json
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

VAE_DIR = REPO_ROOT / "eval/bench_vae_mlx/vae_mlx"
LATENT = REPO_ROOT / "eval/e2/.cache/single_subject_42/latents.safetensors"
REF_MPS_BF16 = REPO_ROOT / "eval/bench_vae_mlx/ref_mps_bf16.npy"


def _rss_mb() -> float:
    # macOS: ru_maxrss is bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "bf16"], required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--ref", default=str(REF_MPS_BF16))
    args = ap.parse_args()

    mx_dtype = {"fp32": mx.float32, "bf16": mx.bfloat16}[args.dtype]

    # --- load latent (normalized space, [C,T,H,W] fp32) -> channels-last [B,T,H,W,C] ---
    with safe_open(str(LATENT), "np") as f:
        z = f.get_tensor("latents").astype(np.float32)
    z_cl = np.transpose(z, (1, 2, 3, 0))[None, ...]  # [1,T,H,W,C]
    print(f"[mlx-bench] dtype={args.dtype} latent channels-last {z_cl.shape}")

    # --- model load ---
    gc.collect()
    mx.clear_cache()
    t0 = time.perf_counter()
    from mlx_diffuser import AutoencoderKLWan, AutoencoderKLWanConfig
    cfg = AutoencoderKLWanConfig(**{
        k: v for k, v in json.load(open(VAE_DIR / "config.json")).items() if k != "class_name"
    })
    model = AutoencoderKLWan(cfg)
    weights = mx.load(str(VAE_DIR / "model.safetensors"))
    model.load_weights(list(weights.items()), strict=True)
    model.set_dtype(mx_dtype)
    mx.eval(model.parameters())
    load_wall = time.perf_counter() - t0
    del weights
    print(f"[mlx-bench] model load+cast: {load_wall:.2f}s")

    # --- denormalize (z*std+mean), matching native Wan2_1_VAE.decode scale ---
    z_mx = mx.array(z_cl, dtype=mx_dtype)
    z_denorm = model.denormalize_latents(z_mx)
    mx.eval(z_denorm)

    # --- warmup (compile) ---
    t0 = time.perf_counter()
    out = model.decode(z_denorm)
    mx.eval(out)
    warmup_wall = time.perf_counter() - t0
    print(f"[mlx-bench] warmup decode: {warmup_wall:.2f}s")

    # --- 3 timed runs ---
    times = []
    out_np = None
    for i in range(3):
        mx.clear_cache()
        t0 = time.perf_counter()
        out = model.decode(z_denorm)
        mx.eval(out)
        dt = time.perf_counter() - t0
        times.append(round(dt, 3))
        print(f"[mlx-bench] run {i}: {dt:.3f}s")
    infer_median = float(np.median(times))
    peak_rss_mb = round(_rss_mb(), 1)

    # output -> [C,T,H,W]
    out_np = np.array(out, copy=True)[0]  # [T,H,W,C]
    out_chw = np.transpose(out_np, (3, 0, 1, 2))  # [C,T,H,W]
    out_chw = np.ascontiguousarray(out_chw).astype(np.float32)

    # --- error vs MPS BF16 reference ---
    error = None
    ref_exists = Path(args.ref).exists()
    if ref_exists:
        ref = np.load(args.ref).astype(np.float32)  # [C,T,H,W]
        diff = np.abs(out_chw - ref)
        error = {
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
        }

    # --- sanity ---
    a = out_chw
    frame_mad = float(np.abs(np.diff(a, axis=1)).mean())
    sanity = {
        "range": [float(a.min()), float(a.max())],
        "mean": float(a.mean()),
        "std": float(a.std()),
        "all_black": bool((a < -0.99).all()),
        "all_white": bool((a > 0.99).all()),
        "frozen": bool(frame_mad < 1e-4),
        "temporal_mad": frame_mad,
    }

    result = {
        "dtype": args.dtype,
        "latent_path": str(LATENT),
        "latent_channels_last_shape": list(z_cl.shape),
        "output_shape": list(out_chw.shape),
        "load_wall_s": round(load_wall, 3),
        "warmup_wall_s": round(warmup_wall, 3),
        "infer_runs_s": times,
        "infer_median_s": round(infer_median, 3),
        "peak_rss_mb": peak_rss_mb,
        "error_vs_mps_bf16": error,
        "sanity": sanity,
    }
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[mlx-bench] DONE: median={infer_median:.3f}s peak_rss={peak_rss_mb}MB "
          f"error={error} sanity={sanity['temporal_mad']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
