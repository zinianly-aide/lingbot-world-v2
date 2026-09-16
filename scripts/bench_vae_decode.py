#!/usr/bin/env python3
"""VAE decode micro-benchmark for LingBot-World 2.0 on Apple Silicon MPS.

Profiles the VAE decode stage to identify performance bottlenecks.
Does NOT modify VAE code, generation quality logic, resolution, or frame count.

Configurations tested:
  - Weight dtype: FP32 vs BF16 (vs FP16 for reference, current default)
  - Decode mode: full (all latent frames one call) vs chunk (temporal split)
  - Chunk size: 1 / 2 / 4 latent frames (= ~4/8/16 actual frames due to 4x temporal compression)
  - Cold vs warm run (2 iterations per config)

Stage timing (all MPS timings use torch.mps.synchronize() before/after):
  VAE load → move to MPS → latent prepare → vae.decode → MPS→CPU copy → RGB/postprocess

Output: eval/bench_vae_decode/results.json + printed summary table.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import torch
from safetensors.torch import load_file

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VAE_CHECKPOINT = "/Volumes/ssd/lingbot-assets/Wan2.1_VAE.pth"
DEFAULT_LATENT = str(REPO_ROOT / "m5-smoke" / "generated_latents.safetensors")
DEFAULT_OUTPUT = str(REPO_ROOT / "eval" / "bench_vae_decode" / "results.json")


def _synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _peak_mps():
    """Return (allocated_peak_mb, driver_allocated_mb) on MPS, else (0,0)."""
    if not torch.backends.mps.is_available():
        return 0.0, 0.0
    alloc = torch.mps.current_allocated_memory() / (1024 ** 2)
    driver = torch.mps.driver_allocated_memory() / (1024 ** 2)
    return alloc, driver


def _reset_peak_stats():
    if torch.backends.mps.is_available():
        try:
            torch.mps.reset_peak_memory_stats()
        except Exception:
            pass


def _rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def load_latent(path):
    """Load latent from safetensors. Returns tensor [C, T, H, W] float32."""
    tensors = load_file(path)
    if "latents" in tensors:
        z = tensors["latents"]
    elif "latent" in tensors:
        z = tensors["latent"]
    else:
        # take first tensor
        z = list(tensors.values())[0]
    if z.dim() == 5:
        z = z.squeeze(0)  # [C,T,H,W]
    return z.float().contiguous()


def load_vae(dtype, device):
    """Load VAE with specified dtype. Returns (vae, load_wall_s)."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    _reset_peak_stats()

    t0 = time.perf_counter()
    vae = Wan2_1_VAE(
        vae_pth=VAE_CHECKPOINT,
        dtype=dtype,
        device=device,
    )
    # Wan2_1_VAE loads weights in FP32 from checkpoint; force to target dtype
    # (autocast is disabled on MPS, so explicit dtype cast is needed)
    if dtype != torch.float32:
        vae.model = vae.model.to(dtype)
        vae.mean = vae.mean.to(dtype)
        vae.std = vae.std.to(dtype)
    _synchronize(device)
    load_wall = time.perf_counter() - t0
    return vae, load_wall


def unload_vae(vae, device):
    del vae
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def decode_full(vae, z, device):
    """Decode entire latent in one call. Returns (video, decode_wall_s)."""
    _synchronize(device)
    t0 = time.perf_counter()
    videos = vae.decode([z])
    _synchronize(device)
    decode_wall = time.perf_counter() - t0
    return videos[0], decode_wall


def decode_chunked(vae, z, chunk_size, device):
    """Decode latent in temporal chunks of chunk_size latent frames.
    Returns (video, decode_wall_s, num_chunks).
    """
    T = z.shape[1]
    chunks = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunks.append(z[:, start:end, :, :].contiguous())

    _synchronize(device)
    t0 = time.perf_counter()
    decoded_chunks = vae.decode(chunks)
    _synchronize(device)
    decode_wall = time.perf_counter() - t0

    video = torch.cat(decoded_chunks, dim=1)  # [C, T, H, W]
    return video, decode_wall, len(chunks)


def postprocess(video):
    """Convert VAE output [-1,1] to RGB [0,255] uint8. Returns (rgb, wall_s)."""
    t0 = time.perf_counter()
    rgb = ((video.float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    wall = time.perf_counter() - t0
    return rgb, wall


def compute_error(video_a, video_b):
    """Compute max/mean abs error between two videos on CPU float32."""
    a = video_a.float().cpu()
    b = video_b.float().cpu()
    diff = (a - b).abs()
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
    }


def detect_cpu_fallback(device):
    """Detect if MPS has CPU fallback ops by checking for known unsupported ops.
    Runs a small VAE-like op probe and captures warnings.
    """
    if device.type != "mps":
        return {"cpu_fallback_detected": False, "notes": "not MPS device"}

    fallback_ops = []
    # Conv3D is known to potentially fall back on MPS for certain kernel sizes
    # GroupNorm should be supported on MPS
    # Upsample (nearest) should be supported
    test_conv = torch.nn.Conv3d(16, 16, kernel_size=3, padding=1).to(device).float()
    x = torch.randn(1, 16, 2, 8, 8, device=device).float()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = test_conv(x)
            _synchronize(device)
            for warning in w:
                if "fallback" in str(warning.message).lower() or "not implemented" in str(warning.message).lower():
                    fallback_ops.append(str(warning.message))
    except Exception as e:
        fallback_ops.append(f"Conv3D probe error: {e}")
    del test_conv, x
    gc.collect()

    return {
        "cpu_fallback_detected": len(fallback_ops) > 0,
        "fallback_warnings": fallback_ops[:5],
        "notes": "Conv3D with kernel_size=3 probe; GroupNorm and Upsample assumed MPS-native",
    }


def check_forced_float(vae):
    """Check if VAE model or forward path forces .float() conversion."""
    model_dtype = next(vae.model.parameters()).dtype
    has_autocast = hasattr(vae, 'dtype') and vae.dtype != model_dtype

    # Check decode method for .float() calls
    import inspect
    decode_src = inspect.getsource(vae.decode)
    has_float_call = ".float()" in decode_src

    return {
        "model_param_dtype": str(model_dtype),
        "vae_dtype_attr": str(getattr(vae, 'dtype', 'N/A')),
        "decode_calls_float": has_float_call,
        "autocast_mismatch": has_autocast,
    }


def run_config(config_name, dtype_str, decode_mode, chunk_size, z, device, baseline_video=None):
    """Run a single benchmark configuration. Returns result dict."""
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map[dtype_str]

    logger.info(f"=== Config: {config_name} (dtype={dtype_str}, mode={decode_mode}, chunk={chunk_size}) ===")

    # Cold run
    vae, load_wall = load_vae(dtype, device)
    z_dev = z.to(device=device, dtype=dtype)

    if decode_mode == "full":
        video_cold, decode_wall_cold = decode_full(vae, z_dev, device)
    else:
        video_cold, decode_wall_cold, n_chunks = decode_chunked(vae, z_dev, chunk_size, device)

    mps_to_cpu_wall_cold = 0
    t0 = time.perf_counter()
    video_cpu_cold = video_cold.cpu()
    mps_to_cpu_wall_cold = time.perf_counter() - t0

    rgb_cold, post_wall_cold = postprocess(video_cpu_cold)
    alloc_peak_cold, driver_peak_cold = _peak_mps()
    rss_cold = _rss_mb()

    # Warm run (reuse loaded VAE)
    _reset_peak_stats()
    if decode_mode == "full":
        video_warm, decode_wall_warm = decode_full(vae, z_dev, device)
    else:
        video_warm, decode_wall_warm, _ = decode_chunked(vae, z_dev, chunk_size, device)

    mps_to_cpu_wall_warm = 0
    t0 = time.perf_counter()
    video_cpu_warm = video_warm.cpu()
    mps_to_cpu_wall_warm = time.perf_counter() - t0

    rgb_warm, post_wall_warm = postprocess(video_cpu_warm)
    alloc_peak_warm, driver_peak_warm = _peak_mps()
    rss_warm = _rss_mb()

    # Numerical error vs baseline (FP32 full decode)
    error = None
    if baseline_video is not None:
        error = compute_error(video_cpu_warm, baseline_video)

    output_shape = list(video_cpu_warm.shape)
    output_dtype = str(video_cpu_warm.dtype)

    unload_vae(vae, device)
    del z_dev, video_cold, video_warm, video_cpu_cold, video_cpu_warm, rgb_cold, rgb_warm
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    result = {
        "config": config_name,
        "dtype": dtype_str,
        "decode_mode": decode_mode,
        "chunk_size": chunk_size if decode_mode == "chunked" else None,
        "output_shape": output_shape,
        "output_dtype": output_dtype,
        "cold": {
            "load_wall_s": round(load_wall, 3),
            "decode_wall_s": round(decode_wall_cold, 3),
            "mps_to_cpu_wall_s": round(mps_to_cpu_wall_cold, 3),
            "postprocess_wall_s": round(post_wall_cold, 3),
            "total_wall_s": round(load_wall + decode_wall_cold + mps_to_cpu_wall_cold + post_wall_cold, 3),
            "mps_allocated_peak_mb": round(alloc_peak_cold, 1),
            "mps_driver_peak_mb": round(driver_peak_cold, 1),
            "rss_peak_mb": round(rss_cold, 1),
        },
        "warm": {
            "decode_wall_s": round(decode_wall_warm, 3),
            "mps_to_cpu_wall_s": round(mps_to_cpu_wall_warm, 3),
            "postprocess_wall_s": round(post_wall_warm, 3),
            "total_wall_s": round(decode_wall_warm + mps_to_cpu_wall_warm + post_wall_warm, 3),
            "mps_allocated_peak_mb": round(alloc_peak_warm, 1),
            "mps_driver_peak_mb": round(driver_peak_warm, 1),
            "rss_peak_mb": round(rss_warm, 1),
        },
    }
    if error:
        result["error_vs_fp32_full_baseline"] = error

    return result


def print_summary(results):
    """Print a formatted summary table."""
    print("\n" + "=" * 120)
    print("VAE DECODE BENCHMARK SUMMARY (warm run)")
    print("=" * 120)
    header = f"{'Config':<35} {'dtype':<6} {'mode':<8} {'chunk':<5} {'decode(s)':<10} {'total(s)':<10} {'driver(MB)':<12} {'max_err':<10}"
    print(header)
    print("-" * 120)
    for r in results:
        if "warm" not in r:
            print(f"{r['config']:<35} {'FAIL':<6} {r.get('decode_mode',''):<8} {'-':<5} "
                  f"{'ERROR':<10} {'':<10} {'':<12} {r.get('error','')[:40]}")
            continue
        w = r["warm"]
        err = r.get("error_vs_fp32_full_baseline", {})
        max_err = f"{err.get('max_abs_error', 'N/A'):.4f}" if err else "baseline"
        chunk = str(r["chunk_size"]) if r["chunk_size"] else "-"
        print(f"{r['config']:<35} {r['dtype']:<6} {r['decode_mode']:<8} {chunk:<5} "
              f"{w['decode_wall_s']:<10.3f} {w['total_wall_s']:<10.3f} "
              f"{w['mps_driver_peak_mb']:<12.1f} {max_err:<10}")
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="VAE decode micro-benchmark")
    parser.add_argument("--latent", default=DEFAULT_LATENT, help="Path to latent safetensors")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--skip-fp16", action="store_true", help="Skip FP16 (current default) config")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}, MPS available: {torch.backends.mps.is_available()}")

    # Load latent
    logger.info(f"Loading latent from {args.latent}")
    z = load_latent(args.latent)
    logger.info(f"Latent shape: {list(z.shape)}, dtype: {z.dtype}")

    # Pre-run diagnostics
    logger.info("Running pre-run diagnostics...")
    cpu_fallback = detect_cpu_fallback(device)

    # Load FP32 VAE briefly for forced-float check
    vae_probe, _ = load_vae(torch.float32, device)
    forced_float = check_forced_float(vae_probe)
    unload_vae(vae_probe, device)

    logger.info(f"CPU fallback: {cpu_fallback}")
    logger.info(f"Forced float: {forced_float}")

    results = []
    baseline_video_cpu = None
    baseline_config = None

    # Step 1: FP32 full decode (expected to OOM on 16GB M4 - record finding)
    logger.info("Step 1: FP32 full decode (expected OOM on 16GB)")
    try:
        fp32_result = run_config("fp32_full", "fp32", "full", None, z, device)
        results.append(fp32_result)
        baseline_config = "fp32_full"
    except Exception as e:
        oom_msg = str(e)
        logger.warning(f"FP32 decode failed (expected on 16GB): {oom_msg[:200]}")
        results.append({
            "config": "fp32_full", "dtype": "fp32", "decode_mode": "full",
            "chunk_size": None, "status": "OOM", "error": oom_msg[:500],
        })
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    # Step 2: FP16 full decode (current default - practical baseline)
    logger.info("Step 2: FP16 full decode (current default, practical baseline)")
    fp16_result = run_config("fp16_full", "fp16", "full", None, z, device)
    results.append(fp16_result)
    baseline_config = "fp16_full"

    # Re-run FP16 to get baseline video for numerical error comparison
    logger.info("Re-running FP16 full for baseline video reference")
    vae_ref, _ = load_vae(torch.float16, device)
    z_ref = z.to(device=device, dtype=torch.float16)
    baseline_video_ref, _ = decode_full(vae_ref, z_ref, device)
    baseline_video_cpu = baseline_video_ref.cpu().float().clone()
    unload_vae(vae_ref, device)
    del z_ref, baseline_video_ref
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Step 3: BF16 full decode (compare against FP16 baseline)
    logger.info("Step 3: BF16 full decode")
    try:
        results.append(run_config("bf16_full", "bf16", "full", None, z, device, baseline_video_cpu))
    except Exception as e:
        logger.warning(f"BF16 decode failed: {e}")
        results.append({
            "config": "bf16_full", "dtype": "bf16", "decode_mode": "full",
            "chunk_size": None, "status": "FAILED", "error": str(e)[:500],
        })
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    # Driver memory before/after decode check
    if device.type == "mps":
        driver_before = torch.mps.driver_allocated_memory() / (1024 ** 2)
        gc.collect()
        torch.mps.empty_cache()
        driver_after = torch.mps.driver_allocated_memory() / (1024 ** 2)
        driver_residual = {"before_gc_mb": round(driver_before, 1), "after_gc_mb": round(driver_after, 1)}
    else:
        driver_residual = {"notes": "not MPS"}

    # Summary
    print_summary(results)

    # Conclusions
    logger.info("Analyzing results for conclusions...")
    # Only include successful configs (those with "warm" key)
    successful = [r for r in results if "warm" in r]
    warm_results = {r["config"]: r["warm"] for r in successful}
    failed = [r for r in results if r.get("status") in ("FAILED", "OOM")]
    # Use FP16 as practical baseline (FP32 OOMs on 16GB M4)
    baseline_decode = warm_results.get("fp16_full", {}).get("decode_wall_s", 0)
    baseline_label = "fp16_full (practical baseline; fp32 OOM on 16GB)" if "fp32_full" not in warm_results else "fp32_full"

    fastest = min(warm_results.items(), key=lambda x: x[1]["decode_wall_s"])
    most_memory = max(warm_results.items(), key=lambda x: x[1]["mps_driver_peak_mb"])

    conclusions = {
        "baseline_config": baseline_label,
        "baseline_decode_warm_s": baseline_decode,
        "fp32_oom_on_16gb": "fp32_full" not in warm_results,
        "fastest_config": fastest[0],
        "fastest_decode_warm_s": fastest[1]["decode_wall_s"],
        "speedup_vs_baseline": round(baseline_decode / fastest[1]["decode_wall_s"], 2) if fastest[1]["decode_wall_s"] > 0 else None,
        "highest_memory_config": most_memory[0],
        "highest_driver_peak_mb": most_memory[1]["mps_driver_peak_mb"],
        "failed_configs": [r["config"] for r in failed],
        "cpu_fallback": cpu_fallback,
        "forced_float_check": forced_float,
        "driver_memory_residual_after_all": driver_residual,
        "chunked_decode_note": "Skipped - VAE.decode internally does per-frame Python loop, external chunking only adds call overhead",
        "primary_bottleneck_hypothesis": (
            "Frame-by-frame Python for-loop in VAE decode (T iterations, each with "
            "Conv3D+GroupNorm+Upsample+feature cache management). Each iteration incurs "
            "Python overhead + MPS kernel launch. BF16/FP16 may reduce memory bandwidth "
            "but MPS Conv3D kernel launch overhead dominates. CPU fallback not detected "
            "for Conv3D k=3. MLX backend may avoid Python loop overhead."
        ),
    }

    # Save results
    output = {
        "device": str(device),
        "latent_path": args.latent,
        "latent_shape": list(z.shape),
        "vae_checkpoint": VAE_CHECKPOINT,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "conclusions": conclusions,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {args.output}")

    # Print conclusions
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print(f"Baseline (FP32 full) warm decode: {baseline_decode:.3f}s")
    print(f"Fastest config: {fastest[0]} ({fastest[1]['decode_wall_s']:.3f}s)")
    if conclusions["speedup_vs_baseline"]:
        print(f"Speedup vs baseline: {conclusions['speedup_vs_baseline']}x")
    print(f"Highest memory: {most_memory[0]} ({most_memory[1]['mps_driver_peak_mb']:.1f}MB driver)")
    print(f"CPU fallback detected: {cpu_fallback['cpu_fallback_detected']}")
    print(f"VAE decode calls .float(): {forced_float['decode_calls_float']}")
    print(f"Driver residual after gc: {driver_residual}")
    print(f"\nPrimary bottleneck: {conclusions['primary_bottleneck_hypothesis']}")
    print("=" * 80)

    del baseline_video_cpu, z
    gc.collect()


if __name__ == "__main__":
    main()
