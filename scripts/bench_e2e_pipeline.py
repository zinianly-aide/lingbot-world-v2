#!/usr/bin/env python3
"""End-to-end inference pipeline profiling for LingBot-World 2.0 on Apple M4.

Splits the full i2v pipeline into stages and measures wall time + peak memory
for each. Uses cached prompt_embeds and image_condition to avoid re-running
text encoder / VAE encode, keeping profiling focused on the generation path.

Stages profiled:
  1. prompt_embeds load (cached, no UMT5 forward)
  2. image_condition load (cached, no VAE encode)
  3. DiT load + generate-latents (denoising)
  4. DiT unload
  5. VAE load + decode
  6. MPS→CPU copy + postprocess + MP4 write

All MPS timings use torch.mps.synchronize() before/after.
Cold run (1st) vs warm run (2nd) recorded separately.

Output: eval/bench_e2e/results.json + printed summary table.
"""

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Fixed G0.7 config for profiling
CKPT_DIR = "/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/snapshots/7e36a5f919f86cb4255cc9bfc30adb44963fbde1"
ASSETS_DIR = "/Volumes/ssd/lingbot-assets"
VAE_CKPT = f"{ASSETS_DIR}/Wan2.1_VAE.pth"
# Prompt text is only used when prompt embeds are NOT cached; we reuse the
# exact e2 proven prompt file for the single_subject scene (cosmetic here,
# since --prompt_embeds_file overrides text encoding).
PROMPT = open(REPO_ROOT / "eval/g0.7/single_subject/prompt_A.txt").read().strip()
IMAGE = "examples/03/image.jpg"
ACTION_PATH = "examples/03"
FRAME_NUM = 13
SIZE = "832*480"  # proven e2 size; effective video is 832x464 (lat_h=58)
CHUNK_SIZE = 4
SEED = 42

# Cached artifacts mandated for profiling (reuse, do not recompute):
#   prompt_embeds : e2 MiniCPM+Adapter embeddings (matches lingbot 1.3B DiT)
#   image_condition : G0.7 cached VAE image condition
#   latents : e2 cached DiT output for seed=42 ([16,4,58,104] fp32)
PROMPT_EMBEDS_CACHE = "eval/e2/embeddings/single_subject_minicpm.safetensors"
IMAGE_CONDITION_CACHE = "eval/g0.7/.cache/image_condition_single_subject.safetensors"
LATENTS_CACHE = "eval/e2/.cache/single_subject_42/latents.safetensors"


def _sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _peak_mps():
    if not torch.backends.mps.is_available():
        return 0.0, 0.0
    return (
        torch.mps.current_allocated_memory() / (1024 ** 2),
        torch.mps.driver_allocated_memory() / (1024 ** 2),
    )


def _rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _reset_peak():
    if torch.backends.mps.is_available():
        try:
            torch.mps.reset_peak_memory_stats()
        except Exception:
            pass


def run_substage(cmd, stage_name, timeout=1800):
    """Run a sub-stage in an independent subprocess. Returns (wall_s, peak_driver_mb, returncode, stdout_tail)."""
    logger.info(f"  Running sub-stage: {stage_name}")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT)
        )
        wall = time.perf_counter() - t0
        stdout_tail = result.stdout[-2000:] if result.stdout else ""
        stderr_tail = result.stderr[-2000:] if result.stderr else ""
        return {
            "wall_s": round(wall, 3),
            "returncode": result.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "wall_s": round(time.perf_counter() - t0, 3),
            "returncode": -1,
            "stdout_tail": "",
            "stderr_tail": "TIMEOUT",
            "success": False,
        }


def profile_dit_stage(prompt_embeds_file, image_condition_file, output_latents, run_label):
    """Profile DiT load + generate-latents in a subprocess."""
    cmd = [
        sys.executable, "scripts/g07_gen_stage.py",
        "--result-json", f"eval/bench_e2e/.tmp/dit_{run_label}.json",
        "--repo-root", str(REPO_ROOT),
        "--",
        "--ckpt_dir", CKPT_DIR,
        "--assets_dir", ASSETS_DIR,
        "--device", "mps",
        "--task", "i2v-1.3B",
        "--infer_mode", "causal_fast",
        "--image", IMAGE,
        "--action_path", ACTION_PATH,
        "--frame_num", str(FRAME_NUM),
        "--size", SIZE,
        "--chunk_size", str(CHUNK_SIZE),
        "--base_seed", str(SEED),
        "--local_attn_size", "-1",
        "--stage", "generate-latents",
        "--prompt", PROMPT,
        "--prompt_embeds_file", prompt_embeds_file,
        "--image_condition_file", image_condition_file,
        "--output_latents_file", output_latents,
    ]
    return run_substage(cmd, f"DiT generate-latents ({run_label})")


def profile_vae_decode_stage(latents_file, output_video, run_label):
    """Profile VAE load + decode in a subprocess."""
    cmd = [
        sys.executable, "scripts/g07_gen_stage.py",
        "--result-json", f"eval/bench_e2e/.tmp/vae_{run_label}.json",
        "--repo-root", str(REPO_ROOT),
        "--",
        "--ckpt_dir", CKPT_DIR,
        "--assets_dir", ASSETS_DIR,
        "--device", "mps",
        "--task", "i2v-1.3B",
        "--infer_mode", "causal_fast",
        "--image", IMAGE,
        "--action_path", ACTION_PATH,
        "--frame_num", str(FRAME_NUM),
        "--size", SIZE,
        "--chunk_size", str(CHUNK_SIZE),
        "--base_seed", str(SEED),
        "--local_attn_size", "-1",
        "--stage", "decode",
        "--prompt", PROMPT,
        "--latents_file", latents_file,
        "--save_file", output_video,
    ]
    return run_substage(cmd, f"VAE decode ({run_label})")


def profile_prompt_embeds_load(filepath):
    """Profile loading cached prompt embeds (in-process, fast)."""
    from safetensors.torch import load_file
    _reset_peak()
    t0 = time.perf_counter()
    data = load_file(filepath)
    _sync()
    wall = time.perf_counter() - t0
    alloc, driver = _peak_mps()
    shape = {k: list(v.shape) for k, v in data.items()}
    dtype = {k: str(v.dtype) for k, v in data.items()}
    del data
    gc.collect()
    return {
        "wall_s": round(wall, 4),
        "mps_allocated_peak_mb": round(alloc, 1),
        "mps_driver_peak_mb": round(driver, 1),
        "shape": shape,
        "dtype": dtype,
        "rss_mb": round(_rss_mb(), 1),
    }


def profile_image_condition_load(filepath):
    """Profile loading cached image condition (in-process, fast)."""
    from safetensors.torch import load_file
    _reset_peak()
    t0 = time.perf_counter()
    data = load_file(filepath)
    _sync()
    wall = time.perf_counter() - t0
    alloc, driver = _peak_mps()
    shape = {k: list(v.shape) for k, v in data.items()}
    dtype = {k: str(v.dtype) for k, v in data.items()}
    del data
    gc.collect()
    return {
        "wall_s": round(wall, 4),
        "mps_allocated_peak_mb": round(alloc, 1),
        "mps_driver_peak_mb": round(driver, 1),
        "shape": shape,
        "dtype": dtype,
        "rss_mb": round(_rss_mb(), 1),
    }


def find_cached_artifacts():
    """Return the mandated cached artifacts (do not auto-discover alternatives)."""
    prompt_embeds = str(REPO_ROOT / PROMPT_EMBEDS_CACHE)
    image_condition = str(REPO_ROOT / IMAGE_CONDITION_CACHE)
    latents = str(REPO_ROOT / LATENTS_CACHE)
    return prompt_embeds, image_condition, latents


def main():
    parser = argparse.ArgumentParser(description="E2E inference pipeline profiling")
    parser.add_argument("--output", default="eval/bench_e2e/results.json")
    parser.add_argument("--skip-dit", action="store_true", help="Skip DiT stage (use cached latents)")
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE decode stage")
    args = parser.parse_args()

    os.makedirs("eval/bench_e2e/.tmp", exist_ok=True)
    device = torch.device("mps")
    logger.info(f"Device: {device}, MPS available: {torch.backends.mps.is_available()}")

    # Find cached artifacts
    prompt_embeds, image_condition, latents = find_cached_artifacts()
    logger.info(f"Cached prompt_embeds: {prompt_embeds}")
    logger.info(f"Cached image_condition: {image_condition}")
    logger.info(f"Cached latents: {latents}")

    results = {
        "device": "mps",
        "config": {
            "frame_num": FRAME_NUM,
            "size": SIZE,
            "chunk_size": CHUNK_SIZE,
            "seed": SEED,
            "prompt": PROMPT,
            "image": IMAGE,
        },
        "cached_artifacts": {
            "prompt_embeds": prompt_embeds,
            "image_condition": image_condition,
            "latents": latents,
        },
        "stages": {},
    }

    # Stage 1: prompt_embeds load
    if prompt_embeds and os.path.exists(prompt_embeds):
        logger.info("Stage 1: prompt_embeds load")
        results["stages"]["prompt_embeds_load"] = profile_prompt_embeds_load(prompt_embeds)
    else:
        logger.warning("No cached prompt_embeds found - skipping load profile")
        results["stages"]["prompt_embeds_load"] = {"skipped": True, "reason": "no cache"}

    # Stage 2: image_condition load
    if image_condition and os.path.exists(image_condition):
        logger.info("Stage 2: image_condition load")
        results["stages"]["image_condition_load"] = profile_image_condition_load(image_condition)
    else:
        logger.warning("No cached image_condition found - skipping load profile")
        results["stages"]["image_condition_load"] = {"skipped": True, "reason": "no cache"}

    # Stage 3: DiT generate-latents (cold + warm)
    if not args.skip_dit and prompt_embeds and image_condition:
        dit_latents = "eval/bench_e2e/.tmp/dit_latents.safetensors"
        logger.info("Stage 3: DiT generate-latents (cold)")
        results["stages"]["dit_generate_latents_cold"] = profile_dit_stage(
            prompt_embeds, image_condition, dit_latents, "cold"
        )
        logger.info("Stage 3: DiT generate-latents (warm)")
        results["stages"]["dit_generate_latents_warm"] = profile_dit_stage(
            prompt_embeds, image_condition, dit_latents, "warm"
        )
        # Use DiT-generated latents for VAE stage if no cached latents
        if latents is None and os.path.exists(dit_latents):
            latents = dit_latents
    else:
        logger.info("Skipping DiT stage")
        results["stages"]["dit_generate_latents_cold"] = {"skipped": True}
        results["stages"]["dit_generate_latents_warm"] = {"skipped": True}

    # Stage 4: VAE decode (cold + warm)
    if not args.skip_vae and latents and os.path.exists(latents):
        output_video = "eval/bench_e2e/.tmp/bench_output.mp4"
        logger.info("Stage 4: VAE decode (cold)")
        results["stages"]["vae_decode_cold"] = profile_vae_decode_stage(
            latents, output_video, "cold"
        )
        logger.info("Stage 4: VAE decode (warm)")
        results["stages"]["vae_decode_warm"] = profile_vae_decode_stage(
            latents, output_video, "warm"
        )
    else:
        logger.info("Skipping VAE decode stage")
        results["stages"]["vae_decode_cold"] = {"skipped": True}
        results["stages"]["vae_decode_warm"] = {"skipped": True}

    # Bottleneck analysis (warm run)
    logger.info("Analyzing bottlenecks...")
    warm_stages = {}
    for key, val in results["stages"].items():
        if key.endswith("_warm") and isinstance(val, dict) and "wall_s" in val:
            warm_stages[key.replace("_warm", "")] = val["wall_s"]

    if warm_stages:
        total = sum(warm_stages.values())
        bottleneck = max(warm_stages, key=warm_stages.get)
        results["bottleneck_analysis"] = {
            "total_warm_wall_s": round(total, 3),
            "stage_breakdown_warm": {k: {"wall_s": v, "pct": round(v / total * 100, 1)} for k, v in warm_stages.items()},
            "top1_bottleneck": bottleneck,
            "top1_pct": round(warm_stages[bottleneck] / total * 100, 1),
        }

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")

    # Print summary
    print("\n" + "=" * 90)
    print("E2E PIPELINE PROFILING SUMMARY")
    print("=" * 90)
    print(f"{'Stage':<35} {'Wall(s)':<12} {'Driver(MB)':<12} {'Notes'}")
    print("-" * 90)
    for key, val in results["stages"].items():
        if isinstance(val, dict) and "wall_s" in val:
            driver = val.get("mps_driver_peak_mb", val.get("peak_driver_mb", "N/A"))
            notes = "OK" if val.get("success", True) else f"RC={val.get('returncode')}"
            print(f"{key:<35} {val['wall_s']:<12.3f} {str(driver):<12} {notes}")
        elif isinstance(val, dict) and val.get("skipped"):
            print(f"{key:<35} {'SKIPPED':<12} {'':<12} {val.get('reason', '')}")

    if "bottleneck_analysis" in results:
        ba = results["bottleneck_analysis"]
        print("-" * 90)
        print(f"Total warm wall: {ba['total_warm_wall_s']:.3f}s")
        print(f"Top 1 bottleneck: {ba['top1_bottleneck']} ({ba['top1_pct']}%)")
        for stage, info in ba["stage_breakdown_warm"].items():
            print(f"  {stage}: {info['wall_s']:.3f}s ({info['pct']}%)")
    print("=" * 90)


if __name__ == "__main__":
    main()
