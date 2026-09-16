#!/usr/bin/env python3
"""G0.7 Real A/B/C Evaluation Framework — main orchestrator.

Generates a fixed matrix of videos (5 scenes × 3 seeds × 3 variants) on
Apple M4 MPS using the staged pipeline.  The only variable across A/B/C is
the text conditioning fed to UMT5.

Usage:
    # Dry run: single_subject scene, A/B/C, seed=42 only
    python scripts/run_g07_eval.py --dry-run

    # Full run
    python scripts/run_g07_eval.py

    # Filter
    python scripts/run_g07_eval.py --scene indoor --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from world_condition import (  # noqa: E402
    MiniCPMVPerceiver,
    WorldDescription,
    compose_compact_world_prompt,
    compose_world_prompt,
    save_world_condition,
    save_world_prompt,
)

EVAL_DIR = REPO_ROOT / "eval" / "g0.7"
CONFIG_PATH = EVAL_DIR / "config.json"
RESULTS_PATH = EVAL_DIR / "results.jsonl"
METRICS_PROMPT_PATH = EVAL_DIR / "metrics_prompt.jsonl"
CACHE_DIR = EVAL_DIR / ".cache"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("g07")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def count_umt5_tokens(text: str, tokenizer_path: str) -> int:
    """Count UMT5 tokenizer tokens without loading the model."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return len(tokenizer.encode(text, add_special_tokens=True))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_existing_results() -> dict[str, dict[str, Any]]:
    """Read results.jsonl into a dict keyed by scene_id|variant|seed."""
    results: dict[str, dict[str, Any]] = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = f"{rec['scene_id']}|{rec['variant']}|{rec['seed']}"
                results[key] = rec
    return results


def write_results_all(results: dict[str, dict[str, Any]]) -> None:
    """Rewrite results.jsonl from the in-memory dict."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for rec in results.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# VLM world condition (once per scene)
# ---------------------------------------------------------------------------

def generate_world_condition(scene: dict[str, Any], cfg: dict[str, Any]) -> tuple[WorldDescription, str]:
    """Run MiniCPM-V once per scene. Returns (world, raw_text)."""
    scene_dir = EVAL_DIR / scene["id"]
    wc_path = scene_dir / "world_condition.json"
    raw_path = scene_dir / "raw_vlm_output.txt"

    if wc_path.exists() and raw_path.exists():
        log.info("World condition already exists for %s, loading cache.", scene["id"])
        from world_condition import load_world_condition
        return load_world_condition(str(wc_path)), raw_path.read_text(encoding="utf-8")

    log.info("Running MiniCPM-V for scene %s ...", scene["id"])
    image_path = REPO_ROOT / scene["image"]
    perceiver = MiniCPMVPerceiver(
        model_name="mlx-community/MiniCPM-V-4.6-4bit",
        backend="mlx",
        device="cpu",
        max_new_tokens=1500,
    )
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        result = perceiver.analyze(img, user_prompt=scene["user_prompt"])
        world = result.world
        raw_text = result.raw_text or ""
        if result.error:
            log.warning("VLM error for %s: %s", scene["id"], result.error)
    finally:
        perceiver.release()

    scene_dir.mkdir(parents=True, exist_ok=True)
    save_world_condition(world, str(wc_path))
    raw_path.write_text(raw_text, encoding="utf-8")
    log.info("Saved world_condition.json and raw_vlm_output.txt for %s", scene["id"])
    return world, raw_text


# ---------------------------------------------------------------------------
# Prompt building (A/B/C)
# ---------------------------------------------------------------------------

def build_variant_prompt(
    variant: str,
    world: WorldDescription,
    user_prompt: str,
) -> str:
    if variant == "A":
        return user_prompt
    elif variant == "B":
        return compose_world_prompt(world, user_prompt)
    elif variant == "C":
        return compose_compact_world_prompt(world, user_prompt)
    raise ValueError(f"Unknown variant: {variant}")


def write_prompts_and_metrics(
    scene: dict[str, Any],
    world: WorldDescription,
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build A/B/C prompts, save prompt_X.txt, write metrics_prompt.jsonl."""
    scene_dir = EVAL_DIR / scene["id"]
    tokenizer_path = cfg["paths"]["tokenizer_path"]
    max_tokens = cfg["umt5_max_tokens"]
    world_chars = len(json.dumps(world.to_dict(), ensure_ascii=False))
    entity_count = len(world.main_entities)

    prompt_info: dict[str, dict[str, Any]] = {}

    for variant in cfg["variants"]:
        prompt = build_variant_prompt(variant, world, scene["user_prompt"])
        prompt_file = scene_dir / f"prompt_{variant}.txt"
        save_world_prompt(prompt, str(prompt_file))

        token_count = count_umt5_tokens(prompt, tokenizer_path)
        truncated = token_count > max_tokens
        record = {
            "scene": scene["id"],
            "variant": variant,
            "prompt_chars": len(prompt),
            "token_count_before_truncation": token_count,
            "token_count_after_truncation": min(token_count, max_tokens),
            "truncated": truncated,
            "world_condition_chars": world_chars,
            "entity_count": entity_count,
        }
        append_jsonl(METRICS_PROMPT_PATH, record)
        log.info(
            "  %s/%s: chars=%d tokens=%d truncated=%s",
            scene["id"], variant, len(prompt), token_count, truncated,
        )
        prompt_info[variant] = {
            "prompt": prompt,
            "prompt_file": str(prompt_file),
            "token_count": token_count,
            "truncated": truncated,
        }

    return prompt_info


# ---------------------------------------------------------------------------
# Staged generation
# ---------------------------------------------------------------------------

def run_stage(
    stage: str,
    cfg: dict[str, Any],
    scene: dict[str, Any],
    variant: str,
    seed: int,
    prompt_file: str,
    prompt_embeds_file: str,
    image_condition_file: str,
    latents_file: str,
    output_video: str,
    work_dir: Path,
) -> dict[str, Any]:
    """Run one generation stage via g07_gen_stage.py wrapper."""
    gen = cfg["generation"]
    paths = cfg["paths"]

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "g07_gen_stage.py"),
        "--result-json", str(work_dir / f"stage_{stage}.json"),
        "--repo-root", str(REPO_ROOT),
        "--",
        "--ckpt_dir", paths["checkpoint_dir"],
        "--assets_dir", paths["assets_dir"],
        "--device", gen["device"],
        "--task", gen["task"],
        "--infer_mode", gen["infer_mode"],
        "--image", str(REPO_ROOT / scene["image"]),
        "--action_path", str(REPO_ROOT / scene["action_path"]),
        "--frame_num", str(gen["frame_num"]),
        "--size", gen["size"],
        "--chunk_size", str(gen["chunk_size"]),
        "--base_seed", str(seed),
        "--local_attn_size", str(gen["local_attn_size"]),
        "--stage", stage,
    ]
    if gen.get("sequential_load"):
        # sequential_load is auto-enabled for MPS in WanI2VCausal;
        # it is not a CLI argument of generate.py.
        pass

    # Prompt: always pass actual prompt text (used for validation/logging).
    # For generate-latents, --prompt_embeds_file provides the actual embedding.
    with open(prompt_file, encoding="utf-8") as f:
        prompt_text = f.read().strip()
    cmd.extend(["--prompt", prompt_text])

    if stage == "generate-latents":
        cmd.extend(["--prompt_embeds_file", prompt_embeds_file])

    if stage == "encode-image":
        cmd.extend(["--dump_image_condition", image_condition_file])
    elif stage == "generate-latents":
        cmd.extend(["--image_condition_file", image_condition_file])
        cmd.extend(["--output_latents_file", latents_file])
    elif stage == "decode":
        cmd.extend(["--latents_file", latents_file])
        cmd.extend(["--save_file", output_video])

    log.info("  Stage %s: %s", stage, " ".join(cmd[:8]) + " ...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))

    result_json_path = work_dir / f"stage_{stage}.json"
    stage_result: dict[str, Any] = {"status": "FAIL", "error": "no result json", "elapsed_seconds": 0}
    if result_json_path.exists():
        with open(result_json_path, encoding="utf-8") as f:
            stage_result = json.load(f)

    if stage_result.get("status") != "PASS":
        log.error("  Stage %s FAILED: %s", stage, stage_result.get("error"))
        log.error("  stdout tail: %s", result.stdout[-500:] if result.stdout else "")
        log.error("  stderr tail: %s", result.stderr[-500:] if result.stderr else "")

    return stage_result


def encode_prompt_embedding(
    prompt_file: str,
    output_file: str,
    cfg: dict[str, Any],
    variant: str = "",
    scene: str = "",
) -> tuple[bool, dict[str, Any]]:
    """Pre-compute UMT5 prompt embedding in an independent subprocess.

    Returns (success, metadata_dict). The subprocess exits after encoding,
    releasing all mmap pages and model memory.
    """
    if os.path.exists(output_file):
        log.info("  Prompt embeds already cached: %s", output_file)
        return True, {"cached": True}

    with open(prompt_file, encoding="utf-8") as f:
        prompt_text = f.read().strip()

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "encode_prompt.py"),
        "--prompt", prompt_text,
        "--assets_dir", cfg["paths"]["assets_dir"],
        "--output", output_file,
        "--device", "cpu",
        "--dtype", "bfloat16",
    ]
    if variant:
        cmd.extend(["--variant", variant])
    if scene:
        cmd.extend(["--scene", scene])

    log.info("  Encoding prompt embeds (independent UMT5 subprocess)...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    metadata = {
        "return_code": result.returncode,
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
    }
    if result.returncode != 0:
        log.error("  Prompt encoding FAILED (return_code=%d):", result.returncode)
        log.error("  STDOUT tail: %s", metadata["stdout_tail"])
        log.error("  STDERR tail: %s", metadata["stderr_tail"])
        return False, metadata
    if not os.path.exists(output_file):
        log.error("  Prompt encoding returned 0 but output file missing: %s", output_file)
        return False, metadata
    log.info("  Prompt embeds saved: %s", output_file)
    return True, metadata


# ---------------------------------------------------------------------------
# Sanity metrics
# ---------------------------------------------------------------------------

def compute_sanity_metrics(video_path: str) -> dict[str, Any]:
    """Compute objective sanity metrics for a generated video."""
    metrics: dict[str, Any] = {
        "frame_count": None,
        "resolution": None,
        "all_black": False,
        "all_white": False,
        "frozen_video": False,
        "nan_inf_corruption": False,
        "temporal_mad_mean": None,
        "temporal_mad_max": None,
        "per_frame_mean": [],
        "per_frame_std": [],
    }

    try:
        import numpy as np
    except ImportError:
        log.warning("numpy not available, skipping sanity metrics")
        return metrics

    # Try to read video frames
    frames = None
    try:
        import imageio.v3 as iio
        frames = iio.imread(video_path)
    except Exception:
        pass

    if frames is None:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_list = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if frame_list:
                frames = np.stack(frame_list)
        except Exception:
            pass

    if frames is None or len(frames) == 0:
        log.warning("  Could not read video frames for sanity metrics: %s", video_path)
        return metrics

    metrics["frame_count"] = int(len(frames))
    metrics["resolution"] = f"{frames.shape[2]}x{frames.shape[1]}"

    # Normalize to [0, 1]
    if frames.dtype == np.uint8:
        frames_f = frames.astype(np.float32) / 255.0
    else:
        frames_f = frames.astype(np.float32)

    # NaN/Inf check
    if not np.all(np.isfinite(frames_f)):
        metrics["nan_inf_corruption"] = True

    # Per-frame mean/std
    for i in range(len(frames_f)):
        metrics["per_frame_mean"].append(float(np.mean(frames_f[i])))
        metrics["per_frame_std"].append(float(np.std(frames_f[i])))

    # All-black / all-white
    overall_mean = float(np.mean(frames_f))
    metrics["all_black"] = overall_mean < 0.02
    metrics["all_white"] = overall_mean > 0.98

    # Temporal MAD (mean absolute difference between adjacent frames)
    if len(frames_f) > 1:
        diffs = np.abs(np.diff(frames_f, axis=0))
        mad_per_frame = np.mean(diffs, axis=(1, 2, 3))
        metrics["temporal_mad_mean"] = float(np.mean(mad_per_frame))
        metrics["temporal_mad_max"] = float(np.max(mad_per_frame))
        # Frozen video: very low temporal difference
        metrics["frozen_video"] = metrics["temporal_mad_mean"] < 0.001

    return metrics


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def should_skip(
    existing: dict[str, Any] | None,
    scene: dict[str, Any],
    variant: str,
    seed: int,
    cfg: dict[str, Any],
) -> tuple[bool, str]:
    """Check if an existing result can be skipped (hash match)."""
    if existing is None:
        return False, "no existing result"
    if existing.get("status") not in ("PASS", "SKIPPED"):
        return False, "existing result is FAIL"
    if not os.path.exists(existing.get("output_video", "")):
        return False, "output video missing"

    # Verify input hash hasn't changed
    input_hash = sha256_file(REPO_ROOT / scene["image"])
    if existing.get("input_image_sha256") != input_hash:
        return False, "input image hash changed"

    # Verify prompt hash hasn't changed
    prompt_file = EVAL_DIR / scene["id"] / f"prompt_{variant}.txt"
    if prompt_file.exists():
        prompt_hash = sha256_file(prompt_file)
        if existing.get("prompt_sha256") != prompt_hash:
            return False, "prompt hash changed"

    # Verify config matches
    if existing.get("frame_num") != cfg["generation"]["frame_num"]:
        return False, "frame_num changed"
    if existing.get("seed") != seed:
        return False, "seed changed"

    return True, "hash match, skipping"


def generate_one(
    scene: dict[str, Any],
    variant: str,
    seed: int,
    world: WorldDescription,
    prompt_info: dict[str, Any],
    cfg: dict[str, Any],
    existing_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Generate one video and return its manifest entry."""
    key = f"{scene['id']}|{variant}|{seed}"
    existing = existing_results.get(key)

    skip, reason = should_skip(existing, scene, variant, seed, cfg)
    if skip:
        log.info("SKIP %s (seed=%d): %s", scene["id"], seed, reason)
        existing["status"] = "SKIPPED"
        return existing

    log.info("GEN %s/%s (seed=%d) ...", scene["id"], variant, seed)

    scene_dir = EVAL_DIR / scene["id"]
    videos_dir = scene_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    work_dir = CACHE_DIR / f"{scene['id']}_{variant}_{seed}"
    work_dir.mkdir(parents=True, exist_ok=True)

    output_video = str(videos_dir / f"{variant}_seed{seed}.mp4")
    prompt_file = prompt_info[variant]["prompt_file"]
    prompt_embeds_file = str(CACHE_DIR / f"prompt_embeds_{scene['id']}_{variant}.safetensors")
    image_condition_file = str(CACHE_DIR / f"image_condition_{scene['id']}.safetensors")
    latents_file = str(work_dir / "latents.safetensors")

    # Remove old output if exists (re-generating)
    if os.path.exists(output_video):
        os.remove(output_video)

    gen_start = time.time()
    total_mps_peak = 0
    total_driver_peak = 0

    # Step 1: Encode prompt embeddings (independent UMT5 subprocess, cached per scene+variant)
    enc_ok, enc_meta = encode_prompt_embedding(prompt_file, prompt_embeds_file, cfg,
                                                  variant=variant, scene=scene["id"])
    if not enc_ok:
        return _fail_entry(scene, variant, seed, prompt_info, cfg,
                           "prompt encoding failed",
                           failure_kind="prompt_encode_failed",
                           return_code=enc_meta.get("return_code"),
                           stderr_tail=enc_meta.get("stderr_tail", ""))

    # Step 2: Encode image condition (cached per scene)
    if not os.path.exists(image_condition_file):
        log.info("  Encoding image condition (VAE)...")
        sr = run_stage("encode-image", cfg, scene, variant, seed,
                       prompt_file, prompt_embeds_file, image_condition_file,
                       latents_file, output_video, work_dir)
        total_mps_peak = max(total_mps_peak, sr.get("peak_mps_allocated_bytes", 0))
        total_driver_peak = max(total_driver_peak, sr.get("peak_driver_allocated_bytes", 0))
        if sr.get("status") != "PASS":
            return _fail_entry(scene, variant, seed, prompt_info, cfg, f"encode-image failed: {sr.get('error')}")
    else:
        log.info("  Image condition cached: %s", image_condition_file)

    # Step 3: Generate latents (DiT)
    log.info("  Generating latents (DiT)...")
    sr = run_stage("generate-latents", cfg, scene, variant, seed,
                   prompt_file, prompt_embeds_file, image_condition_file,
                   latents_file, output_video, work_dir)
    total_mps_peak = max(total_mps_peak, sr.get("peak_mps_allocated_bytes", 0))
    total_driver_peak = max(total_driver_peak, sr.get("peak_driver_allocated_bytes", 0))
    if sr.get("status") != "PASS":
        return _fail_entry(scene, variant, seed, prompt_info, cfg, f"generate-latents failed: {sr.get('error')}")

    # Step 4: Decode latents to video (VAE)
    log.info("  Decoding latents to video (VAE)...")
    sr = run_stage("decode", cfg, scene, variant, seed,
                   prompt_file, prompt_embeds_file, image_condition_file,
                   latents_file, output_video, work_dir)
    total_mps_peak = max(total_mps_peak, sr.get("peak_mps_allocated_bytes", 0))
    total_driver_peak = max(total_driver_peak, sr.get("peak_driver_allocated_bytes", 0))
    if sr.get("status") != "PASS":
        return _fail_entry(scene, variant, seed, prompt_info, cfg, f"decode failed: {sr.get('error')}")

    gen_time = round(time.time() - gen_start, 2)

    if not os.path.exists(output_video):
        return _fail_entry(scene, variant, seed, prompt_info, cfg, "output video not found after decode")

    # Compute sanity metrics
    log.info("  Computing sanity metrics...")
    sanity = compute_sanity_metrics(output_video)

    # Build manifest entry
    wc_file = scene_dir / "world_condition.json"
    entry = {
        "scene_id": scene["id"],
        "variant": variant,
        "seed": seed,
        "status": "PASS",
        "input_image": str(REPO_ROOT / scene["image"]),
        "input_image_sha256": sha256_file(REPO_ROOT / scene["image"]),
        "prompt_file": prompt_file,
        "prompt_sha256": sha256_file(prompt_file),
        "world_condition_file": str(wc_file),
        "world_condition_sha256": sha256_file(wc_file) if wc_file.exists() else None,
        "prompt_token_count": prompt_info[variant]["token_count"],
        "truncated": prompt_info[variant]["truncated"],
        "checkpoint_id": os.path.basename(cfg["paths"]["checkpoint_dir"]),
        "baseline_commit": cfg["baseline_commit"],
        "frame_num": cfg["generation"]["frame_num"],
        "chunk_size": cfg["generation"]["chunk_size"],
        "requested_size": cfg["generation"]["size"],
        "actual_size": sanity.get("resolution"),
        "generation_time": gen_time,
        "peak_mps_allocated": total_mps_peak,
        "peak_driver_allocated": total_driver_peak,
        "output_video": output_video,
        "output_sha256": sha256_file(output_video),
        "sanity_metrics": sanity,
    }

    log.info("  DONE in %.1fs, %s, %s frames",
             gen_time,
             sanity.get("resolution", "?"),
             sanity.get("frame_count", "?"))
    return entry


def _fail_entry(
    scene: dict[str, Any],
    variant: str,
    seed: int,
    prompt_info: dict[str, Any],
    cfg: dict[str, Any],
    error: str,
    failure_kind: str = "generation_failed",
    return_code: int | None = None,
    signal: int | None = None,
    stderr_tail: str = "",
) -> dict[str, Any]:
    wc_file = EVAL_DIR / scene["id"] / "world_condition.json"
    return {
        "scene_id": scene["id"],
        "variant": variant,
        "seed": seed,
        "status": "FAIL",
        "failure_kind": failure_kind,
        "error": error,
        "return_code": return_code,
        "signal": signal,
        "stderr_tail": stderr_tail[:3000] if stderr_tail else "",
        "input_image": str(REPO_ROOT / scene["image"]),
        "input_image_sha256": sha256_file(REPO_ROOT / scene["image"]),
        "prompt_file": prompt_info[variant]["prompt_file"],
        "prompt_sha256": sha256_file(prompt_info[variant]["prompt_file"]),
        "world_condition_file": str(wc_file),
        "world_condition_sha256": sha256_file(wc_file) if wc_file.exists() else None,
        "prompt_token_count": prompt_info[variant]["token_count"],
        "truncated": prompt_info[variant]["truncated"],
        "checkpoint_id": os.path.basename(cfg["paths"]["checkpoint_dir"]),
        "baseline_commit": cfg["baseline_commit"],
        "frame_num": cfg["generation"]["frame_num"],
        "chunk_size": cfg["generation"]["chunk_size"],
        "requested_size": cfg["generation"]["size"],
        "output_video": None,
        "output_sha256": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="G0.7 A/B/C evaluation runner.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only run single_subject scene, A/B/C, seed=42.")
    parser.add_argument("--scene", type=str, default=None,
                        help="Filter to a single scene id.")
    parser.add_argument("--variant", type=str, default=None, choices=["A", "B", "C"],
                        help="Filter to a single variant.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Filter to a single seed.")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM world condition generation (use existing cache).")
    args = parser.parse_args()

    cfg = load_config()
    existing_results = read_existing_results()

    # Determine active scenes/variants/seeds
    scenes = cfg["scenes"]
    if args.dry_run:
        scenes = [s for s in scenes if s["id"] == "single_subject"]
        seeds = [42]
        variants = ["A", "B", "C"]
        log.info("DRY RUN: single_subject, A/B/C, seed=42 (3 videos)")
    else:
        if args.scene:
            scenes = [s for s in scenes if s["id"] == args.scene]
        seeds = [args.seed] if args.seed is not None else cfg["seeds"]
        variants = [args.variant] if args.variant else cfg["variants"]

    total = len(scenes) * len(variants) * len(seeds)
    log.info("Plan: %d scenes × %d variants × %d seeds = %d videos",
             len(scenes), len(variants), len(seeds), total)

    # Clear metrics_prompt for this run (rebuild fresh)
    if METRICS_PROMPT_PATH.exists():
        METRICS_PROMPT_PATH.unlink()

    # Phase 1: VLM world condition + prompt building for all scenes
    all_prompt_info: dict[str, dict[str, Any]] = {}
    all_worlds: dict[str, WorldDescription] = {}

    for scene in scenes:
        log.info("=== Scene: %s ===", scene["id"])
        if args.skip_vlm:
            from world_condition import load_world_condition
            wc_path = EVAL_DIR / scene["id"] / "world_condition.json"
            world = load_world_condition(str(wc_path)) if wc_path.exists() else WorldDescription()
        else:
            world, _ = generate_world_condition(scene, cfg)
        all_worlds[scene["id"]] = world

        prompt_info = write_prompts_and_metrics(scene, world, cfg)
        all_prompt_info[scene["id"]] = prompt_info

    # Phase 2: Generate videos (each prompt encode and video gen is independent subprocess)
    log.info("\n=== Generation Phase ===")
    pass_count = 0
    fail_count = 0
    skip_count = 0

    for scene in scenes:
        for variant in variants:
            for seed in seeds:
                entry = generate_one(
                    scene, variant, seed,
                    all_worlds[scene["id"]],
                    all_prompt_info[scene["id"]],
                    cfg,
                    existing_results,
                )
                key = f"{scene['id']}|{variant}|{seed}"
                existing_results[key] = entry

                if entry["status"] == "PASS":
                    pass_count += 1
                elif entry["status"] == "SKIPPED":
                    skip_count += 1
                else:
                    fail_count += 1

                # Write incrementally after each video
                write_results_all(existing_results)

    log.info("\n=== Summary ===")
    log.info("PASS: %d, FAIL: %d, SKIPPED: %d, Total: %d",
             pass_count, fail_count, skip_count, total)
    log.info("Results: %s", RESULTS_PATH)
    log.info("Prompt metrics: %s", METRICS_PROMPT_PATH)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
