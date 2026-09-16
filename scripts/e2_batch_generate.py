#!/usr/bin/env python3
"""E2 batch: generate 15 MiniCPM5+Adapter videos (5 scenes × 3 seeds).

Reuses:
  - G0.7 cached image conditions (eval/g0.7/.cache/image_condition_<scene>.safetensors)
  - Pre-computed MiniCPM embeddings (eval/e2/embeddings/<scene>_minicpm.safetensors)
  - Same generation params as G0.7 A variant

Each video = generate-latents (DiT) + decode (VAE), each in independent subprocess.
Supports resume: skips videos that already pass sanity checks.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.e1_encode_with_adapter import compute_sanity_metrics_from_frames  # noqa: E402

EVAL_DIR = REPO_ROOT / "eval" / "e2"
CACHE_DIR = EVAL_DIR / ".cache"
VIDEOS_DIR = EVAL_DIR / "videos"
EMBEDDINGS_DIR = EVAL_DIR / "embeddings"
RESULTS_PATH = EVAL_DIR / "results_minicpm.jsonl"

# G0.7 config (reuse paths)
CKPT_DIR = "/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/snapshots/7e36a5f919f86cb4255cc9bfc30adb44963fbde1"
ASSETS_DIR = "/Volumes/ssd/lingbot-assets"
CHECKPOINT_ID = "7e36a5f919f86cb4255cc9bfc30adb44963fbde1"

SCENES = {
    "single_subject": {
        "image": "examples/03/image.jpg",
        "action_path": "examples/03",
        "prompt_file": "eval/g0.7/single_subject/prompt_A.txt",
    },
    "spatial": {
        "image": "examples/01/image.jpg",
        "action_path": "examples/01",
        "prompt_file": "eval/g0.7/spatial/prompt_A.txt",
    },
    "indoor": {
        "image": "eval/g0.7/eval_assets/indoor/image.jpg",
        "action_path": "eval/g0.7/eval_assets/indoor",
        "prompt_file": "eval/g0.7/indoor/prompt_A.txt",
    },
    "outdoor": {
        "image": "examples/04/image.jpg",
        "action_path": "examples/04",
        "prompt_file": "eval/g0.7/outdoor/prompt_A.txt",
    },
    "camera_motion": {
        "image": "examples/00/image.jpg",
        "action_path": "examples/00",
        "prompt_file": "eval/g0.7/camera_motion/prompt_A.txt",
    },
}
SEEDS = [42, 123, 2026]

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("e2_batch")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_stage(stage: str, extra_args: list[str], work_dir: Path) -> dict:
    """Run one generate.py stage via g07_gen_stage.py wrapper."""
    result_json = work_dir / f"stage_{stage}.json"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "g07_gen_stage.py"),
        "--result-json", str(result_json),
        "--repo-root", str(REPO_ROOT),
        "--",
        "--ckpt_dir", CKPT_DIR,
        "--assets_dir", ASSETS_DIR,
        "--device", "mps",
        "--task", "i2v-1.3B",
        "--infer_mode", "causal_fast",
        "--frame_num", "13",
        "--size", "832*480",
        "--chunk_size", "4",
        "--local_attn_size", "-1",
        "--stage", stage,
    ] + extra_args

    log.info("  Stage %s: %s ...", stage, " ".join(cmd[:8]))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))

    stage_result = {"status": "FAIL", "error": "no result json", "elapsed_seconds": 0,
                    "peak_driver_allocated_bytes": 0}
    if result_json.exists():
        with open(result_json) as f:
            stage_result = json.load(f)
    if stage_result.get("status") != "PASS":
        log.error("  Stage %s FAILED: %s", stage, stage_result.get("error"))
        log.error("  stderr tail: %s", result.stderr[-500:] if result.stderr else "")
    return stage_result


def video_passes_sanity(video_path: str) -> tuple[bool, dict]:
    """Check if video exists and passes basic sanity."""
    if not os.path.exists(video_path):
        return False, {}
    try:
        import imageio.v3 as iio
        frames = iio.imread(video_path)
        m = compute_sanity_metrics_from_frames(frames)
        ok = (m["frame_count"] == 13 and not m["all_black"] and
              not m["all_white"] and not m["frozen_video"] and
              not m["nan_inf_corruption"])
        return ok, m
    except Exception as e:
        log.warning("  Sanity check error: %s", e)
        return False, {}


def load_existing_results() -> dict:
    results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results[f"{r['scene_id']}|{r['seed']}"] = r
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    scenes = [args.scene] if args.scene else list(SCENES.keys())
    seeds = [args.seed] if args.seed else SEEDS

    existing = load_existing_results()
    pass_count = fail_count = skip_count = 0

    for scene_id in scenes:
        scene = SCENES[scene_id]
        image_path = str(REPO_ROOT / scene["image"])
        action_path = str(REPO_ROOT / scene["action_path"])
        prompt_file = str(REPO_ROOT / scene["prompt_file"])

        with open(prompt_file) as f:
            prompt_text = f.read().strip()

        embed_path = str(EMBEDDINGS_DIR / f"{scene_id}_minicpm.safetensors")
        embed_json = embed_path.replace(".safetensors", ".json")
        with open(embed_json) as f:
            embed_meta = json.load(f)

        # G0.7 cached image condition
        image_cond = str(REPO_ROOT / "eval" / "g0.7" / ".cache" /
                         f"image_condition_{scene_id}.safetensors")
        assert os.path.isfile(image_cond), f"Image condition missing: {image_cond}"

        for seed in seeds:
            key = f"{scene_id}|{seed}"
            output_video = str(VIDEOS_DIR / f"{scene_id}_seed{seed}_minicpm.mp4")

            # Resume check
            if key in existing and existing[key].get("status") == "PASS":
                ok, _ = video_passes_sanity(output_video)
                if ok:
                    log.info("SKIP %s (seed=%d): already PASS", scene_id, seed)
                    skip_count += 1
                    continue

            log.info("GEN %s (seed=%d) ...", scene_id, seed)
            work_dir = CACHE_DIR / f"{scene_id}_{seed}"
            work_dir.mkdir(parents=True, exist_ok=True)
            latents_file = str(work_dir / "latents.safetensors")

            base_args = [
                "--image", image_path,
                "--action_path", action_path,
                "--base_seed", str(seed),
                "--prompt", prompt_text,
            ]

            gen_start = time.time()
            total_peak_driver = 0

            # Stage 1: generate-latents (reuse cached image condition)
            sr = run_stage("generate-latents", base_args + [
                "--prompt_embeds_file", embed_path,
                "--image_condition_file", image_cond,
                "--output_latents_file", latents_file,
            ], work_dir)
            total_peak_driver = max(total_peak_driver, sr.get("peak_driver_allocated_bytes", 0))
            if sr.get("status") != "PASS":
                rec = {
                    "scene_id": scene_id, "seed": seed, "status": "FAIL",
                    "failure_kind": "generate_latents_failed",
                    "error": sr.get("error"),
                    "stderr_tail": "",
                }
                with open(RESULTS_PATH, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                fail_count += 1
                continue

            # Stage 2: decode
            sr = run_stage("decode", base_args + [
                "--latents_file", latents_file,
                "--save_file", output_video,
            ], work_dir)
            total_peak_driver = max(total_peak_driver, sr.get("peak_driver_allocated_bytes", 0))
            gen_time = round(time.time() - gen_start, 2)

            if sr.get("status") != "PASS" or not os.path.exists(output_video):
                rec = {
                    "scene_id": scene_id, "seed": seed, "status": "FAIL",
                    "failure_kind": "decode_failed",
                    "error": sr.get("error") or "video not found",
                    "generation_time": gen_time,
                    "peak_driver_allocated": total_peak_driver,
                }
                with open(RESULTS_PATH, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                fail_count += 1
                continue

            # Sanity metrics
            ok, sanity = video_passes_sanity(output_video)
            status = "PASS" if ok else "FAIL"
            if not ok:
                rec["failure_kind"] = "sanity_check_failed"

            rec = {
                "scene_id": scene_id, "seed": seed, "status": status,
                "vae_dtype": os.environ.get("LINGBOT_VAE_DTYPE", "fp16"),
                "prompt_text": prompt_text,
                "prompt_sha256": embed_meta.get("prompt_sha256"),
                "context_file": embed_path,
                "context_sha256": None,  # filled from encode result if available
                "context_shape": embed_meta.get("shape"),
                "context_dtype": embed_meta.get("dtype"),
                "encode_wall_time": None,
                "encode_peak_mps_mb": None,
                "input_image": image_path,
                "input_image_sha256": sha256_file(image_path),
                "checkpoint_id": CHECKPOINT_ID,
                "frame_num": 13, "chunk_size": 4,
                "requested_size": "832*480",
                "actual_size": sanity.get("resolution"),
                "generation_time": gen_time,
                "peak_driver_allocated": total_peak_driver,
                "output_video": output_video,
                "output_sha256": sha256_file(output_video),
                "sanity_metrics": sanity,
            }

            # Fill encode metrics from encode result JSON
            enc_result_path = EMBEDDINGS_DIR / f"{scene_id}_encode_result.json"
            if enc_result_path.exists():
                with open(enc_result_path) as f:
                    enc = json.load(f)
                rec["encode_wall_time"] = enc.get("encode_wall_time_seconds")
                rec["encode_peak_mps_mb"] = enc.get("peak_mps_mb")
                rec["context_sha256"] = enc.get("context_sha256")

            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")

            if status == "PASS":
                pass_count += 1
                log.info("  DONE in %.1fs, %s, mad_mean=%.4f",
                         gen_time, sanity.get("resolution"),
                         sanity.get("temporal_mad_mean", 0))
            else:
                fail_count += 1

    log.info("=== Summary: PASS=%d FAIL=%d SKIP=%d ===", pass_count, fail_count, skip_count)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
