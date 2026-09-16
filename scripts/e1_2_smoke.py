#!/usr/bin/env python3
"""E1.2 smoke: single_subject, seed=42, MiniCPM5 + E1.2 best adapter.

Mirrors scripts/e2_batch_generate.py staged call for the single_subject scene,
but uses the E1.2-encoded prompt embedding.  Outputs one mp4 + a stats JSON.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = "/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/snapshots/7e36a5f919f86cb4255cc9bfc30adb44963fbde1"
ASSETS_DIR = "/Volumes/ssd/lingbot-assets"

PROMPT = "Move the camera slowly forward while keeping the lone tree stable and centered."
IMAGE = str(REPO_ROOT / "examples/03/image.jpg")
ACTION_PATH = str(REPO_ROOT / "examples/03")
EMBED = str(REPO_ROOT / "eval/e1.2/embeddings/single_subject_minicpm.safetensors")
IMAGECOND = str(REPO_ROOT / "eval/g0.7/.cache/image_condition_single_subject.safetensors")
OUT_DIR = REPO_ROOT / "eval/e1.2/smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO = str(OUT_DIR / "single_subject_seed42_e12.mp4")


def run_stage(stage, extra):
    work = OUT_DIR / "work"
    work.mkdir(parents=True, exist_ok=True)
    rj = work / f"stage_{stage}.json"
    cmd = [sys.executable, str(REPO_ROOT / "scripts/g07_gen_stage.py"),
           "--result-json", str(rj), "--repo-root", str(REPO_ROOT), "--",
           "--ckpt_dir", CKPT_DIR, "--assets_dir", ASSETS_DIR, "--device", "mps",
           "--task", "i2v-1.3B", "--infer_mode", "causal_fast",
           "--frame_num", "13", "--size", "832*480", "--chunk_size", "4",
           "--local_attn_size", "-1", "--stage", stage] + extra
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    res = json.load(open(rj)) if rj.exists() else {"status": "FAIL", "error": "no json"}
    res["wall_s"] = round(time.time() - t0, 1)
    if res.get("status") != "PASS":
        print(f"  STAGE {stage} FAIL: {res.get('error')}")
        print("  stderr tail:", (r.stderr or "")[-800:])
    return res


base = ["--image", IMAGE, "--action_path", ACTION_PATH,
        "--base_seed", "42", "--prompt", PROMPT]
latents = str(OUT_DIR / "work" / "latents.safetensors")

print("Stage 1: generate-latents ...")
r1 = run_stage("generate-latents", base + ["--prompt_embeds_file", EMBED,
               "--image_condition_file", IMAGECOND, "--output_latents_file", latents])
print("  ->", r1.get("status"), r1.get("wall_s"), "s")

print("Stage 2: decode ...")
r2 = run_stage("decode", base + ["--latents_file", latents, "--save_file", VIDEO])
print("  ->", r2.get("status"), r2.get("wall_s"), "s")

# stats
stats = {"stage1": r1, "stage2": r2, "video": VIDEO,
         "exists": os.path.exists(VIDEO)}
if os.path.exists(VIDEO):
    import imageio.v3 as iio
    import numpy as np
    frames = iio.imread(VIDEO).astype(np.float32) / 255.0
    diffs = np.abs(frames[1:] - frames[:-1])
    stats.update({"frame_count": int(frames.shape[0]),
                  "h": int(frames.shape[1]), "w": int(frames.shape[2]),
                  "global_mean": float(frames.mean()),
                  "global_std": float(frames.std()),
                  "temporal_mad": float(diffs.mean())})
    # save a 3-frame thumb
    from PIL import Image
    idxs = [0, frames.shape[0]//2, frames.shape[0]-1]
    combo = np.concatenate([(frames[i]*255).astype(np.uint8) for i in idxs], axis=1)
    Image.fromarray(combo).save(str(OUT_DIR / "smoke_thumb.png"))
json.dump(stats, open(OUT_DIR / "smoke_stats.json", "w"), indent=2)
print(json.dumps({k: stats[k] for k in ["frame_count","global_mean","global_std","temporal_mad"] if k in stats}, indent=2))
