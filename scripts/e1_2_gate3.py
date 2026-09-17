#!/usr/bin/env python3
"""E1.2 gate3: 3 scenes (spatial/camera_motion/indoor) x seed42.

Staged generate-latents -> decode, BF16 VAE, reusing G0.7 image_condition
caches.  Writes videos + per-frame stats + frame{1,7,13} samples + a results
jsonl.  Does NOT touch DiT/text_embedding/UMT5/VAE architecture.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = "/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/snapshots/7e36a5f919f86cb4255cc9bfc30adb44963fbde1"
ASSETS_DIR = "/Volumes/ssd/lingbot-assets"

SCENES = {
    "spatial":       {"image": "examples/01/image.jpg", "action_path": "examples/01"},
    "camera_motion": {"image": "examples/00/image.jpg", "action_path": "examples/00"},
    "indoor":        {"image": "eval/g0.7/eval_assets/indoor/image.jpg",
                      "action_path": "eval/g0.7/eval_assets/indoor"},
}
OUT = REPO_ROOT / "eval/e1.2/gate3"
(OUT / "videos").mkdir(parents=True, exist_ok=True)
(OUT / "frames").mkdir(parents=True, exist_ok=True)


def run_stage(stage, work, extra):
    rj = work / f"stage_{stage}.json"
    cmd = [sys.executable, str(REPO_ROOT / "scripts/g07_gen_stage.py"),
           "--result-json", str(rj), "--repo-root", str(REPO_ROOT), "--",
           "--ckpt_dir", CKPT_DIR, "--assets_dir", ASSETS_DIR, "--device", "mps",
           "--task", "i2v-1.3B", "--infer_mode", "causal_fast",
           "--frame_num", "13", "--size", "832*480", "--chunk_size", "4",
           "--local_attn_size", "-1", "--stage", stage] + extra
    t0 = time.time()
    env = dict(os.environ); env["LINGBOT_VAE_DTYPE"] = "bf16"
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)
    res = json.load(open(rj)) if rj.exists() else {"status": "FAIL", "error": "no json"}
    res["wall_s"] = round(time.time() - t0, 1)
    if res.get("status") != "PASS":
        print(f"  STAGE {stage} FAIL: {res.get('error')}\n  {(r.stderr or '')[-500:]}")
    return res


results = []
for scene, cfg in SCENES.items():
    print(f"=== {scene} ===")
    prompt = open(REPO_ROOT / f"eval/g0.7/{scene}/prompt_A.txt").read().strip()
    image = str(REPO_ROOT / cfg["image"])
    action = str(REPO_ROOT / cfg["action_path"])
    embed = str(REPO_ROOT / f"eval/e1.2/embeddings/{scene}_e12.safetensors")
    imgcond = str(REPO_ROOT / f"eval/g0.7/.cache/image_condition_{scene}.safetensors")
    video = str(OUT / "videos" / f"{scene}_seed42_e12.mp4")
    work = OUT / "work" / scene; work.mkdir(parents=True, exist_ok=True)
    latents = str(work / "latents.safetensors")

    base = ["--image", image, "--action_path", action, "--base_seed", "42", "--prompt", prompt]
    t_start = time.time()
    s1 = run_stage("generate-latents", work, base + ["--prompt_embeds_file", embed,
                   "--image_condition_file", imgcond, "--output_latents_file", latents])
    peak = s1.get("peak_driver_allocated_bytes", 0)
    s2 = run_stage("decode", work, base + ["--latents_file", latents, "--save_file", video])
    peak = max(peak, s2.get("peak_driver_allocated_bytes", 0))
    gen_time = round(time.time() - t_start, 1)

    rec = {"scene": scene, "seed": 42, "video": video,
           "stage1": s1.get("status"), "stage2": s2.get("status"),
           "gen_time_s": gen_time, "peak_driver_bytes": peak}
    if os.path.exists(video):
        fr = iio.imread(video).astype(np.float32) / 255.0
        diffs = np.abs(fr[1:] - fr[:-1])
        rec.update({"frame_count": int(fr.shape[0]), "h": int(fr.shape[1]),
                    "w": int(fr.shape[2]),
                    "global_mean": round(float(fr.mean()), 4),
                    "global_std": round(float(fr.std()), 4),
                    "temporal_mad": round(float(diffs.mean()), 4)})
        for fnum in [0, 6, 12]:
            Image.fromarray((fr[fnum] * 255).astype(np.uint8)).save(
                str(OUT / "frames" / f"{scene}_frame{fnum+1}.png"))
    results.append(rec)
    print("  ->", {k: rec[k] for k in ["stage1", "stage2", "global_mean", "global_std", "temporal_mad"] if k in rec})

with open(OUT / "results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
json.dump(results, open(OUT / "results_raw.json", "w"), indent=2)
print("DONE")
