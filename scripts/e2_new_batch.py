#!/usr/bin/env python3
"""E2-new full paired eval: 5 scenes x 3 seeds = 15 E1.2 videos.

Reuses 4 already-generated videos (gate3 seed42 x3 + single_subject smoke seed42),
generates the other 11 with staged g07_gen_stage.py (BF16 VAE, seed=given,
13 frames, 832*480, G0.7 image_condition cache).  Records objective stats.
UMT5 baselines are NOT touched (reused from eval/g0.7/.../A_seed*.mp4).
"""
import json
import os
import shutil
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
    "single_subject": {"image": "examples/03/image.jpg", "action_path": "examples/03"},
    "spatial":        {"image": "examples/01/image.jpg", "action_path": "examples/01"},
    "camera_motion":  {"image": "examples/00/image.jpg", "action_path": "examples/00"},
    "indoor":         {"image": "eval/g0.7/eval_assets/indoor/image.jpg",
                       "action_path": "eval/g0.7/eval_assets/indoor"},
    "outdoor":        {"image": "examples/04/image.jpg", "action_path": "examples/04"},
}
SEEDS = [42, 123, 2026]

# Already-generated E1.2 videos to reuse (adapter SHA + params identical).
REUSE = {
    ("single_subject", 42): "eval/e1.2/smoke/single_subject_seed42_e12.mp4",
    ("spatial", 42):        "eval/e1.2/gate3/videos/spatial_seed42_e12.mp4",
    ("camera_motion", 42):  "eval/e1.2/gate3/videos/camera_motion_seed42_e12.mp4",
    ("indoor", 42):         "eval/e1.2/gate3/videos/indoor_seed42_e12.mp4",
}

OUT = REPO_ROOT / "eval/e2_new"
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
        print(f"    STAGE {stage} FAIL: {res.get('error')}\n    {(r.stderr or '')[-400:]}")
    return res


def video_stats(path):
    fr = iio.imread(str(path)).astype(np.float32) / 255.0
    d = np.abs(fr[1:] - fr[:-1])
    return {"frame_count": int(fr.shape[0]), "h": int(fr.shape[1]), "w": int(fr.shape[2]),
            "global_mean": round(float(fr.mean()), 4),
            "global_std": round(float(fr.std()), 4),
            "temporal_mad": round(float(d.mean()), 4)}, fr


records = []
for scene, cfg in SCENES.items():
    prompt = open(REPO_ROOT / f"eval/g0.7/{scene}/prompt_A.txt").read().strip()
    image = str(REPO_ROOT / cfg["image"])
    action = str(REPO_ROOT / cfg["action_path"])
    embed = str(REPO_ROOT / f"eval/e1.2/embeddings/{scene}_e12.safetensors")
    imgcond = str(REPO_ROOT / f"eval/g0.7/.cache/image_condition_{scene}.safetensors")

    for seed in SEEDS:
        vout = OUT / "videos" / f"{scene}_seed{seed}_e12.mp4"
        rec = {"scene": scene, "seed": seed, "video": str(vout), "encoder": "minicpm5+e1.2_adapter"}
        reused = False

        if (scene, seed) in REUSE:
            src = REPO_ROOT / REUSE[(scene, seed)]
            if src.exists():
                shutil.copyfile(src, vout)
                reused = True
                rec["reused_from"] = str(src.relative_to(REPO_ROOT))
                print(f"  REUSE {scene} seed{seed}")

        if not reused:
            work = OUT / "work" / f"{scene}_{seed}"; work.mkdir(parents=True, exist_ok=True)
            latents = str(work / "latents.safetensors")
            base = ["--image", image, "--action_path", action, "--base_seed", str(seed), "--prompt", prompt]
            t0 = time.time()
            s1 = run_stage("generate-latents", work, base + ["--prompt_embeds_file", embed,
                           "--image_condition_file", imgcond, "--output_latents_file", latents])
            peak = s1.get("peak_driver_allocated_bytes", 0)
            s2 = run_stage("decode", work, base + ["--latents_file", latents, "--save_file", str(vout)])
            peak = max(peak, s2.get("peak_driver_allocated_bytes", 0))
            rec["gen_time_s"] = round(time.time() - t0, 1)
            rec["peak_driver_bytes"] = peak
            rec["stage1"], rec["stage2"] = s1.get("status"), s2.get("status")
            print(f"  GEN   {scene} seed{seed}  ({rec.get('gen_time_s')}s)")

        if vout.exists():
            st, fr = video_stats(vout)
            rec.update(st)
            # frame samples 1/7/13
            for fn in [0, 6, 12]:
                Image.fromarray((fr[fn] * 255).astype(np.uint8)).save(
                    str(OUT / "frames" / f"{scene}_seed{seed}_frame{fn+1}.png"))
        records.append(rec)

with open(OUT / "results.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"DONE. {len(records)} records -> {OUT/'results.jsonl'}")
