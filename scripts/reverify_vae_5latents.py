#!/usr/bin/env python3
"""Re-verify MLX BF16 VAE decode vs MPS BF16 across 5 scenes.

For each scene's cached DiT latent:
  1. Run bench_vae_mps.py  (MPS BF16, warmup+3 median) -> reference .npy + result json
  2. Run bench_vae_mlx.py  (MLX BF16, warmup+3 median)   -> result json (error vs ref)
Each stage runs in its own subprocess to avoid MPS/MLX contention.

Writes eval/bench_vae_mlx/reverify_5latents.json with per-scene rows + summary.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCENES = ["single_subject", "spatial", "indoor", "outdoor", "camera_motion"]
WORK = REPO_ROOT / "eval/bench_vae_mlx/reverify"
OUT_JSON = REPO_ROOT / "eval/bench_vae_mlx/reverify_5latents.json"


def run(cmd, log_path):
    print(f"  $ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    with open(log_path, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    wall = time.perf_counter() - t0
    return r.returncode, wall


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in SCENES:
        latent = REPO_ROOT / f"eval/e2/.cache/{scene}_42/latents.safetensors"
        print(f"\n=== {scene} ({latent}) ===", flush=True)
        if not latent.exists():
            rows.append({"scene": scene, "status": "FAIL", "error": f"missing latent {latent}"})
            continue

        mps_video = WORK / f"{scene}_mps.npy"
        mps_result = WORK / f"{scene}_mps.json"
        mlx_result = WORK / f"{scene}_mlx.json"

        # 1) MPS BF16 reference
        rc, wall = run([
            sys.executable, "scripts/bench_vae_mps.py",
            "--latent", str(latent),
            "--out-video", str(mps_video),
            "--result", str(mps_result),
        ], WORK / f"{scene}_mps.log")
        if rc != 0:
            rows.append({"scene": scene, "status": "FAIL", "stage": "mps", "log": str(WORK / f"{scene}_mps.log")})
            continue
        mps = json.loads(mps_result.read_text())

        # 2) MLX BF16 (error vs MPS ref)
        rc, wall = run([
            sys.executable, "scripts/bench_vae_mlx.py",
            "--dtype", "bf16",
            "--latent", str(latent),
            "--ref", str(mps_video),
            "--result", str(mlx_result),
        ], WORK / f"{scene}_mlx.log")
        if rc != 0:
            rows.append({"scene": scene, "status": "FAIL", "stage": "mlx", "log": str(WORK / f"{scene}_mlx.log")})
            continue
        mlx = json.loads(mlx_result.read_text())

        speedup = mps["infer_median_s"] / mlx["infer_median_s"]
        rows.append({
            "scene": scene,
            "status": "OK",
            "resolution": f"{mps['output_shape'][-1]}x{mps['output_shape'][-2]}",
            "mps_median_s": mps["infer_median_s"],
            "mps_peak_driver_mb": mps["peak_driver_mb"],
            "mlx_median_s": mlx["infer_median_s"],
            "mlx_peak_rss_mb": mlx["peak_rss_mb"],
            "mlx_metal_cache_mb": mlx.get("metal_cache_mb"),
            "max_abs_error": mlx["error_vs_mps_bf16"]["max_abs_error"],
            "mean_abs_error": mlx["error_vs_mps_bf16"]["mean_abs_error"],
            "speedup_mps_over_mlx": round(speedup, 3),
            "pct_faster": round((1.0 - mlx["infer_median_s"] / mps["infer_median_s"]) * 100, 1),
            "sanity_ok": (not mlx["sanity"]["all_black"] and not mlx["sanity"]["all_white"]
                          and not mlx["sanity"]["frozen"]),
            "mps_temporal_mad": mps["temporal_mad"],
            "mlx_temporal_mad": mlx["sanity"]["temporal_mad"],
        })

    # Summary over OK rows
    ok = [r for r in rows if r.get("status") == "OK"]
    if ok:
        summary = {
            "n_ok": len(ok),
            "avg_mps_median_s": round(sum(r["mps_median_s"] for r in ok) / len(ok), 3),
            "avg_mlx_median_s": round(sum(r["mlx_median_s"] for r in ok) / len(ok), 3),
            "avg_pct_faster": round(sum(r["pct_faster"] for r in ok) / len(ok), 1),
            "speedup_min": round(min(r["speedup_mps_over_mlx"] for r in ok), 3),
            "speedup_max": round(max(r["speedup_mps_over_mlx"] for r in ok), 3),
            "avg_max_abs_error": round(sum(r["max_abs_error"] for r in ok) / len(ok), 5),
            "avg_mean_abs_error": round(sum(r["mean_abs_error"] for r in ok) / len(ok), 5),
        }
    else:
        summary = {"n_ok": 0}

    out = {"scenes": rows, "summary": summary}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
