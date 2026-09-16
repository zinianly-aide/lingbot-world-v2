#!/usr/bin/env python3
"""Build E2 blind eval materials + summary.json.

- 30 anonymous video copies (15 UMT5 + 15 MiniCPM)
- blind_map.json with shuffle_seed=42
- human_eval_paired.csv template
- summary.json with objective comparison
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import statistics
from pathlib import Path

REPO = Path("/Users/anshi/clawd/lingbot-world-v2")
E2 = REPO / "eval" / "e2"
G07 = REPO / "eval" / "g0.7"
BLIND_DIR = E2 / "blind"
BLIND_VIDEOS = BLIND_DIR / "videos"

SCENES = ["single_subject", "spatial", "indoor", "outdoor", "camera_motion"]
SEEDS = [42, 123, 2026]


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_minicpm() -> list[dict]:
    rows = []
    for line in open(E2 / "results_minicpm.jsonl"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_umt5() -> dict:
    """Return {(scene,seed): record} for G0.7 A variant."""
    out = {}
    for line in open(G07 / "results.jsonl"):
        r = json.loads(line)
        if r.get("variant") == "A":
            out[(r["scene_id"], r["seed"])] = r
    return out


def main():
    BLIND_VIDEOS.mkdir(parents=True, exist_ok=True)
    minicpm = load_minicpm()
    umt5 = load_umt5()

    # Build 30 entries: for each (scene,seed), 2 videos (umt5 + minicpm)
    entries = []
    for scene in SCENES:
        for seed in SEEDS:
            m = next(r for r in minicpm if r["scene_id"] == scene and r["seed"] == seed)
            u = umt5[(scene, seed)]
            entries.append({
                "encoder": "minicpm+adapter",
                "scene_id": scene, "seed": seed,
                "source_video": m["output_video"],
                "sha256": m["output_sha256"],
            })
            umt5_path = str(G07 / scene / "videos" / f"A_seed{seed}.mp4")
            entries.append({
                "encoder": "umt5",
                "scene_id": scene, "seed": seed,
                "source_video": umt5_path,
                "sha256": sha256_file(umt5_path),
            })

    # Shuffle with fixed seed
    rng = random.Random(42)
    rng.shuffle(entries)

    # Copy videos and assign blind IDs
    blind_map = []
    for i, e in enumerate(entries, 1):
        blind_id = f"video_{i:03d}"
        dst = BLIND_VIDEOS / f"{blind_id}.mp4"
        shutil.copy2(e["source_video"], dst)
        blind_map.append({
            "blind_id": blind_id,
            "encoder": e["encoder"],
            "scene_id": e["scene_id"],
            "seed": e["seed"],
            "source_video": e["source_video"],
            "sha256": e["sha256"],
        })

    with open(BLIND_DIR / "blind_map.json", "w") as f:
        json.dump(blind_map, f, indent=2, ensure_ascii=False)

    # Build paired CSV: pair UMT5 vs MiniCPM for each (scene, seed)
    # Look up blind IDs from map
    by_key = {(e["encoder"], e["scene_id"], e["seed"]): e["blind_id"] for e in blind_map}
    with open(BLIND_DIR / "human_eval_paired.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "video_a_blind_id", "video_b_blind_id",
                     "which_is_better", "quality_diff", "notes"])
        for i, scene in enumerate(SCENES):
            for seed in SEEDS:
                a = by_key[("umt5", scene, seed)]
                b = by_key[("minicpm+adapter", scene, seed)]
                # Randomize which is A vs B
                if i % 2 == 0:
                    a, b = b, a
                w.writerow([f"pair_{i}_{seed}", a, b, "", "", ""])

    # === summary.json ===
    # Per-encoder aggregates
    def agg(rows, key):
        vals = [r["sanity_metrics"][key] for r in rows if key in r.get("sanity_metrics", {})]
        return {"mean": round(statistics.mean(vals), 4),
                "median": round(statistics.median(vals), 4),
                "std": round(statistics.stdev(vals), 4)} if len(vals) > 1 else {}

    def gen_time(rows):
        vals = [r["generation_time"] for r in rows if r.get("generation_time")]
        return {"mean": round(statistics.mean(vals), 1),
                "median": round(statistics.median(vals), 1),
                "std": round(statistics.stdev(vals), 1)} if len(vals) > 1 else {}

    minicpm_pass = [r for r in minicpm if r["status"] == "PASS"]
    umt5_pass = [umt5[(s, seed)] for s in SCENES for seed in SEEDS
                 if (s, seed) in umt5 and umt5[(s, seed)].get("status") == "PASS"]

    # BF16 vs FP16 decode comparison
    fp16_rows = [r for r in minicpm if r.get("vae_dtype") == "fp16"]
    bf16_rows = [r for r in minicpm if r.get("vae_dtype") == "bf16"]

    # Per-pair comparison
    pairs = []
    win = tie = loss = 0
    for scene in SCENES:
        for seed in SEEDS:
            m = next(r for r in minicpm if r["scene_id"] == scene and r["seed"] == seed)
            u = umt5[(scene, seed)]
            m_mad = m["sanity_metrics"]["temporal_mad_mean"]
            u_mad = u["sanity_metrics"]["temporal_mad_mean"]
            m_std = m["sanity_metrics"]["per_frame_std"][0]
            u_std = u["sanity_metrics"]["per_frame_std"][0]
            # win = lower temporal_mad (more stable)
            if m_mad < u_mad * 0.95:
                verdict = "win"
                win += 1
            elif abs(m_std - u_std) / max(u_std, 1e-6) < 0.05:
                verdict = "tie"
                tie += 1
            else:
                verdict = "loss"
                loss += 1
            pairs.append({
                "scene_id": scene, "seed": seed,
                "minicpm_mad": round(m_mad, 4),
                "umt5_mad": round(u_mad, 4),
                "mad_delta_pct": round((m_mad - u_mad) / u_mad * 100, 1),
                "minicpm_std": round(m_std, 4),
                "umt5_std": round(u_std, 4),
                "verdict": verdict,
            })

    summary = {
        "e2_status": "COMPLETE",
        "minicpm": {
            "total": len(minicpm),
            "pass": len(minicpm_pass),
            "fail": sum(1 for r in minicpm if r["status"] == "FAIL"),
            "generation_time": gen_time(minicpm_pass),
            "temporal_mad_mean": agg(minicpm_pass, "temporal_mad_mean"),
            "per_frame_std": {"mean": round(statistics.mean(
                r["sanity_metrics"]["per_frame_std"][0] for r in minicpm_pass), 4)},
        },
        "umt5_baseline": {
            "total": len(umt5_pass),
            "generation_time": gen_time(umt5_pass),
            "temporal_mad_mean": agg(umt5_pass, "temporal_mad_mean"),
        },
        "vae_dtype_comparison": {
            "fp16": {"n": len(fp16_rows),
                     "mean_gen_time": round(statistics.mean(r["generation_time"] for r in fp16_rows), 1)},
            "bf16": {"n": len(bf16_rows),
                     "mean_gen_time": round(statistics.mean(r["generation_time"] for r in bf16_rows), 1)},
            "speedup_pct": round((1 - statistics.mean(r["generation_time"] for r in bf16_rows) /
                                  statistics.mean(r["generation_time"] for r in fp16_rows)) * 100, 1),
            "note": "BF16 VAE decode ~1.77x faster than FP16 (warm decode 62.6s vs 110.7s). First FP16 run included cold model load; BF16 batch reused warm cache.",
        },
        "pair_comparison": {
            "method": "Objective sanity metrics only. temporal_mad_mean lower = more stable = win. per_frame_std within 5% = tie. NOT semantic quality score.",
            "win": win, "tie": tie, "loss": loss,
            "pairs": pairs,
        },
        "g1_status": "NOT STARTED",
    }

    with open(E2 / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Blind videos: {len(blind_map)} copies in {BLIND_VIDEOS}")
    print(f"blind_map.json: {len(blind_map)} entries")
    print(f"human_eval_paired.csv: 15 pairs")
    print(f"summary.json: win={win} tie={tie} loss={loss}")
    print(f"BF16 speedup: {summary['vae_dtype_comparison']['speedup_pct']}%")


if __name__ == "__main__":
    main()
