#!/usr/bin/env python3
"""Summarize E2-new paired blind eval from human_eval_paired.csv.

Usage:
    python scripts/summarize_e2_new_paired.py \
        --csv eval/e2_new/blind/human_eval_paired.csv \
        --map eval/e2_new/blind/blind_map.json

Reads the scored CSV (0-5 per dimension + win/tie/loss per pair), maps anon
video ids back to scene/seed/encoder via blind_map.json, and prints:
  - overall win/tie/loss (UMT5 vs MiniCPM+E1.2)
  - per-scene and per-seed win/tie/loss
  - mean per-dimension scores (left/right both summed then averaged by encoder)
Dry mode: if CSV has no scores, reports how many pairs are missing.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIMS = ["semantic_fidelity", "spatial_consistency", "camera_motion",
        "temporal_stability", "hallucination", "overall_preference"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="eval/e2_new/blind/human_eval_paired.csv")
    ap.add_argument("--map", default="eval/e2_new/blind/blind_map.json")
    args = ap.parse_args()

    bm = json.load(open(REPO_ROOT / args.map))
    vmap = bm["videos"]  # anon_noext? key is video_001.mp4
    pairs = {p["pair_id"]: p for p in bm["pairs"]}

    rows = list(csv.DictReader(open(REPO_ROOT / args.csv)))
    scored = [r for r in rows if (r.get("overall_preference") or "").strip() or (r.get("win") or "").strip()]
    print(f"Pairs total: {len(rows)}, scored: {len(scored)}")
    if not scored:
        print("DRY: no scores yet. Fill human_eval_paired.csv then re-run.")
        return

    # For each scored pair, determine which encoder won.
    # win column: user fills "left" / "right" / "tie".  We map left/right to encoders.
    wlt = {"umt5": 0, "e12": 0, "tie": 0}
    by_scene = defaultdict(lambda: {"umt5": 0, "e12": 0, "tie": 0})
    by_seed = defaultdict(lambda: {"umt5": 0, "e12": 0, "tie": 0})
    dim_scores = defaultdict(list)  # encoder -> [dim values]

    for r in scored:
        pid = r["pair_id"]
        p = pairs[pid]
        left_enc = vmap[p["left_video"] + ".mp4"]["encoder"]
        right_enc = vmap[p["right_video"] + ".mp4"]["encoder"]
        scene = vmap[p["left_video"] + ".mp4"]["scene"]
        seed = vmap[p["left_video"] + ".mp4"]["seed"]
        w = (r.get("win") or "").strip().lower()
        if w == "left":
            winner = left_enc
        elif w == "right":
            winner = right_enc
        else:
            winner = "tie"
        wlt[winner] += 1
        by_scene[scene][winner] += 1
        by_seed[str(seed)][winner] += 1
        # dimension scores: average left+right per encoder
        for d in DIMS:
            v = (r.get(d) or "").strip()
            if v:
                dim_scores[left_enc].append(float(v))
                dim_scores[right_enc].append(float(v))

    print("\n=== Overall win/tie/loss (UMT5 vs MiniCPM+E1.2) ===")
    tot = sum(wlt.values())
    print(f"  UMT5 wins: {wlt['umt5']}  E1.2 wins: {wlt['e12']}  ties: {wlt['tie']}  (n={tot})")
    print("\n=== Per-scene ===")
    for s, v in sorted(by_scene.items()):
        print(f"  {s:16s} umt5={v['umt5']} e12={v['e12']} tie={v['tie']}")
    print("\n=== Per-seed ===")
    for s, v in sorted(by_seed.items()):
        print(f"  seed={s:6s} umt5={v['umt5']} e12={v['e12']} tie={v['tie']}")
    print("\n=== Mean per-dimension (by encoder) ===")
    for enc in ["umt5", "e12"]:
        if dim_scores[enc]:
            print(f"  {enc}: n={len(dim_scores[enc])} mean={sum(dim_scores[enc])/len(dim_scores[enc]):.2f}")


if __name__ == "__main__":
    main()
