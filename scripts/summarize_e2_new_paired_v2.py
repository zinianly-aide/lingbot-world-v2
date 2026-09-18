#!/usr/bin/env python3
"""Summarize E2-new paired blind eval using per-side X/Y scores from JSON.

CSV win column: left/right/tie (left=X, right=Y).
JSON holds full *_X/*_Y dimension scores for true per-encoder means.

Usage:
    python scripts/summarize_e2_new_paired_v2.py
    python scripts/summarize_e2_new_paired_v2.py \
        --json eval/e2_new/blind/human_eval_paired.json \
        --csv eval/e2_new/blind/human_eval_paired.csv \
        --map eval/e2_new/blind/blind_map.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIMS = [
    "semantic_fidelity",
    "spatial_consistency",
    "camera_motion",
    "temporal_stability",
    "hallucination",
    "overall_preference",
]


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="eval/e2_new/blind/human_eval_paired.json")
    ap.add_argument("--csv", default="eval/e2_new/blind/human_eval_paired.csv")
    ap.add_argument("--map", default="eval/e2_new/blind/blind_map.json")
    args = ap.parse_args()

    bm = json.load(open(REPO_ROOT / args.map, encoding="utf-8"))
    vmap = bm["videos"]
    pairs = {str(p["pair_id"]): p for p in bm["pairs"]}

    scores = json.load(open(REPO_ROOT / args.json, encoding="utf-8"))
    csv_win: dict[str, str] = {}
    with open(REPO_ROOT / args.csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = (row.get("win") or "").strip().lower()
            if w:
                csv_win[row["pair_id"].strip()] = w

    wlt = {"umt5": 0, "e12": 0, "tie": 0}
    by_scene = defaultdict(lambda: {"umt5": 0, "e12": 0, "tie": 0})
    by_seed = defaultdict(lambda: {"umt5": 0, "e12": 0, "tie": 0})
    dim_scores: dict[str, dict[str, list[float]]] = {
        "umt5": {d: [] for d in DIMS},
        "e12": {d: [] for d in DIMS},
    }
    unblinded_rows = []

    for rec in scores:
        pid = rec["pair_id"]
        p = pairs.get(pid)
        if p is None:
            # allow pair_01 style keys in map
            alt = str(int(pid.split("_")[-1]))
            p = pairs.get(alt)
        if p is None:
            print(f"WARN: {pid} not in blind_map pairs, skip")
            continue

        left_id = rec.get("left_video") or p["left_video"]
        right_id = rec.get("right_video") or p["right_video"]
        left_key = left_id if left_id.endswith(".mp4") else left_id + ".mp4"
        right_key = right_id if right_id.endswith(".mp4") else right_id + ".mp4"
        left_enc = vmap[left_key]["encoder"]
        right_enc = vmap[right_key]["encoder"]
        scene = vmap[left_key]["scene"]
        seed = vmap[left_key]["seed"]

        verdict = (rec.get("verdict") or "").strip().lower()
        w = csv_win.get(pid)
        if not w:
            if verdict == "x_win":
                w = "left"
            elif verdict == "y_win":
                w = "right"
            elif verdict == "tie":
                w = "tie"
            else:
                w = "tie"

        if w == "left":
            winner = left_enc
        elif w == "right":
            winner = right_enc
        else:
            winner = "tie"

        wlt[winner] += 1
        by_scene[scene][winner] += 1
        by_seed[str(seed)][winner] += 1

        for d in DIMS:
            xv = rec.get(f"{d}_X")
            yv = rec.get(f"{d}_Y")
            if xv is not None:
                dim_scores[left_enc][d].append(float(xv))
            if yv is not None:
                dim_scores[right_enc][d].append(float(yv))

        unblinded_rows.append({
            "pair_id": pid,
            "scene": scene,
            "seed": seed,
            "left_video": left_id,
            "right_video": right_id,
            "left_encoder": left_enc,
            "right_encoder": right_enc,
            "verdict": verdict or w,
            "winner_encoder": winner,
            "reason": rec.get("reason", ""),
            "scores_X": {d: rec.get(f"{d}_X") for d in DIMS},
            "scores_Y": {d: rec.get(f"{d}_Y") for d in DIMS},
        })

    print(f"Pairs scored: {len(scores)}")
    print("\n=== Overall win/tie/loss (UMT5 vs MiniCPM+E1.2) ===")
    tot = sum(wlt.values())
    print(f"  UMT5 wins: {wlt['umt5']}  E1.2 wins: {wlt['e12']}  ties: {wlt['tie']}  (n={tot})")

    print("\n=== Per-scene ===")
    for s, v in sorted(by_scene.items()):
        print(f"  {s:16s} umt5={v['umt5']} e12={v['e12']} tie={v['tie']}")

    print("\n=== Per-seed ===")
    for s, v in sorted(by_seed.items()):
        print(f"  seed={s:6s} umt5={v['umt5']} e12={v['e12']} tie={v['tie']}")

    print("\n=== Mean per-dimension (by encoder, X/Y aware) ===")
    for enc in ["umt5", "e12"]:
        print(f"  {enc}:")
        for d in DIMS:
            m = mean(dim_scores[enc][d])
            n = len(dim_scores[enc][d])
            print(f"    {d:22s} n={n:2d} mean={m:.3f}" if m is not None else f"    {d:22s} n=0 mean=N/A")

    print("\n=== Per-pair unblinded ===")
    for row in unblinded_rows:
        print(
            f"  {row['pair_id']} {row['scene']}/{row['seed']}: "
            f"{row['left_encoder']}({row['left_video']}) vs {row['right_encoder']}({row['right_video']}) "
            f"→ {row['winner_encoder']} | {row['reason']}"
        )

    out = {
        "overall_wlt": wlt,
        "per_scene": dict(by_scene),
        "per_seed": {k: v for k, v in by_seed.items()},
        "dimension_means": {
            enc: {d: mean(dim_scores[enc][d]) for d in DIMS} for enc in ["umt5", "e12"]
        },
        "pairs": unblinded_rows,
    }
    out_path = REPO_ROOT / "eval/e2_new/blind/summary.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
