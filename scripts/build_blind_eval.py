#!/usr/bin/env python3
"""Build blind evaluation materials for G0.7 A/B/C comparison.

Reads results.jsonl, creates anonymous video IDs (video_001, video_002, ...),
generates a blind mapping (blind_map.json), and produces a human_eval.csv
template with all scoring dimensions.  The variant (A/B/C) is hidden during
scoring and revealed only through blind_map.json after scoring is complete.

Usage:
    python scripts/build_blind_eval.py
    python scripts/build_blind_eval.py --seed 12345
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval" / "g0.7"
RESULTS_PATH = EVAL_DIR / "results.jsonl"
BLIND_MAP_PATH = EVAL_DIR / "blind_map.json"
HUMAN_EVAL_PATH = EVAL_DIR / "human_eval.csv"

POSITIVE_DIMS = [
    "identity_consistency",
    "attribute_preservation",
    "spatial_layout",
    "environment_consistency",
    "camera_continuity",
    "temporal_stability",
    "intent_fidelity",
]
HALLUCINATION_DIM = "hallucination"
ALL_DIMS = POSITIVE_DIMS + [HALLUCINATION_DIM]

DIM_DESCRIPTIONS = {
    "identity_consistency": "Does the main subject retain its identity across frames? (0-5, higher=better)",
    "attribute_preservation": "Are subject attributes (color, shape, appearance) preserved? (0-5, higher=better)",
    "spatial_layout": "Is the spatial arrangement of objects preserved? (0-5, higher=better)",
    "environment_consistency": "Is the environment/background consistent? (0-5, higher=better)",
    "camera_continuity": "Is the camera motion smooth and continuous? (0-5, higher=better)",
    "temporal_stability": "Is the video temporally stable (no flickering/jitter)? (0-5, higher=better)",
    "intent_fidelity": "Does the video follow the user's requested action? (0-5, higher=better)",
    "hallucination": "NEGATIVE: degree of unsupported new content / semantic drift. (0=none, 5=severe)",
}


def load_results() -> list[dict[str, Any]]:
    records = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build blind evaluation materials.")
    parser.add_argument("--shuffle-seed", type=int, default=42,
                        help="Seed for random shuffling of blind IDs.")
    parser.add_argument("--include-failed", action="store_true",
                        help="Include failed generations in the blind set (default: skip).")
    args = parser.parse_args()

    records = load_results()
    if not records:
        print("No results found. Run run_g07_eval.py first.", file=sys.stderr)
        return 1

    # Filter to passing videos
    passing = [r for r in records if r.get("status") in ("PASS", "SKIPPED")]
    if not args.include_failed:
        records = passing
    else:
        records = passing + [r for r in records if r.get("status") == "FAIL"]

    if not records:
        print("No passing videos found.", file=sys.stderr)
        return 1

    # Create blind IDs with deterministic shuffle
    rng = random.Random(args.shuffle_seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    blind_map: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []

    for blind_idx, record_idx in enumerate(indices, start=1):
        rec = records[record_idx]
        blind_id = f"video_{blind_idx:03d}"
        video_id = f"{rec['scene_id']}_{rec['variant']}_seed{rec['seed']}"

        blind_map[blind_id] = {
            "video_id": video_id,
            "scene_id": rec["scene_id"],
            "variant": rec["variant"],
            "seed": rec["seed"],
            "output_video": rec.get("output_video"),
            "status": rec.get("status", "UNKNOWN"),
            "prompt_token_count": rec.get("prompt_token_count"),
            "truncated": rec.get("truncated"),
        }

        row = {
            "blind_id": blind_id,
            "video_id": video_id,
            "scene_id": rec["scene_id"],
            "seed": rec["seed"],
            # Note: variant intentionally omitted for blind scoring
            "output_video": rec.get("output_video", ""),
            "status": rec.get("status", "UNKNOWN"),
        }
        for dim in ALL_DIMS:
            row[dim] = ""
        row["positive_score"] = ""
        row["adjusted_score"] = ""
        row["notes"] = ""
        csv_rows.append(row)

    # Write blind_map.json
    BLIND_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BLIND_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "shuffle_seed": args.shuffle_seed,
            "total_videos": len(blind_map),
            "mapping": blind_map,
            "dimension_descriptions": DIM_DESCRIPTIONS,
            "scoring_instructions": (
                "Score each video 0-5 on all dimensions. hallucination is NEGATIVE "
                "(0=no hallucination, 5=severe). Do NOT look at blind_map.json until "
                "all scoring is complete. positive_score = mean of 7 positive dimensions. "
                "adjusted_score = positive_score - 0.5 * hallucination."
            ),
        }, f, ensure_ascii=False, indent=2)

    # Write human_eval.csv
    fieldnames = ["blind_id", "video_id", "scene_id", "seed", "output_video", "status"]
    fieldnames += ALL_DIMS
    fieldnames += ["positive_score", "adjusted_score", "notes"]

    with open(HUMAN_EVAL_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"Blind evaluation materials built:")
    print(f"  Videos: {len(blind_map)}")
    print(f"  Blind map: {BLIND_MAP_PATH}")
    print(f"  Human eval CSV: {HUMAN_EVAL_PATH}")
    print(f"  Shuffle seed: {args.shuffle_seed}")
    print(f"\nScoring instructions:")
    print(f"  1. Open human_eval.csv")
    print(f"  2. For each blind_id, watch the video and score 0-5 on all dimensions")
    print(f"  3. Do NOT open blind_map.json until all scores are entered")
    print(f"  4. After scoring, run: python scripts/summarize_g07_eval.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
