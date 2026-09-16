#!/usr/bin/env python3
"""Summarize E2 paired human eval results.

Usage:
  python scripts/summarize_e2_paired.py [--csv eval/e2/blind/human_eval_paired.csv]
                                        [--map eval/e2/blind/blind_map.json]

Dry mode (no scores filled): reports missing rows, no conclusions.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
E2 = REPO / "eval" / "e2"

DIMENSIONS = [
    "intent_fidelity", "identity_consistency", "spatial_consistency",
    "temporal_stability", "camera_motion_fidelity", "hallucination",
]
EXPECTED_COLS = ["pair_id", "video_a", "video_b"] + DIMENSIONS + ["overall_preference", "notes"]


def validate(csv_path: Path, map_path: Path) -> tuple[list[dict], dict]:
    """Strict validation. Returns (rows, blind_map) or raises SystemExit."""
    errors = []

    with open(map_path) as f:
        blind_map = json.load(f)
    by_id = {e["blind_id"]: e for e in blind_map}

    # Check map uniqueness
    if len(by_id) != 30:
        errors.append(f"blind_map has {len(by_id)} entries, expected 30")
    for e in blind_map:
        if e["blind_id"] not in by_id:
            errors.append(f"duplicate blind_id: {e['blind_id']}")

    # Check pairs in map are same scene/seed
    from collections import defaultdict
    pairs = defaultdict(list)
    for e in blind_map:
        pairs[(e["scene_id"], e["seed"])].append(e)
    for key, entries in pairs.items():
        if len(entries) != 2:
            errors.append(f"pair {key} has {len(entries)} entries, expected 2")
        else:
            encoders = {e["encoder"] for e in entries}
            if encoders != {"umt5", "minicpm+adapter"}:
                errors.append(f"pair {key} encoders mismatch: {encoders}")

    # Read CSV
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []

    if header != EXPECTED_COLS:
        errors.append(f"CSV columns mismatch. Expected {EXPECTED_COLS}, got {header}")

    if len(rows) != 15:
        errors.append(f"CSV has {len(rows)} rows, expected 15")

    pair_ids = [r["pair_id"] for r in rows]
    if len(set(pair_ids)) != len(pair_ids):
        errors.append("duplicate pair_id in CSV")

    # Validate each row
    filled = 0
    for r in rows:
        pid = r["pair_id"]
        # Resolve blind IDs
        a_id = r["video_a"].split("/")[-1].replace(".mp4", "")
        b_id = r["video_b"].split("/")[-1].replace(".mp4", "")
        if a_id not in by_id:
            errors.append(f"{pid}: video_a blind_id {a_id} not in map")
        if b_id not in by_id:
            errors.append(f"{pid}: video_b blind_id {b_id} not in map")
        if a_id in by_id and b_id in by_id:
            a_info = by_id[a_id]
            b_info = by_id[b_id]
            if a_info["scene_id"] != b_info["scene_id"] or a_info["seed"] != b_info["seed"]:
                errors.append(f"{pid}: A/B not same scene/seed: {a_info['scene_id']}/{a_info['seed']} vs {b_info['scene_id']}/{b_info['seed']}")
            if a_info["encoder"] == b_info["encoder"]:
                errors.append(f"{pid}: A/B same encoder: {a_info['encoder']}")

        # Check score ranges
        for dim in DIMENSIONS:
            val = r.get(dim, "").strip()
            if val:
                try:
                    v = int(val)
                    if not (0 <= v <= 5):
                        errors.append(f"{pid}: {dim}={v} out of range 0-5")
                except ValueError:
                    errors.append(f"{pid}: {dim}='{val}' not an integer")

        op = r.get("overall_preference", "").strip()
        if op:
            try:
                v = int(op)
                if not (-2 <= v <= 2):
                    errors.append(f"{pid}: overall_preference={v} out of range -2..+2")
            except ValueError:
                errors.append(f"{pid}: overall_preference='{op}' not an integer")

        # Check if filled
        if op:
            filled += 1

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    return rows, blind_map, filled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(E2 / "blind" / "human_eval_paired.csv"))
    parser.add_argument("--map", default=str(E2 / "blind" / "blind_map.json"))
    args = parser.parse_args()

    rows, blind_map, filled = validate(Path(args.csv), Path(args.map))
    by_id = {e["blind_id"]: e for e in blind_map}

    if filled < 15:
        print(f"DRY MODE: {filled}/15 pairs scored. Human evaluation not complete.")
        print("Fill in human_eval_paired.csv to get conclusions.")
        return

    # Decode: for each pair, determine which encoder is A and which is B
    win_minicpm = tie = win_umt5 = 0
    per_scene = {}
    per_seed = {}
    dim_scores = {"umt5": {d: [] for d in DIMENSIONS}, "minicpm+adapter": {d: [] for d in DIMENSIONS}}

    for r in rows:
        a_id = r["video_a"].split("/")[-1].replace(".mp4", "")
        b_id = r["video_b"].split("/")[-1].replace(".mp4", "")
        a_info = by_id[a_id]
        b_info = by_id[b_id]
        scene = a_info["scene_id"]
        seed = a_info["seed"]

        # overall_preference: -2..+2, negative = A better, positive = B better
        op = int(r["overall_preference"])

        # Determine encoder of A and B
        a_enc = a_info["encoder"]
        b_enc = b_info["encoder"]

        # If op < 0: A is better; op > 0: B is better; op == 0: tie
        if op == 0:
            tie += 1
            scene_verdict = "tie"
        elif (op < 0 and a_enc == "minicpm+adapter") or (op > 0 and b_enc == "minicpm+adapter"):
            win_minicpm += 1
            scene_verdict = "minicpm"
        else:
            win_umt5 += 1
            scene_verdict = "umt5"

        per_scene.setdefault(scene, {"minicpm": 0, "umt5": 0, "tie": 0})
        per_scene[scene][scene_verdict] += 1
        per_seed.setdefault(seed, {"minicpm": 0, "umt5": 0, "tie": 0})
        per_seed[seed][scene_verdict] += 1

        # Per-dimension scores
        for dim in DIMENSIONS:
            v = int(r[dim])
            dim_scores[a_enc][dim].append(v)
            dim_scores[b_enc][dim].append(v)

    # Output
    print("=== E2 Paired Human Eval Summary ===")
    print(f"\nOverall: MiniCPM={win_minicpm}, Tie={tie}, UMT5={win_umt5}")
    print(f"\nPer-scene:")
    for scene, v in sorted(per_scene.items()):
        print(f"  {scene:20s}: MiniCPM={v['minicpm']}, Tie={v['tie']}, UMT5={v['umt5']}")
    print(f"\nPer-seed:")
    for seed, v in sorted(per_seed.items()):
        print(f"  {seed:5d}: MiniCPM={v['minicpm']}, Tie={v['tie']}, UMT5={v['umt5']}")
    print(f"\nPer-dimension mean scores:")
    for dim in DIMENSIONS:
        u = statistics.mean(dim_scores["umt5"][dim])
        m = statistics.mean(dim_scores["minicpm+adapter"][dim])
        print(f"  {dim:25s}: UMT5={u:.2f}, MiniCPM={m:.2f}, delta={m-u:+.2f}")

    print(f"\nNote: temporal_mad etc. in metrics_objective.json are auxiliary, not part of this summary.")


if __name__ == "__main__":
    main()
