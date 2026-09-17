#!/usr/bin/env python3
"""Build the anonymized 30-video blind eval set for E2-new.

Reads eval/e2_new/results.jsonl (15 E1.2 videos) + matches each to its G0.7
UMT5 baseline.  Produces:
  eval/e2_new/blind/videos/video_001..030.mp4   (anonymized copies)
  eval/e2_new/blind/blind_map.json               (secret: id -> scene/seed/encoder)
  eval/e2_new/blind/human_eval_paired.csv        (template, to be filled by user)

Anonymity contract: the video filenames and the CSV contain NO encoder name,
NO scene name, NO seed number, NO absolute path.  Pairing is 15 pairs of
(UMT5 baseline, MiniCPM+E1.2), with left/right shuffled per pair so the
rater cannot infer which is which.
"""
import json
import os
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
E2NEW = REPO_ROOT / "eval/e2_new"
BLIND = E2NEW / "blind"
(BLIND / "videos").mkdir(parents=True, exist_ok=True)


def main():
    records = [json.loads(l) for l in open(E2NEW / "results.jsonl") if l.strip()]
    assert len(records) == 15, f"expected 15 E1.2 records, got {len(records)}"

    g = random.Random(20260917)
    # Build 15 pairs
    pairs = []
    for r in records:
        scene, seed = r["scene"], r["seed"]
        umt5 = REPO_ROOT / f"eval/g0.7/{scene}/videos/A_seed{seed}.mp4"
        e12 = REPO_ROOT / r["video"]
        assert umt5.exists(), f"missing baseline {umt5}"
        assert e12.exists(), f"missing e12 {e12}"
        # shuffle left/right
        order = ["umt5", "e12"]
        g.shuffle(order)
        pairs.append({"scene": scene, "seed": seed, order[0]: None, order[1]: None})
        pairs[-1]["_left_enc"] = order[0]
        pairs[-1]["_right_enc"] = order[1]

    # Assign anonymous ids 001..030
    flat = []  # (pair_idx, encoder, path)
    for i, p in enumerate(pairs):
        for enc_key, src in [("umt5", REPO_ROOT / f"eval/g0.7/{p['scene']}/videos/A_seed{p['seed']}.mp4"),
                             ("e12", REPO_ROOT / next(r["video"] for r in records if r["scene"] == p["scene"] and r["seed"] == p["seed"]))]:
            flat.append((i, enc_key, src))
    # shuffle the overall presentation order of the 30 videos (still keep pair grouping in CSV)
    g.shuffle(flat)

    mapping = {}
    for idx, (pi, enc, src) in enumerate(flat, start=1):
        anon = f"video_{idx:03d}.mp4"
        shutil.copyfile(src, BLIND / "videos" / anon)
        mapping[anon] = {"pair": pi + 1, "encoder": enc,
                         "scene": pairs[pi]["scene"], "seed": pairs[pi]["seed"]}

    # Pair table for CSV: pair rows, each referencing its two anon ids (left/right per shuffle)
    pair_rows = []
    for i, p in enumerate(pairs):
        left_anon = [k for k, v in mapping.items() if v["pair"] == i + 1 and v["encoder"] == p["_left_enc"]][0]
        right_anon = [k for k, v in mapping.items() if v["pair"] == i + 1 and v["encoder"] == p["_right_enc"]][0]
        pair_rows.append({"pair_id": f"pair_{i+1:02d}",
                          "left": left_anon.replace(".mp4", ""),
                          "right": right_anon.replace(".mp4", "")})

    # blind_map.json (secret)
    json.dump({"videos": mapping, "pairs": pair_rows,
               "note": "SECRET mapping. Do not distribute with blind set."},
              open(BLIND / "blind_map.json", "w"), indent=2)

    # CSV template (no scene/seed/encoder leaks)
    cols = ["pair_id", "left_video", "right_video",
            "semantic_fidelity", "spatial_consistency", "camera_motion",
            "temporal_stability", "hallucination", "overall_preference",
            "win", "tie_loss", "notes"]
    lines = [",".join(cols)]
    for pr in pair_rows:
        lines.append(f"{pr['pair_id']},{pr['left']},{pr['right']},,,,,,,")
    open(BLIND / "human_eval_paired.csv", "w").write("\n".join(lines) + "\n")

    print(f"Wrote {len(flat)} blind videos, {len(pairs)} pairs")
    print(f"  map: {BLIND/'blind_map.json'}")
    print(f"  csv: {BLIND/'human_eval_paired.csv'}")


if __name__ == "__main__":
    main()
