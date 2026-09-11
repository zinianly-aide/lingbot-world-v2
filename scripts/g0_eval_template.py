#!/usr/bin/env python3
"""Create the manual-scoring ledger for the required G0 A/B/C matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCENES = [
    "single_subject_color",
    "multi_object_layout",
    "indoor_structure",
    "outdoor_complex",
    "strong_camera_motion",
]
VARIANTS = ["A_original", "B_full_world", "C_compact_world"]
SCORES = [
    "subject_identity",
    "object_attributes",
    "spatial_layout",
    "environment_consistency",
    "camera_continuity",
    "temporal_stability",
    "user_intent_fidelity",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="g0_eval.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    records = []
    for scene in SCENES:
        for seed in args.seeds:
            for variant in VARIANTS:
                records.append(
                    {
                        "scene": scene,
                        "seed": seed,
                        "variant": variant,
                        "status": "NOT RUN",
                        "video": None,
                        "world_condition": None,
                        "world_prompt": None,
                        "prompt_tokens": {
                            "original": None,
                            "world": None,
                            "final_umt5": None,
                            "limit": 512,
                            "truncated": None,
                        },
                        "scores_0_to_5": {key: None for key in SCORES},
                        "hallucination_penalty": {
                            "score_0_to_5": None,
                            "new_subject": None,
                            "wrong_attribute": None,
                            "wrong_spatial_relation": None,
                            "unsupported_vlm_fact": None,
                        },
                        "notes": "Awaiting real model generation and human review",
                    }
                )
    payload = {
        "status": "NOT RUN",
        "planned_matrix": {
            "scenes": SCENES,
            "seeds": args.seeds,
            "variants": VARIANTS,
            "planned_runs": len(records),
            "completed_runs": 0,
        },
        "records": records,
        "automated_metrics_are_auxiliary": True,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} NOT RUN records to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

