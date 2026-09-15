#!/usr/bin/env python3
"""G0.7 A/B/C evaluation summarizer.

Reads results.jsonl and metrics_prompt.jsonl, computes per-variant mean/
median/std across all dimensions, per-scene and per-seed breakdowns, and
pairwise differences (B-A, C-A, C-B).  Also applies the G1 Entry Gate.

Usage:
    python scripts/summarize_g07_eval.py
    python scripts/summarize_g07_eval.py --output eval/g0.7/summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval" / "g0.7"
RESULTS_PATH = EVAL_DIR / "results.jsonl"
METRICS_PATH = EVAL_DIR / "metrics_prompt.jsonl"
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


def load_results() -> list[dict[str, Any]]:
    records = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_prompt_metrics() -> list[dict[str, Any]]:
    records = []
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_human_scores() -> dict[str, dict[str, float]]:
    """Load human_eval.csv into {video_id: {dim: score}}."""
    scores: dict[str, dict[str, float]] = {}
    if not HUMAN_EVAL_PATH.exists():
        return scores
    import csv
    with open(HUMAN_EVAL_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row.get("video_id", "")
            if not video_id:
                continue
            dim_scores: dict[str, float] = {}
            for dim in ALL_DIMS:
                val = row.get(dim, "").strip()
                if val:
                    try:
                        dim_scores[dim] = float(val)
                    except ValueError:
                        pass
            if dim_scores:
                scores[video_id] = dim_scores
    return scores


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "std": None, "count": 0}
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def compute_variant_stats(
    records: list[dict[str, Any]],
    human_scores: dict[str, dict[str, float]],
    variant: str,
) -> dict[str, Any]:
    """Compute stats for one variant across all passing videos."""
    dim_values: dict[str, list[float]] = {d: [] for d in ALL_DIMS}
    adjusted_values: list[float] = []
    sanity_mad: list[float] = []

    for rec in records:
        if rec.get("variant") != variant or rec.get("status") not in ("PASS", "SKIPPED"):
            continue
        video_id = f"{rec['scene_id']}_{rec['variant']}_seed{rec['seed']}"
        hs = human_scores.get(video_id, {})

        for dim in ALL_DIMS:
            if dim in hs:
                dim_values[dim].append(hs[dim])

        # Adjusted score
        pos_vals = [hs[d] for d in POSITIVE_DIMS if d in hs]
        if pos_vals and HALLUCINATION_DIM in hs:
            positive = statistics.mean(pos_vals)
            adjusted = positive - hs[HALLUCINATION_DIM] * 0.5
            adjusted_values.append(adjusted)

        # Sanity
        sm = rec.get("sanity_metrics", {})
        if sm.get("temporal_mad_mean") is not None:
            sanity_mad.append(sm["temporal_mad_mean"])

    result = {
        "dimensions": {d: stats(dim_values[d]) for d in ALL_DIMS},
        "adjusted_score": stats(adjusted_values),
        "sanity_temporal_mad_mean": stats(sanity_mad),
    }
    return result


def compute_pairwise_diffs(
    variant_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute B-A, C-A, C-B differences for each dimension and adjusted score."""
    pairs = [("B", "A"), ("C", "A"), ("C", "B")]
    diffs: dict[str, dict[str, Any]] = {}
    for v1, v2 in pairs:
        key = f"{v1}-{v2}"
        dim_diffs: dict[str, Any] = {}
        for dim in ALL_DIMS:
            m1 = variant_stats[v1]["dimensions"][dim]["mean"]
            m2 = variant_stats[v2]["dimensions"][dim]["mean"]
            if m1 is not None and m2 is not None:
                dim_diffs[dim] = round(m1 - m2, 3)
            else:
                dim_diffs[dim] = None
        a1 = variant_stats[v1]["adjusted_score"]["mean"]
        a2 = variant_stats[v2]["adjusted_score"]["mean"]
        dim_diffs["adjusted_score"] = round(a1 - a2, 3) if a1 is not None and a2 is not None else None
        diffs[key] = dim_diffs
    return diffs


def compute_per_scene(
    records: list[dict[str, Any]],
    human_scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """Per-scene breakdown: mean adjusted score per variant."""
    scenes = sorted(set(r["scene_id"] for r in records))
    result: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        scene_result: dict[str, Any] = {}
        for variant in ["A", "B", "C"]:
            adj_values: list[float] = []
            dim_values: dict[str, list[float]] = {d: [] for d in ALL_DIMS}
            for rec in records:
                if rec["scene_id"] != scene or rec["variant"] != variant:
                    continue
                if rec.get("status") not in ("PASS", "SKIPPED"):
                    continue
                video_id = f"{rec['scene_id']}_{rec['variant']}_seed{rec['seed']}"
                hs = human_scores.get(video_id, {})
                for dim in ALL_DIMS:
                    if dim in hs:
                        dim_values[dim].append(hs[dim])
                pos_vals = [hs[d] for d in POSITIVE_DIMS if d in hs]
                if pos_vals and HALLUCINATION_DIM in hs:
                    adj_values.append(statistics.mean(pos_vals) - hs[HALLUCINATION_DIM] * 0.5)
            scene_result[variant] = {
                "adjusted_score": stats(adj_values),
                "dimensions": {d: stats(dim_values[d]) for d in ALL_DIMS},
            }
        result[scene] = scene_result
    return result


def compute_per_seed(
    records: list[dict[str, Any]],
    human_scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """Per-seed breakdown: mean adjusted score per variant."""
    seeds = sorted(set(r["seed"] for r in records))
    result: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        seed_result: dict[str, Any] = {}
        for variant in ["A", "B", "C"]:
            adj_values: list[float] = []
            for rec in records:
                if rec["seed"] != seed or rec["variant"] != variant:
                    continue
                if rec.get("status") not in ("PASS", "SKIPPED"):
                    continue
                video_id = f"{rec['scene_id']}_{rec['variant']}_seed{rec['seed']}"
                hs = human_scores.get(video_id, {})
                pos_vals = [hs[d] for d in POSITIVE_DIMS if d in hs]
                if pos_vals and HALLUCINATION_DIM in hs:
                    adj_values.append(statistics.mean(pos_vals) - hs[HALLUCINATION_DIM] * 0.5)
            seed_result[variant] = stats(adj_values)
        result[str(seed)] = seed_result
    return result


def g1_entry_gate(
    records: list[dict[str, Any]],
    human_scores: dict[str, dict[str, float]],
    per_scene: dict[str, dict[str, Any]],
    per_seed: dict[str, dict[str, Any]],
    prompt_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Strict G1 Entry Gate. Returns G1 JUSTIFIED or G1 NOT JUSTIFIED."""
    # Check if we have human scores at all
    if not human_scores:
        return {
            "decision": "G1 NOT STARTED",
            "reason": "No human evaluation scores available yet.",
            "checks": {},
        }

    # Check 1: C improves over A in at least 4/5 scenes
    scenes_improved = 0
    scene_details = {}
    for scene, sdata in per_scene.items():
        c_mean = sdata["C"]["adjusted_score"]["mean"]
        a_mean = sdata["A"]["adjusted_score"]["mean"]
        if c_mean is not None and a_mean is not None and c_mean > a_mean:
            scenes_improved += 1
            scene_details[scene] = {"C": c_mean, "A": a_mean, "improved": True}
        else:
            scene_details[scene] = {"C": c_mean, "A": a_mean, "improved": False}
    check_scenes = scenes_improved >= 4

    # Check 2: identity_consistency does not decrease for C vs A
    def global_dim_mean(variant: str, dim: str) -> float | None:
        vals = []
        for rec in records:
            if rec["variant"] != variant or rec.get("status") not in ("PASS", "SKIPPED"):
                continue
            video_id = f"{rec['scene_id']}_{rec['variant']}_seed{rec['seed']}"
            hs = human_scores.get(video_id, {})
            if dim in hs:
                vals.append(hs[dim])
        return statistics.mean(vals) if vals else None

    c_identity = global_dim_mean("C", "identity_consistency")
    a_identity = global_dim_mean("A", "identity_consistency")
    check_identity = c_identity is not None and a_identity is not None and c_identity >= a_identity - 0.1

    # Check 3: intent_fidelity does not decrease
    c_intent = global_dim_mean("C", "intent_fidelity")
    a_intent = global_dim_mean("A", "intent_fidelity")
    check_intent = c_intent is not None and a_intent is not None and c_intent >= a_intent - 0.1

    # Check 4: hallucination does not significantly increase (lower is better)
    c_hall = global_dim_mean("C", "hallucination")
    a_hall = global_dim_mean("A", "hallucination")
    check_hallucination = c_hall is not None and a_hall is not None and c_hall <= a_hall + 0.5

    # Check 5: at least 2/3 seeds show C > A trend
    seeds_improved = 0
    for seed, sdata in per_seed.items():
        c_mean = sdata["C"]["mean"]
        a_mean = sdata["A"]["mean"]
        if c_mean is not None and a_mean is not None and c_mean > a_mean:
            seeds_improved += 1
    check_seeds = seeds_improved >= 2

    # Check 6: C truncation better than B or token budget more stable
    b_trunc = [m for m in prompt_metrics if m["variant"] == "B"]
    c_trunc = [m for m in prompt_metrics if m["variant"] == "C"]
    b_trunc_count = sum(1 for m in b_trunc if m["truncated"])
    c_trunc_count = sum(1 for m in c_trunc if m["truncated"])
    b_max_tokens = max((m["token_count_before_truncation"] for m in b_trunc), default=0)
    c_max_tokens = max((m["token_count_before_truncation"] for m in c_trunc), default=0)
    check_tokens = (c_trunc_count <= b_trunc_count) or (c_max_tokens < b_max_tokens)

    all_checks = {
        "C_improves_in_4of5_scenes": {"pass": check_scenes, "detail": f"{scenes_improved}/5 scenes improved"},
        "identity_not_decreased": {"pass": check_identity, "detail": f"C={c_identity}, A={a_identity}"},
        "intent_fidelity_not_decreased": {"pass": check_intent, "detail": f"C={c_intent}, A={a_intent}"},
        "hallucination_not_significantly_increased": {"pass": check_hallucination, "detail": f"C={c_hall}, A={a_hall}"},
        "trend_consistent_in_2of3_seeds": {"pass": check_seeds, "detail": f"{seeds_improved}/3 seeds improved"},
        "C_token_budget_better_than_B": {"pass": check_tokens, "detail": f"B trunc={b_trunc_count}, max={b_max_tokens}; C trunc={c_trunc_count}, max={c_max_tokens}"},
    }

    all_pass = all(c["pass"] for c in all_checks.values())

    # Also check if B improves but C doesn't (compact prompt design issue)
    b_scenes_improved = sum(
        1 for scene, sdata in per_scene.items()
        if sdata["B"]["adjusted_score"]["mean"] is not None
        and sdata["A"]["adjusted_score"]["mean"] is not None
        and sdata["B"]["adjusted_score"]["mean"] > sdata["A"]["adjusted_score"]["mean"]
    )

    if all_pass:
        decision = "G1 JUSTIFIED"
        reason = "C (compact world prompt) shows stable improvement over A across all gate criteria."
    elif b_scenes_improved >= 3 and scenes_improved < 3:
        decision = "G1 NOT JUSTIFIED"
        reason = "B (full world) shows improvement but C (compact) does not — compact prompt design needs optimization before G1."
    else:
        decision = "G1 NOT JUSTIFIED"
        reason = "Neither B nor C shows stable improvement over A. Optimize World JSON schema / VLM prompt / compact composition before G1."

    return {
        "decision": decision,
        "reason": reason,
        "checks": all_checks,
        "scene_details": scene_details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize G0.7 A/B/C evaluation results.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: print to stdout).")
    args = parser.parse_args()

    records = load_results()
    prompt_metrics = load_prompt_metrics()
    human_scores = load_human_scores()

    if not records:
        print("No results found. Run run_g07_eval.py first.", file=sys.stderr)
        return 1

    # Status overview
    status_counts: dict[str, int] = {}
    for r in records:
        s = r.get("status", "UNKNOWN")
        status_counts[s] = status_counts.get(s, 0) + 1

    # Per-variant stats
    variant_stats = {v: compute_variant_stats(records, human_scores, v) for v in ["A", "B", "C"]}

    # Pairwise diffs
    pairwise = compute_pairwise_diffs(variant_stats)

    # Per-scene and per-seed
    per_scene = compute_per_scene(records, human_scores)
    per_seed = compute_per_seed(records, human_scores)

    # Prompt token stats
    token_stats: dict[str, Any] = {}
    for variant in ["A", "B", "C"]:
        vmetrics = [m for m in prompt_metrics if m["variant"] == variant]
        token_stats[variant] = {
            "mean_tokens": round(statistics.mean(m["token_count_before_truncation"] for m in vmetrics), 1) if vmetrics else None,
            "max_tokens": max((m["token_count_before_truncation"] for m in vmetrics), default=None),
            "truncated_count": sum(1 for m in vmetrics if m["truncated"]),
            "mean_chars": round(statistics.mean(m["prompt_chars"] for m in vmetrics), 1) if vmetrics else None,
        }

    # G1 gate
    gate = g1_entry_gate(records, human_scores, per_scene, per_seed, prompt_metrics)

    summary = {
        "eval_name": "g0.7-real-abc-eval",
        "total_records": len(records),
        "status_counts": status_counts,
        "human_scores_available": len(human_scores),
        "variant_stats": variant_stats,
        "pairwise_differences": pairwise,
        "per_scene": per_scene,
        "per_seed": per_seed,
        "prompt_token_stats": token_stats,
        "g1_entry_gate": gate,
    }

    output_text = json.dumps(summary, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"Summary written to {args.output}")
    else:
        print(output_text)

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"Status: {status_counts}")
    print(f"Human scores: {len(human_scores)} videos scored")
    for variant in ["A", "B", "C"]:
        adj = variant_stats[variant]["adjusted_score"]["mean"]
        print(f"  Variant {variant}: adjusted_score mean = {adj}")
    print(f"  B-A adjusted: {pairwise['B-A']['adjusted_score']}")
    print(f"  C-A adjusted: {pairwise['C-A']['adjusted_score']}")
    print(f"  C-B adjusted: {pairwise['C-B']['adjusted_score']}")
    print(f"\nG1 Gate: {gate['decision']}")
    print(f"  Reason: {gate['reason']}")
    for check_name, check_data in gate["checks"].items():
        print(f"  [{'PASS' if check_data['pass'] else 'FAIL'}] {check_name}: {check_data['detail']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
