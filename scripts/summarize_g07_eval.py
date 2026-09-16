#!/usr/bin/env python3
"""G0.7 A/B/C evaluation summarizer (blind-id unblinding).

Reads human_eval.csv by blind_id, unblinds via blind_map.json, merges with
results.jsonl, computes per-variant / per-scene / per-seed mean/median/std,
pairwise differences (B-A, C-A, C-B), paired (scene+seed) deltas with
wins/ties/losses, and applies the G1 Entry Gate.

Strict validation: if any scores are present, ALL 45 rows must have all 8
dimensions filled with 0-5 values, no duplicate blind_ids, all unblindable.
If zero scores are present, reports G1 NOT STARTED.

Usage:
    python scripts/summarize_g07_eval.py
    python scripts/summarize_g07_eval.py --output eval/g0.7/summary.json
"""
from __future__ import annotations

import argparse
import csv
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
BLIND_MAP_PATH = EVAL_DIR / "blind_map.json"

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


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

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


def load_blind_map() -> dict[str, Any]:
    """Load blind_map.json. Returns the mapping dict {blind_id: {...}}."""
    if not BLIND_MAP_PATH.exists():
        return {}
    with open(BLIND_MAP_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("mapping", {})


def load_and_validate_human_scores(
    blind_map: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Load human_eval.csv by blind_id and strictly validate.

    Returns {blind_id: {dim: score}}.
    If zero scores are present, returns empty dict (NOT STARTED path).
    If partial/invalid scores, prints error and exits with code 1.
    """
    if not HUMAN_EVAL_PATH.exists():
        return {}

    with open(HUMAN_EVAL_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check required columns
    required = {"blind_id"} | set(ALL_DIMS)
    missing_cols = required - set(reader.fieldnames or [])
    if missing_cols:
        print(f"ERROR: human_eval.csv missing columns: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    # Detect if any scores are present
    any_score = False
    for row in rows:
        for dim in ALL_DIMS:
            if row.get(dim, "").strip():
                any_score = True
                break
        if any_score:
            break

    if not any_score:
        return {}  # No scores yet — NOT STARTED path

    # --- Strict validation (scores present) ---
    errors: list[str] = []

    # 45 rows
    if len(rows) != 45:
        errors.append(f"Expected 45 rows, found {len(rows)}")

    # No duplicate blind_ids
    blind_ids = [row["blind_id"].strip() for row in rows]
    dupes = [b for b in set(blind_ids) if blind_ids.count(b) > 1]
    if dupes:
        errors.append(f"Duplicate blind_ids: {dupes}")

    scores: dict[str, dict[str, float]] = {}
    for row in rows:
        bid = row["blind_id"].strip()
        if not bid:
            errors.append("Row with empty blind_id")
            continue

        # All 8 dimensions non-empty and in 0-5
        dim_scores: dict[str, float] = {}
        for dim in ALL_DIMS:
            val = row.get(dim, "").strip()
            if not val:
                errors.append(f"{bid}: dimension '{dim}' is empty")
                continue
            try:
                fval = float(val)
            except ValueError:
                errors.append(f"{bid}: dimension '{dim}' is not a number: '{val}'")
                continue
            if fval < 0 or fval > 5:
                errors.append(f"{bid}: dimension '{dim}'={fval} out of range [0,5]")
                continue
            dim_scores[dim] = fval

        # All blind_ids must unblind
        if bid not in blind_map:
            errors.append(f"{bid}: not found in blind_map.json")
        elif len(dim_scores) == 8:
            scores[bid] = dim_scores

    if errors:
        print("ERROR: human_eval.csv validation failed:", file=sys.stderr)
        for e in errors[:20]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors", file=sys.stderr)
        sys.exit(1)

    return scores


def build_score_lookup(
    human_scores: dict[str, dict[str, float]],
    blind_map: dict[str, Any],
) -> dict[tuple[str, str, int], dict[str, float]]:
    """Map blind_id scores to (scene_id, variant, seed) lookup.

    Returns {(scene_id, variant, seed): {dim: score}}.
    """
    lookup: dict[tuple[str, str, int], dict[str, float]] = {}
    for bid, dim_scores in human_scores.items():
        info = blind_map.get(bid, {})
        key = (info.get("scene_id", ""), info.get("variant", ""), info.get("seed", 0))
        lookup[key] = dim_scores
    return lookup


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "std": None, "count": 0}
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def adjusted_from_scores(dim_scores: dict[str, float]) -> float | None:
    """Compute positive_score and adjusted_score from dimension scores."""
    pos_vals = [dim_scores[d] for d in POSITIVE_DIMS if d in dim_scores]
    if len(pos_vals) == 7 and HALLUCINATION_DIM in dim_scores:
        positive = statistics.mean(pos_vals)
        return positive - dim_scores[HALLUCINATION_DIM] * 0.5
    return None


# ---------------------------------------------------------------------------
# Per-variant / per-scene / per-seed
# ---------------------------------------------------------------------------

def compute_variant_stats(
    records: list[dict[str, Any]],
    score_lookup: dict[tuple[str, str, int], dict[str, float]],
    variant: str,
) -> dict[str, Any]:
    dim_values: dict[str, list[float]] = {d: [] for d in ALL_DIMS}
    adjusted_values: list[float] = []
    sanity_mad: list[float] = []

    for rec in records:
        if rec.get("variant") != variant or rec.get("status") not in ("PASS", "SKIPPED"):
            continue
        key = (rec["scene_id"], rec["variant"], rec["seed"])
        hs = score_lookup.get(key, {})

        for dim in ALL_DIMS:
            if dim in hs:
                dim_values[dim].append(hs[dim])

        adj = adjusted_from_scores(hs)
        if adj is not None:
            adjusted_values.append(adj)

        sm = rec.get("sanity_metrics", {})
        if sm.get("temporal_mad_mean") is not None:
            sanity_mad.append(sm["temporal_mad_mean"])

    return {
        "dimensions": {d: stats(dim_values[d]) for d in ALL_DIMS},
        "adjusted_score": stats(adjusted_values),
        "sanity_temporal_mad_mean": stats(sanity_mad),
    }


def compute_pairwise_diffs(
    variant_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
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
    score_lookup: dict[tuple[str, str, int], dict[str, float]],
) -> dict[str, dict[str, Any]]:
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
                key = (rec["scene_id"], rec["variant"], rec["seed"])
                hs = score_lookup.get(key, {})
                for dim in ALL_DIMS:
                    if dim in hs:
                        dim_values[dim].append(hs[dim])
                adj = adjusted_from_scores(hs)
                if adj is not None:
                    adj_values.append(adj)
            scene_result[variant] = {
                "adjusted_score": stats(adj_values),
                "dimensions": {d: stats(dim_values[d]) for d in ALL_DIMS},
            }
        result[scene] = scene_result
    return result


def compute_per_seed(
    records: list[dict[str, Any]],
    score_lookup: dict[tuple[str, str, int], dict[str, float]],
) -> dict[str, dict[str, Any]]:
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
                key = (rec["scene_id"], rec["variant"], rec["seed"])
                hs = score_lookup.get(key, {})
                adj = adjusted_from_scores(hs)
                if adj is not None:
                    adj_values.append(adj)
            seed_result[variant] = stats(adj_values)
        result[str(seed)] = seed_result
    return result


# ---------------------------------------------------------------------------
# Paired (scene + seed) deltas & wins/ties/losses
# ---------------------------------------------------------------------------

def compute_paired_deltas(
    records: list[dict[str, Any]],
    score_lookup: dict[tuple[str, str, int], dict[str, float]],
) -> dict[str, Any]:
    """For each (scene_id, seed), compare A/B/C adjusted scores and dims.

    Returns per-pair deltas, wins/ties/losses overall and per scene.
    """
    # Group records by (scene_id, seed)
    groups: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for rec in records:
        if rec.get("status") not in ("PASS", "SKIPPED"):
            continue
        key = (rec["scene_id"], rec["seed"])
        vkey = (rec["scene_id"], rec["variant"], rec["seed"])
        hs = score_lookup.get(vkey)
        if hs is None:
            continue
        if key not in groups:
            groups[key] = {}
        groups[key][rec["variant"]] = hs

    pairs = [("B", "A"), ("C", "A"), ("C", "B")]
    pair_deltas: dict[str, list[dict[str, Any]]] = {f"{v1}-{v2}": [] for v1, v2 in pairs}
    wins_overall: dict[str, dict[str, int]] = {
        f"{v1}-{v2}": {"win": 0, "tie": 0, "loss": 0} for v1, v2 in pairs
    }
    wins_per_scene: dict[str, dict[str, dict[str, int]]] = {}

    for (scene, seed), variants in sorted(groups.items()):
        if scene not in wins_per_scene:
            wins_per_scene[scene] = {
                f"{v1}-{v2}": {"win": 0, "tie": 0, "loss": 0} for v1, v2 in pairs
            }
        for v1, v2 in pairs:
            pkey = f"{v1}-{v2}"
            if v1 not in variants or v2 not in variants:
                continue
            s1 = variants[v1]
            s2 = variants[v2]
            adj1 = adjusted_from_scores(s1)
            adj2 = adjusted_from_scores(s2)
            if adj1 is None or adj2 is None:
                continue

            delta_adj = round(adj1 - adj2, 3)
            dim_deltas: dict[str, float] = {}
            for dim in ALL_DIMS:
                if dim in s1 and dim in s2:
                    dim_deltas[dim] = round(s1[dim] - s2[dim], 3)

            pair_deltas[pkey].append({
                "scene": scene, "seed": seed,
                "adjusted_delta": delta_adj,
                "dimension_deltas": dim_deltas,
            })

            if delta_adj > 0:
                wins_overall[pkey]["win"] += 1
                wins_per_scene[scene][pkey]["win"] += 1
            elif delta_adj == 0:
                wins_overall[pkey]["tie"] += 1
                wins_per_scene[scene][pkey]["tie"] += 1
            else:
                wins_overall[pkey]["loss"] += 1
                wins_per_scene[scene][pkey]["loss"] += 1

    return {
        "paired_deltas": pair_deltas,
        "wins_overall": wins_overall,
        "wins_per_scene": wins_per_scene,
        "total_pairs": {k: len(v) for k, v in pair_deltas.items()},
    }


# ---------------------------------------------------------------------------
# G1 Entry Gate (logic unchanged — input now from score_lookup)
# ---------------------------------------------------------------------------

def g1_entry_gate(
    records: list[dict[str, Any]],
    score_lookup: dict[tuple[str, str, int], dict[str, float]],
    per_scene: dict[str, dict[str, Any]],
    per_seed: dict[str, dict[str, Any]],
    prompt_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Strict G1 Entry Gate. Returns G1 JUSTIFIED or G1 NOT JUSTIFIED."""
    if not score_lookup:
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

    def global_dim_mean(variant: str, dim: str) -> float | None:
        vals = []
        for rec in records:
            if rec["variant"] != variant or rec.get("status") not in ("PASS", "SKIPPED"):
                continue
            key = (rec["scene_id"], rec["variant"], rec["seed"])
            hs = score_lookup.get(key, {})
            if dim in hs:
                vals.append(hs[dim])
        return statistics.mean(vals) if vals else None

    c_identity = global_dim_mean("C", "identity_consistency")
    a_identity = global_dim_mean("A", "identity_consistency")
    check_identity = c_identity is not None and a_identity is not None and c_identity >= a_identity - 0.1

    c_intent = global_dim_mean("C", "intent_fidelity")
    a_intent = global_dim_mean("A", "intent_fidelity")
    check_intent = c_intent is not None and a_intent is not None and c_intent >= a_intent - 0.1

    c_hall = global_dim_mean("C", "hallucination")
    a_hall = global_dim_mean("A", "hallucination")
    check_hallucination = c_hall is not None and a_hall is not None and c_hall <= a_hall + 0.5

    seeds_improved = 0
    for seed, sdata in per_seed.items():
        c_mean = sdata["C"]["mean"]
        a_mean = sdata["A"]["mean"]
        if c_mean is not None and a_mean is not None and c_mean > a_mean:
            seeds_improved += 1
    check_seeds = seeds_improved >= 2

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize G0.7 A/B/C evaluation results.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: print to stdout).")
    args = parser.parse_args()

    records = load_results()
    prompt_metrics = load_prompt_metrics()
    blind_map = load_blind_map()
    human_scores = load_and_validate_human_scores(blind_map)
    score_lookup = build_score_lookup(human_scores, blind_map)

    if not records:
        print("No results found. Run run_g07_eval.py first.", file=sys.stderr)
        return 1

    status_counts: dict[str, int] = {}
    for r in records:
        s = r.get("status", "UNKNOWN")
        status_counts[s] = status_counts.get(s, 0) + 1

    variant_stats = {v: compute_variant_stats(records, score_lookup, v) for v in ["A", "B", "C"]}
    pairwise = compute_pairwise_diffs(variant_stats)
    per_scene = compute_per_scene(records, score_lookup)
    per_seed = compute_per_seed(records, score_lookup)
    paired = compute_paired_deltas(records, score_lookup)

    token_stats: dict[str, Any] = {}
    for variant in ["A", "B", "C"]:
        vmetrics = [m for m in prompt_metrics if m["variant"] == variant]
        token_stats[variant] = {
            "mean_tokens": round(statistics.mean(m["token_count_before_truncation"] for m in vmetrics), 1) if vmetrics else None,
            "max_tokens": max((m["token_count_before_truncation"] for m in vmetrics), default=None),
            "truncated_count": sum(1 for m in vmetrics if m["truncated"]),
            "mean_chars": round(statistics.mean(m["prompt_chars"] for m in vmetrics), 1) if vmetrics else None,
        }

    gate = g1_entry_gate(records, score_lookup, per_scene, per_seed, prompt_metrics)

    summary = {
        "eval_name": "g0.7-real-abc-eval",
        "total_records": len(records),
        "status_counts": status_counts,
        "human_scores_available": len(human_scores),
        "variant_stats": variant_stats,
        "pairwise_differences": pairwise,
        "paired_deltas": paired,
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
    if paired["wins_overall"]:
        print("\nPaired wins/ties/losses (adjusted_score):")
        for pkey, wt in paired["wins_overall"].items():
            print(f"  {pkey}: W={wt['win']} T={wt['tie']} L={wt['loss']} (n={paired['total_pairs'][pkey]})")
    print(f"\nG1 Gate: {gate['decision']}")
    print(f"  Reason: {gate['reason']}")
    for check_name, check_data in gate["checks"].items():
        print(f"  [{'PASS' if check_data['pass'] else 'FAIL'}] {check_name}: {check_data['detail']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
