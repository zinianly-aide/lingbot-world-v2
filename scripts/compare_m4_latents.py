#!/usr/bin/env python3
"""Compare staged and full M4 generated-latent caches.

Exact (bitwise) equality is the default. Tolerances are opt-in and are never
silently relaxed, so this can be used as the M4 full-vs-staged equivalence gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Allow execution as ``python scripts/compare_m4_latents.py`` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.staged_cache import load_generated_latents


_METADATA_FIELDS = (
    "format_version",
    "checkpoint_id",
    "seed",
    "requested_frame_num",
    "aligned_frame_num",
    "chunk_size",
    "h",
    "w",
    "lat_f",
    "lat_h",
    "lat_w",
    "dtype",
)


def compare_files(staged_path: str, full_path: str, *, atol: float = 0.0, rtol: float = 0.0) -> dict:
    staged, staged_meta = load_generated_latents(staged_path)
    full, full_meta = load_generated_latents(full_path)

    issues: list[str] = []
    if staged.shape != full.shape:
        issues.append(f"shape mismatch: staged={tuple(staged.shape)} full={tuple(full.shape)}")
    if staged.dtype != full.dtype:
        issues.append(f"dtype mismatch: staged={staged.dtype} full={full.dtype}")

    for field in _METADATA_FIELDS:
        left = getattr(staged_meta, field)
        right = getattr(full_meta, field)
        if left != right:
            issues.append(f"metadata {field} mismatch: staged={left!r} full={right!r}")

    staged_finite = bool(torch.isfinite(staged).all().item())
    full_finite = bool(torch.isfinite(full).all().item())
    if not staged_finite:
        issues.append("staged latents contain NaN/Inf")
    if not full_finite:
        issues.append("full latents contain NaN/Inf")

    exact = False
    close = False
    max_abs_diff = None
    mean_abs_diff = None
    if staged.shape == full.shape and staged.dtype == full.dtype and staged_finite and full_finite:
        exact = bool(torch.equal(staged, full))
        diff = (staged.to(torch.float32) - full.to(torch.float32)).abs()
        if diff.numel():
            max_abs_diff = float(diff.max().item())
            mean_abs_diff = float(diff.mean().item())
        else:
            max_abs_diff = 0.0
            mean_abs_diff = 0.0
        close = exact if atol == 0.0 and rtol == 0.0 else bool(
            torch.allclose(staged, full, atol=atol, rtol=rtol)
        )
        if not close:
            issues.append(
                f"latent values differ: max_abs_diff={max_abs_diff:.9g} "
                f"mean_abs_diff={mean_abs_diff:.9g} atol={atol:g} rtol={rtol:g}"
            )

    return {
        "equivalent": not issues and close,
        "exact": exact,
        "atol": atol,
        "rtol": rtol,
        "shape": list(staged.shape),
        "dtype": str(staged.dtype),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "issues": issues,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staged", help="staged generated_latents.safetensors")
    parser.add_argument("full", help="full-mode generated_latents.safetensors")
    parser.add_argument("--atol", type=float, default=0.0, help="absolute tolerance; default 0 (exact)")
    parser.add_argument("--rtol", type=float, default=0.0, help="relative tolerance; default 0 (exact)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.atol < 0 or args.rtol < 0:
        print("ERROR: tolerances must be non-negative", file=sys.stderr)
        return 2

    try:
        result = compare_files(args.staged, args.full, atol=args.atol, rtol=args.rtol)
    except Exception as exc:
        print(f"M4_EQUIVALENCE ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["equivalent"]:
        mode = "bitwise exact" if result["exact"] else "within explicit tolerance"
        print(
            f"M4_EQUIVALENCE PASS ({mode}): shape={tuple(result['shape'])} "
            f"dtype={result['dtype']} max_abs_diff={result['max_abs_diff']:.9g} "
            f"mean_abs_diff={result['mean_abs_diff']:.9g}"
        )
    else:
        print("M4_EQUIVALENCE FAIL")
        for issue in result["issues"]:
            print(f"  - {issue}")

    return 0 if result["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
