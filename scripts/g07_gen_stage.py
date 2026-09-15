#!/usr/bin/env python3
"""Run one generate.py stage in-process and record peak MPS memory + timing.

This wrapper executes generate.py via runpy so that torch.mps peak memory
stats live in the same process.  Results are written to a JSON file.

Usage:
    python scripts/g07_gen_stage.py --result-json /tmp/stage.json --repo-root /path/to/repo -- \
        --checkpoint_dir ... --stage encode-image ...
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a generate.py stage with MPS memory tracking.")
    parser.add_argument("--result-json", required=True, help="Path to write the result JSON.")
    parser.add_argument("--repo-root", required=True, help="Absolute path to the repository root.")
    args, generate_args = parser.parse_known_args()
    # Strip the "--" separator if present (parse_known_args leaves it in the remainder)
    if generate_args and generate_args[0] == "--":
        generate_args = generate_args[1:]

    repo_root = os.path.abspath(args.repo_root)
    sys.path.insert(0, repo_root)
    os.chdir(repo_root)

    import torch

    peak_mps = 0
    peak_driver = 0
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        try:
            torch.mps.reset_peak_memory_stats()
        except Exception:
            pass

    sys.argv = ["generate.py"] + generate_args

    start = time.time()
    status = "PASS"
    error = None
    try:
        runpy.run_path(os.path.join(repo_root, "generate.py"), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            status = "FAIL"
            error = f"SystemExit({exc.code})"
    except Exception as exc:
        status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - start

    if torch.backends.mps.is_available():
        try:
            peak_mps = int(torch.mps.max_memory_allocated())
        except Exception:
            peak_mps = int(torch.mps.current_allocated_memory())
        try:
            peak_driver = int(torch.mps.driver_allocated_memory())
        except Exception:
            peak_driver = 0

    result = {
        "status": status,
        "error": error,
        "elapsed_seconds": round(elapsed, 2),
        "peak_mps_allocated_bytes": peak_mps,
        "peak_driver_allocated_bytes": peak_driver,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.result_json)), exist_ok=True)
    with open(args.result_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
