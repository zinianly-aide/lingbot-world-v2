#!/usr/bin/env python3
"""Run the VLM-only G0 smoke test and write reproducible run metadata."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="MiniCPM-V 4.6 G0 smoke test")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output-dir", default="g0-smoke")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    info: dict[str, object] = {
        "status": "NOT RUN",
        "model": args.model,
        "transformers_version": _version("transformers"),
        "torch_version": _version("torch"),
        "device": args.device,
        "dtype": None,
        "load_time_s": None,
        "inference_time_s": None,
        "peak_memory": None,
        "memory_after_release": None,
        "fallback": None,
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        from PIL import Image
        from world_condition import MiniCPMVPerceiver, save_world_condition, save_world_prompt
    except Exception as exc:
        info["error"] = f"dependency import failed: {type(exc).__name__}: {exc}"
        _write(output_dir / "run_info.json", info)
        return 2
    if not Path(args.image).is_file():
        info["error"] = f"image not found: {args.image}"
        _write(output_dir / "run_info.json", info)
        return 2

    try:
        import torch

        if args.device == "auto":
            info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if str(info["device"]).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
    except Exception as exc:
        info["error"] = f"torch unavailable: {type(exc).__name__}: {exc}"
        _write(output_dir / "run_info.json", info)
        return 2

    perceiver = MiniCPMVPerceiver(model_name=args.model, device=args.device)
    try:
        image = Image.open(args.image).convert("RGB")
        load_start = time.perf_counter()
        perceiver.load()
        info["load_time_s"] = round(time.perf_counter() - load_start, 3)
        info["dtype"] = str(getattr(perceiver.model, "dtype", None))
        infer_start = time.perf_counter()
        result = perceiver.analyze(image, user_prompt=args.prompt)
        info["inference_time_s"] = round(time.perf_counter() - infer_start, 3)
        info["fallback"] = result.used_fallback
        info["error"] = result.error
        save_world_condition(result.world, output_dir / "world_condition.json")
        save_world_prompt(
            __import__("world_condition").compose_world_prompt(result.world, args.prompt),
            output_dir / "world_prompt.txt",
        )
        try:
            import torch

            if str(info["device"]).startswith("cuda"):
                info["peak_memory"] = {"cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
            else:
                info["peak_memory"] = {"ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        except Exception:
            pass
        info["status"] = "PASS" if not result.used_fallback else "FALLBACK"
        return 0 if not result.used_fallback else 1
    except Exception as exc:
        info["error"] = f"smoke failed: {type(exc).__name__}: {exc}"
        return 2
    finally:
        perceiver.release()
        try:
            import torch

            if str(info["device"]).startswith("cuda"):
                info["memory_after_release"] = {"cuda_allocated_bytes": torch.cuda.memory_allocated()}
            else:
                info["memory_after_release"] = {"ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        except Exception:
            pass
        _write(output_dir / "run_info.json", info)


if __name__ == "__main__":
    raise SystemExit(main())
