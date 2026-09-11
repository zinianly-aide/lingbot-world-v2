#!/usr/bin/env python3
"""Run the VLM-only G0 smoke test and write reproducible run metadata.

Supports two perception backends:
  --vlm_backend transformers  (full BF16, requires recent transformers)
  --vlm_backend mlx           (4-bit quantised, requires mlx-vlm on Apple Silicon)

Also supports --world_condition_file to skip VLM entirely and reuse a
cached structured world description.
"""

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


def _mlx_version() -> str | None:
    try:
        import mlx.core as mx

        return getattr(mx, "__version__", "unknown")
    except Exception:
        return None


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="MiniCPM-V 4.6 G0 smoke test")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=None, help="Override model name for the chosen backend")
    parser.add_argument("--vlm_backend", choices=["transformers", "mlx"], default="transformers")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output-dir", default="g0-smoke")
    parser.add_argument(
        "--world_condition_file",
        default=None,
        help="Skip VLM and load a cached world_condition.json",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Default model per backend
    if args.model is None:
        args.model = {
            "transformers": "openbmb/MiniCPM-V-4.6",
            "mlx": "mlx-community/MiniCPM-V-4.6-4bit",
        }[args.vlm_backend]

    info: dict[str, object] = {
        "status": "NOT RUN",
        "backend": args.vlm_backend,
        "model": args.model,
        "transformers_version": _version("transformers"),
        "mlx_version": _mlx_version(),
        "mlx_vlm_version": _version("mlx_vlm"),
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

    # --world_condition_file: skip VLM entirely
    if args.world_condition_file:
        try:
            from world_condition import (
                WorldDescription,
                compose_world_prompt,
                load_world_condition,
                save_world_condition,
                save_world_prompt,
            )
        except Exception as exc:
            info["error"] = f"dependency import failed: {type(exc).__name__}: {exc}"
            _write(output_dir / "run_info.json", info)
            return 2
        wc_path = Path(args.world_condition_file)
        if not wc_path.is_file():
            info["error"] = f"world_condition_file not found: {wc_path}"
            _write(output_dir / "run_info.json", info)
            return 2
        try:
            world = load_world_condition(wc_path)
            save_world_condition(world, output_dir / "world_condition.json")
            save_world_prompt(
                compose_world_prompt(world, args.prompt),
                output_dir / "world_prompt.txt",
            )
            info["status"] = "PASS (cached)"
            info["fallback"] = False
            _write(output_dir / "run_info.json", info)
            return 0
        except Exception as exc:
            info["error"] = f"cached world load failed: {type(exc).__name__}: {exc}"
            _write(output_dir / "run_info.json", info)
            return 2

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

    perceiver = MiniCPMVPerceiver(
        model_name=args.model,
        device=args.device,
        backend=args.vlm_backend,
    )
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
            info["memory_after_release"] = {
                "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            }
        except Exception:
            pass
        _write(output_dir / "run_info.json", info)


if __name__ == "__main__":
    raise SystemExit(main())
