#!/usr/bin/env python3
"""E0 Benchmark: UMT5-XXL vs MiniCPM5-2B text encoder.

Measures, per encoder, in an isolated subprocess (so large model weights do
not accumulate across encoders in the parent):

  * checkpoint load wall time
  * one prompt-encode wall time
  * peak RSS (ru_maxrss), peak torch.mps.current_allocated_memory(),
    peak torch.mps.driver_allocated_memory()
  * output shape / dtype

UMT5 path: reuses the M3.3 low-memory loader from scripts/encode_prompt.py
(mmap + meta + tensor assign).  Output: [L, 4096] bf16.

MiniCPM5 path: AutoModel.from_pretrained(openbmb/MiniCPM5-2B, torch_dtype=bf16),
tokenizer from HF, forward -> last_hidden_state [L, 2048] bf16.

If the MiniCPM5 checkpoint is still downloading (``.incomplete`` blobs present,
or no safetensors weights in the snapshot), the MiniCPM5 worker reports
"MiniCPM5 not available, skipping" and only UMT5 is measured.

Usage (parent mode):
    python scripts/e0_benchmark.py \
        --prompt "A red car is parked beside a tree." \
        --output /tmp/e0_benchmark.json

Worker mode (internal; invoked by the parent):
    python scripts/e0_benchmark.py --worker umt5    --result-json /tmp/umt5.json
    python scripts/e0_benchmark.py --worker minicpm5 --result-json /tmp/mcp.json
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DEFAULT_PROMPT = "A red car is parked beside a tree."
DEFAULT_UMT5_CHECKPOINT = "/Volumes/ssd/lingbot-assets/models_t5_umt5-xxl-enc-bf16.pth"
DEFAULT_UMT5_TOKENIZER = "/Volumes/ssd/lingbot-assets/google/umt5-xxl"
DEFAULT_MINICPM5_DIR = "/Volumes/ssd/huggingface/hub/models--openbmb--MiniCPM5-2B"
DEFAULT_DEVICE = "mps"


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def mps_allocated_mb() -> float:
    import torch
    if not torch.backends.mps.is_available():
        return 0.0
    return torch.mps.current_allocated_memory() / (1024 * 1024)


def mps_driver_mb() -> float:
    import torch
    if not torch.backends.mps.is_available():
        return 0.0
    try:
        return torch.mps.driver_allocated_memory() / (1024 * 1024)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# MiniCPM5 availability detection
# ---------------------------------------------------------------------------

def minicpm5_available(model_dir: str) -> tuple[bool, str]:
    """Return (ok, reason).  ok=False if checkpoint still downloading."""
    if not os.path.isdir(model_dir):
        return False, f"model dir missing: {model_dir}"
    blobs = os.path.join(model_dir, "blobs")
    if not os.path.isdir(blobs):
        return False, f"blobs dir missing: {blobs}"
    incomplete = [f for f in os.listdir(blobs) if f.endswith(".incomplete")]
    if incomplete:
        size_gb = 0.0
        for f in incomplete:
            try:
                size_gb += os.path.getsize(os.path.join(blobs, f)) / (1024 ** 3)
            except OSError:
                pass
        return False, f"download in progress ({len(incomplete)} incomplete blob(s), ~{size_gb:.2f} GB)"
    # Must have at least one real safetensors weight.
    has_weight = any(
        f.endswith(".safetensors") and not f.endswith(".incomplete")
        for f in os.listdir(blobs)
    )
    if not has_weight:
        return False, "no .safetensors weight blobs found"
    return True, "ok"


# ---------------------------------------------------------------------------
# Worker: UMT5 (reuses encode_prompt low-memory loader)
# ---------------------------------------------------------------------------

def run_umt5_worker(prompt: str, device: str, result_json: str) -> dict:
    import torch
    from scripts.encode_prompt import (
        load_umt5_low_memory,
        UMT5_XXL_ENCODER_CFG,
    )
    from wan.modules.tokenizers import HuggingfaceTokenizer
    from wan.configs.wan_i2v_1_3B import i2v_1_3B as config

    out: dict = {"encoder": "UMT5-XXL", "status": "PASS"}
    try:
        text_len = config.text_len
        # Pre-load RSS / reset MPS peak.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            try:
                torch.mps.reset_peak_memory_stats()
            except Exception:
                pass

        t_load0 = time.time()
        model = load_umt5_low_memory(DEFAULT_UMT5_CHECKPOINT, torch.bfloat16)
        model = model.to(torch.device(device))
        t_load = time.time() - t_load0

        tokenizer = HuggingfaceTokenizer(
            name=DEFAULT_UMT5_TOKENIZER, seq_len=text_len, clean="whitespace")
        ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)

        # Warmup + timed encode.
        with torch.no_grad():
            _ = model(ids, mask)
            t_enc0 = time.time()
            context = model(ids, mask)
            t_enc = time.time() - t_enc0

        seq_lens = mask.gt(0).sum(dim=1).long()
        ctx = context[0][:seq_lens[0]]
        out.update({
            "load_wall_s": round(t_load, 3),
            "encode_wall_s": round(t_enc, 3),
            "output_shape": list(ctx.shape),
            "output_dtype": str(ctx.dtype).replace("torch.", ""),
            "rss_peak_mb": round(rss_mb(), 1),
            "mps_alloc_peak_mb": round(mps_allocated_mb(), 1),
            "mps_driver_mb": round(mps_driver_mb(), 1),
        })
    except Exception as exc:  # noqa: BLE001
        out["status"] = "FAIL"
        out["error"] = f"{type(exc).__name__}: {exc}"

    os.makedirs(os.path.dirname(os.path.abspath(result_json)), exist_ok=True)
    with open(result_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


# ---------------------------------------------------------------------------
# Worker: MiniCPM5-2B (last hidden state)
# ---------------------------------------------------------------------------

def run_minicpm5_worker(prompt: str, device: str, result_json: str) -> dict:
    import torch
    out: dict = {"encoder": "MiniCPM5-2B", "status": "PASS"}
    try:
        ok, reason = minicpm5_available(DEFAULT_MINICPM5_DIR)
        if not ok:
            out["status"] = "SKIP"
            out["reason"] = f"MiniCPM5 not available, skipping: {reason}"
            with open(result_json, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
            return out

        from transformers import AutoModel, AutoTokenizer

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            try:
                torch.mps.reset_peak_memory_stats()
            except Exception:
                pass

        t_load0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MINICPM5_DIR, local_files_only=True)
        model = AutoModel.from_pretrained(
            DEFAULT_MINICPM5_DIR,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        t_load = time.time() - t_load0

        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            _ = model(**enc)  # warmup
            t_enc0 = time.time()
            outputs = model(**enc)
            t_enc = time.time() - t_enc0

        hidden = outputs.last_hidden_state[0].float()
        out.update({
            "load_wall_s": round(t_load, 3),
            "encode_wall_s": round(t_enc, 3),
            "output_shape": list(hidden.shape),
            "output_dtype": "torch.float32",  # adapter expects fp32-in / adapter casts
            "rss_peak_mb": round(rss_mb(), 1),
            "mps_alloc_peak_mb": round(mps_allocated_mb(), 1),
            "mps_driver_mb": round(mps_driver_mb(), 1),
        })
    except Exception as exc:  # noqa: BLE001
        out["status"] = "FAIL"
        out["error"] = f"{type(exc).__name__}: {exc}"

    os.makedirs(os.path.dirname(os.path.abspath(result_json)), exist_ok=True)
    with open(result_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


# ---------------------------------------------------------------------------
# Parent: spawn workers, collect, print table
# ---------------------------------------------------------------------------

def spawn_worker(worker: str, prompt: str, device: str, tmp_dir: str) -> dict:
    result_json = os.path.join(tmp_dir, f"e0_bench_{worker}.json")
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--worker", worker,
        "--prompt", prompt,
        "--device", device,
        "--result-json", result_json,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0 and not os.path.exists(result_json):
        return {
            "encoder": worker, "status": "FAIL",
            "error": f"subprocess exit {proc.returncode}: {proc.stderr[-800:]}",
        }
    with open(result_json, "r", encoding="utf-8") as fh:
        return json.load(fh)


def print_table(results: list[dict]) -> None:
    cols = ["encoder", "status", "load_wall_s", "encode_wall_s",
            "output_shape", "output_dtype", "rss_peak_mb",
            "mps_alloc_peak_mb", "mps_driver_mb", "reason", "error"]
    rows = []
    for r in results:
        rows.append({c: r.get(c, "") for c in cols})
    # Print as a simple aligned table.
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main() -> int:
    parser = argparse.ArgumentParser(description="E0 encoder benchmark: UMT5 vs MiniCPM5-2B.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", default=None, help="JSON output path.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--worker", choices=["umt5", "minicpm5"], default=None,
                        help="Internal worker mode (parent omits this).")
    parser.add_argument("--result-json", default=None,
                        help="Worker: where to write per-worker JSON.")
    args = parser.parse_args()

    if args.worker:
        # Worker mode: run one encoder.
        if args.worker == "umt5":
            run_umt5_worker(args.prompt, args.device, args.result_json)
        else:
            run_minicpm5_worker(args.prompt, args.device, args.result_json)
        return 0

    # Parent mode.
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="e0_bench_")
    results = []
    print(f"Prompt: {args.prompt}")
    print(f"Device: {args.device}\n")

    # UMT5 worker.
    print(">>> Running UMT5-XXL worker (subprocess)...")
    results.append(spawn_worker("umt5", args.prompt, args.device, tmp_dir))

    # MiniCPM5 worker.
    print(">>> Running MiniCPM5-2B worker (subprocess)...")
    results.append(spawn_worker("minicpm5", args.prompt, args.device, tmp_dir))

    print()
    print_table(results)

    final = {
        "prompt": args.prompt,
        "device": args.device,
        "results": results,
    }
    out_path = args.output or os.path.join(REPO_ROOT, "e0_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(final, fh, indent=2)
    print(f"\nJSON written to: {out_path}")
    # Non-zero if either encoder hard-failed (SKIP is OK).
    failed = [r for r in results if r.get("status") == "FAIL"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
