#!/usr/bin/env python3
"""E2: Encode a text prompt with MiniCPM5-2B + trained TextAdapter.

Produces a Wan-DiT-compatible prompt embedding (.safetensors + sidecar .json)
in the exact format expected by ``wan.utils.prompt_embedding.load_prompt_embedding``:

    context tensor : [64, 4096] bfloat16   (adapter output, squeezed + cast)
    sidecar .json  : prompt_sha256, dtype, shape, hidden_dim=4096,
                     text_len=64, model_id="minicpm5-2b+adapter",
                     format_version="1.0", token_count, adapter_weights, ...

This script runs as an independent subprocess.  MiniCPM5 (bf16, frozen) and
the TextAdapter (fp32, frozen) are loaded, the prompt is encoded, the embedding
is saved, and then every model object is explicitly deleted and MPS cache is
emptied before the process exits.  The LingBot DiT is NEVER loaded here.

Usage:
    python scripts/e1_encode_with_adapter.py \
        --prompt "Move the camera slowly forward." \
        --adapter-weights eval/e1/best_adapter.safetensors \
        --output eval/e2/embeddings/single_subject_minicpm.safetensors \
        --device mps

Resume:
    If --output already exists and its sidecar JSON prompt_sha256 matches the
    given prompt, the encoding is SKIPPED (exit 0, result status="SKIPPED").
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from wan.adapters.text_adapter import TextAdapter  # noqa: E402
from wan.utils.prompt_embedding import (  # noqa: E402
    FORMAT_VERSION,
    load_prompt_embedding,
    prompt_sha256,
    save_prompt_embedding,
)

MODEL_ID = "minicpm5-2b+adapter"
DEFAULT_MINICPM5_DIR = (
    "/Volumes/ssd/huggingface/hub/models--openbmb--MiniCPM5-2B/snapshots/"
    "12a3808a956f869c767195e9266b59c4d21d92e2"
)
DEFAULT_ADAPTER_WEIGHTS = os.path.join(
    REPO_ROOT, "eval", "e1", "best_adapter.safetensors"
)
DEFAULT_ADAPTER_CONFIG = os.path.join(
    REPO_ROOT, "eval", "e1", "adapter_final", "config.json"
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2: encode prompt with MiniCPM5+Adapter.")
    p.add_argument("--prompt", type=str, default=None, help="Prompt text (mutually exclusive with --prompt-file).")
    p.add_argument("--prompt-file", type=str, default=None, help="Path to a UTF-8 text file containing the prompt.")
    p.add_argument("--adapter-weights", type=str, default=DEFAULT_ADAPTER_WEIGHTS,
                   help="Path to best_adapter.safetensors.")
    p.add_argument("--adapter-config", type=str, default=DEFAULT_ADAPTER_CONFIG,
                   help="Path to adapter config.json.")
    p.add_argument("--minicpm5-dir", type=str, default=DEFAULT_MINICPM5_DIR,
                   help="Path to MiniCPM5-2B snapshot dir.")
    p.add_argument("--output", type=str, required=True, help="Output .safetensors path.")
    p.add_argument("--device", type=str, default="mps", choices=["mps", "cpu", "cuda"])
    p.add_argument("--result-json", type=str, default=None,
                   help="Optional path to write the result JSON (also printed to stdout).")
    p.add_argument("--max-prompt-len", type=int, default=512,
                   help="Max tokenized length for MiniCPM5 tokenizer.")
    return p.parse_args()


def compute_sanity_metrics_from_frames(frames):
    """Compute sanity metrics from a numpy array of video frames [T,H,W,C].

    Pure-numpy, no video I/O — importable in unit tests with mock tensors.
    Mirrors the logic in scripts/run_g07_eval.py::compute_sanity_metrics.

    Args:
        frames: np.ndarray of shape [T, H, W, C], dtype uint8 or float.

    Returns:
        dict with frame_count, resolution, all_black, all_white, frozen_video,
        nan_inf_corruption, temporal_mad_mean, temporal_mad_max,
        per_frame_mean, per_frame_std.
    """
    import numpy as np

    metrics = {
        "frame_count": None,
        "resolution": None,
        "all_black": False,
        "all_white": False,
        "frozen_video": False,
        "nan_inf_corruption": False,
        "temporal_mad_mean": None,
        "temporal_mad_max": None,
        "per_frame_mean": [],
        "per_frame_std": [],
    }
    if frames is None or len(frames) == 0:
        return metrics

    frames = np.asarray(frames)
    metrics["frame_count"] = int(len(frames))
    metrics["resolution"] = f"{frames.shape[2]}x{frames.shape[1]}"

    if frames.dtype == np.uint8:
        frames_f = frames.astype(np.float32) / 255.0
    else:
        frames_f = frames.astype(np.float32)

    if not np.all(np.isfinite(frames_f)):
        metrics["nan_inf_corruption"] = True

    for i in range(len(frames_f)):
        metrics["per_frame_mean"].append(float(np.mean(frames_f[i])))
        metrics["per_frame_std"].append(float(np.std(frames_f[i])))

    overall_mean = float(np.mean(frames_f))
    metrics["all_black"] = overall_mean < 0.02
    metrics["all_white"] = overall_mean > 0.98

    if len(frames_f) > 1:
        diffs = np.abs(np.diff(frames_f, axis=0))
        mad_per_frame = np.mean(diffs, axis=(1, 2, 3))
        metrics["temporal_mad_mean"] = float(np.mean(mad_per_frame))
        metrics["temporal_mad_max"] = float(np.max(mad_per_frame))
        metrics["frozen_video"] = metrics["temporal_mad_mean"] < 0.001

    return metrics


def load_adapter_config(config_path: str) -> dict:
    """Read the adapter training config.json and return architecture params."""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return {
        "hidden_dim": cfg["hidden_dim"],
        "output_dim": cfg["output_dim"],
        "num_queries": cfg["num_queries"],
        "num_resampler_layers": cfg["num_resampler_layers"],
        "num_heads": cfg.get("num_heads", 8),
        "ffn_mult": cfg.get("ffn_mult", 4),
        "dtype": cfg.get("dtype", "float32"),
    }


def save_minicpm_embedding(
    output_path: str,
    context: torch.Tensor,
    prompt: str,
    token_count: int,
    adapter_weights_path: str,
    adapter_config_path: str,
) -> dict:
    """Save a MiniCPM5+Adapter context in the Wan prompt-embedding format.

    This is the testable core: it calls ``save_prompt_embedding`` (which writes
    the safetensors + base sidecar JSON) and then overrides the sidecar JSON
    so that ``model_id`` is "minicpm5-2b+adapter" and E2-specific fields are
    present.  The resulting file round-trips through ``load_prompt_embedding``.

    Args:
        output_path: target .safetensors path.
        context: tensor of shape [64, 4096] (already squeezed + cast).
        prompt: original prompt text.
        token_count: MiniCPM5 tokenizer token count.
        adapter_weights_path: path to the adapter weights .safetensors.
        adapter_config_path: path to the adapter config.json.

    Returns:
        The metadata dict written to the sidecar JSON.
    """
    # text_len=64 because the adapter always emits exactly 64 query tokens.
    save_prompt_embedding(
        output_path=output_path,
        context=context,
        prompt=prompt,
        text_len=64,
        dtype=context.dtype,
    )
    json_path = output_path.replace(".safetensors", ".json")
    with open(json_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["model_id"] = MODEL_ID
    meta["text_len"] = 64
    meta["token_count"] = token_count
    meta["adapter_weights"] = os.path.abspath(adapter_weights_path)
    meta["adapter_weights_sha256"] = sha256_file(adapter_weights_path)
    meta["adapter_config"] = os.path.abspath(adapter_config_path)
    meta["encoder"] = "minicpm5-2b+adapter"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def load_adapter(config_path: str, weights_path: str, device: torch.device) -> TextAdapter:
    """Build TextAdapter from config.json and load trained weights."""
    cfg = load_adapter_config(config_path)
    adapter = TextAdapter(
        hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["output_dim"],
        num_queries=cfg["num_queries"],
        num_resampler_layers=cfg["num_resampler_layers"],
        num_heads=cfg["num_heads"],
        ffn_mult=cfg["ffn_mult"],
    )
    state = load_file(weights_path)
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if missing:
        logging.warning("Adapter load: %d missing keys (first 3): %s", len(missing), missing[:3])
    if unexpected:
        logging.warning("Adapter load: %d unexpected keys (first 3): %s", len(unexpected), unexpected[:3])
    adapter = adapter.float().to(device)
    adapter.eval()
    adapter.requires_grad_(False)
    return adapter


def check_resume(output_path: str, prompt: str) -> bool:
    """Return True if output exists and sidecar prompt_sha256 matches."""
    if not os.path.isfile(output_path):
        return False
    json_path = output_path.replace(".safetensors", ".json")
    if not os.path.isfile(json_path):
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        return False
    return meta.get("prompt_sha256") == prompt_sha256(prompt)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("e2_encode")

    # ---- resolve prompt text ----
    if args.prompt is not None and args.prompt_file is not None:
        log.error("--prompt and --prompt-file are mutually exclusive.")
        return 2
    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as fh:
            prompt = fh.read().strip()
    else:
        log.error("Either --prompt or --prompt-file is required.")
        return 2

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ---- resume check ----
    if check_resume(output_path, prompt):
        log.info("SKIP: %s already exists with matching prompt_sha256.", output_path)
        result = {
            "status": "SKIPPED",
            "output": output_path,
            "prompt_sha256": prompt_sha256(prompt),
            "reason": "existing output with matching prompt_sha256",
        }
        print(json.dumps(result, indent=2))
        if args.result_json:
            with open(args.result_json, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
        return 0

    device = torch.device(args.device)
    if device.type == "mps":
        torch.mps.empty_cache()
        try:
            torch.mps.reset_peak_memory_stats()
        except Exception:
            pass

    wall_start = time.time()
    result: dict = {
        "status": "FAIL",
        "output": output_path,
        "prompt_sha256": prompt_sha256(prompt),
        "model_id": MODEL_ID,
    }

    try:
        # ---- load MiniCPM5 (bf16, frozen) ----
        from transformers import AutoModel, AutoTokenizer
        log.info("Loading MiniCPM5 tokenizer (use_fast=False) from %s ...", args.minicpm5_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            args.minicpm5_dir, local_files_only=True, use_fast=False)
        log.info("Loading MiniCPM5 model (bf16, frozen) ...")
        minicpm = AutoModel.from_pretrained(
            args.minicpm5_dir, torch_dtype=torch.bfloat16, local_files_only=True)
        minicpm = minicpm.to(device)
        minicpm.eval()
        minicpm.requires_grad_(False)

        # ---- load TextAdapter (fp32, frozen) ----
        log.info("Loading TextAdapter from %s (weights=%s) ...",
                 args.adapter_config, args.adapter_weights)
        adapter = load_adapter(args.adapter_config, args.adapter_weights, device)

        # ---- tokenize ----
        enc = tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_prompt_len,
        )
        token_count = int(enc["attention_mask"].sum().item())
        enc = {k: v.to(device) for k, v in enc.items()}
        log.info("Tokenized prompt: %d tokens, input_ids shape=%s",
                 token_count, tuple(enc["input_ids"].shape))

        # ---- forward MiniCPM5 ----
        log.info("Running MiniCPM5 forward ...")
        fwd_start = time.time()
        with torch.no_grad():
            out = minicpm(**enc)
        hidden = out.last_hidden_state  # [1, L, 2048] bf16
        log.info("MiniCPM5 forward done in %.1fs, hidden shape=%s dtype=%s",
                 time.time() - fwd_start, tuple(hidden.shape), hidden.dtype)

        # ---- forward adapter ----
        log.info("Running TextAdapter forward ...")
        adapter_start = time.time()
        with torch.no_grad():
            ctx = adapter(hidden, enc["attention_mask"])  # [1, 64, 4096] fp32
        log.info("Adapter forward done in %.1fs, ctx shape=%s dtype=%s",
                 time.time() - adapter_start, tuple(ctx.shape), ctx.dtype)

        # ---- cast + squeeze to [64, 4096] bf16 (match UMT5 context layout) ----
        ctx_bf16 = ctx[0].to(torch.bfloat16).contiguous()  # [64, 4096]
        assert ctx_bf16.shape == (64, 4096), f"unexpected shape {ctx_bf16.shape}"
        assert ctx_bf16.dtype == torch.bfloat16
        log.info("Final context: shape=%s dtype=%s all_finite=%s",
                 tuple(ctx_bf16.shape), ctx_bf16.dtype,
                 bool(torch.isfinite(ctx_bf16).all().item()))

        # ---- save via shared helper, then verify round-trip ----
        meta = save_minicpm_embedding(
            output_path=output_path,
            context=ctx_bf16,
            prompt=prompt,
            token_count=token_count,
            adapter_weights_path=args.adapter_weights,
            adapter_config_path=args.adapter_config,
        )

        # ---- verify round-trip through the shared loader ----
        loaded_ctx, loaded_meta = load_prompt_embedding(
            output_path, expected_prompt=prompt, expected_hidden_dim=4096,
            max_text_len=512,
        )
        assert tuple(loaded_ctx.shape) == (64, 4096), f"round-trip shape {loaded_ctx.shape}"
        assert loaded_meta["format_version"] == FORMAT_VERSION
        assert loaded_meta["hidden_dim"] == 4096
        assert loaded_meta["model_id"] == MODEL_ID
        log.info("Round-trip load OK: shape=%s dtype=%s",
                 tuple(loaded_ctx.shape), loaded_ctx.dtype)

        # ---- memory metrics ----
        peak_mps = peak_driver = 0
        if device.type == "mps":
            try:
                peak_mps = int(torch.mps.max_memory_allocated())
            except Exception:
                peak_mps = int(torch.mps.current_allocated_memory())
            try:
                peak_driver = int(torch.mps.driver_allocated_memory())
            except Exception:
                peak_driver = 0

        wall_total = time.time() - wall_start
        # context tensor sha256 over raw bytes
        ctx_bytes = ctx_bf16.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        result.update({
            "status": "PASS",
            "encode_wall_time_seconds": round(wall_total, 2),
            "peak_mps_allocated_bytes": peak_mps,
            "peak_driver_allocated_bytes": peak_driver,
            "peak_mps_mb": round(peak_mps / (1024 * 1024), 1),
            "peak_driver_mb": round(peak_driver / (1024 * 1024), 1),
            "context_shape": list(ctx_bf16.shape),
            "context_dtype": "bfloat16",
            "token_count": token_count,
            "prompt_chars": len(prompt),
            "metadata": loaded_meta,
            "context_sha256": hashlib.sha256(ctx_bytes).hexdigest(),
        })

    except Exception as exc:
        wall_total = time.time() - wall_start
        result.update({
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "encode_wall_time_seconds": round(wall_total, 2),
        })
        log.exception("Encoding failed.")
    finally:
        # ---- aggressive unload: MiniCPM5 + Adapter must NOT coexist with LingBot ----
        for name in ("minicpm", "adapter", "tokenizer"):
            if name in dir():
                try:
                    del globals()[name]
                except Exception:
                    pass
        # Also delete from locals if they leaked
        for var in list(globals().keys()):
            if var in ("minicpm", "adapter", "tokenizer"):
                try:
                    del globals()[var]
                except Exception:
                    pass
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(json.dumps(result, indent=2))
    if args.result_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.result_json)), exist_ok=True)
        with open(args.result_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
