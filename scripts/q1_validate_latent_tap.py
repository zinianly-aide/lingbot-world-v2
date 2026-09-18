#!/usr/bin/env python3
"""Q1 real-model gate for the non-invasive causal latent tap.

Re-runs the frozen E1.2 single-subject smoke latent generation with the tap
installed and compares the generated latent tensor against the existing smoke
baseline. The sink records metadata only; it never copies x0 to CPU.

This intentionally validates *non-interference*. The current 13-frame smoke
with chunk_size=4 contains one DiT latent chunk, so multi-chunk ordering remains
covered by unit tests and must be exercised again with a longer Q2 run.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.streaming import tap_causal_latent_chunks
from wan.utils.device import set_autocast_device_type
from wan.utils.staged_cache import load_generated_latents, sha256_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = "/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/snapshots/7e36a5f919f86cb4255cc9bfc30adb44963fbde1"
DEFAULT_ASSETS = "/Volumes/ssd/lingbot-assets"
DEFAULT_PROMPT = "Move the camera slowly forward while keeping the lone tree stable and centered."


class MetadataSink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def on_latent_chunk(self, event, latent) -> None:
        # Do not detach/copy/synchronize. Q1 is specifically checking that an
        # observing sink can stay off the data path.
        payload = event.to_dict()
        payload["device"] = str(latent.device)
        payload["requiresGrad"] = bool(latent.requires_grad)
        self.events.append(payload)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--assets-dir", default=DEFAULT_ASSETS)
    p.add_argument("--image", default=str(REPO_ROOT / "examples/03/image.jpg"))
    p.add_argument("--action-path", default=str(REPO_ROOT / "examples/03"))
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompt-embeds", default=str(REPO_ROOT / "eval/e1.2/embeddings/single_subject_minicpm.safetensors"))
    p.add_argument("--image-condition", default=str(REPO_ROOT / "eval/g0.7/.cache/image_condition_single_subject.safetensors"))
    p.add_argument("--baseline-latents", default=str(REPO_ROOT / "eval/e1.2/smoke/work/latents.safetensors"))
    p.add_argument("--output-latents", default=str(REPO_ROOT / "eval/quest-streaming/q1_tap/latents_tapped.safetensors"))
    p.add_argument("--report", default=str(REPO_ROOT / "eval/quest-streaming/q1_tap/report.json"))
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame-num", type=int, default=13)
    p.add_argument("--chunk-size", type=int, default=4)
    return p.parse_args()


def require(path: str, label: str) -> Path:
    result = Path(path)
    if not result.exists():
        raise SystemExit(f"{label} not found: {result}")
    return result


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")

    require(args.ckpt_dir, "checkpoint")
    require(args.assets_dir, "assets")
    require(args.image, "image")
    require(args.action_path, "action path")
    require(args.prompt_embeds, "prompt embedding")
    require(args.image_condition, "image condition")
    require(args.baseline_latents, "baseline latents")

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    set_autocast_device_type(device.type)

    baseline, baseline_meta = load_generated_latents(args.baseline_latents)
    expected_chunks = math.ceil(int(baseline.shape[1]) / args.chunk_size)

    cfg = WAN_CONFIGS["i2v-1.3B"]
    pipe = wan.WanI2VCausal(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=False,
        local_attn_size=-1,
        sink_size=0,
        infer_mode="causal_fast",
        assets_dir=args.assets_dir,
        prompt_embeds_file=args.prompt_embeds,
        sequential_load=True,
    )

    # The tap wraps the live DiT object, so load it explicitly before entering
    # the context. generate-latents will still unload it normally at the end.
    pipe.load_dit()
    sink = MetadataSink()
    output_path = Path(args.output_latents)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    started = time.perf_counter()
    with tap_causal_latent_chunks(
        pipe,
        sink,
        generation_id="q1-e12-seed42",
        seed=args.seed,
        total_chunks=expected_chunks,
        fail_open=False,
    ):
        pipe.generate(
            args.prompt,
            image,
            action_path=args.action_path,
            chunk_size=args.chunk_size,
            max_area=480 * 832,
            frame_num=args.frame_num,
            seed=args.seed,
            offload_model=True,
            stage="generate-latents",
            image_condition_file=args.image_condition,
            output_latents_file=str(output_path),
        )
    elapsed = time.perf_counter() - started

    tapped, tapped_meta = load_generated_latents(str(output_path))
    if baseline.shape != tapped.shape:
        max_abs = float("inf")
        mean_abs = float("inf")
        exact = False
    else:
        diff = (baseline.float() - tapped.float()).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        exact = bool(torch.equal(baseline, tapped))

    event_indices = [int(e["chunk_index"]) for e in sink.events]
    event_starts = [int(e["latent_start"]) for e in sink.events]
    expected_indices = list(range(expected_chunks))
    events_ok = event_indices == expected_indices
    starts_ok = event_starts == [i * args.chunk_size for i in expected_indices]

    report = {
        "gate": "Q1_LATENT_TAP_EQUIVALENCE",
        "pass": bool(exact and events_ok and starts_ok),
        "note": (
            "13-frame/chunk_size=4 smoke contains one DiT chunk; use a longer "
            "generation before claiming multi-chunk real-model cadence."
        ),
        "seed": args.seed,
        "frameNum": args.frame_num,
        "chunkSize": args.chunk_size,
        "expectedChunks": expected_chunks,
        "eventCount": len(sink.events),
        "events": sink.events,
        "eventIndicesOk": events_ok,
        "eventStartsOk": starts_ok,
        "baselineShape": list(baseline.shape),
        "tappedShape": list(tapped.shape),
        "baselineSha256": sha256_tensor(baseline),
        "tappedSha256": sha256_tensor(tapped),
        "exactTensorEqual": exact,
        "maxAbs": max_abs,
        "meanAbs": mean_abs,
        "elapsedSec": elapsed,
        "baselineMetadata": baseline_meta.to_dict(),
        "tappedMetadata": tapped_meta.to_dict(),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
