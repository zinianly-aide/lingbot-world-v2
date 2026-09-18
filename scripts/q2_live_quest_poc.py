#!/usr/bin/env python3
"""Q2 live POC: causal DiT chunks -> progressive VAE -> paced bridge -> Quest.

This is intentionally a gated POC for M4 16GB. It does not alter model code,
signaling, or the Quest receiver. It first creates a matching image-condition
cache, then explicitly co-resides DiT + VAE only for the progressive run.

Use only after Q1 and Q1.5 pass. OOM/coexistence failure is a valid Q2 result;
do not disable the MPS high-watermark guard to force it through.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.streaming import (
    BufferedFrameBridgePublisher,
    FrameBridgeServer,
    LatestFrameStore,
    ProgressiveWanVaeDecoder,
    tap_causal_latent_chunks,
)
from wan.utils.device import set_autocast_device_type
from wan.utils.staged_cache import load_image_condition

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = "/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/snapshots/7e36a5f919f86cb4255cc9bfc30adb44963fbde1"
DEFAULT_ASSETS = "/Volumes/ssd/lingbot-assets"
DEFAULT_PROMPT = "Move the camera slowly forward while keeping the lone tree stable and centered."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--assets-dir", default=DEFAULT_ASSETS)
    p.add_argument("--image", default=str(REPO_ROOT / "examples/03/image.jpg"))
    p.add_argument("--action-path", default=str(REPO_ROOT / "examples/03"))
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompt-embeds", default=str(REPO_ROOT / "eval/e1.2/embeddings/single_subject_minicpm.safetensors"))
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--vae-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame-num", type=int, default=33,
                   help="33 with chunk-size 4 yields two DiT chunks / 29 aligned RGB frames")
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--bridge-host", default="127.0.0.1")
    p.add_argument("--bridge-port", type=int, default=8765)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--buffer-frames", type=int, default=120)
    p.add_argument("--tail-seconds", type=float, default=1.0)
    p.add_argument("--work-dir", default=str(REPO_ROOT / "eval/quest-streaming/q2_live"))
    return p.parse_args()


def require(path: str, label: str) -> Path:
    result = Path(path)
    if not result.exists():
        raise SystemExit(f"{label} not found: {result}")
    return result


def memory_stats(device: torch.device) -> dict[str, float]:
    if device.type == "mps":
        return {
            "allocatedGB": torch.mps.current_allocated_memory() / 1e9,
            "driverGB": torch.mps.driver_allocated_memory() / 1e9,
        }
    if device.type == "cuda":
        return {
            "allocatedGB": torch.cuda.memory_allocated(device) / 1e9,
            "reservedGB": torch.cuda.memory_reserved(device) / 1e9,
        }
    return {}


class TrackingSink:
    def __init__(self, decoder, publisher, device: torch.device) -> None:
        self.decoder = decoder
        self.publisher = publisher
        self.device = device
        self.chunks: list[dict] = []

    def on_latent_chunk(self, event, latent) -> None:
        t0 = time.monotonic()
        frames = self.decoder.decode_chunk(latent)
        decode_sec = time.monotonic() - t0
        enqueue_t0 = time.monotonic()
        enqueued = self.publisher.publish_chunk(frames)
        enqueue_sec = time.monotonic() - enqueue_t0
        self.chunks.append({
            "event": event.to_dict(),
            "rgbShape": [int(v) for v in frames.shape],
            "decodeSec": decode_sec,
            "jpegEnqueueSec": enqueue_sec,
            "enqueuedFrames": enqueued,
            "memoryAfterChunk": memory_stats(self.device),
        })


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0 or args.frame_num <= 0:
        raise SystemExit("frame-num and chunk-size must be > 0")

    require(args.ckpt_dir, "checkpoint")
    require(args.assets_dir, "assets")
    require(args.image, "image")
    require(args.action_path, "action path")
    require(args.prompt_embeds, "prompt embedding")

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    set_autocast_device_type(device.type)
    os.environ["LINGBOT_VAE_DTYPE"] = args.vae_dtype

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    image_condition = work_dir / "image_condition.safetensors"
    output_latents = work_dir / "latents.safetensors"
    report_path = work_dir / "report.json"

    cfg = WAN_CONFIGS["i2v-1.3B"]
    image = Image.open(args.image).convert("RGB")
    store = LatestFrameStore()
    store.set_state("generating")
    server = FrameBridgeServer(args.bridge_host, args.bridge_port, store=store)
    server_thread = threading.Thread(target=server.serve_forever, name="qps-frame-bridge", daemon=True)
    server_thread.start()

    publisher = None
    decoder = None
    pipe = None
    report: dict = {
        "gate": "Q2_LIVE_PROGRESSIVE_QUEST",
        "pass": False,
        "bridge": f"http://{args.bridge_host}:{args.bridge_port}",
        "frameNumRequested": args.frame_num,
        "chunkSize": args.chunk_size,
        "seed": args.seed,
        "vaeDtype": args.vae_dtype,
    }

    try:
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

        # Build an image-condition cache matching this longer, multi-chunk run.
        prep_started = time.monotonic()
        pipe.generate(
            args.prompt,
            image,
            action_path=args.action_path,
            chunk_size=args.chunk_size,
            max_area=480 * 832,
            frame_num=args.frame_num,
            seed=args.seed,
            offload_model=True,
            stage="encode-image",
            dump_image_condition=str(image_condition),
        )
        report["prepareImageConditionSec"] = time.monotonic() - prep_started
        _, condition_meta = load_image_condition(str(image_condition))
        total_chunks = (condition_meta.lat_f + args.chunk_size - 1) // args.chunk_size
        report["alignedFrameNum"] = int(condition_meta.aligned_frame_num)
        report["latentFrames"] = int(condition_meta.lat_f)
        report["totalChunks"] = int(total_chunks)
        if total_chunks < 2:
            raise RuntimeError(
                f"Q2 requires at least 2 DiT chunks; got {total_chunks}. "
                "Increase --frame-num or reduce --chunk-size."
            )

        # Explicit coexistence gate. No watermark override is used.
        pipe.load_dit()
        report["memoryAfterDit"] = memory_stats(device)
        pipe.load_vae()
        report["memoryAfterDitPlusVae"] = memory_stats(device)

        decoder = ProgressiveWanVaeDecoder(pipe.vae).start()
        publisher = BufferedFrameBridgePublisher(
            store,
            fps=float(cfg.sample_fps),
            jpeg_quality=args.jpeg_quality,
            max_frames=args.buffer_frames,
        )
        sink = TrackingSink(decoder, publisher, device)

        # generate-latents normally asserts VAE is absent in sequential mode.
        # For this isolated Q2 experiment both models are intentionally loaded,
        # so temporarily disable automatic stage load/unload. Model math is not
        # changed; the flag is restored in finally.
        pipe.sequential_load = False
        generation_started = time.monotonic()
        with tap_causal_latent_chunks(
            pipe,
            sink,
            generation_id=f"q2-seed{args.seed}",
            seed=args.seed,
            total_chunks=total_chunks,
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
                image_condition_file=str(image_condition),
                output_latents_file=str(output_latents),
            )
        generation_finished = time.monotonic()
        pipe.sequential_load = True

        drained = publisher.wait_empty(timeout=60.0)
        playback_stats = publisher.stats()
        store.set_state("completed")
        if args.tail_seconds > 0:
            time.sleep(args.tail_seconds)

        first_publish = playback_stats.first_publish_monotonic
        ttff = None if first_publish is None else first_publish - generation_started
        generation_sec = generation_finished - generation_started
        decoder_stats = decoder.stats()

        report.update({
            "generationSec": generation_sec,
            "ttffSec": ttff,
            "firstFrameBeforeGenerationEnd": bool(
                first_publish is not None and first_publish < generation_finished
            ),
            "chunks": sink.chunks,
            "chunkEvents": len(sink.chunks),
            "progressiveDecoder": {
                "latentFrames": decoder_stats.latent_frames,
                "outputFrames": decoder_stats.output_frames,
                "chunks": decoder_stats.chunks,
            },
            "playback": {
                "enqueued": playback_stats.enqueued,
                "published": playback_stats.published,
                "queueDepth": playback_stats.queue_depth,
                "maxQueueDepth": playback_stats.max_queue_depth,
                "drained": drained,
            },
            "memoryAfterGeneration": memory_stats(device),
            "bridgeStatus": store.status(),
        })

        report["pass"] = bool(
            len(sink.chunks) == total_chunks
            and decoder_stats.output_frames == condition_meta.aligned_frame_num
            and playback_stats.published == decoder_stats.output_frames
            and drained
            and report["firstFrameBeforeGenerationEnd"]
        )
    except Exception as exc:
        store.set_state("failed")
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["memoryAtFailure"] = memory_stats(device)
    finally:
        if pipe is not None:
            pipe.sequential_load = True
        if decoder is not None:
            decoder.close()
        if publisher is not None:
            publisher.close(drain=False, timeout=2.0)
        if pipe is not None:
            try:
                if getattr(pipe, "vae", None) is not None:
                    pipe.unload_vae()
            except Exception:
                pass
            try:
                if getattr(pipe, "model", None) is not None:
                    pipe.unload_dit()
            except Exception:
                pass
        server.shutdown()
        server_thread.join(timeout=2.0)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0 if report.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
