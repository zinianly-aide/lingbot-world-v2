#!/usr/bin/env python3
"""Q2 live POC: causal DiT chunks -> progressive VAE -> paced bridge -> Quest.

This is a gated POC for Apple MPS and single-GPU CUDA validation. It does not alter model code,
signaling, or the Quest receiver. It first creates a matching image-condition
cache, then explicitly co-resides DiT + VAE only for the progressive run.

Use only after Q1 and Q1.5 pass. OOM/coexistence failure is a valid Q2 result;
do not disable the MPS high-watermark guard to force it through.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.streaming import (
    BufferedFrameBridgePublisher,
    FrameBridgeServer,
    LatestFrameStore,
    ProgressiveWanVaeDecoder,
)
from wan.streaming.frame_publisher_async import AsyncFrameBridgePublisher
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
    p.add_argument("--max-area", type=int, default=480*832, help="H*W budget for spatial resolution")
    p.add_argument("--streaming-profile", choices=["performance", "interactive", "none"],
                    default="none", help="preset defaults: performance=chunk4 (C3), interactive=chunk2 (C5)")
    p.add_argument("--continuous-publish", action="store_true",
                   help="C6: publish every decoded VAE frame during generation instead of deferring tail frames")
    p.add_argument("--raw-queue-frames", type=int, default=6,
                   help="C6 raw CUDA/MPS frame queue depth before JPEG encoding")
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
    """Q2.5-C3: first frame immediate, tail staged as MPS uint8 tensors."""

    def __init__(self, decoder, downstream_publisher, device: torch.device, fps: float = 16.0, jpeg_quality: int = 90) -> None:
        self.decoder = decoder
        self.downstream = downstream_publisher
        self.device = device
        self.fps = float(fps)
        self.jpeg_quality = int(jpeg_quality)
        self.sequence = 0
        self.chunks: list[dict] = []
        self.total_readbacks = 0
        self._pending_tail_u8: list = []
        self._flush_sec = 0.0
        self._flush_jpeg_sec = 0.0
        self._uint8_convert_sec = 0.0
        self._deferred_bytes = 0
        self._deferred_count = 0
        self._preview_times: list[float] = []
        self._generation_start: float | None = None

    def _frame_to_jpeg(self, frame_hwc_u8) -> bytes:
        from io import BytesIO
        from PIL import Image
        out = BytesIO()
        Image.fromarray(frame_hwc_u8, mode="RGB").save(
            out, format="JPEG", quality=self.jpeg_quality, optimize=False
        )
        return out.getvalue()

    def _publish_numpy_frames(self, frames_hwc_u8):
        import numpy as np
        n = frames_hwc_u8.shape[0]
        for i in range(n):
            seq = self.sequence
            self.sequence += 1
            pts_ms = seq * 1000.0 / self.fps
            jpeg = self._frame_to_jpeg(np.ascontiguousarray(frames_hwc_u8[i]))
            self.downstream.publish_jpeg_bytes(jpeg, seq)

    def on_latent_chunk(self, event, latent) -> None:
        if self._generation_start is None:
            self._generation_start = time.monotonic()
        before_cleanup = memory_stats(self.device)
        if self.device.type == "mps":
            import gc
            torch.mps.synchronize()
            gc.collect()
            torch.mps.empty_cache()
            torch.mps.synchronize()
        after_cleanup = memory_stats(self.device)

        chunk_t0 = time.monotonic()
        first_slice_t = None
        first_frame_readback_sec = 0.0
        first_frame_jpeg_sec = 0.0
        total_frames = 0

        for idx, frame_slice in enumerate(self.decoder.decode_chunk_iter(latent)):
            if first_slice_t is None:
                first_slice_t = time.monotonic()
            if idx == 0:
                n_frames = int(frame_slice.shape[1])
                t0 = time.monotonic()
                frame_u8 = (
                    frame_slice.detach()
                    .clamp(-1, 1).add(1.0).mul(127.5).round()
                    .to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
                )
                first_frame_readback_sec = time.monotonic() - t0
                self.total_readbacks += 1
                t0 = time.monotonic()
                self._publish_numpy_frames(frame_u8)
                first_frame_jpeg_sec = time.monotonic() - t0
                self._preview_times.append(time.monotonic())
                total_frames += n_frames
                del frame_slice, frame_u8
            else:
                # Convert to uint8 ON MPS immediately, release float ref
                n_frames = int(frame_slice.shape[1])
                t0 = time.monotonic()
                # [3,N,H,W] -> [N,H,W,3] uint8 contiguous on MPS
                u8 = (
                    frame_slice.detach()
                    .clamp(-1, 1).add(1.0).mul(127.5).round()
                    .to(torch.uint8).permute(1, 2, 3, 0).contiguous()
                )
                self._uint8_convert_sec += time.monotonic() - t0
                self._pending_tail_u8.append(u8)
                self._deferred_bytes += u8.numel()
                self._deferred_count += n_frames
                total_frames += n_frames
                del frame_slice, u8

        decode_sec = time.monotonic() - chunk_t0
        after_decode = memory_stats(self.device)
        self.chunks.append({
            "event": event.to_dict(),
            "decodeSec": decode_sec,
            "firstSliceMs": (first_slice_t - chunk_t0) * 1000.0 if first_slice_t else None,
            "firstFrameReadbackSec": first_frame_readback_sec,
            "firstFrameJpegSec": first_frame_jpeg_sec,
            "deferredTailFrames": total_frames - 1 if first_slice_t else total_frames,
            "readbacksThisChunk": 1,
            "enqueuedFrames": total_frames,
            "beforeDecodeCleanup": before_cleanup,
            "afterDecodeCleanup": after_cleanup,
            "afterDecode": after_decode,
            "memoryAfterChunk": memory_stats(self.device),
        })

    def flush(self) -> None:
        if not self._pending_tail_u8:
            return
        t0 = time.monotonic()
        tail_cat = torch.cat(self._pending_tail_u8, dim=0)  # [N,H,W,3] uint8 MPS
        del self._pending_tail_u8
        self._pending_tail_u8 = []
        tail_u8 = tail_cat.cpu().numpy()
        del tail_cat
        self._flush_sec = time.monotonic() - t0
        self.total_readbacks += 1
        t0 = time.monotonic()
        self._publish_numpy_frames(tail_u8)
        self._flush_jpeg_sec = time.monotonic() - t0
        del tail_u8


class ContinuousTrackingSink:
    """C6: decode and enqueue every RGB frame as soon as its latent chunk is ready."""

    def __init__(self, decoder, async_publisher, device: torch.device) -> None:
        self.decoder = decoder
        self.publisher = async_publisher
        self.device = device
        self.sequence = 0
        self.chunks: list[dict] = []
        self.total_readbacks = 0
        self._generation_start: float | None = None
        self._preview_times: list[float] = []
        self._flush_sec = 0.0
        self._flush_jpeg_sec = 0.0
        self._uint8_convert_sec = 0.0
        self._deferred_bytes = 0
        self._deferred_count = 0

    def on_latent_chunk(self, event, latent) -> None:
        if self._generation_start is None:
            self._generation_start = time.monotonic()
        started = time.monotonic()
        frame_count = 0
        first_submit = None
        for frame_slice in self.decoder.decode_chunk_iter(latent):
            for i in range(int(frame_slice.shape[1])):
                if first_submit is None:
                    first_submit = time.monotonic()
                    self._preview_times.append(first_submit)
                self.publisher.submit_frame(frame_slice[:, i])
                frame_count += 1
                self.sequence += 1
        self.chunks.append({
            "event": event.to_dict(),
            "decodeSec": time.monotonic() - started,
            "firstSliceMs": None if first_submit is None else (first_submit - started) * 1000.0,
            "enqueuedFrames": frame_count,
            "memoryAfterChunk": memory_stats(self.device),
        })

    def flush(self) -> None:
        started = time.monotonic()
        self.publisher.close()
        self._flush_sec = time.monotonic() - started

    def async_stats(self) -> dict:
        return self.publisher.stats()


PROFILE_DEFAULTS = {
    "performance": {"chunk_size": 4, "max_area": 384*672},
    "interactive": {"chunk_size": 2, "max_area": 384*672},
}

def main() -> int:
    args = parse_args()
    if args.streaming_profile in PROFILE_DEFAULTS:
        prof = PROFILE_DEFAULTS[args.streaming_profile]
        # CLI explicit values only override if user passed them; since argparse
        # uses defaults, we apply profile defaults only when the arg equals its
        # default. Simple approach: profile sets defaults, CLI wins.
        if args.chunk_size == 4 and args.streaming_profile == "interactive":
            args.chunk_size = prof["chunk_size"]
        if args.max_area == 480*832 and args.streaming_profile in ("performance", "interactive"):
            args.max_area = prof["max_area"]
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
            max_area=args.max_area,
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
        downstream_publisher = BufferedFrameBridgePublisher(
            store,
            fps=float(cfg.sample_fps),
            jpeg_quality=args.jpeg_quality,
            max_frames=args.buffer_frames,
        )
        publisher = downstream_publisher
        async_publisher = None
        if args.continuous_publish:
            async_publisher = AsyncFrameBridgePublisher(
                downstream_publisher,
                fps=float(cfg.sample_fps),
                jpeg_quality=args.jpeg_quality,
                raw_queue_frames=args.raw_queue_frames,
            ).start()
            sink = ContinuousTrackingSink(decoder, async_publisher, device)
        else:
            sink = TrackingSink(
                decoder, downstream_publisher, device,
                fps=float(cfg.sample_fps), jpeg_quality=args.jpeg_quality,
            )

        # generate-latents normally asserts VAE is absent in sequential mode.
        # For this isolated Q2 experiment both models are intentionally loaded.
        pipe.sequential_load = False
        # Direct seam: set sink on the pipe instead of wrapping model.forward.
        pipe.latent_chunk_sink = sink
        generation_started = time.monotonic()
        pipe.generate(
            args.prompt,
            image,
            action_path=args.action_path,
            chunk_size=args.chunk_size,
            max_area=args.max_area,
            frame_num=args.frame_num,
            seed=args.seed,
            offload_model=True,
            stage="generate-latents",
            image_condition_file=str(image_condition),
            output_latents_file=str(output_latents),
        )
        pipe.latent_chunk_sink = None
        sink.flush()
        generation_finished = time.monotonic()
        pipe.sequential_load = True

        drained = downstream_publisher.wait_empty(timeout=60.0)
        playback_stats = downstream_publisher.stats()
        report["streamingProfile"] = args.streaming_profile
        report["continuousPublish"] = bool(args.continuous_publish)
        if args.continuous_publish:
            report["asyncPublisher"] = sink.async_stats()
        report["chunkSize"] = args.chunk_size
        report["maxArea"] = args.max_area
        report["totalReadbacks"] = sink.total_readbacks
        report["flushReadbackSec"] = sink._flush_sec
        report["flushJpegSec"] = sink._flush_jpeg_sec
        report["uint8ConvertSec"] = sink._uint8_convert_sec
        report["deferredUint8MB"] = round(sink._deferred_bytes / 1e6, 1)
        report["deferredFrameCount"] = sink._deferred_count
        report["previewCount"] = len(sink._preview_times)
        if sink._generation_start is not None:
            report["previewTimestampsRel"] = [round(t - sink._generation_start, 1) for t in sink._preview_times]
        if len(sink._preview_times) >= 2:
            report["previewIntervalSec"] = round(sink._preview_times[1] - sink._preview_times[0], 1)
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
            downstream_publisher.close(drain=False, timeout=2.0)
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
