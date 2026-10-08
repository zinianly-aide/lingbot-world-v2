#!/usr/bin/env python3
"""C7 GPU experiment: hold DiT/VAE weights in memory across generated segments.

Unlike quest_continuous_world.py (subprocess-per-segment fallback), this stays
in one CUDA process, reusing the loaded model, VAE and cached text embedding.
The causal KV and progressive VAE feature caches still reset at boundaries;
the last JPEG becomes the next segment's image condition.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.quest_continuous_world import aligned_frames, npy_first_dimension


class OffsetStore:
    """Translate per-segment publisher indices into global monotonic indices."""

    def __init__(self, downstream, offset: int, fps: float) -> None:
        if offset < 0 or fps <= 0:
            raise ValueError("offset >= 0 and FPS > 0 required")
        self.downstream = downstream
        self.offset = offset
        self.fps = fps

    def publish(self, jpeg: bytes, sequence: int, pts_ms: float) -> None:
        global_sequence = self.offset + sequence
        self.downstream.publish(jpeg, global_sequence, global_sequence * 1000.0 / self.fps)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--assets-dir", type=Path, required=True)
    p.add_argument("--image", type=Path, default=ROOT / "examples/03/image.jpg")
    p.add_argument("--action-path", type=Path, default=ROOT / "examples/03")
    p.add_argument("--prompt-embeds", type=Path,
                   default=ROOT / "eval/e1.2/embeddings/single_subject_minicpm.safetensors")
    p.add_argument("--prompt", default="Move the camera slowly forward while keeping the lone tree stable and centered.")
    p.add_argument("--frame-num", type=int, default=129)
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--local-attn-size", type=int, default=16)
    p.add_argument("--sink-size", type=int, default=0)
    p.add_argument("--max-area", type=int, default=258048)
    p.add_argument("--output-fps", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-segments", type=int, default=2, help="0=unbounded")
    p.add_argument("--bridge-port", type=int, default=8766)
    p.add_argument("--hold-seconds", type=float, default=0.0)
    p.add_argument("--retain-segments", type=int, default=3,
                   help="Number of recent segment working directories to keep")
    p.add_argument("--work-dir", type=Path, default=ROOT / "eval/quest-streaming/c7_reuse")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def validate(a) -> int:
    if a.bridge_port in (8765, 18765) or not 1 <= a.bridge_port <= 65535:
        raise ValueError("bridge-port is invalid or reserved for C5")
    if a.frame_num < 9 or a.frame_num % 4 != 1 or a.chunk_size <= 0:
        raise ValueError("frame-num must be 4n+1 with at least two latent chunks")
    if a.local_attn_size != -1 and (
        a.local_attn_size < 2 * a.chunk_size
        or a.sink_size < 0
        or a.sink_size + a.chunk_size > a.local_attn_size
    ):
        raise ValueError("rolling attention window too small")
    if a.output_fps <= 0 or a.max_area <= 0 or a.max_segments < 0 or a.hold_seconds < 0 or a.retain_segments < 2:
        raise ValueError("invalid positive numeric parameter")
    if not a.image.is_file():
        raise ValueError("source image missing")
    if npy_first_dimension(a.action_path / "poses.npy") < a.frame_num:
        raise ValueError("action path has insufficient camera poses")
    if not a.prompt_embeds.is_file():
        raise ValueError("prompt embeddings missing")
    aligned = aligned_frames(a.frame_num, a.chunk_size)
    if aligned < 5:
        raise ValueError("too few aligned RGB frames")
    return aligned


def run(a) -> int:
    expected_frames = validate(a)
    if a.dry_run:
        print(json.dumps({"mode":"C7_GPU_REUSE", "alignedFrames":expected_frames,
                          "bridge": f"http://127.0.0.1:{a.bridge_port}",
                          "maxSegments":a.max_segments}, indent=2))
        return 0

    import torch
    from PIL import Image
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.streaming import BufferedFrameBridgePublisher, FrameBridgeServer, LatestFrameStore, ProgressiveWanVaeDecoder
    from wan.streaming.frame_publisher_async import AsyncFrameBridgePublisher
    from wan.utils.device import set_autocast_device_type
    from wan.utils.staged_cache import load_image_condition
    from scripts.q2_live_quest_poc import ContinuousTrackingSink, memory_stats

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this experimental runner")
    device = torch.device("cuda")
    set_autocast_device_type(device.type)
    cfg = WAN_CONFIGS["i2v-1.3B"]
    a.work_dir.mkdir(parents=True, exist_ok=True)
    store = LatestFrameStore()
    server = FrameBridgeServer("127.0.0.1", a.bridge_port, store=store)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    pipe = None
    manifest = []
    image = Image.open(a.image).convert("RGB")
    offset = 0
    previous_published_at = None

    try:
        store.set_state("generating")
        pipe = wan.WanI2VCausal(
            config=cfg, checkpoint_dir=str(a.ckpt_dir), device_id=device,
            rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=False,
            convert_model_dtype=False, local_attn_size=a.local_attn_size,
            sink_size=a.sink_size, infer_mode="causal_fast",
            assets_dir=str(a.assets_dir), prompt_embeds_file=str(a.prompt_embeds),
            sequential_load=True,
        )
        pipe.load_dit()
        pipe.load_vae()
        pipe.sequential_load = False
        print("C7 reused DiT and VAE loaded", flush=True)
        segment = 0
        while not a.max_segments or segment < a.max_segments:
            started = time.monotonic()
            segment_dir = a.work_dir / f"segment-{segment:04d}"
            segment_dir.mkdir(exist_ok=True)
            image_cache = segment_dir / "image_condition.safetensors"
            store.set_state("generating")
            pipe.generate(
                a.prompt, image, action_path=str(a.action_path),
                chunk_size=a.chunk_size, max_area=a.max_area,
                frame_num=a.frame_num, seed=a.seed + segment,
                offload_model=True, stage="encode-image",
                dump_image_condition=str(image_cache),
            )
            _, metadata = load_image_condition(str(image_cache))
            if metadata.aligned_frame_num != expected_frames:
                raise RuntimeError("aligned frame count changed between segments")

            decoder = ProgressiveWanVaeDecoder(pipe.vae).start()
            base_publisher = BufferedFrameBridgePublisher(
                OffsetStore(store, offset, a.output_fps),
                fps=a.output_fps, jpeg_quality=90, max_frames=120,
            )
            encoder = AsyncFrameBridgePublisher(
                base_publisher, fps=a.output_fps, jpeg_quality=90,
                raw_queue_frames=8,
            ).start()
            sink = ContinuousTrackingSink(decoder, encoder, device)
            generated_at = time.monotonic()
            pipe.latent_chunk_sink = sink
            try:
                pipe.generate(
                    a.prompt, image, action_path=str(a.action_path),
                    chunk_size=a.chunk_size, max_area=a.max_area,
                    frame_num=a.frame_num, seed=a.seed + segment,
                    offload_model=True, stage="generate-latents",
                    image_condition_file=str(image_cache),
                )
                sink.flush()
                drained = base_publisher.wait_empty(timeout=60)
            finally:
                pipe.latent_chunk_sink = None
                encoder.close()
                base_publisher.close(drain=False, timeout=5)
                decoder.close()

            stats = base_publisher.stats()
            snap = store.snapshot()
            if not drained or stats.published != expected_frames or snap.sequence != offset + expected_frames - 1:
                raise RuntimeError(f"segment {segment} integrity failure: published={stats.published} seq={snap.sequence}")
            first = stats.first_publish_monotonic
            gap = (first - previous_published_at) if previous_published_at is not None and first else None
            previous_published_at = stats.last_publish_monotonic

            # Use only the final committed image as the next I2V condition.
            image = Image.open(io.BytesIO(snap.jpeg)).convert("RGB")
            if segment >= 2:
                previous_dir = a.work_dir / f"segment-{segment-2:04d}" / "image_condition.safetensors"
                previous_dir.unlink(missing_ok=True)
            image.save(segment_dir / "last_frame.png")
            result = {
                "segment": segment, "frames": expected_frames, "offset": offset,
                "generationSec": round(time.monotonic() - generated_at, 3),
                "segmentWallSec": round(time.monotonic() - started, 3),
                "ttffFromGenerationSec": round(first-generated_at, 3) if first else None,
                "boundaryGapSec": round(gap, 3) if gap is not None else None,
                "maxQueueDepth": stats.max_queue_depth,
                "memory": memory_stats(device),
                "lastSeq": snap.sequence,
            }
            manifest.append(result)
            (a.work_dir / "session.json").write_text(json.dumps(manifest, indent=2)+"\n")
            print("C7_REUSE_PASS", json.dumps(result), flush=True)
            offset += expected_frames
            segment += 1
            gc.collect()
            torch.cuda.empty_cache()
        store.set_state("completed")
        if a.hold_seconds:
            time.sleep(a.hold_seconds)
        return 0
    finally:
        if pipe is not None:
            pipe.latent_chunk_sink = None
            pipe.sequential_load = True
            if getattr(pipe, "vae", None) is not None:
                pipe.unload_vae()
            if getattr(pipe, "model", None) is not None:
                pipe.unload_dit()
        server.shutdown()
        worker.join(timeout=3)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
