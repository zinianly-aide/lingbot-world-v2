#!/usr/bin/env python3
"""C7 buffered continuous mode: DiT/VAE reused + overlapping playback and generation.

The model's causal KV still resets per segment, but the network bridge, encoder
and playback pacing threads persist. The next segment begins preparing before
the previous segment's buffered frames are all played.
"""
from __future__ import annotations

import gc
import io
import json
import shutil
from collections import deque
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.quest_world_reuse import parse_args, validate


class TimedStore:
    """Record actual bridge frame cadence without retaining an unbounded trace."""

    def __init__(self, downstream, *, fps: float, segment_frames: int) -> None:
        self.downstream = downstream
        self.fps = fps
        self.segment_frames = segment_frames
        self.first_sequence = None
        self.last_sequence = None
        self.last_time = None
        self.interval_count = 0
        self.max_gap_ms = 0.0
        self.late_intervals = 0
        self.recent_boundary_gaps_ms = deque(maxlen=10)
        self._lock = threading.Lock()

    def publish(self, jpeg: bytes, sequence: int, pts_ms: float) -> None:
        # Update telemetry only after the frame was actually accepted.
        self.downstream.publish(jpeg, sequence, pts_ms)
        now = time.monotonic()
        with self._lock:
            if self.last_sequence is not None:
                if sequence != self.last_sequence + 1:
                    raise RuntimeError("non-monotonic or dropped sequence at timed bridge")
                interval_ms = (now - self.last_time) * 1000.0
                self.max_gap_ms = max(self.max_gap_ms, interval_ms)
                if interval_ms > 1500.0 / self.fps:
                    self.late_intervals += 1
                if sequence % self.segment_frames == 0:
                    self.recent_boundary_gaps_ms.append(round(interval_ms, 2))
                self.interval_count += 1
            else:
                self.first_sequence = sequence
            self.last_sequence = sequence
            self.last_time = now

    def stats(self) -> dict:
        with self._lock:
            return {
                "framesObserved": self.interval_count + (1 if self.first_sequence is not None else 0),
                "maxInterframeGapMs": round(self.max_gap_ms, 2),
                "lateIntervals": self.late_intervals,
                "recentBoundaryGapsMs": list(self.recent_boundary_gaps_ms),
            }


class SegmentSink:
    """Publish each decoded frame and retain final decoded RGB image for next I2V segment."""

    def __init__(self, decoder, publisher) -> None:
        self.decoder = decoder
        self.publisher = publisher
        self.frames = 0
        self.chunks = 0
        self.last_frame = None

    def on_latent_chunk(self, event, latent) -> None:
        for frames in self.decoder.decode_chunk_iter(latent):
            for i in range(frames.shape[1]):
                self.publisher.submit_frame(frames[:, i])
                self.frames += 1
            self.last_frame = frames[:, -1].detach().clone()
        self.chunks += 1


def model_rgb_to_image(frame):
    import torch
    from PIL import Image
    arr = (
        frame.detach().clamp(-1, 1).add(1).mul(127.5).round()
        .to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(arr).convert("RGB")


def run(args) -> int:
    expected = validate(args)
    if args.dry_run:
        print(json.dumps({"mode":"C7_BUFFERED_CONTINUOUS",
                          "framesPerSegment":expected,
                          "outputFps":args.output_fps,
                          "maxSegments":args.max_segments},indent=2))
        return 0
    import torch
    from PIL import Image
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.streaming import FrameBridgeServer, LatestFrameStore, BufferedFrameBridgePublisher, ProgressiveWanVaeDecoder
    from wan.streaming.frame_publisher_async import AsyncFrameBridgePublisher
    from wan.utils.device import set_autocast_device_type
    from wan.utils.staged_cache import load_image_condition
    from scripts.q2_live_quest_poc import memory_stats

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    device = torch.device("cuda")
    set_autocast_device_type(device.type)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cfg = WAN_CONFIGS["i2v-1.3B"]
    store = LatestFrameStore()
    server = FrameBridgeServer("127.0.0.1", args.bridge_port, store=store)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    pipe = None
    pacer = None
    encoder = None
    timed_store = None
    manifest = []
    image = Image.open(args.image).convert("RGB")
    prev_end = None
    try:
        store.set_state("generating")
        pipe = wan.WanI2VCausal(
            config=cfg, checkpoint_dir=str(args.ckpt_dir), device_id=device,
            rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=False,
            convert_model_dtype=False, local_attn_size=args.local_attn_size,
            sink_size=args.sink_size, infer_mode="causal_fast",
            assets_dir=str(args.assets_dir), prompt_embeds_file=str(args.prompt_embeds),
            sequential_load=True,
        )
        pipe.load_dit()
        pipe.load_vae()
        pipe.sequential_load = False

        # One single paced queue and encoder across ALL generated segments.
        # Source FrameStore sequence and PTS never reset, and queued old frames
        # can play while the next segment is being prepared.
        timed_store = TimedStore(store, fps=args.output_fps, segment_frames=expected)
        pacer = BufferedFrameBridgePublisher(timed_store, fps=args.output_fps,
                                             jpeg_quality=90, max_frames=48)
        encoder = AsyncFrameBridgePublisher(pacer, fps=args.output_fps,
                                            jpeg_quality=90, raw_queue_frames=8).start()
        print("C7 buffered persistent bridge + model loaded",flush=True)

        segment = 0
        while not args.max_segments or segment < args.max_segments:
            started = time.monotonic()
            d = args.work_dir / f"segment-{segment:04d}"
            d.mkdir(exist_ok=True)
            cache = d / "image_condition.safetensors"
            pipe.generate(
                args.prompt, image, action_path=str(args.action_path),
                chunk_size=args.chunk_size, max_area=args.max_area,
                frame_num=args.frame_num, seed=args.seed + segment,
                offload_model=True, stage="encode-image",
                dump_image_condition=str(cache),
            )
            _, meta = load_image_condition(str(cache))
            if meta.aligned_frame_num != expected:
                raise RuntimeError("unexpected segment alignment")

            decoder = ProgressiveWanVaeDecoder(pipe.vae).start()
            sink = SegmentSink(decoder, encoder)
            pipe.latent_chunk_sink = sink
            gen_started = time.monotonic()
            try:
                pipe.generate(
                    args.prompt, image, action_path=str(args.action_path),
                    chunk_size=args.chunk_size, max_area=args.max_area,
                    frame_num=args.frame_num, seed=args.seed + segment,
                    offload_model=True, stage="generate-latents",
                    image_condition_file=str(cache),
                )
            finally:
                pipe.latent_chunk_sink = None
                decoder.close()

            if sink.frames != expected or sink.last_frame is None:
                raise RuntimeError(f"segment {segment}: expected {expected} frames, got {sink.frames}")
            # Crucial: do NOT drain the playback queue here. Start the next
            # model iteration while frames from this one continue playing.
            image = model_rgb_to_image(sink.last_frame)
            image.save(d / "last_frame.png")
            del sink
            del decoder
            segment_end = time.monotonic()
            rec = {
                "segment": segment, "frames": expected,
                "firstGlobalSequence": segment * expected,
                "lastGlobalSequence": (segment + 1) * expected - 1,
                "generationSec": round(segment_end - gen_started, 3),
                "segmentWallSec": round(segment_end - started, 3),
                "gapBetweenGenerationEndSec": round(started - prev_end, 3) if prev_end else None,
                "encoderEnqueued": encoder.sequence,
                "bridgePublished": store.snapshot().sequence + 1,
                "queuedFrames": pacer.stats().queue_depth,
                "memory": memory_stats(device),
            }
            manifest.append(rec)
            (args.work_dir / "session.json").write_text(json.dumps(manifest,indent=2)+"\n")
            print("C7_BUFFERED_SEGMENT_PASS",json.dumps(rec),flush=True)
            prev_end = segment_end
            segment += 1
            old_index = segment - args.retain_segments
            if old_index >= 0:
                old_dir = args.work_dir / f"segment-{old_index:04d}"
                if old_dir.is_dir():
                    shutil.rmtree(old_dir)
            gc.collect()
            torch.cuda.empty_cache()

        encoder.close()           # Flush raw frames through JPEG encoder
        if not pacer.wait_empty(timeout=180):
            raise RuntimeError("frame pacer did not drain")
        final_seq = store.snapshot().sequence
        if final_seq != segment * expected - 1:
            raise RuntimeError(f"expected final global seq={segment*expected-1}, got {final_seq}")
        store.set_state("completed")
        print("C7_BUFFERED_PASS",json.dumps({
            "segments":segment, "frames":segment*expected,
            "lastSeq":final_seq,"queueMaxDepth":pacer.stats().max_queue_depth,
            "cadence":timed_store.stats(),
        }),flush=True)
        if args.hold_seconds:
            time.sleep(args.hold_seconds)
        return 0
    finally:
        if pipe is not None:
            pipe.latent_chunk_sink = None
            pipe.sequential_load = True
            if getattr(pipe, "vae", None) is not None:
                pipe.unload_vae()
            if getattr(pipe, "model", None) is not None:
                pipe.unload_dit()
        if encoder is not None:
            encoder.close()
        if pacer is not None:
            pacer.close(drain=False, timeout=4)
        server.shutdown()
        server_thread.join(timeout=3)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
