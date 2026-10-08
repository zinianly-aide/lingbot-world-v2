"""Async two-stage frame publisher: VAE thread only enqueues raw frames.

Stage A (VAE thread): ``submit_frame(frame)`` reserves a sequence number and
pushes the raw MPS tensor into a bounded raw queue. It returns immediately.

Stage B (encoder worker, single thread): pops raw frames, does
MPS->CPU sync, uint8 conversion, JPEG encode, pushes encoded bytes to the
existing encoded queue consumed by the pacer/LatestFrameStore.

Single encoder worker guarantees order. Bounded raw queue gives backpressure
to the VAE thread without dropping frames.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import torch
from PIL import Image

from .buffered_publisher import BufferedFrameBridgePublisher


@dataclass
class PublishTiming:
    vaeSliceDecodeSec: float = 0.0
    submitFrameSec: float = 0.0
    rawQueueWaitSec: float = 0.0
    mpsToCpuSec: float = 0.0
    jpegEncodeSec: float = 0.0
    encodedQueueWaitSec: float = 0.0
    publishLatencySec: float = 0.0


@dataclass
class TimingStats:
    samples: list[PublishTiming] = field(default_factory=list)

    def add(self, t: PublishTiming) -> None:
        self.samples.append(t)

    def summarize(self) -> dict[str, dict[str, float]]:
        if not self.samples:
            return {}
        out: dict[str, dict[str, float]] = {}
        for key in (
            "submitFrameSec",
            "rawQueueWaitSec",
            "mpsToCpuSec",
            "jpegEncodeSec",
            "encodedQueueWaitSec",
            "publishLatencySec",
        ):
            vals = sorted(getattr(s, key) for s in self.samples)
            n = len(vals)
            out[key] = {
                "p50": vals[n // 2],
                "p95": vals[min(n - 1, int(n * 0.95))],
                "max": vals[-1],
                "total": sum(vals),
            }
        return out


class AsyncFrameBridgePublisher:
    """Two-stage async publisher.

    VAE thread calls :meth:`submit_frame` (raw ``[3,H,W]`` MPS tensor) and
    returns immediately. A background encoder worker does MPS->CPU + JPEG,
    then pushes bytes to :class:`LatestFrameStore` at the configured fps.
    """

    def __init__(
        self,
        downstream: BufferedFrameBridgePublisher,
        *,
        fps: float = 24.0,
        jpeg_quality: int = 90,
        raw_queue_frames: int = 3,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self.downstream = downstream
        self.fps = float(fps)
        self.jpeg_quality = int(jpeg_quality)
        self.raw_queue_frames = int(raw_queue_frames)

        self.sequence = 0
        self._raw_q: queue.Queue[Any] = queue.Queue(maxsize=self.raw_queue_frames)
        self._encoder_thread: threading.Thread | None = None
        self._encoder_stop = threading.Event()
        self._encoder_error: BaseException | None = None

        self.timing = TimingStats()
        self.raw_queue_max_depth = 0
        self.raw_queue_blocked_sec = 0.0
        self._lock = threading.Lock()

    # ---------- encoder worker ----------
    def _encoder_loop(self) -> None:
        try:
            while True:
                item = self._raw_q.get()
                if item is None:
                    self._raw_q.task_done()
                    return
                seq, frame, submit_t = item
                try:
                    self._encode_one(seq, frame, submit_t)
                finally:
                    self._raw_q.task_done()
        except BaseException as e:  # noqa: BLE001
            self._encoder_error = e
            logging.exception("encoder worker crashed")

    def _encode_one(self, seq: int, frame: torch.Tensor, submit_t: float) -> None:
        t = PublishTiming()
        t.rawQueueWaitSec = time.monotonic() - submit_t

        # MPS -> CPU + uint8
        t0 = time.monotonic()
        image = (
            frame.detach()
            .float()
            .clamp(-1, 1)
            .add(1.0)
            .mul(127.5)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        t.mpsToCpuSec = time.monotonic() - t0

        # JPEG
        t0 = time.monotonic()
        out = BytesIO()
        Image.fromarray(image, mode="RGB").save(
            out, format="JPEG", quality=self.jpeg_quality, optimize=False
        )
        jpeg_bytes = out.getvalue()
        t.jpegEncodeSec = time.monotonic() - t0

        # hand off JPEG bytes to the existing paced downstream publisher
        t0 = time.monotonic()
        self.downstream.publish_jpeg_bytes(jpeg_bytes, seq)
        t.publishLatencySec = time.monotonic() - t0

        self.timing.add(t)

    # ---------- VAE thread API ----------
    def start(self) -> "AsyncFrameBridgePublisher":
        if self._encoder_thread is not None:
            return self
        self._encoder_thread = threading.Thread(
            target=self._encoder_loop, name="q2-encoder", daemon=True
        )
        self._encoder_thread.start()
        return self

    def submit_frame(self, frame: torch.Tensor) -> int:
        """Reserve sequence and enqueue a raw ``[3,H,W]`` frame (non-blocking-ish)."""
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(f"frame must be [3,H,W], got {tuple(frame.shape)}")
        if self._encoder_error is not None:
            raise RuntimeError("encoder worker failed") from self._encoder_error

        seq = self.sequence
        self.sequence += 1

        t0 = time.monotonic()
        try:
            self._raw_q.put((seq, frame.detach(), t0), timeout=30.0)
        except queue.Full:
            self.raw_queue_blocked_sec += time.monotonic() - t0
            raise RuntimeError("raw queue stayed full for 30s")
        self.raw_queue_blocked_sec += time.monotonic() - t0

        with self._lock:
            self.raw_queue_max_depth = max(
                self.raw_queue_max_depth, self._raw_q.qsize()
            )
        return seq

    def publish_chunk(self, frames: torch.Tensor) -> int:
        """Backwards-compatible: submit each frame in ``[3,T,H,W]``."""
        if frames.ndim != 4 or frames.shape[0] != 3:
            raise ValueError(f"frames must be [3,T,H,W], got {tuple(frames.shape)}")
        published = 0
        for i in range(frames.shape[1]):
            self.submit_frame(frames[:, i])
            published += 1
        return published

    def close(self) -> None:
        """Stop encoder, drain queues, ensure no tail frames dropped."""
        if self._encoder_thread is None:
            return
        # signal stop
        self._raw_q.put(None)
        self._encoder_thread.join(timeout=60.0)
        self._encoder_thread = None
        if self._encoder_error is not None:
            raise self._encoder_error

    def stats(self) -> dict[str, Any]:
        return {
            "published": self.sequence,
            "rawQueueMaxDepth": self.raw_queue_max_depth,
            "rawQueueBlockedSec": self.raw_queue_blocked_sec,
            "timing": self.timing.summarize(),
        }
