"""Paced frame publisher for progressive generated chunks.

A VAE chunk yields several RGB frames at once. Writing all of them directly to
LatestFrameStore would overwrite intermediate frames before the macOS sender can
poll them. This buffer JPEG-encodes the chunk, queues it, and publishes one
frame per playback interval.

When generation is slower than playback the last frame simply remains visible;
the next arriving chunk resumes at normal cadence instead of being burst out.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from io import BytesIO

import torch
from PIL import Image

from .frame_bridge import LatestFrameStore


@dataclass(frozen=True)
class BufferedPublisherStats:
    enqueued: int
    published: int
    queue_depth: int
    max_queue_depth: int
    first_publish_monotonic: float | None
    last_publish_monotonic: float | None


@dataclass(frozen=True)
class _EncodedFrame:
    sequence: int
    pts_ms: float
    jpeg: bytes


class BufferedFrameBridgePublisher:
    """Bounded, lossless POC playback queue with explicit backpressure."""

    def __init__(
        self,
        store: LatestFrameStore,
        *,
        fps: float = 16.0,
        jpeg_quality: int = 90,
        max_frames: int = 120,
        enqueue_timeout: float = 10.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if max_frames <= 0:
            raise ValueError("max_frames must be > 0")
        self.store = store
        self.fps = float(fps)
        self.jpeg_quality = int(jpeg_quality)
        self.enqueue_timeout = float(enqueue_timeout)
        self._queue: queue.Queue[_EncodedFrame | None] = queue.Queue(maxsize=max_frames)
        self._lock = threading.Lock()
        self._next_sequence = 0
        self._enqueued = 0
        self._published = 0
        self._max_queue_depth = 0
        self._first_publish: float | None = None
        self._last_publish: float | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="qps-frame-pacer",
            daemon=True,
        )
        self._worker.start()

    def _jpeg(self, frame: torch.Tensor) -> bytes:
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(f"frame must be [3,H,W], got {tuple(frame.shape)}")
        array = (
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
        output = BytesIO()
        Image.fromarray(array).save(
            output,
            format="JPEG",
            quality=self.jpeg_quality,
            optimize=False,
        )
        return output.getvalue()

    def publish_chunk(self, frames: torch.Tensor) -> int:
        """Encode and enqueue ``[3,T,H,W]`` frames in temporal order.

        The bounded queue deliberately applies backpressure instead of dropping
        generated frames. A timeout is treated as a pipeline failure so a slow
        consumer cannot grow memory without bound.
        """
        if self._closed:
            raise RuntimeError("publisher is closed")
        if frames.ndim != 4 or frames.shape[0] != 3:
            raise ValueError(f"frames must be [3,T,H,W], got {tuple(frames.shape)}")

        count = 0
        for i in range(frames.shape[1]):
            with self._lock:
                sequence = self._next_sequence
                self._next_sequence += 1
            item = _EncodedFrame(
                sequence=sequence,
                pts_ms=sequence * 1000.0 / self.fps,
                jpeg=self._jpeg(frames[:, i]),
            )
            try:
                self._queue.put(item, timeout=self.enqueue_timeout)
            except queue.Full as exc:
                raise TimeoutError("frame playback buffer is full") from exc
            with self._lock:
                self._enqueued += 1
                self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
            count += 1
        return count

    def _run(self) -> None:
        period = 1.0 / self.fps
        last_wall: float | None = None
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                now = time.monotonic()
                if last_wall is not None:
                    target = last_wall + period
                    if target > now:
                        time.sleep(target - now)
                self.store.publish(item.jpeg, item.sequence, item.pts_ms)
                published_at = time.monotonic()
                last_wall = published_at
                with self._lock:
                    self._published += 1
                    if self._first_publish is None:
                        self._first_publish = published_at
                    self._last_publish = published_at
            finally:
                self._queue.task_done()

    def stats(self) -> BufferedPublisherStats:
        with self._lock:
            return BufferedPublisherStats(
                enqueued=self._enqueued,
                published=self._published,
                queue_depth=self._queue.qsize(),
                max_queue_depth=self._max_queue_depth,
                first_publish_monotonic=self._first_publish,
                last_publish_monotonic=self._last_publish,
            )

    def wait_empty(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.02)
        return self._queue.unfinished_tasks == 0

    def close(self, *, drain: bool = True, timeout: float = 30.0) -> None:
        if self._closed:
            return
        if drain:
            self.wait_empty(timeout=timeout)
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # If drain timed out, remove one stale queued frame so the worker can
            # observe the sentinel and terminate rather than leaking a thread.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(None)
        self._worker.join(timeout=timeout)
