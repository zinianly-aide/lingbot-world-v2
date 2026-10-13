"""Paced frame publisher for progressive generated chunks.

A VAE chunk yields several RGB frames at once. Writing all of them directly to
LatestFrameStore would overwrite intermediate frames before the macOS sender can
poll them. This buffer JPEG-encodes the chunk, queues it, and publishes one
frame per playback interval.

When generation is slower than playback the last frame simply remains visible;
the next arriving chunk resumes at normal cadence instead of being burst out.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from io import BytesIO

import torch
from PIL import Image

from .frame_bridge import LatestFrameStore
from .playback_policy import PlaybackController, PlaybackOptions


@dataclass(frozen=True)
class BufferedPublisherStats:
    enqueued: int
    published: int
    queue_depth: int
    max_queue_depth: int
    first_publish_monotonic: float | None
    last_publish_monotonic: float | None
    dropped_frames: int = 0
    dropped_by_age: int = 0
    underflow_events: int = 0
    last_frame_age_ms: float | None = None
    max_frame_age_ms: float = 0.0
    published_fps: float = 0.0


@dataclass(frozen=True)
class _EncodedFrame:
    sequence: int
    pts_ms: float
    jpeg: bytes
    enqueued_monotonic: float


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
        policy: PlaybackOptions | None = None,
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
        self.policy = policy or PlaybackOptions(
            mode="fixed", fps=fps, min_fps=fps, max_fps=fps
        )
        if self.policy.fps != self.fps:
            raise ValueError("policy fps must match publisher media timebase fps")
        self._controller = PlaybackController(self.policy)
        self._queue: queue.Queue[_EncodedFrame | None] = queue.Queue(maxsize=max_frames)
        self._lock = threading.Lock()
        self._next_sequence = 0
        self._enqueued = 0
        self._published = 0
        self._max_queue_depth = 0
        self._first_publish: float | None = None
        self._last_publish: float | None = None
        self._closed = False
        self._worker_error: BaseException | None = None
        self._dropped_frames = 0
        self._dropped_by_age = 0
        self._underflow_events = 0
        self._last_frame_age_ms: float | None = None
        self._max_frame_age_ms = 0.0
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

    def publish_jpeg_bytes(self, jpeg: bytes, sequence: int) -> None:
        """Encode already-encoded JPEG bytes (used by external async encoder)."""
        if self._closed:
            raise RuntimeError("publisher is closed")
        item = _EncodedFrame(
            sequence=sequence,
            pts_ms=sequence * 1000.0 / self.fps,
            jpeg=jpeg,
            enqueued_monotonic=time.monotonic(),
        )
        self._enqueue(item)

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
                enqueued_monotonic=time.monotonic(),
            )
            self._enqueue(item)
            count += 1
        return count

    def _enqueue(self, item: _EncodedFrame) -> None:
        if self._closed:
            raise RuntimeError("publisher is closed")
        if self._worker_error is not None:
            raise RuntimeError("playback worker failed") from self._worker_error
        if self.policy.mode == "interactive":
            # Never wait behind a full encoded queue. Evict the oldest pending
            # frame, preserving the most recent view of the world.
            while True:
                try:
                    self._queue.put_nowait(item)
                    break
                except queue.Full:
                    try:
                        discarded = self._queue.get_nowait()
                    except queue.Empty:
                        continue  # consumer already took the queued item
                    if discarded is None:
                        self._queue.task_done()
                        raise RuntimeError("publisher already stopping")
                    self._queue.task_done()
                    with self._lock:
                        self._dropped_frames += 1
        else:
            deadline = time.monotonic() + self.enqueue_timeout
            while True:
                if self._worker_error is not None:
                    raise RuntimeError("playback worker failed") from self._worker_error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("frame playback buffer is full")
                try:
                    self._queue.put(item, timeout=min(0.25, remaining))
                    break
                except queue.Full:
                    continue
        with self._lock:
            self._enqueued += 1
            self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())

    def _run(self) -> None:
        try:
            self._consume_frames()
        except BaseException as exc:
            self._worker_error = exc
            logging.exception("frame playback worker crashed")

    def _consume_frames(self) -> None:
        last_wall: float | None = None
        while True:
            before_get = time.monotonic()
            item = self._queue.get()
            try:
                if item is None:
                    return
                now = time.monotonic()
                period = self._controller.period(depth=self._queue.qsize(), now=now)
                if last_wall is not None:
                    if now - before_get > 1.5 * period:
                        with self._lock:
                            self._underflow_events += 1
                    target = last_wall + period
                    if target > now:
                        time.sleep(target - now)
                now = time.monotonic()
                age_ms = max(0.0, (now - item.enqueued_monotonic) * 1000.0)
                if self._controller.should_drop(age_ms=age_ms) and not self._queue.empty():
                    with self._lock:
                        self._dropped_frames += 1
                        self._dropped_by_age += 1
                    continue
                self.store.publish(item.jpeg, item.sequence, item.pts_ms)
                last_wall = time.monotonic()
                with self._lock:
                    self._published += 1
                    self._last_frame_age_ms = age_ms
                    self._max_frame_age_ms = max(self._max_frame_age_ms, age_ms)
                    if self._first_publish is None:
                        self._first_publish = last_wall
                    self._last_publish = last_wall
            finally:
                self._queue.task_done()

    def stats(self) -> BufferedPublisherStats:
        with self._lock:
            elapsed = (self._last_publish - self._first_publish) if (
                self._first_publish is not None and self._last_publish is not None
            ) else 0.0
            return BufferedPublisherStats(
                enqueued=self._enqueued,
                published=self._published,
                queue_depth=self._queue.qsize(),
                max_queue_depth=self._max_queue_depth,
                first_publish_monotonic=self._first_publish,
                last_publish_monotonic=self._last_publish,
                dropped_frames=self._dropped_frames,
                dropped_by_age=self._dropped_by_age,
                underflow_events=self._underflow_events,
                last_frame_age_ms=self._last_frame_age_ms,
                max_frame_age_ms=self._max_frame_age_ms,
                published_fps=(self._published - 1) / elapsed if elapsed > 0 else 0.0,
            )

    def metrics(self) -> dict:
        s = self.stats()
        with self._queue.mutex:
            oldest = self._queue.queue[0] if self._queue.queue else None
        return {
            **self._controller.snapshot(),
            "enqueued": s.enqueued,
            "published": s.published,
            "queueDepth": s.queue_depth,
            "queueMaxDepth": s.max_queue_depth,
            "oldestQueueAgeMs": (round(max(0., (time.monotonic() - oldest.enqueued_monotonic) * 1000.), 2)
                                 if isinstance(oldest, _EncodedFrame) else 0.0),
            "lastPublishedFrameAgeMs": s.last_frame_age_ms,
            "maxPublishedFrameAgeMs": round(s.max_frame_age_ms, 2),
            "droppedFrames": s.dropped_frames,
            "droppedByAge": s.dropped_by_age,
            "underflowEvents": s.underflow_events,
            "averagePublishedFps": round(s.published_fps, 3),
        }

    def wait_empty(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._worker_error is not None:
                raise RuntimeError("playback worker failed") from self._worker_error
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.02)
        if self._worker_error is not None:
            raise RuntimeError("playback worker failed") from self._worker_error
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
        if self._worker_error is not None:
            raise RuntimeError("playback worker failed") from self._worker_error
