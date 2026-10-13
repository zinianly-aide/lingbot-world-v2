"""CPU-only playback policy. Defaults to the legacy C7 fixed-rate behavior.

A video renderer/pacer can consume frames at a different rate from generation.
Smooth keeps every frame (delaying the viewer under overload); interactive may
skip frames that waited too long (trading completeness for reduced lag).
Model inference rate, published FPS and Quest panel refresh are distinct metrics.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class PlaybackOptions:
    mode: str = "fixed"  # fixed | smooth | interactive
    fps: float = 5.0  # initial pacer FPS; also immutable media PTS time base
    min_fps: float = 5.0
    max_fps: float = 8.0
    target_buffer_ms: float = 700.0
    max_frame_age_ms: float = 1000.0
    rate_step_fps: float = 0.5
    update_interval_sec: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "smooth", "interactive"}:
            raise ValueError("playback mode must be fixed, smooth or interactive")
        if not (0 < self.min_fps <= self.fps <= self.max_fps <= 120):
            raise ValueError("require 0 < min_fps <= fps <= max_fps <= 120")
        if self.target_buffer_ms <= 0 or self.max_frame_age_ms <= 0:
            raise ValueError("buffer and frame age limits must be positive")
        if self.rate_step_fps <= 0 or self.update_interval_sec <= 0:
            raise ValueError("rate update step/interval must be positive")


class PlaybackController:
    """Deterministic rate controller, fed with measured encoded-queue depth.

    In smooth, persistently full queues speed up playback, while near-empty
    queues slow it down. We do not claim this guarantees model throughput.
    Interactive is fixed at maximum requested playout rate: low lag matters
    more than displaying all of the frames produced by the model.
    """

    def __init__(self, options: PlaybackOptions):
        self.options = options
        self._lock = threading.Lock()
        self._fps = options.max_fps if options.mode == "interactive" else options.fps
        self._last_update: float | None = None
        self._adjustments = 0

    def period(self, *, depth: int, now: float) -> float:
        if depth < 0:
            raise ValueError("negative queue depth")
        with self._lock:
            o = self.options
            if o.mode == "smooth" and (self._last_update is None or now - self._last_update >= o.update_interval_sec):
                desired_depth = max(2, round(o.target_buffer_ms * self._fps / 1000.0))
                if depth >= desired_depth + 2:
                    adjusted = min(o.max_fps, self._fps + o.rate_step_fps)
                elif depth <= max(1, desired_depth - 2):
                    adjusted = max(o.min_fps, self._fps - o.rate_step_fps)
                else:
                    adjusted = self._fps
                if adjusted != self._fps:
                    self._adjustments += 1
                    self._fps = adjusted
                self._last_update = now
            return 1.0 / self._fps

    def should_drop(self, *, age_ms: float) -> bool:
        return self.options.mode == "interactive" and age_ms > self.options.max_frame_age_ms

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "mode": self.options.mode,
                "currentFps": round(self._fps, 3),
                "mediaTimebaseFps": self.options.fps,
                "minFps": self.options.min_fps,
                "maxFps": self.options.max_fps,
                "targetBufferMs": self.options.target_buffer_ms,
                "maxFrameAgeMs": self.options.max_frame_age_ms,
                "rateAdjustments": self._adjustments,
            }
