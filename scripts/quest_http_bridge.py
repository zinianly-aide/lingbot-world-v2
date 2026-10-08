"""Forward C6 generated JPEG frames to a separately owned persistent bridge.

Standard-library only, suitable for a child generation process. The parent
session owns the listening port and can start a fresh model process without
dropping the Mac Sender's HTTP endpoint.
"""
from __future__ import annotations

import json
from urllib.request import Request, urlopen


def request_json(url: str, *, payload: bytes | None = None,
                 headers: dict[str, str] | None = None, timeout: float = 5.0) -> dict:
    req = Request(url, data=payload, headers=headers or {},
                  method="POST" if payload is not None else "GET")
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class RemoteFrameStore:
    """Translate segment-local frame indices to monotonic session indices."""

    def __init__(self, bridge_url: str, *, sequence_offset: int,
                 output_fps: float, timeout: float = 5.0) -> None:
        if sequence_offset < 0 or output_fps <= 0:
            raise ValueError("sequence_offset must be >=0 and output_fps >0")
        self.url = bridge_url.rstrip("/")
        self.sequence_offset = sequence_offset
        self.output_fps = output_fps
        self.timeout = timeout
        self._last_local_sequence = -1

    def set_state(self, state: str) -> None:
        if state not in {"idle", "generating", "streaming", "completed", "failed"}:
            raise ValueError("invalid bridge state")
        request_json(self.url + "/v1/state",
                     payload=json.dumps({"state": state}).encode(),
                     headers={"Content-Type": "application/json"},
                     timeout=self.timeout)

    def publish(self, jpeg: bytes, sequence: int, pts_ms: float) -> None:
        if sequence <= self._last_local_sequence:
            raise ValueError("nonmonotonic segment sequence")
        global_sequence = self.sequence_offset + sequence
        global_pts_ms = global_sequence * (1000.0 / self.output_fps)
        body = request_json(
            self.url + "/v1/frame",
            payload=jpeg,
            headers={
                "Content-Type": "image/jpeg",
                "X-QPS-Frame-Seq": str(global_sequence),
                "X-QPS-PTS-Ms": f"{global_pts_ms:.3f}",
            },
            timeout=self.timeout,
        )
        if body.get("sequence") != global_sequence:
            raise RuntimeError("persistent bridge rejected frame sequence")
        self._last_local_sequence = sequence

    def status(self) -> dict:
        return request_json(self.url + "/v1/status", timeout=self.timeout)
