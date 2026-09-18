"""Localhost frame bridge used by the Quest streaming POC.

Producer side POSTs JPEG frames.  QuestPhoneStream's macOS sender polls the
latest frame and turns it into a Canvas MediaStream, reusing the existing WebRTC
session/signaling path.

This is intentionally a localhost-only POC transport.  It is not a replacement
for the existing QuestPhoneStream protocol.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MAX_FRAME_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class FrameSnapshot:
    jpeg: bytes | None
    sequence: int
    pts_ms: float
    updated_monotonic: float
    state: str


class LatestFrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._sequence = -1
        self._pts_ms = 0.0
        self._updated_monotonic = 0.0
        self._state = "idle"

    def publish(self, jpeg: bytes, sequence: int, pts_ms: float) -> None:
        if not jpeg:
            raise ValueError("jpeg frame is empty")
        if len(jpeg) > MAX_FRAME_BYTES:
            raise ValueError(f"jpeg frame exceeds {MAX_FRAME_BYTES} bytes")
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("payload is not a complete JPEG")
        with self._lock:
            if sequence <= self._sequence:
                return
            self._jpeg = bytes(jpeg)
            self._sequence = int(sequence)
            self._pts_ms = float(pts_ms)
            self._updated_monotonic = time.monotonic()
            self._state = "streaming"

    def set_state(self, state: str) -> None:
        state = state.strip().lower()
        if state not in {"idle", "generating", "streaming", "completed", "failed"}:
            raise ValueError(f"unsupported state: {state}")
        with self._lock:
            self._state = state
            self._updated_monotonic = time.monotonic()

    def snapshot(self) -> FrameSnapshot:
        with self._lock:
            return FrameSnapshot(
                jpeg=self._jpeg,
                sequence=self._sequence,
                pts_ms=self._pts_ms,
                updated_monotonic=self._updated_monotonic,
                state=self._state,
            )

    def status(self) -> dict[str, Any]:
        snap = self.snapshot()
        age_ms = None
        if snap.updated_monotonic:
            age_ms = max(0.0, (time.monotonic() - snap.updated_monotonic) * 1000.0)
        return {
            "version": 1,
            "state": snap.state,
            "sequence": snap.sequence,
            "ptsMs": snap.pts_ms,
            "frameAvailable": snap.jpeg is not None,
            "ageMs": age_ms,
        }


class FrameBridgeServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, store: LatestFrameStore | None = None) -> None:
        self.store = store or LatestFrameStore()
        handler = self._make_handler(self.store)
        self.httpd = ThreadingHTTPServer((host, port), handler)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    def serve_forever(self) -> None:
        self.httpd.serve_forever(poll_interval=0.25)

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    @staticmethod
    def _make_handler(store: LatestFrameStore):
        class Handler(BaseHTTPRequestHandler):
            server_version = "LingBotFrameBridge/1"

            def log_message(self, format: str, *args: object) -> None:
                return

            def _cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, X-QPS-Frame-Seq, X-QPS-PTS-Ms")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

            def _json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self._cors()
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(HTTPStatus.NO_CONTENT)
                self._cors()
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                if self.path.split("?", 1)[0] == "/healthz":
                    self._json(HTTPStatus.OK, {"ok": True, **store.status()})
                    return
                if self.path.split("?", 1)[0] == "/v1/status":
                    self._json(HTTPStatus.OK, store.status())
                    return
                if self.path.split("?", 1)[0] == "/v1/frame.jpg":
                    snap = store.snapshot()
                    if snap.jpeg is None:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "no_frame"})
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(snap.jpeg)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-QPS-Frame-Seq", str(snap.sequence))
                    self.send_header("X-QPS-PTS-Ms", f"{snap.pts_ms:.3f}")
                    self._cors()
                    self.end_headers()
                    self.wfile.write(snap.jpeg)
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

            def do_POST(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length < 0 or length > MAX_FRAME_BYTES:
                    self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "payload_too_large"})
                    return
                body = self.rfile.read(length)
                try:
                    if path == "/v1/frame":
                        seq = int(self.headers.get("X-QPS-Frame-Seq", "0"))
                        pts_ms = float(self.headers.get("X-QPS-PTS-Ms", "0"))
                        store.publish(body, seq, pts_ms)
                        self._json(HTTPStatus.ACCEPTED, store.status())
                        return
                    if path == "/v1/state":
                        payload = json.loads(body.decode("utf-8"))
                        store.set_state(str(payload.get("state", "")))
                        self._json(HTTPStatus.OK, store.status())
                        return
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        return Handler
