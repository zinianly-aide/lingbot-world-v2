#!/usr/bin/env python3
"""Replay an existing MP4 into the localhost Quest frame bridge.

This closes Q0 without touching model code: generate a normal LingBot MP4,
start ``quest_frame_bridge.py``, then replay it while the QuestPhoneStream macOS
sender consumes the bridge as a MediaStream source.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"


def post_state(base_url: str, state: str) -> None:
    body = json.dumps({"state": state}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/state",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=2) as resp:
        resp.read()


def post_frame(base_url: str, jpeg: bytes, sequence: int, pts_ms: float) -> None:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/frame",
        data=jpeg,
        headers={
            "Content-Type": "image/jpeg",
            "X-QPS-Frame-Seq": str(sequence),
            "X-QPS-PTS-Ms": f"{pts_ms:.3f}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=2) as resp:
        resp.read()


def iter_mjpeg(stdout):
    buf = bytearray()
    while True:
        chunk = stdout.read(64 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        while True:
            start = buf.find(SOI)
            if start < 0:
                if len(buf) > 1:
                    del buf[:-1]
                break
            end = buf.find(EOI, start + 2)
            if end < 0:
                if start > 0:
                    del buf[:start]
                break
            end += 2
            frame = bytes(buf[start:end])
            del buf[:end]
            yield frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--bridge", default="http://127.0.0.1:8765")
    ap.add_argument("--fps", type=float, default=12.0)
    args = ap.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"video not found: {args.input}")
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")

    cmd = [
        "ffmpeg", "-v", "error", "-i", str(args.input),
        "-vf", f"fps={args.fps}",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout unavailable")

    period = 1.0 / args.fps
    started = time.monotonic()
    post_state(args.bridge, "streaming")
    try:
        for sequence, jpeg in enumerate(iter_mjpeg(proc.stdout)):
            target = started + sequence * period
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            post_frame(args.bridge, jpeg, sequence, sequence * period * 1000.0)
    except Exception:
        post_state(args.bridge, "failed")
        proc.terminate()
        raise
    finally:
        proc.stdout.close()
        proc.wait()

    if proc.returncode:
        post_state(args.bridge, "failed")
        raise SystemExit(proc.returncode)
    post_state(args.bridge, "completed")


if __name__ == "__main__":
    main()
