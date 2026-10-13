#!/usr/bin/env python3
"""Collect bounded C7 bridge samples over localhost/SSH forward (no GPU required).

Uses optional /v1/metrics added in C7.2; writes JSONL for a later GPU/Quest
session. This records *cloud-side* queue residence time, not end-to-end Quest
latency. No telemetry is captured from the headset without an explicit client.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("C7 bridge metrics must use a local http://localhost/127.0.0.1 SSH endpoint")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("URL must contain only scheme, host and port")


def sample(url: str, *, timeout: float = 2) -> dict:
    validate_url(url)
    with urlopen(url.rstrip("/") + "/v1/metrics", timeout=timeout) as response:
        data = json.load(response)
    if "playback" not in data or "bridge" not in data:
        raise ValueError("endpoint returned an invalid C7 metrics payload")
    return data


def summarize(samples: list[dict]) -> dict:
    ok = [s for s in samples if "metrics" in s]
    playback = [s["metrics"]["playback"] for s in ok if s["metrics"].get("playback", {}).get("published") is not None]
    ages = [m["oldestQueueAgeMs"] for m in playback if m.get("oldestQueueAgeMs") is not None]
    depths = [m["queueDepth"] for m in playback if m.get("queueDepth") is not None]
    if not playback:
        return {"samples": len(samples), "successes": len(ok), "errors": len(samples) - len(ok)}
    first, last = playback[0], playback[-1]
    return {
        "samples": len(samples), "successes": len(ok), "errors": len(samples) - len(ok),
        "firstPublished": first["published"], "lastPublished": last["published"],
        "maxQueueDepth": max(depths) if depths else None,
        "maxObservedOldestQueueAgeMs": round(max(ages), 2) if ages else None,
        "maxReportedPublishedFrameAgeMs": max(m.get("maxPublishedFrameAgeMs", 0) for m in playback),
        "droppedFrames": last.get("droppedFrames"),
        "underflowEvents": last.get("underflowEvents"),
        "lastPlaybackFps": last.get("currentFps"),
        "lastAveragePublishedFps": last.get("averagePublishedFps"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:18766")
    p.add_argument("--duration", type=float, default=60.0, help="bounded collection seconds")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    validate_url(args.url)
    if not (0 < args.duration <= 1800) or not (.2 <= args.interval <= 60):
        p.error("duration must be 0..1800s and interval between 0.2..60s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    deadline = time.monotonic() + args.duration
    with args.out.open("w", encoding="utf-8") as stream:
        while time.monotonic() < deadline:
            item = {"sampleUtc": datetime.now(timezone.utc).isoformat(),
                    "sampleMonotonicSec": time.monotonic()}
            try:
                item["metrics"] = sample(args.url, timeout=min(3.0, args.interval))
            except (URLError, ValueError, OSError) as exc:
                item["error"] = str(exc)
            samples.append(item)
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
    print(json.dumps(summarize(samples), indent=2, ensure_ascii=False))
    return 0 if any("metrics" in s for s in samples) else 2


if __name__ == "__main__":
    raise SystemExit(main())
