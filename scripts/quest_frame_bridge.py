#!/usr/bin/env python3
"""Serve the localhost frame bridge used by QuestPhoneStream AI-video POC."""
from __future__ import annotations

import argparse

from wan.streaming.frame_bridge import FrameBridgeServer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    server = FrameBridgeServer(args.host, args.port)
    host, port = server.address
    print(f"LingBot frame bridge: http://{host}:{port}", flush=True)
    print("GET /healthz  GET /v1/status  GET /v1/frame.jpg  POST /v1/frame", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
