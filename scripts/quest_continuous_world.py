#!/usr/bin/env python3
"""C7 preview: indefinitely generate NEW causal clips into one persistent bridge.

A fresh process is used per bounded segment. This is a safe, restartable
*segmented* world session, not cross-segment KV continuity. Input for segment
N+1 is the final decoded JPEG from segment N. The bridge and global frame
sequence never reset, so Mac Sender/Quest need not reconnect.

Requires CUDA for real generation. --dry-run works without GPU/model weights.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.quest_http_bridge import request_json


def npy_first_dimension(path: Path) -> int:
    with path.open("rb") as fd:
        if fd.read(6) != b"\x93NUMPY":
            raise ValueError(f"invalid numpy file: {path}")
        major, minor = fd.read(2)
        if major not in (1, 2, 3):
            raise ValueError(f"unsupported numpy version in {path}")
        header_size = int.from_bytes(fd.read(2 if major == 1 else 4), "little")
        if header_size > 65536:
            raise ValueError("unexpectedly large numpy header")
        header = ast.literal_eval(fd.read(header_size).decode("utf-8"))
    return int(header["shape"][0])


def aligned_frames(frame_num: int, chunk_size: int) -> int:
    if frame_num <= 0 or chunk_size <= 0:
        raise ValueError('invalid alignment inputs')
    count = (frame_num - 1) // 4 + 1
    count -= count % chunk_size
    return (count - 1) * 4 + 1 if count else 0


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


@dataclass(frozen=True)
class SessionPlan:
    python: str
    script: Path
    ckpt_dir: Path
    assets_dir: Path
    image: Path
    action_path: Path
    prompt_embeds: Path
    prompt: str
    work_dir: Path
    bridge_url: str
    frame_num: int = 257
    chunk_size: int = 4
    local_attn_size: int = 16
    sink_size: int = 0
    max_area: int = 258048
    output_fps: float = 8.0
    seed: int = 42

    def validate(self) -> None:
        if self.frame_num < 9 or self.frame_num % 4 != 1:
            raise ValueError("frame-num must be >=9 and 4n+1")
        if self.chunk_size <= 0 or self.frame_num <= self.chunk_size * 4:
            raise ValueError("frame-num must span at least two chunks")
        if self.local_attn_size != -1 and (
            self.local_attn_size < 2 * self.chunk_size
            or self.sink_size < 0
            or self.sink_size + self.chunk_size > self.local_attn_size
        ):
            raise ValueError("rolling window must fit at least two chunks and the sink")
        if self.local_attn_size == -1 and self.sink_size:
            raise ValueError("sink-size requires rolling window")
        if self.output_fps <= 0 or self.max_area <= 0:
            raise ValueError("output-fps and max-area must be positive")
        if not self.image.exists() or not self.action_path.is_dir():
            raise ValueError("image and action directory must exist")
        poses = self.action_path / "poses.npy"
        intrinsics = self.action_path / "intrinsics.npy"
        if not poses.is_file() or not intrinsics.is_file():
            raise ValueError("camera poses.npy and intrinsics.npy are required")
        if npy_first_dimension(poses) < self.frame_num:
            raise ValueError("camera trajectory shorter than frame-num")
        if npy_first_dimension(intrinsics) < 1:
            raise ValueError("invalid camera intrinsics")
        if aligned_frames(self.frame_num, self.chunk_size) < 2:
            raise ValueError("frame/chunk settings leave no output")

    def segment_command(self, segment: int, *, image: Path, offset: int) -> list[str]:
        if segment < 0 or offset < 0:
            raise ValueError("negative segment or offset")
        return [
            self.python, str(self.script),
            "--device", "cuda", "--vae-dtype", "bf16",
            "--ckpt-dir", str(self.ckpt_dir), "--assets-dir", str(self.assets_dir),
            "--image", str(image), "--action-path", str(self.action_path),
            "--prompt", self.prompt, "--prompt-embeds", str(self.prompt_embeds),
            "--frame-num", str(self.frame_num), "--chunk-size", str(self.chunk_size),
            "--local-attn-size", str(self.local_attn_size),
            "--sink-size", str(self.sink_size),
            "--max-area", str(self.max_area), "--seed", str(self.seed + segment),
            "--continuous-publish", "--raw-queue-frames", "8",
            "--buffer-frames", "120",
            "--output-fps", str(self.output_fps),
            "--external-bridge-url", self.bridge_url,
            "--sequence-offset", str(offset),
            "--tail-seconds", "0",
            "--work-dir", str(self.work_dir / f"segment-{segment:04d}"),
        ]


def get_jpeg(url: str) -> bytes:
    with urlopen(url.rstrip("/") + "/v1/frame.jpg", timeout=5) as response:
        jpeg = response.read()
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise RuntimeError("persistent bridge returned an invalid JPEG")
    return jpeg


def run_segments(plan: SessionPlan, *, max_segments: int = 0,
                 child_timeout: float = 900.0, retain_segments: int = 3) -> int:
    """Run model segments against an already-started bridge, stop on first error."""
    plan.validate()
    if max_segments < 0 or retain_segments < 2:
        raise ValueError("max-segments >=0 and retain-segments >=2 required")
    plan.work_dir.mkdir(parents=True, exist_ok=True)
    image = plan.image
    offset = int(request_json(plan.bridge_url + "/v1/status")["sequence"]) + 1
    if offset != 0:
        raise RuntimeError("bridge is not fresh; use a new port to avoid stale frame state")

    segment = 0
    manifest = []
    while not max_segments or segment < max_segments:
        cmd = plan.segment_command(segment, image=image, offset=offset)
        segment_dir = plan.work_dir / f"segment-{segment:04d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        with (segment_dir / "generator.log").open("w") as log:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = proc.wait(timeout=child_timeout)
            except subprocess.TimeoutExpired as exc:
                stop_process(proc)
                raise RuntimeError(f"segment {segment} timed out") from exc
            except BaseException:
                stop_process(proc)
                raise
        report_path = segment_dir / "report.json"
        if returncode != 0 or not report_path.is_file():
            request_json(plan.bridge_url + "/v1/state",
                         payload=b'{"state":"failed"}',
                         headers={"Content-Type": "application/json"})
            raise RuntimeError(
                f"segment {segment} failed (exit {returncode}); see {segment_dir / 'generator.log'}"
            )
        report = json.loads(report_path.read_text())
        frames = int(report.get("playback", {}).get("published", 0))
        bridge_status = request_json(plan.bridge_url + "/v1/status")
        if not report.get("pass") or frames <= 0 or bridge_status["sequence"] != offset + frames - 1:
            raise RuntimeError(f"segment {segment} failed frame/sequence integrity checks")
        image = segment_dir / "last_frame.jpg"
        image.write_bytes(get_jpeg(plan.bridge_url))
        manifest.append({
            "segment": segment, "frames": frames, "offset": offset,
            "generationSec": report.get("generationSec"),
            "ttffSec": report.get("ttffSec"),
            "lastImage": str(image),
        })
        old_index = segment - retain_segments
        if old_index >= 0:
            old_dir = plan.work_dir / f"segment-{old_index:04d}"
            if old_dir.is_dir():
                shutil.rmtree(old_dir)
                manifest[old_index]["lastImage"] = None
                manifest[old_index]["pruned"] = True
        (plan.work_dir / "session.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"C7 segment {segment} PASS: {frames} new frames, global seq {offset}..{offset+frames-1}", flush=True)
        offset += frames
        segment += 1
    return segment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", required=True, type=Path)
    p.add_argument("--assets-dir", required=True, type=Path)
    p.add_argument("--image", type=Path, default=ROOT / "examples/03/image.jpg")
    p.add_argument("--action-path", type=Path, default=ROOT / "examples/03")
    p.add_argument("--prompt-embeds", type=Path,
                   default=ROOT / "eval/e1.2/embeddings/single_subject_minicpm.safetensors")
    p.add_argument("--prompt", default="Move the camera slowly forward while keeping the lone tree stable and centered.")
    p.add_argument("--bridge-host", default="127.0.0.1")
    p.add_argument("--bridge-port", default=8766, type=int)
    p.add_argument("--frame-num", default=257, type=int)
    p.add_argument("--chunk-size", default=4, type=int)
    p.add_argument("--local-attn-size", default=16, type=int)
    p.add_argument("--sink-size", default=0, type=int)
    p.add_argument("--max-area", default=258048, type=int)
    p.add_argument("--output-fps", default=8.0, type=float)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--max-segments", default=0, type=int,
                   help="0 = run continuously until Ctrl+C; other values for gated tests")
    p.add_argument("--child-timeout", default=900.0, type=float)
    p.add_argument("--retain-segments", default=3, type=int,
                   help="Keep N latest segment work folders, minimum 2")
    p.add_argument("--work-dir", type=Path, default=ROOT / "eval/quest-streaming/world_session")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate arguments and print first two commands without loading a model")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.bridge_host != "127.0.0.1":
        raise SystemExit("C7 persistent bridge must bind only 127.0.0.1")
    if args.bridge_port in (8765, 18765):
        raise SystemExit("C5 port is reserved, use 8766 or another dedicated port")
    if args.child_timeout <= 0 or args.retain_segments < 2:
        raise SystemExit("child-timeout must be positive and retain-segments >=2")
    plan = SessionPlan(
        python=sys.executable, script=ROOT / "scripts/q2_live_quest_poc.py",
        ckpt_dir=args.ckpt_dir, assets_dir=args.assets_dir,
        image=args.image, action_path=args.action_path,
        prompt_embeds=args.prompt_embeds, prompt=args.prompt,
        work_dir=args.work_dir,
        bridge_url=f"http://{args.bridge_host}:{args.bridge_port}",
        frame_num=args.frame_num, chunk_size=args.chunk_size,
        local_attn_size=args.local_attn_size, sink_size=args.sink_size,
        max_area=args.max_area, output_fps=args.output_fps, seed=args.seed,
    )
    plan.validate()
    if args.dry_run:
        print(json.dumps({
            "mode": "segmented-continuous", "gpuRequired": False,
            "note": "segment boundaries reset causal KV and may briefly freeze",
            "alignedFramesPerSegment": aligned_frames(plan.frame_num, plan.chunk_size),
            "commands": [plan.segment_command(i, image=plan.image, offset=i * aligned_frames(plan.frame_num, plan.chunk_size))
                         for i in range(2)]
        }, indent=2))
        return 0

    # Lazy import: --dry-run and controller tests do not need GPU dependencies.
    from wan.streaming.frame_bridge import FrameBridgeServer, LatestFrameStore
    store = LatestFrameStore()
    server = FrameBridgeServer(args.bridge_host, args.bridge_port, store=store)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Persistent LingBot bridge: {plan.bridge_url}", flush=True)
    previous_termination_handler = signal.getsignal(signal.SIGTERM)
    def on_termination(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_termination)
    try:
        count = run_segments(plan, max_segments=args.max_segments,
                             child_timeout=args.child_timeout,
                             retain_segments=args.retain_segments)
        print(f"Session complete ({count} segments).", flush=True)
        return 0
    except KeyboardInterrupt:
        print("Session stopped by user.", flush=True)
        return 130
    except Exception as exc:
        store.set_state("failed")
        print(f"C7 session failed: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous_termination_handler)
        server.shutdown()
        server_thread.join(timeout=3.0)


if __name__ == "__main__":
    raise SystemExit(main())
