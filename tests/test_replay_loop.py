"""Integration tests for quest_stream_replay --loop monotonic sequence.

The replay CLI streams JPEG frames into the localhost FrameBridge. The bridge
drops any frame whose sequence is <= the last accepted one, so a naive shell
loop that restarts the script (resetting sequence to 0) freezes on the last
frame after the first loop. --loop must keep sequence and PTS monotonically
increasing across ffmpeg restarts inside one process.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wan.streaming.frame_bridge import FrameBridgeServer, LatestFrameStore  # noqa: E402


def _http_get(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _make_test_mp4(path: Path, seconds: float = 0.4, fps: int = 10) -> None:
    """Tiny ffmpeg-generated test pattern, deterministic, no audio."""
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=256x256:rate={fps}",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


class ReplayLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = LatestFrameStore()
        self.server = FrameBridgeServer("127.0.0.1", 0, store=self.store)
        import threading
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.address
        self.bridge = f"http://{self.host}:{self.port}"

        self.workdir = Path("/tmp/q0-replay-loop")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.mp4 = self.workdir / "loop.mp4"
        _make_test_mp4(self.mp4, seconds=0.4, fps=10)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2.0)

    def _run_replay(self, extra_args: list[str], timeout_sec: float) -> subprocess.Popen:
        return subprocess.Popen(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "quest_stream_replay.py"),
                str(self.mp4),
                "--bridge", self.bridge,
                "--fps", "10",
                *extra_args,
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def test_loop_keeps_sequence_monotonic_across_restarts(self) -> None:
        """Two loops inside one process: sequence must keep growing past the
        frame count of a single ffmpeg run, and the bridge must accept every
        frame (no out-of-order drops)."""
        # A single ffmpeg run emits 0.4s * 10fps = 4 frames.
        # Let --loop run for ~1.5s so it restarts ffmpeg at least twice.
        proc = self._run_replay(["--loop"], timeout_sec=1.5)
        try:
            time.sleep(1.6)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        snap = self.store.snapshot()
        self.assertEqual(snap.state, "streaming", "loop replay should leave state=streaming")
        # Single run = 4 frames. After >=2 restarts we must be past frame 4.
        self.assertGreater(
            snap.sequence, 4,
            f"sequence={snap.sequence} did not grow past one ffmpeg run; "
            "loop likely reset sequence to 0 and bridge dropped restarts",
        )
        # Bridge itself only accepts strictly-increasing sequence; if we reached
        # snap.sequence without out-of-order drops it already proves monotonicity.

    def test_new_replay_process_keeps_sequence_growing(self) -> None:
        """After a non-loop replay finishes, a second replay process must keep
        sequence growing (this is the regression that motivated --loop)."""
        first = self._run_replay([], timeout_sec=1.0)
        first.wait(timeout=5.0)
        seq_after_first = self.store.snapshot().sequence
        self.assertGreaterEqual(seq_after_first, 0)

        # Without --loop the old shell-loop pattern restarted the script, which
        # reset sequence to 0. With the fix the bridge must still accept the
        # new process (it picks up from the last accepted sequence).
        second = self._run_replay(["--loop"], timeout_sec=0.8)
        try:
            time.sleep(0.8)
        finally:
            second.terminate()
            try:
                second.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                second.kill()
                second.wait()

        seq_after_second = self.store.snapshot().sequence
        self.assertGreater(
            seq_after_second, seq_after_first,
            f"second replay sequence {seq_after_second} did not exceed first {seq_after_first}",
        )


if __name__ == "__main__":
    unittest.main()
