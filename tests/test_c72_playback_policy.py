"""CPU integration tests for C7.2. No model weights, CUDA, or Quest required.

Isolated import path avoids importing unrelated Wan/DiT dependencies during
local Mac testing; CI also executes this module with the same code.
"""
from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from urllib.request import urlopen
from scripts.collect_quest_playback_metrics import sample, summarize, validate_url

_ROOT = Path(__file__).resolve().parent.parent
_PKG = "_c72_streaming_testpkg"
if _PKG not in sys.modules:
    package = types.ModuleType(_PKG)
    package.__path__ = [str(_ROOT / "wan/streaming")]
    sys.modules[_PKG] = package

policy_module = importlib.import_module(_PKG + ".playback_policy")
bridge_module = importlib.import_module(_PKG + ".frame_bridge")
publisher_module = importlib.import_module(_PKG + ".buffered_publisher")
async_module = importlib.import_module(_PKG + ".frame_publisher_async")
PlaybackOptions = policy_module.PlaybackOptions
PlaybackController = policy_module.PlaybackController
FrameBridgeServer = bridge_module.FrameBridgeServer
LatestFrameStore = bridge_module.LatestFrameStore
BufferedFrameBridgePublisher = publisher_module.BufferedFrameBridgePublisher
TimingStats = async_module.TimingStats
PublishTiming = async_module.PublishTiming
JPEG = b"\xff\xd8cpu-test-encoded-frame\xff\xd9"


class PlaybackPolicyTests(unittest.TestCase):
    def test_fixed_is_legacy_rate_and_never_drops(self):
        ctrl = PlaybackController(PlaybackOptions(mode="fixed", fps=5, min_fps=5, max_fps=5))
        for n in (0, 48, 0, 20):
            self.assertAlmostEqual(ctrl.period(depth=n, now=float(n)), .2)
        self.assertFalse(ctrl.should_drop(age_ms=999999))
        self.assertEqual(ctrl.snapshot()["rateAdjustments"], 0)

    def test_smooth_rate_hysteresis_and_clamps(self):
        ctrl = PlaybackController(PlaybackOptions(
            mode="smooth", fps=5, min_fps=5, max_fps=8,
            update_interval_sec=1.0, target_buffer_ms=700,
        ))
        ctrl.period(depth=20, now=0)
        self.assertAlmostEqual(ctrl.snapshot()["currentFps"], 5.5)
        ctrl.period(depth=20, now=.1)
        self.assertAlmostEqual(ctrl.snapshot()["currentFps"], 5.5)
        for i in range(1, 20):
            ctrl.period(depth=20, now=i + .1)
        self.assertEqual(ctrl.snapshot()["currentFps"], 8)
        for i in range(20, 50):
            ctrl.period(depth=0, now=i + .1)
        self.assertEqual(ctrl.snapshot()["currentFps"], 5)

    def test_interactive_age_and_invalid_options(self):
        opts = PlaybackOptions(mode="interactive", fps=5, min_fps=5, max_fps=8,
                               max_frame_age_ms=500)
        ctrl = PlaybackController(opts)
        self.assertAlmostEqual(ctrl.period(depth=12, now=1), .125)
        self.assertFalse(ctrl.should_drop(age_ms=499))
        self.assertTrue(ctrl.should_drop(age_ms=501))
        for bad in (dict(mode="unknown"), dict(fps=0), dict(min_fps=9),
                    dict(max_frame_age_ms=0), dict(target_buffer_ms=-1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                PlaybackOptions(**bad)


class PlaybackPublisherTests(unittest.TestCase):
    def test_fixed_retains_all_frames_and_pts(self):
        store = LatestFrameStore()
        pub = BufferedFrameBridgePublisher(store, fps=70, max_frames=5)
        try:
            for i in range(10):
                pub.publish_jpeg_bytes(JPEG, i)
            self.assertTrue(pub.wait_empty(timeout=4))
            metrics = pub.metrics()
            self.assertEqual(metrics["enqueued"], 10)
            self.assertEqual(metrics["published"], 10)
            self.assertEqual(metrics["droppedFrames"], 0)
            self.assertEqual(metrics["mode"], "fixed")
            self.assertEqual(store.snapshot().sequence, 9)
            self.assertAlmostEqual(store.snapshot().pts_ms, 9 * 1000 / 70)
        finally:
            pub.close(timeout=4)

    def test_interactive_evicts_stale_frames_keeps_latest_seq(self):
        store = LatestFrameStore()
        policy = PlaybackOptions(mode="interactive", fps=5, min_fps=5,
                                 max_fps=8, max_frame_age_ms=400)
        pub = BufferedFrameBridgePublisher(store, fps=5, max_frames=3, policy=policy)
        try:
            for i in range(30):
                pub.publish_jpeg_bytes(JPEG, i)
            self.assertTrue(pub.wait_empty(timeout=5))
            m = pub.metrics()
            self.assertEqual(m["enqueued"], 30)
            self.assertEqual(store.snapshot().sequence, 29)
            self.assertGreater(m["droppedFrames"], 0)
            self.assertEqual(m["published"] + m["droppedFrames"], m["enqueued"])
            self.assertEqual(m["mode"], "interactive")
            self.assertLessEqual(m["queueMaxDepth"], 3)
        finally:
            pub.close(timeout=5)

    def test_interactive_expired_frame_is_skipped_but_last_shows(self):
        store = LatestFrameStore()
        policy = PlaybackOptions(mode="interactive", fps=5, min_fps=5,
                                 max_fps=8, max_frame_age_ms=1)
        pub = BufferedFrameBridgePublisher(store, fps=5, max_frames=16, policy=policy)
        try:
            for i in range(16):
                pub.publish_jpeg_bytes(JPEG, i)
            self.assertTrue(pub.wait_empty(timeout=5))
            self.assertEqual(store.snapshot().sequence, 15)
            self.assertGreater(pub.stats().dropped_by_age, 0)
        finally:
            pub.close(timeout=5)

    def test_smooth_changes_rate_with_backlog(self):
        store = LatestFrameStore()
        opts = PlaybackOptions(mode="smooth", fps=5, min_fps=5, max_fps=8,
                               update_interval_sec=.02)
        pub = BufferedFrameBridgePublisher(store, fps=5, max_frames=48,
                                            policy=opts)
        try:
            for i in range(25):
                pub.publish_jpeg_bytes(JPEG, i)
            self.assertTrue(pub.wait_empty(timeout=8))
            self.assertEqual(pub.stats().dropped_frames, 0)
            self.assertEqual(store.snapshot().sequence, 24)
            self.assertGreater(pub.metrics()["rateAdjustments"], 0)
        finally:
            pub.close(timeout=8)

    def test_http_metrics_opt_in_without_changing_existing_status(self):
        store = LatestFrameStore()
        pub = BufferedFrameBridgePublisher(store, fps=100, max_frames=3)
        server = FrameBridgeServer("127.0.0.1", 0, store=store,
                                   metrics_provider=lambda: {"playback": pub.metrics()})
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:" + str(server.address[1])
        try:
            pub.publish_jpeg_bytes(JPEG, 0)
            self.assertTrue(pub.wait_empty(timeout=3))
            with urlopen(base + "/v1/metrics", timeout=3) as stream:
                d = json.loads(stream.read())
            self.assertEqual(d["playback"]["published"], 1)
            with urlopen(base + "/v1/status", timeout=3) as stream:
                status = json.loads(stream.read())
            self.assertEqual(status["sequence"], 0)
            self.assertNotIn("playback", status)
        finally:
            server.shutdown()
            thread.join(timeout=3)
            pub.close(timeout=3)

    def test_metrics_collector_uses_actual_http_and_restricts_hosts(self):
        store = LatestFrameStore()
        server = FrameBridgeServer("127.0.0.1", 0, store=store,
                                   metrics_provider=lambda: {
                                       "bridge": store.status(),
                                       "playback": {"published": 12, "queueDepth": 2,
                                                    "oldestQueueAgeMs": 240,
                                                    "maxPublishedFrameAgeMs": 450,
                                                    "droppedFrames": 1,
                                                    "underflowEvents": 0,
                                                    "currentFps": 6},
                                   })
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = sample("http://127.0.0.1:" + str(server.address[1]))
            self.assertEqual(result["playback"]["published"], 12)
            summary = summarize([{"metrics": result}, {"error": "temporary"}])
            self.assertEqual(summary["maxQueueDepth"], 2)
            self.assertEqual(summary["errors"], 1)
            for url in ("https://127.0.0.1:8766", "http://example.com:8766",
                        "http://127.0.0.1:8766/v1/state",
                        "http://127.0.0.1:8766/?token=secret"):
                with self.assertRaises(ValueError):
                    validate_url(url)
        finally:
            server.shutdown()
            thread.join(timeout=2)

    def test_worker_failure_is_reported_not_hung(self):
        class FailingStore:
            def publish(self, *args):
                raise RuntimeError("test sink failure")
        pub = BufferedFrameBridgePublisher(FailingStore(), fps=100, max_frames=2)
        pub.publish_jpeg_bytes(JPEG, 0)
        from unittest.mock import patch
        with patch.object(publisher_module.logging, "exception"):
            with self.assertRaisesRegex(RuntimeError, "playback worker failed"):
                pub.wait_empty(timeout=2)
            with self.assertRaisesRegex(RuntimeError, "playback worker failed"):
                pub.close(timeout=2)

    def test_interactive_timed_store_allows_only_forward_skips(self):
        from scripts.quest_world_buffered import TimedStore
        class Sink:
            def publish(self, *args): pass
        tracker = TimedStore(Sink(), fps=8, segment_frames=125, allow_sequence_gaps=True)
        tracker.publish(JPEG, 0, 0)
        tracker.publish(JPEG, 3, 375)
        self.assertEqual(tracker.stats()["skippedSequences"], 2)
        with self.assertRaises(RuntimeError):
            tracker.publish(JPEG, 3, 375)

    def test_runner_dry_run_works_without_wan_model_dependencies(self):
        runner = _ROOT / "scripts/quest_world_buffered.py"
        base = [sys.executable, str(runner), "--ckpt-dir", "/unused",
                "--assets-dir", "/unused", "--prompt-embeds", str(_ROOT / "examples/03/image.jpg"),
                "--output-fps", "5", "--dry-run"]
        for mode, queue_size in (("fixed", 48), ("smooth", 48), ("interactive", 12)):
            with self.subTest(mode=mode):
                result = subprocess.run(base + ["--playback-mode", mode],
                                        cwd=_ROOT, text=True, capture_output=True, timeout=8)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["playbackPolicy"], mode)
                self.assertEqual(payload["encodedQueueFrames"], queue_size)
                self.assertEqual(payload["framesPerSegment"], 125)

    def test_bounded_async_encoder_timing_memory(self):
        ts = TimingStats()
        for _ in range(20_000):
            ts.add(PublishTiming(jpegEncodeSec=.02))
        self.assertEqual(ts.total_samples, 20_000)
        self.assertEqual(len(ts.samples), 512)
        self.assertAlmostEqual(ts.summarize()["jpegEncodeSec"]["p95"], .02)


if __name__ == "__main__":
    unittest.main()
