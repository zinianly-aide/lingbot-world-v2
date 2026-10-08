"""GPU-free C7 contract tests for persistent Bridge + segmented generation."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError

from scripts.quest_continuous_world import ROOT, SessionPlan, run_segments, aligned_frames, npy_first_dimension
from scripts.quest_http_bridge import RemoteFrameStore

# Import frame_bridge.py without loading the heavy wan.streaming package.
name = "quest_bridge_standalone"
spec = importlib.util.spec_from_file_location(
    name, ROOT / "wan/streaming/frame_bridge.py")
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
FrameBridgeServer = module.FrameBridgeServer
LatestFrameStore = module.LatestFrameStore

JPEG = b"\xff\xd8synthetic-jpeg-for-unit-test\xff\xd9"

MOCK_GENERATOR = r"""
import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

p = argparse.ArgumentParser()
for name in ('external-bridge-url','sequence-offset','output-fps','work-dir','image'):
    p.add_argument('--'+name, required=True)
args, _ = p.parse_known_args()

sys.path.insert(0, str(Path.cwd()))
from scripts.quest_http_bridge import RemoteFrameStore
s = RemoteFrameStore(args.external_bridge_url,
                     sequence_offset=int(args.sequence_offset),
                     output_fps=float(args.output_fps))
assert Path(args.image).is_file()
s.set_state('generating')
for i in range(3):
    s.publish(b'\xff\xd8synthetic-jpeg-for-unit-test\xff\xd9', i, 0)
s.set_state('completed')
out = Path(args.work_dir)
out.mkdir(parents=True, exist_ok=True)
(out/'report.json').write_text(json.dumps({
    'pass': True,
    'generationSec': 1,
    'ttffSec': 0.1,
    'playback': {'published': 3},
}))
"""


class C7BridgeTests(unittest.TestCase):
    def setUp(self):
        self.bridge = FrameBridgeServer("127.0.0.1", 0, store=LatestFrameStore())
        self.thread = threading.Thread(target=self.bridge.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.bridge.address
        self.base = f"http://{host}:{port}"

    def tearDown(self):
        self.bridge.shutdown()
        self.thread.join(timeout=2)

    def test_forward_offsets_sequences_and_pts(self):
        first = RemoteFrameStore(self.base, sequence_offset=0, output_fps=8)
        first.publish(JPEG, 0, 0)
        first.publish(JPEG, 1, 125)
        second = RemoteFrameStore(self.base, sequence_offset=2, output_fps=8)
        second.publish(JPEG, 0, 0)
        self.assertEqual(second.status()["sequence"], 2)
        self.assertEqual(second.status()["ptsMs"], 250.0)
        self.assertTrue(second.status()["frameAvailable"])
        with self.assertRaises(ValueError):
            second.publish(JPEG, 0, 0)
        with self.assertRaises(RuntimeError):
            RemoteFrameStore(self.base, sequence_offset=1, output_fps=8).publish(JPEG, 0, 0)

    def test_two_segments_keep_same_bridge_and_input_last_frame(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fake_script = root / "fake_generator.py"
            fake_script.write_text(MOCK_GENERATOR)
            plan = SessionPlan(
                python=sys.executable,
                script=fake_script,
                ckpt_dir=root / "not_loaded",
                assets_dir=root / "not_loaded",
                image=ROOT / "examples/03/image.jpg",
                action_path=ROOT / "examples/03",
                prompt_embeds=root / "not_loaded",
                prompt="test",
                work_dir=root / "session",
                bridge_url=self.base,
                frame_num=33,
            )
            self.assertEqual(run_segments(plan, max_segments=2, child_timeout=10), 2)
            self.assertEqual(self.bridge.store.status()["sequence"], 5)
            self.assertEqual(self.bridge.store.status()["ptsMs"], 625)
            manifest = json.loads((root/"session/session.json").read_text())
            self.assertEqual([seg["offset"] for seg in manifest], [0, 3])
            self.assertEqual((root/"session/segment-0000/last_frame.jpg").read_bytes(), JPEG)
            self.assertEqual((root/"session/segment-0001/last_frame.jpg").read_bytes(), JPEG)

    def test_segment_pruning_bounds_disk_growth(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fake_script = root / "fake_generator.py"
            fake_script.write_text(MOCK_GENERATOR)
            plan = SessionPlan(
                python=sys.executable, script=fake_script,
                ckpt_dir=root, assets_dir=root,
                image=ROOT / "examples/03/image.jpg", action_path=ROOT / "examples/03",
                prompt_embeds=root, prompt="test", work_dir=root / "session",
                bridge_url=self.base, frame_num=33,
            )
            self.assertEqual(run_segments(plan, max_segments=4, retain_segments=2), 4)
            self.assertFalse((root/"session/segment-0000").exists())
            self.assertFalse((root/"session/segment-0001").exists())
            self.assertTrue((root/"session/segment-0002/last_frame.jpg").exists())
            self.assertTrue((root/"session/segment-0003/last_frame.jpg").exists())
            self.assertEqual(self.bridge.store.status()["sequence"], 11)

    def test_alignment_and_camera_header_are_readable(self):
        self.assertEqual(aligned_frames(257, 4), 253)
        self.assertEqual(aligned_frames(129, 4), 125)
        self.assertEqual(npy_first_dimension(ROOT / "examples/03/poses.npy"), 269)

    def test_invalid_window_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            plan = SessionPlan(
                python=sys.executable, script=Path("generator.py"),
                ckpt_dir=Path(d), assets_dir=Path(d),
                image=ROOT / "examples/03/image.jpg", action_path=ROOT / "examples/03",
                prompt_embeds=Path(d), prompt="test", work_dir=Path(d),
                bridge_url=self.base, frame_num=257, local_attn_size=4,
            )
            with self.assertRaises(ValueError):
                plan.validate()


if __name__ == "__main__":
    unittest.main()
