import time
import unittest

from scripts.quest_world_buffered import TimedStore, SegmentSink


class MemoryStore:
    def __init__(self):
        self.sequences = []

    def publish(self, jpeg, sequence, pts_ms):
        self.sequences.append(sequence)


class TimedStoreTests(unittest.TestCase):
    def test_actual_cadence_and_boundary_observation(self):
        underlying = MemoryStore()
        t = TimedStore(underlying, fps=5, segment_frames=2)
        t.publish(b"frame0", 0, 0)
        time.sleep(.01)
        t.publish(b"frame1", 1, 200)
        time.sleep(.01)
        t.publish(b"frame2", 2, 400)
        stats = t.stats()
        self.assertEqual(stats["framesObserved"], 3)
        self.assertEqual(len(stats["recentBoundaryGapsMs"]), 1)
        self.assertGreater(stats["maxInterframeGapMs"], 0)
        self.assertEqual(underlying.sequences, [0, 1, 2])
        with self.assertRaises(RuntimeError):
            t.publish(b"bad", 2, 400)

    def test_sink_hands_off_frames_and_retains_last(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        class DummyDecoder:
            def decode_chunk_iter(self, latent):
                yield torch.zeros(3, 1, 2, 2)
                yield torch.ones(3, 2, 2, 2)
        class DummyPublisher:
            def __init__(self):
                self.items = []
            def submit_frame(self, tensor):
                self.items.append(tensor.clone())
        pub = DummyPublisher()
        sink = SegmentSink(DummyDecoder(), pub)
        sink.on_latent_chunk(None, None)
        self.assertEqual(sink.chunks, 1)
        self.assertEqual(sink.frames, 3)
        self.assertEqual(len(pub.items), 3)
        self.assertTrue(torch.equal(sink.last_frame, torch.ones(3, 2, 2)))


if __name__ == "__main__":
    unittest.main()
