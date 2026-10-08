import unittest

from scripts.quest_world_reuse import OffsetStore, parse_args
from scripts.quest_continuous_world import aligned_frames


class _Store:
    def __init__(self):
        self.records = []

    def publish(self, jpeg, sequence, pts_ms):
        self.records.append((sequence, pts_ms, jpeg))


class C7ModelReuseContractTests(unittest.TestCase):
    def test_global_sequence_offset_and_pts(self):
        store = _Store()
        a = OffsetStore(store, offset=253, fps=8)
        a.publish(b"j1", 0, 0)
        a.publish(b"j2", 1, 125)
        self.assertEqual(store.records, [
            (253, 31625.0, b"j1"),
            (254, 31750.0, b"j2"),
        ])

    def test_alignment_matches_causal_generator(self):
        self.assertEqual(aligned_frames(129, 4), 125)
        self.assertEqual(aligned_frames(257, 4), 253)
        with self.assertRaises(ValueError):
            OffsetStore(_Store(), -1, 8)
        with self.assertRaises(ValueError):
            OffsetStore(_Store(), 0, 0)


if __name__ == "__main__":
    unittest.main()
