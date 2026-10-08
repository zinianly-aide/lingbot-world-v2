import unittest

import torch

from scripts.q2_live_quest_poc import ContinuousTrackingSink


class _Event:
    def to_dict(self):
        return {"chunk_index": 0}


class _Decoder:
    def decode_chunk_iter(self, latent):
        yield torch.zeros(3, 1, 2, 2)
        yield torch.ones(3, 2, 2, 2)


class _Publisher:
    def __init__(self):
        self.frames = []
        self.closed = False

    def submit_frame(self, frame):
        self.frames.append(frame.clone())
        return len(self.frames) - 1

    def close(self):
        self.closed = True

    def stats(self):
        return {"published": len(self.frames)}


class C6ContinuousSinkTests(unittest.TestCase):
    def test_publishes_every_decoded_frame_before_flush(self):
        publisher = _Publisher()
        sink = ContinuousTrackingSink(_Decoder(), publisher, torch.device("cpu"))

        sink.on_latent_chunk(_Event(), torch.zeros(16, 2, 1, 1))

        self.assertEqual(len(publisher.frames), 3)
        self.assertEqual(sink.sequence, 3)
        self.assertEqual(sink.chunks[0]["enqueuedFrames"], 3)
        self.assertFalse(publisher.closed)

        sink.flush()
        self.assertTrue(publisher.closed)
        self.assertEqual(sink.async_stats()["published"], 3)


if __name__ == "__main__":
    unittest.main()
