"""Unit tests for the LingBot -> Quest progressive streaming primitives."""
from __future__ import annotations

import contextlib
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from wan.streaming.buffered_publisher import BufferedFrameBridgePublisher
from wan.streaming.frame_bridge import LatestFrameStore
from wan.streaming.frame_publisher import FrameBridgePublisher
from wan.streaming.latent_tap import tap_causal_latent_chunks
from wan.streaming.vae_progressive import ProgressiveWanVaeDecoder


class _CollectSink:
    def __init__(self):
        self.items = []

    def on_latent_chunk(self, event, latent):
        self.items.append((event, latent))


class _FakeDiT(nn.Module):
    def forward(self, x, t, **kwargs):
        return [x[0]]


class _FakePipe:
    infer_mode = "causal_fast"

    def __init__(self):
        self.model = _FakeDiT()

    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep, scheduler):
        return flow_pred + xt


class LatentTapTests(unittest.TestCase):
    def test_emits_only_post_chunk_context_update_and_restores_methods(self):
        pipe = _FakePipe()
        sink = _CollectSink()
        original_convert_func = pipe._convert_flow_pred_to_x0.__func__
        original_forward_func = pipe.model.forward.__func__

        with tap_causal_latent_chunks(
            pipe,
            sink,
            generation_id="g1",
            seed=42,
            total_chunks=2,
            fail_open=False,
        ):
            x0 = pipe._convert_flow_pred_to_x0(
                torch.ones(2, 1, 2, 2),
                torch.ones(2, 1, 2, 2),
                torch.tensor([100.0]),
                object(),
            )
            pipe.model(x=[torch.zeros_like(x0)], t=torch.tensor([100.0]))
            self.assertEqual(len(sink.items), 0)

            pipe.model(x=[x0], t=torch.tensor([0.0]), cross_attn_first_call=False)
            self.assertEqual(len(sink.items), 1)
            event, emitted = sink.items[0]
            self.assertIs(emitted, x0)
            self.assertEqual(event.generation_id, "g1")
            self.assertEqual(event.chunk_index, 0)
            self.assertEqual(event.total_chunks, 2)
            self.assertEqual(event.latent_start, 0)
            self.assertEqual(event.latent_count, 1)
            self.assertEqual(event.seed, 42)

        self.assertIs(pipe._convert_flow_pred_to_x0.__func__, original_convert_func)
        self.assertIs(pipe.model.forward.__func__, original_forward_func)

    def test_teardown_survives_pipe_model_unload(self):
        pipe = _FakePipe()
        sink = _CollectSink()
        model = pipe.model
        original_forward_func = model.forward.__func__
        with tap_causal_latent_chunks(pipe, sink, fail_open=False):
            x0 = pipe._convert_flow_pred_to_x0(
                torch.ones(1, 1, 1, 1),
                torch.zeros(1, 1, 1, 1),
                torch.tensor([1.0]),
                object(),
            )
            model(x=[x0], t=torch.tensor([0.0]))
            pipe.model = None
        self.assertIs(model.forward.__func__, original_forward_func)

    def test_fail_open_does_not_break_generation(self):
        class BrokenSink:
            def on_latent_chunk(self, event, latent):
                raise RuntimeError("bridge down")

        pipe = _FakePipe()
        with tap_causal_latent_chunks(pipe, BrokenSink(), fail_open=True):
            x0 = pipe._convert_flow_pred_to_x0(
                torch.ones(1, 1, 1, 1),
                torch.zeros(1, 1, 1, 1),
                torch.tensor([1.0]),
                object(),
            )
            result = pipe.model(x=[x0], t=torch.tensor([0.0]))
        self.assertTrue(torch.equal(result[0], x0))


class _FakeDecoder:
    def __call__(self, x, feat_cache=None, feat_idx=None):
        prev = feat_cache[0]
        if prev is None:
            prev = torch.zeros_like(x)
        out = x + prev
        feat_cache[0] = x.clone()
        feat_idx[0] += 1
        return out


class _FakeVaeModel:
    z_dim = 2

    def __init__(self):
        self.decoder = _FakeDecoder()
        self._feat_map = []
        self._conv_idx = [0]
        self.clear_count = 0

    def clear_cache(self):
        self._feat_map = [None]
        self._conv_idx = [0]
        self.clear_count += 1

    def conv2(self, z):
        return z


class _FakeVae:
    def __init__(self):
        self.dtype = torch.float32
        self.model = _FakeVaeModel()
        self.scale = [0.0, 1.0]


def _full_fake_decode(vae, latent):
    model = vae.model
    model.clear_cache()
    x = model.conv2(latent.unsqueeze(0))
    outputs = []
    for i in range(x.shape[2]):
        model._conv_idx = [0]
        outputs.append(model.decoder(
            x[:, :, i:i + 1],
            feat_cache=model._feat_map,
            feat_idx=model._conv_idx,
        ))
    out = torch.cat(outputs, dim=2).squeeze(0)
    model.clear_cache()
    return out


class ProgressiveVaeTests(unittest.TestCase):
    @patch("wan.streaming.vae_progressive.autocast_ctx", side_effect=lambda **_: contextlib.nullcontext())
    def test_chunked_decode_matches_single_stream_cache_order(self, _autocast):
        latent = torch.arange(2 * 5 * 2 * 2, dtype=torch.float32).reshape(2, 5, 2, 2)
        vae = _FakeVae()
        expected = _full_fake_decode(vae, latent)

        with ProgressiveWanVaeDecoder(vae) as decoder:
            actual = decoder.decode_chunks(latent.split(2, dim=1))
            stats = decoder.stats()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(stats.latent_frames, 5)
        self.assertEqual(stats.output_frames, 5)
        self.assertEqual(stats.chunks, 3)

    @patch("wan.streaming.vae_progressive.autocast_ctx", side_effect=lambda **_: contextlib.nullcontext())
    def test_rejects_non_video_latent(self, _autocast):
        vae = _FakeVae()
        with ProgressiveWanVaeDecoder(vae) as decoder:
            with self.assertRaises(ValueError):
                decoder.decode_chunk(torch.zeros(2, 4, 4))


class FramePublisherTests(unittest.TestCase):
    def test_publishes_monotonic_jpeg_frames(self):
        store = LatestFrameStore()
        publisher = FrameBridgePublisher(store, fps=20.0, jpeg_quality=80)
        frames = torch.zeros(3, 2, 4, 4)
        self.assertEqual(publisher.publish_chunk(frames), 2)
        snap = store.snapshot()
        self.assertEqual(snap.sequence, 1)
        self.assertAlmostEqual(snap.pts_ms, 50.0)
        self.assertIsNotNone(snap.jpeg)
        self.assertTrue(snap.jpeg.startswith(b"\xff\xd8"))
        self.assertTrue(snap.jpeg.endswith(b"\xff\xd9"))

    def test_buffered_publisher_preserves_chunk_frames_at_playback_cadence(self):
        store = LatestFrameStore()
        publisher = BufferedFrameBridgePublisher(
            store,
            fps=50.0,
            jpeg_quality=80,
            max_frames=4,
        )
        try:
            frames = torch.zeros(3, 3, 4, 4)
            self.assertEqual(publisher.publish_chunk(frames), 3)
            self.assertTrue(publisher.wait_empty(timeout=2.0))
            stats = publisher.stats()
            snap = store.snapshot()
            self.assertEqual(stats.enqueued, 3)
            self.assertEqual(stats.published, 3)
            self.assertGreaterEqual(stats.max_queue_depth, 1)
            self.assertEqual(snap.sequence, 2)
            self.assertAlmostEqual(snap.pts_ms, 40.0)
        finally:
            publisher.close(drain=True, timeout=2.0)


if __name__ == "__main__":
    unittest.main()
