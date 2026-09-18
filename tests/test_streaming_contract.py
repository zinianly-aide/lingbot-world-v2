import json
import threading
import unittest
import urllib.error
import urllib.request

from wan.streaming.events import LatentChunkEvent
from wan.streaming.frame_bridge import FrameBridgeServer, LatestFrameStore


class StreamingContractTests(unittest.TestCase):
    def test_latent_chunk_event_keeps_latent_coordinates(self):
        event = LatentChunkEvent(
            generation_id="g1",
            chunk_index=1,
            total_chunks=3,
            latent_start=3,
            latent_count=3,
            shape=(16, 3, 30, 52),
            dtype="torch.float32",
            seed=42,
            elapsed_ms=1234.5,
        )
        payload = event.to_dict()
        self.assertEqual(payload["latent_start"], 3)
        self.assertEqual(payload["latent_count"], 3)
        self.assertEqual(payload["shape"], [16, 3, 30, 52])

    def test_store_ignores_out_of_order_frames(self):
        store = LatestFrameStore()
        store.publish(b"\xff\xd8new\xff\xd9", 2, 20.0)
        store.publish(b"\xff\xd8old\xff\xd9", 1, 10.0)
        snap = store.snapshot()
        self.assertEqual(snap.sequence, 2)
        self.assertEqual(snap.jpeg, b"\xff\xd8new\xff\xd9")

    def test_http_bridge_round_trip(self):
        server = FrameBridgeServer("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.address
        base = f"http://{host}:{port}"
        try:
            frame = b"\xff\xd8frame\xff\xd9"
            req = urllib.request.Request(
                f"{base}/v1/frame",
                data=frame,
                headers={"X-QPS-Frame-Seq": "7", "X-QPS-PTS-Ms": "42.5"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                self.assertEqual(resp.status, 202)

            with urllib.request.urlopen(f"{base}/v1/status", timeout=2) as resp:
                status = json.loads(resp.read())
            self.assertEqual(status["sequence"], 7)
            self.assertTrue(status["frameAvailable"])

            with urllib.request.urlopen(f"{base}/v1/frame.jpg", timeout=2) as resp:
                self.assertEqual(resp.headers["X-QPS-Frame-Seq"], "7")
                self.assertEqual(resp.read(), frame)
        finally:
            server.shutdown()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
