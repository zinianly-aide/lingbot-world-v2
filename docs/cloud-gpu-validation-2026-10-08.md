# Cloud GPU validation — 2026-10-08

## Environment

AutoDL replacement instance:

- GPU: RTX 4090 48 GB
- driver: 595.71.05
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- Python: 3.12.3
- GPU compute capability: 8.9
- CPU quota: 20 cores
- memory quota: ~96 GiB

This run is a functional/performance baseline for the 4090 48 GB replacement host. It is **not** a replacement for the 5090 benchmark from 2026-10-07.

## Results

| Gate | Result | Key observations |
|---|---|---|
| C0 static | PASS | 26 unit tests; compile PASS |
| C1 13-frame full smoke | PASS | 832x464 actual, 13 frames, DiT chunk ~1.50 s; nvidia-smi peak 14,838 MiB |
| C2 M4-parity | PASS | 29 frames; generation 3.590 s; TTFF 0.865 s; peak 12,072 MiB |
| C3 480x832-class | PASS | 29 frames; generation 5.660 s; TTFF 1.926 s; peak 17,202 MiB |
| C4 cloud -> Mac tunnel | PASS | cloud bridge 8765 -> Mac 18765; JPEG fetch OK; sequence and PTS monotonic |
| C5 Quest E2E | PENDING | macOS sender built/tested; real Quest observation still required |

## Comparison with the 2026-10-07 RTX 5090 run

| Profile | RTX 5090 | RTX 4090 48 GB | 5090 speed advantage |
|---|---:|---:|---:|
| Q2 M4-parity generation | 2.773 s | 3.590 s | 1.29x |
| Q2 M4-parity TTFF | 0.566 s | 0.865 s | 1.53x |
| Q2 480x832-class generation | 3.813 s | 5.660 s | 1.48x |
| Q2 480x832-class TTFF | 1.223 s | 1.926 s | 1.57x |

Both runs used the current stock path without FlashAttention/SageAttention/FP8/torch.compile optimization.

## C4 evidence

The Mac already uses localhost port 8765 for another service, so this run used:

```text
cloud 127.0.0.1:8765
        |
        | SSH local forward
        v
Mac 127.0.0.1:18765
```

Observed over the tunnel:

- health state: `streaming`
- first sampled sequence: `1232`, PTS `102666.667 ms`
- later sampled sequence: `1378`, PTS `114833.333 ms`
- JPEG fetch returned HTTP 200 with a valid image payload

QuestPhoneStream macOS sender on `feat/p3-spatial` also passed 19/19 unit tests and a production build before C5.

## Next

For C5, set the macOS sender bridge URL to:

```text
http://127.0.0.1:18765
```

Select `LingBot AI video · localhost bridge`, start the existing WebRTC session, and record Quest-visible playback, first visible frame, reconnect behavior, and any frame corruption/drop pattern.
