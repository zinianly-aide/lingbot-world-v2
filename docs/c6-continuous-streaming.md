# C6 continuous LingBot → Quest streaming validation

Date: 2026-10-08

## Goal

Validate real LingBot causal chunk generation streamed while generation is still running, without disturbing the existing C5 replay path.

Isolation used for this gate:

- C5 remains on cloud `127.0.0.1:8765` → Mac `127.0.0.1:18765`.
- C6 uses cloud `127.0.0.1:8766` → Mac `127.0.0.1:18766`.
- C6 worktree/copy is independent from the C5 validation directory.

## Implementation

`scripts/q2_live_quest_poc.py` now has an opt-in `--continuous-publish` mode. The default Q2/C3 behavior is unchanged.

In C6 mode:

1. Causal DiT emits completed latent chunks.
2. `ProgressiveWanVaeDecoder.decode_chunk_iter()` decodes temporal slices while retaining causal VAE state.
3. Every decoded RGB frame is immediately handed to `AsyncFrameBridgePublisher`.
4. A bounded raw-frame queue performs device→CPU conversion and JPEG encoding.
5. `BufferedFrameBridgePublisher` paces delivery to the bridge.
6. `--output-fps` can be lower than the model sample FPS so playback matches sustained generation throughput instead of repeatedly starving the playback queue.

## RTX 4090 result

Hardware: RTX 4090 48 GB, PyTorch 2.8.0+cu128.

Configuration:

- requested frames: 129
- aligned RGB frames: 125
- chunk size: 4
- max area: 258048
- output FPS: 8
- bridge: 8766
- continuous publish: enabled

Result: **PASS**

| Metric | Result |
|---|---:|
| Generated RGB frames | 125 / 125 |
| Causal chunks | 8 |
| Generation time | 15.578 s |
| Sustained generated FPS | 8.024 |
| Playback duration | 15.625 s |
| TTFF | 1.091 s |
| First frame before generation completed | yes |
| Playback queue max depth | 12 |
| Raw frame queue max depth | 4 |
| Raw queue blocked time | 0.002 s |
| CUDA allocated after generation | 6.01 GB |
| CUDA reserved after generation | 16.19 GB |

Mac-side bridge sampling during generation confirmed monotonically increasing frames:

```text
seq=6   pts=0.750s
seq=24  pts=3.000s
seq=41  pts=5.125s
seq=59  pts=7.375s
seq=76  pts=9.500s
seq=94  pts=11.750s
seq=110 pts=13.750s
seq=124 pts=15.500s
```

This is not MP4 replay: sequence advanced while the causal generation process was still active.

## Next gate

C6-Quest:

1. Keep C5 available on `18765` as fallback.
2. Start a fresh C6 generation on cloud port `8766`.
3. Point the Mac Sender LingBot bridge to `http://127.0.0.1:18766`.
4. Observe Quest playback for first-frame latency, chunk-boundary stalls, visual continuity and reconnect behavior.
5. If 8 FPS is stable on-headset, test 253 aligned frames at 8 FPS (~31.6 s playback) using the existing 269-frame camera trajectory.
