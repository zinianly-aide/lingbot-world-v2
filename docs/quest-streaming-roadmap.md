# LingBot → QuestPhoneStream Streaming Roadmap

Status: POC track. E1.2 adapter is frozen; G1/world-conditioning work is out of scope.

## Goal

Make LingBot on Apple Silicon publish generated video to QuestPhoneStream with low time-to-first-frame (TTFF), while reusing QuestPhoneStream's existing signaling, WebRTC session, Quest decoder, and SpatialPanel path.

```text
LingBot causal DiT
  -> completed x0 latent chunk
  -> stateful Wan VAE progressive decoder
  -> bounded paced frame buffer
  -> localhost JPEG frame bridge (POC only)
  -> QuestPhoneStream macOS Canvas MediaStream
  -> existing RTCPeerConnection video track
  -> QuestWebRtcReceiver / SpatialPanel
```

The localhost frame bridge is deliberately temporary. It gives us a measurable end-to-end seam before committing to VideoToolbox/zero-copy plumbing.

## Implemented critical path

### Q0 transport seam — code complete, device verification pending

LingBot:
- `scripts/quest_frame_bridge.py` — localhost bridge server.
- `scripts/quest_stream_replay.py` — replay any existing MP4 as JPEG frames.
- `wan/streaming/frame_bridge.py` — latest-frame store, status, sequence/PTS headers, CORS expose headers.

QuestPhoneStream:
- `src/aiVideoSource.ts` converts bridge JPEGs to a Canvas `MediaStream`.
- `src/aiVideoBootstrap.ts` injects `qps-ai-video` as a pseudo capture source and intercepts only that `getUserMedia()` request.
- Existing `renderer.ts`, signaling, `createPeer()`, offer/answer, data channels, Quest receiver and SpatialPanel stay unchanged.

Manual gate:
```bash
python scripts/quest_frame_bridge.py
python scripts/quest_stream_replay.py eval/e1.2/smoke/single_subject_seed42_e12.mp4 --fps 12
```
Then select `LingBot AI video · localhost bridge` in the macOS sender and start the normal Quest session.

Acceptance:
- Quest displays replayed video.
- reconnect/stop does not leak tracks or polling.
- bridge sequence increases monotonically.
- no signaling/schema changes.

### Q1 latent seam — code complete, M4 evidence pending

`wan/streaming/latent_tap.py` installs an opt-in `tap_causal_latent_chunks(...)` context manager around one existing causal-fast generation.

It does **not** rewrite `wan/image2video.py`. It recognizes the generator's existing post-chunk zero-timestep KV update by object identity with the final `x0`, and emits that live tensor only after the KV update succeeds.

Properties:
- no tap by default: zero behavior change.
- no implicit `.cpu()` or disk write.
- sink failure can be fail-open.
- original bound methods are restored even if `generate-latents` unloads the DiT before the context exits.
- unit tests cover event semantics, fail-open and teardown.

Real-model gate is automated by:

```bash
python scripts/q1_validate_latent_tap.py
```

It re-runs the frozen E1.2 smoke with a metadata-only tap and requires exact latent equality / identical SHA256 against the existing baseline. The current 13-frame, chunk-size-4 smoke contains one real DiT chunk, so this proves non-interference; true multi-chunk cadence is exercised by Q2.

### Q1.5 progressive VAE semantics — implementation complete, M4 evidence pending

Important code fact: Wan's VAE already decodes one latent timestep at a time with causal feature caches. The original `WanVAE_.decode()` clears those caches only at call boundaries.

`wan/streaming/vae_progressive.py` mirrors the original decode order while keeping the same feature cache alive across calls. `scripts/q1_5_validate_progressive_vae.py` compares:

```text
full latent -> existing vae.decode()
vs
same latent split into chunks -> ProgressiveWanVaeDecoder
```

It records:
- max/mean absolute error,
- MSE / PSNR,
- output seam errors,
- full vs progressive decode time,
- MPS/CUDA memory.

For the frozen 4-latent-step smoke, run both chunk size 1 and 2 so cache continuity is actually exercised. Do not claim Q1.5 PASS until both real M4 reports pass.

### Q2 live progressive frames — code complete, M4/Quest gate pending

A VAE chunk yields several RGB frames at once. Publishing them directly to a latest-frame slot would overwrite intermediate frames before Electron can poll them. `wan/streaming/buffered_publisher.py` therefore JPEG-encodes into a bounded queue and publishes one frame per playback interval. When generation is slower than playback, the last frame remains visible until the next generated chunk arrives; generated frames are not silently burst/dropped.

`scripts/q2_live_quest_poc.py` wires the full POC:

```text
live x0 chunk
  -> ProgressiveWanVaeDecoder
  -> BufferedFrameBridgePublisher
  -> LatestFrameStore
  -> localhost bridge
  -> QuestPhoneStream AI MediaStream
  -> existing WebRTC / Quest receiver
```

The script uses `frame_num=33, chunk_size=4` by default, which aligns to two real DiT chunks / 29 output frames. It:
- creates a matching image-condition cache,
- explicitly tests DiT+VAE coexistence without changing the MPS watermark guard,
- records TTFF, per-chunk VAE/JPEG times, queue depth and MPS memory,
- requires the first published frame to occur before generation finishes.

A DiT+VAE OOM on M4 16GB is a valid Q2 memory-blocked result. Do not use `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0` to force the gate.

### Q2.5 overlap — not started

Only after Q2 memory gate:

```text
DiT chunk N+1
      ||
VAE decode chunk N
      ||
WebRTC encode/send chunk N-1
```

Measure separately with MPS VAE and MLX VAE. Do not start overlap work if Q2 proves co-residency is not viable.

## Metrics

Record for every real run:
- TTFF,
- DiT chunk cadence,
- progressive VAE latency/chunk,
- JPEG enqueue/playback queue depth,
- bridge-to-canvas latency,
- WebRTC sender/receiver latency,
- received frame sequence/drop count,
- MPS current/driver peak,
- RSS,
- output equivalence hashes/errors.

## What the Agent should NOT redesign

- E1.2 adapter / MiniCPM conditioning.
- G1 / world conditioning.
- signaling or offer/answer/session semantics.
- Spatial Protocol schema.
- Quest receiver/SpatialPanel architecture.
- DiT math / KV-cache update order.
- VAE weights.

## Remaining gates

| Gate | Code | Evidence still required |
|---|---|---|
| Q0 | DONE | Quest real-device replay |
| Q1 | DONE | M4 exact latent-tap equivalence report |
| Q1.5 | DONE | M4 chunk1 + chunk2 full-vs-progressive reports |
| Q2 | DONE | M4 coexistence + first-frame-before-finish + Quest observation |
| Q2.5 | NOT STARTED | overlap profiling only if Q2 permits it |
| Q3 | NOT STARTED | replace polling/canvas only if profiling justifies it |
| Q4 | NOT STARTED | Quest → Mac generation controls |

## Branches / runbook

- LingBot: `feat/quest-streaming-poc`
- QuestPhoneStream: `feat/ai-video-streaming-poc`
- Execution-only instructions: `docs/quest-streaming-runbook.md`

Do not merge either branch until Q0/Q1/Q1.5 evidence is attached and Q2 has an explicit PASS or memory-blocked result.
