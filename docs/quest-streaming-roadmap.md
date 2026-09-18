# LingBot → QuestPhoneStream Streaming Roadmap

Status: POC track. E1.2 adapter is frozen; G1/world-conditioning work is out of scope.

## Goal

Make LingBot on Apple Silicon publish generated video to QuestPhoneStream with low time-to-first-frame (TTFF), while reusing QuestPhoneStream's existing signaling, WebRTC session, Quest decoder, and SpatialPanel path.

```text
LingBot causal DiT
  -> completed x0 latent chunk
  -> stateful Wan VAE progressive decoder
  -> JPEG/RGB frame bridge (POC only)
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

### Q1 latent seam — code complete, generation regression pending

`wan/streaming/latent_tap.py` installs an opt-in `tap_causal_latent_chunks(...)` context manager around one existing causal-fast generation.

It does **not** rewrite `wan/image2video.py`. It recognizes the generator's existing post-chunk zero-timestep KV update by object identity with the final `x0`, and emits that live tensor only after the KV update succeeds.

Properties:
- no tap by default: zero behavior change.
- no implicit `.cpu()` or disk write.
- sink failure can be fail-open.
- original bound methods are restored on exit/failure.
- unit tests cover one-event-per-chunk semantics and restoration.

Required real-model gate:
- same prompt/image/action/seed with tap absent vs tap + no-op sink.
- final latent/video hashes or numeric output must match.
- cross-KV init count remains 1.
- self-KV remains persistent across chunks.
- sink-disabled performance regression <1%.

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

Gate example:
```bash
LINGBOT_VAE_DTYPE=bf16 python scripts/q1_5_validate_progressive_vae.py \
  --latents <generated_latents.safetensors> \
  --vae-pth <Wan2.1_VAE.pth> \
  --device mps --chunk-size 3
```

Do not claim Q1.5 PASS until this runs on a real E1.2 latent file.

### Q2 live progressive frames — wiring complete, memory gate pending

`wan/streaming/pipeline.py` provides `ProgressiveVaeFrameSink`:

```text
live x0 chunk
  -> ProgressiveWanVaeDecoder
  -> FrameBridgePublisher
  -> LatestFrameStore
  -> localhost bridge
  -> QuestPhoneStream AI MediaStream
```

This sink is intentionally synchronous and opt-in. Do **not** enable it on M4 16GB until Q1.5 passes and a DiT+VAE coexistence memory check proves it is safe.

The current M4 path intentionally unloads DiT before full VAE decode. That protection must not be removed just to claim streaming.

### Q2.5 overlap — not started

Only after Q2 memory gate:

```text
DiT chunk N+1
      ||
VAE decode chunk N
      ||
WebRTC encode/send chunk N-1
```

Measure separately with MPS VAE and MLX VAE. Keep `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0` out of the design.

## Metrics

Record for every real run:
- TTFF,
- DiT chunk cadence,
- progressive VAE latency/chunk,
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
| Q1 | DONE | real generation no-op equivalence |
| Q1.5 | DONE | M4 full-vs-progressive VAE report |
| Q2 | DONE | M4 DiT+VAE coexistence + first frame before generation end |
| Q2.5 | NOT STARTED | overlap profiling |
| Q3 | NOT STARTED | replace polling/canvas only if profiling justifies it |
| Q4 | NOT STARTED | Quest → Mac generation controls |

## Branches

- LingBot: `feat/quest-streaming-poc`
- QuestPhoneStream: `feat/ai-video-streaming-poc`

Do not merge either branch until Q0/Q1/Q1.5 evidence is attached.
