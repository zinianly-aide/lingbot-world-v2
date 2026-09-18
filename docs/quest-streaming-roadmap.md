# LingBot → QuestPhoneStream Streaming Roadmap

Status: POC track. E1.2 adapter is frozen; G1/world-conditioning work is out of scope.

## Goal

Make LingBot on Apple Silicon publish generated video to QuestPhoneStream with low time-to-first-frame (TTFF), while reusing QuestPhoneStream's existing signaling, WebRTC session, Quest decoder, and SpatialPanel path.

```text
LingBot causal DiT
  -> latent chunk boundary (already exists)
  -> VAE decode worker
  -> JPEG/RGB frame bridge (POC only)
  -> QuestPhoneStream macOS sender Canvas MediaStream
  -> existing RTCPeerConnection video track
  -> QuestWebRtcReceiver / SpatialPanel
```

The localhost frame bridge is deliberately temporary. It gives us a measurable end-to-end seam before committing to VideoToolbox/zero-copy plumbing.

## Current code facts

- `WanI2VCausal._generate_causal_fast()` already loops over latent chunks.
- The completed chunk is available as `x0` immediately before `pred_latent_chunks.append(x0)`.
- Today all chunks are concatenated and VAE-decoded only after DiT generation finishes.
- On M4 sequential-load mode unloads DiT before loading VAE. That memory behavior must not be removed just to claim streaming.
- QuestPhoneStream macOS sender already publishes any `MediaStream` video track through the existing WebRTC negotiation. Screen capture is only the current source.

## Gates

### Q0 — transport path, no model changes

Purpose: prove `LingBot output -> macOS sender -> Quest` independently from progressive generation.

1. Start bridge:
   ```bash
   python scripts/quest_frame_bridge.py
   ```
2. Replay any existing generated MP4:
   ```bash
   python scripts/quest_stream_replay.py eval/e1.2/smoke/single_subject_seed42_e12.mp4 --fps 12
   ```
3. In QuestPhoneStream macOS sender, use `createAiVideoSource()` from `apps/macos-sender/src/aiVideoSource.ts` instead of desktop capture and pass its `MediaStream` to the existing `createPeer()` path.

Acceptance:
- Quest displays the replayed video.
- Existing signaling/session/data-channel behavior is unchanged.
- Stop/reconnect does not leak tracks.
- Record WebRTC latency and bridge duplicate/error counters.

### Q1 — expose causal latent chunks

Add an optional `latent_chunk_sink=None` to `WanI2VCausal.generate()` and `_generate_causal_fast()`.

Hook location (do not move model math):

```python
pred_latent_chunks.append(x0)

if latent_chunk_sink is not None:
    from wan.streaming.events import LatentChunkEvent
    event = LatentChunkEvent(
        generation_id=generation_id,
        chunk_index=chunk_id,
        total_chunks=num_inference_chunk,
        latent_start=chunk_id * chunk_size,
        latent_count=int(x0.shape[1]),
        shape=tuple(int(v) for v in x0.shape),
        dtype=str(x0.dtype),
        seed=seed,
        elapsed_ms=(time.perf_counter() - stream_started) * 1000.0,
    )
    latent_chunk_sink.on_latent_chunk(event, x0)
```

Rules:
- Default sink is `None`: no CPU copy, no file I/O, no output change.
- Sink owns any detach/copy needed after callback return.
- Causal KV update remains exactly where it is today.
- Start with `causal_fast` only; do not change pretrain behavior for the POC.

Acceptance:
- Existing generation remains bit-identical with sink disabled.
- Event order is 0..N-1 and latent shapes match the chunks that are concatenated today.
- Sink-disabled benchmark regression < 1%.

### Q1.5 — prove VAE chunk-decode semantics

Do **not** assume each latent chunk can be independently VAE-decoded. Wan's temporal VAE may require history/overlap.

For one fixed E1.2 sample:
1. Save full latent output.
2. Decode full latent as reference.
3. Decode chunk/overlap candidates.
4. Compare RGB frames at boundaries and record max/mean error plus visual seam evidence.

Only after equivalence is understood should progressive decoded frames be published to the bridge.

### Q2 — progressive frames

Once Q1.5 passes, connect:

```text
latent_chunk_sink
  -> bounded decode queue
  -> VAE progressive decoder
  -> frame publisher
  -> localhost bridge
  -> Canvas MediaStream
  -> existing WebRTC
```

Required backpressure:
- bounded queue (drop/stop policy explicit),
- generation/session id on every chunk,
- cancellation,
- EOS/completed/failed state,
- stale generation frames rejected.

Primary metrics:
- TTFF,
- latent chunk cadence,
- VAE decode latency/chunk,
- bridge-to-canvas latency,
- WebRTC send/receive latency,
- queue depth,
- MPS current/driver memory peak.

### Q2.5 — overlap only if M4 memory permits

Desired pipeline:

```text
DiT chunk N+1
      ||
VAE decode chunk N
      ||
WebRTC encode/send chunk N-1
```

But current M4 path intentionally unloads DiT before VAE. Do not remove that protection without a memory gate. First benchmark overlap with the existing MPS VAE and the MLX VAE POC separately. Keep `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0` out of the design.

### Q3 — replace POC bridge with production local transport

After TTFF is proven:
- prefer decoded-frame zero/low-copy handoff,
- let Chromium/WebRTC use the platform encoder initially,
- evaluate direct VideoToolbox only if profiling shows browser/canvas copy is material,
- retain localhost bridge as a debug/replay tool.

### Q4 — Quest -> Mac control

Add only after streaming is stable:
- start/cancel generation,
- prompt/image/session id,
- progress and buffer state,
- camera/action updates.

Do not change Spatial Protocol schema in Q0-Q2.

## Target checkpoints

| Gate | Target |
|---|---|
| Q0 | Existing MP4 reaches Quest through AI-video source |
| Q1 | Latent chunk hook, sink-off output unchanged |
| Q1.5 | VAE temporal boundary semantics proven |
| Q2 | First decoded chunk reaches Quest before full generation completes |
| Q2.5 | TTFF < 15-20s without memory pressure |
| Q3 | POC polling/canvas bottlenecks profiled and replaced only if needed |
| Q4 | Quest control closes the interactive loop |

## Branches

- LingBot: `feat/quest-streaming-poc`
- QuestPhoneStream: `feat/ai-video-streaming-poc`

Neither branch should merge model-evaluation/G1 work into the streaming experiment.
