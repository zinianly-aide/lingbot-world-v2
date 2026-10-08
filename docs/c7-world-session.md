# C7: persistent Quest bridge + segmented LingBot WorldSession

Status: **CPU contract tested; RTX 4090 and Quest C7 validation pending**.

This is a practical continuous *segmented* generator. It creates a fresh
Causal DiT generation per segment, conditioning the next segment on the previous
decoded final frame. It does **not** carry DiT / VAE KV across segments, so
boundary stalls and continuity jumps remain possible. Do not describe it as
a fully seamless, infinite causal world.

## What stays alive

- A single bridge HTTP service, bound only to 127.0.0.1:8766.
- Global monotonic frame sequence and PTS, across every generated segment.
- The QuestPhoneStream Mac Sender WebRTC session (point its LingBot bridge to
  http://127.0.0.1:18766 using the usual SSH tunnel).
- Segment orchestration until Ctrl+C or `--max-segments N`.

Each subprocess runs opt-in `q2_live_quest_poc.py --continuous-publish
--external-bridge-url ... --sequence-offset ...`, retaining the existing
causal per-chunk generation, ProgressiveWanVaeDecoder, async encoding and
paced bridge publishing. The model process is recycled at each segment boundary.

## Preflight without GPU

```bash
python scripts/quest_continuous_world.py \
  --ckpt-dir /root/autodl-tmp/models/lingbot-world-v2-1.3b-causal-fast \
  --assets-dir /root/autodl-tmp/models/lingbot-assets \
  --dry-run
python -m unittest -v tests.test_quest_continuous_world
```

The preflight reads camera trajectory metadata and prints the proposed
subprocess commands; it does not load checkpoints.

## AutoDL 4090 gated start (only after approving the cloud test)

Use a separate checkout of the C7 branch from C5. Validate two segments first:

```bash
cd /root/autodl-tmp/lingbot-c7
/root/autodl-tmp/venv/bin/python scripts/quest_continuous_world.py \
  --ckpt-dir /root/autodl-tmp/models/lingbot-world-v2-1.3b-causal-fast \
  --assets-dir /root/autodl-tmp/models/lingbot-assets \
  --frame-num 129 --chunk-size 4 --local-attn-size 16 \
  --output-fps 8 --max-segments 2 \
  --work-dir /root/autodl-tmp/bench/c7-2segments
```

Once GPU smoke / queue / OOM / boundary observations pass, switch to the
normal 253-frame (about 31.6 s at 8 FPS) segment profile:

```bash
/root/autodl-tmp/venv/bin/python scripts/quest_continuous_world.py \
  --ckpt-dir /root/autodl-tmp/models/lingbot-world-v2-1.3b-causal-fast \
  --assets-dir /root/autodl-tmp/models/lingbot-assets \
  --frame-num 257 --chunk-size 4 --local-attn-size 16 \
  --output-fps 8 --max-segments 0 --retain-segments 3 \
  --work-dir /root/autodl-tmp/bench/c7-continuous
```

`--max-segments 0` continues creating *new generated* content until stopped;
it does not replay the same MP4. Each segment may momentarily freeze when
the next model process loads and re-encodes its image condition. The existing
camera trajectory contains 269 poses, supporting 257 requested frames.

## C5 isolation

- **Do not touch C5** cloud port 8765 / Mac tunnel port 18765.
- C7 binds its own cloud port 8766; Mac forwards 18766 to it.
- Mac Sender signaling stays on the existing LAN signaling endpoint.
- After the C7 bridge starts, Mac Sender selects LingBot source with bridge
  URL `http://127.0.0.1:18766`.

## Evidence / acceptance

Require cloud result: 2+ segments pass; session report contains monotonically
increasing offsets; bridge remains HTTP-200 while moving to next segment;
Last JPEG becomes next segment's image; no frame regression or OOM; stop cleans
up child process. Quest observations: no reconnect on boundary, forward visual
continuity and freeze duration measured. Do not declare C7 PASS without GPU and
Quest evidence.

## Safety / cleanup

- `Ctrl+C` or SIGTERM stops the generation subprocess group.
- `--child-timeout` bounds a stalled child (default 900 seconds).
- `--retain-segments` retains N most recent segment directories (default 3).
- Bind the HTTP bridge to localhost; SSH tunnel stays local-only.
- Stop AutoDL when no longer testing to avoid charges.

## 2026-10-09 RTX 4090 validation results

All runs isolated on `127.0.0.1:8766`; C5 ports untouched.

| Gate | Strategy | Frames | Result |
| --- | --- | --- | --- |
| G1 | Process per segment, 125 frames × 2, 8 FPS | 250/250 | PASS, seq 0–249 |
| G2 | Process per segment, 253 frames × 2, 8 FPS | 506/506 | PASS, seq 0–505; CUDA reserved ~13.3 GB |
| G3 | Same process, reuse DiT+VAE, 125 frames × 2, 8 FPS | 250/250 | PASS; 6.23s bridge publishing gap across segment boundary |
| G4 | Same process + overlapping playback, 125 frames × 2, 5 FPS | 250/250 | PASS; one persistent encoder/pacer |
| G5 | Buffered continuous, 125 frames × 3, 5 FPS | 375/375 | PASS; max actual inter-frame gap 271.24ms, 0 intervals above 300ms; boundary intervals 200.07ms and 200.09ms |

G5 end-of-segment GPU allocated memory ~4.0 GB, reserved ~14–15 GB;
bounded encoded queue max depth 48 frames. Content aesthetics, headset
latency, and visual scene continuity still require the physical Quest test.
Do not conflate the full clip's 5 FPS publish cadence with low-latency 8 FPS
interactive output. The buffered queue may add seconds of end-to-end latency.

Recommended experimental entrypoint for the Quest continuity gate:

```bash
cd /root/autodl-tmp/lingbot-c7
/root/autodl-tmp/venv/bin/python scripts/quest_world_buffered.py \
  --ckpt-dir /root/autodl-tmp/models/lingbot-world-v2-1.3b-causal-fast \
  --assets-dir /root/autodl-tmp/models/lingbot-assets \
  --frame-num 129 --chunk-size 4 --local-attn-size 16 \
  --output-fps 5 --max-segments 0 --retain-segments 3 \
  --bridge-port 8766 --work-dir /root/autodl-tmp/bench/c7-live
```

`--max-segments 0` uses the rented GPU continuously until explicitly
stopped. Prefer `--max-segments 3` during bounded acceptance. C7 buffered
mode preserves publisher/Bridge lifetime and starts next generation while
already generated frames remain queued. It resets causal context per segment,
so it **does not** establish seamless world-state continuity.

Stop generation with Ctrl+C/SIGTERM before stopping the AutoDL instance.
Do not start an unbounded GPU run unattended.
