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
