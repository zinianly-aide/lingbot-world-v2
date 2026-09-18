# Quest Streaming POC — Execution Runbook

This runbook is intentionally execution-only. Do not redesign the streaming architecture while running these gates.

Branches:
- LingBot: `feat/quest-streaming-poc`
- QuestPhoneStream: `feat/ai-video-streaming-poc`

Frozen/out of scope:
- E1.2 adapter / MiniCPM conditioning
- G1 / world conditioning
- signaling / offer-answer-session
- Spatial Protocol schema
- Quest receiver / SpatialPanel
- DiT math / KV-cache ordering
- VAE weights
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0`

## 0. Local static checks

LingBot:

```bash
git checkout feat/quest-streaming-poc
git pull --ff-only
python -m unittest -v tests.test_streaming_contract tests.test_quest_streaming
```

QuestPhoneStream macOS sender:

```bash
git checkout feat/ai-video-streaming-poc
git pull --ff-only
cd apps/macos-sender
npm ci
npm test
npm run build
```

Stop on failure. Do not work around a failing test by weakening assertions.

## Q0 — Existing MP4 -> bridge -> existing WebRTC -> Quest

Terminal A in LingBot:

```bash
python scripts/quest_frame_bridge.py
```

Terminal B:

```bash
python scripts/quest_stream_replay.py \
  eval/e1.2/smoke/single_subject_seed42_e12.mp4 \
  --fps 12
```

Mac sender:
1. Start normally.
2. Select `LingBot AI video · localhost bridge`.
3. Leave bridge URL at `http://127.0.0.1:8765`.
4. Click `Start stream`.
5. Use the normal Quest session.

PASS only if:
- Quest receives the video through the existing receiver/SpatialPanel,
- frames advance in order rather than showing only the final frame,
- Stop ends the track/poll loop,
- reconnect works,
- no signaling/schema changes are needed.

Record: PASS/FAIL + observed Quest behavior. Do not modify model code during Q0.

## Q1 — Real-model latent tap non-interference

The frozen 13-frame E1.2 smoke is deliberately reused so the tapped run can be compared against an existing baseline latent tensor.

```bash
python scripts/q1_validate_latent_tap.py
```

Read:

```text
eval/quest-streaming/q1_tap/report.json
```

PASS requires:
- `pass=true`
- `exactTensorEqual=true`
- baseline/tapped SHA256 identical
- `maxAbs=0`
- event indices/start offsets correct

Important: `frame_num=13, chunk_size=4` contains only **one DiT chunk**. This gate proves non-interference, not real multi-chunk cadence.

Do not edit `wan/image2video.py` if this fails. Report the JSON and error first.

## Q1.5 — Progressive VAE equivalence

Use the exact frozen smoke latent tensor, but deliberately split its latent time axis into smaller decoder chunks to test cache continuity.

Run both:

```bash
LINGBOT_VAE_DTYPE=bf16 python scripts/q1_5_validate_progressive_vae.py \
  --latents eval/e1.2/smoke/work/latents.safetensors \
  --vae-pth /Volumes/ssd/lingbot-assets/Wan2.1_VAE.pth \
  --device mps --dtype bf16 --chunk-size 1 \
  --output eval/quest-streaming/q1_5_chunk1.json
```

```bash
LINGBOT_VAE_DTYPE=bf16 python scripts/q1_5_validate_progressive_vae.py \
  --latents eval/e1.2/smoke/work/latents.safetensors \
  --vae-pth /Volumes/ssd/lingbot-assets/Wan2.1_VAE.pth \
  --device mps --dtype bf16 --chunk-size 2 \
  --output eval/quest-streaming/q1_5_chunk2.json
```

PASS requires both JSON files to report `pass=true` and no boundary-specific error spike.

If this fails, stop before Q2. Do not add ad-hoc overlap frames or change VAE weights.

## Q2 — Live multi-chunk generation -> progressive VAE -> paced Quest stream

Prerequisites: Q0, Q1 and Q1.5 PASS.

Stop the standalone Q0 bridge first; Q2 owns port 8765 itself.

Start the QuestPhoneStream macOS sender and select `LingBot AI video · localhost bridge`. Then run:

```bash
LINGBOT_VAE_DTYPE=bf16 python scripts/q2_live_quest_poc.py \
  --device mps \
  --vae-dtype bf16 \
  --frame-num 33 \
  --chunk-size 4
```

Output:

```text
eval/quest-streaming/q2_live/report.json
```

Why `33 / 4`: after Wan temporal alignment this produces 29 RGB frames from 8 latent timesteps, i.e. **two real DiT chunks**. The first progressive decoded chunk can therefore reach Quest before the second DiT chunk finishes.

The script deliberately:
- creates a matching image-condition cache,
- loads DiT and VAE together only for this gate,
- keeps the MPS watermark guard intact,
- taps final x0 chunks after KV update,
- keeps VAE causal feature cache across chunks,
- JPEG-encodes decoded frames,
- puts them into a bounded playback buffer,
- publishes at the model playback FPS instead of overwriting the latest-frame slot in a burst,
- records TTFF, chunk decode time, queue depth and MPS memory.

PASS requires:
- `pass=true`
- `totalChunks >= 2`
- `chunkEvents == totalChunks`
- progressive output frame count equals aligned frame count
- playback publishes every decoded frame
- `firstFrameBeforeGenerationEnd=true`
- Quest visibly starts receiving frames before the full generation finishes
- no structure/temporal corruption at the chunk boundary

If DiT+VAE coexistence OOMs, that is a valid **Q2 BLOCKED_BY_MEMORY** result. Do not set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0` and do not weaken the memory guard.

## After Q2

Do not start Q2.5 overlap automatically.

Return only:
- Q0 PASS/FAIL
- Q1 report JSON summary
- Q1.5 chunk1/chunk2 report summaries
- Q2 report summary + Quest observation
- exact commit SHAs used
- any failing command/log tail

No merge, no G1, no protocol redesign.
