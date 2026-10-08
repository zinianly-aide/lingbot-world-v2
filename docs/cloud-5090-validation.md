# RTX 5090 Cloud Validation Runbook

This gate prepares the LingBot 1.3B causal-fast path for a single RTX 5090 before reconnecting the Quest pipeline.

## Scope

Keep the transport architecture frozen while validating CUDA:

- LingBot world generation may change only for correctness/measurement.
- Do not redesign signaling, Spatial Protocol, Quest receiver, or WebRTC session handling.
- Do not start FP8/SageAttention/torch.compile optimization until the stock CUDA baseline is reproducible.
- Do not merge this branch based on one successful cloud run; record the reports first.

## Known-good environment

First AutoDL validation on 2026-10-07 used:

- RTX 5090 32 GB
- PyTorch 2.8.0+cu128
- CUDA runtime 12.8
- Python 3.12
- Ubuntu 22.04
- one GPU / `WORLD_SIZE=1` / `ulysses_size=1`

Reference observations from that run are **not hard acceptance thresholds**:

| Gate | Reference result |
|---|---:|
| 13-frame full smoke, 832x464 actual | 16.24 s cold wall time |
| single DiT chunk in that smoke | ~1.08 s |
| full-smoke nvidia-smi peak | ~15.0 GB |
| Q2 M4-parity profile, 29 output frames | 2.77 s generation |
| Q2 M4-parity TTFF | 0.566 s |
| Q2 M4-parity nvidia-smi peak | ~11.7 GB |
| Q2 480x832-class profile, 29 frames | 3.81 s generation |
| Q2 480x832-class TTFF | 1.22 s |

## Before starting the GPU

Keep the AutoDL instance shut down while editing. When ready, boot the GPU instance and set paths:

```bash
export CKPT_DIR=/root/autodl-tmp/models/lingbot-world-v2-1.3b-causal-fast
export ASSETS_DIR=/root/autodl-tmp/models/lingbot-assets
export PYTHON=/root/autodl-tmp/venv/bin/python
```

Run the non-GPU contract gate first:

```bash
bash scripts/cloud_cuda_validate.sh static
```

Then run the GPU gates separately so a failure does not waste the rest of the paid session:

```bash
bash scripts/cloud_cuda_validate.sh smoke13
bash scripts/cloud_cuda_validate.sh q2-exact
bash scripts/cloud_cuda_validate.sh q2-480p
```

Each run writes logs, reports and a 200 ms `nvidia-smi` trace under `eval/cloud-5090/<timestamp>/` unless `OUT_ROOT` is overridden.

## Gate definitions

### C0 — static

PASS requires Python compilation plus:

- `tests.test_m36_staged`
- `tests.test_streaming_contract`
- `tests.test_quest_streaming`
- `tests.test_replay_loop`

This specifically guards the two failures found during the first CUDA bring-up:

1. `scripts/q2_live_quest_poc.py` must compile.
2. `stage=full` must continue into VAE decode instead of sharing the `generate-latents` early return.

### C1 — full smoke

PASS requires:

- exit code 0,
- a non-empty `out.mp4`,
- 13 decoded frames when `ffprobe` is available,
- no CUDA OOM or NaN/Inf failure.

### C2 — exact M4-parity profile

Configuration is intentionally matched to the existing M4 Q2 report:

- requested frames: 33
- aligned output frames: 29
- chunk size: 2
- max area: 258048
- seed: 123
- BF16 VAE

PASS requires Q2 JSON `pass=true`, all 29 frames published, queue drained, and first frame published before generation completes.

### C3 — 480x832-class target

Configuration:

- requested frames: 33
- aligned output frames: 29
- chunk size: 4
- max area: 399360
- seed: 42
- BF16 VAE

Record generation time, TTFF, chunk decode times, peak VRAM, GPU utilization and power.

## After CUDA gates pass

Do the network/Quest gates in one paid session:

1. Start Q2/bridge on the cloud host.
2. SSH-forward cloud `127.0.0.1:8765` to any free localhost port on the Mac. Use `127.0.0.1:8765` only when it is free; otherwise choose another localhost port such as `18765`. The macOS sender accepts any localhost HTTP(S) port via its bridge URL input.
3. Verify Mac `aiVideoSource.ts` sees `/healthz` and monotonically increasing frame sequence.
4. Start the existing QuestPhoneStream WebRTC session without changing signaling or Spatial Protocol.
5. Record cloud generation TTFF, cloud-to-Mac bridge latency, Mac sender behavior and Quest-visible continuity/reconnect behavior.
6. Shut the GPU instance down immediately after evidence is saved.

Only after this baseline is repeatable should work begin on persistent WorldSession, `torch.compile`, FP8/SageAttention, NVENC/direct WebRTC, or action-conditioned closed-loop control.
