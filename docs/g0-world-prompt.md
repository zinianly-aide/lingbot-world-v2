# G0: MiniCPM-V world-prompt conditioning

This is an optional, observation-only stage. With no new flags, `generate.py`
uses the original prompt byte-for-byte and the existing
`prompt -> UMT5 -> WanI2VCausal` path. When enabled, rank 0 performs one image
query, releases MiniCPM-V, and broadcasts the resulting text to the other
workers. `WanI2VCausal`, `wan/modules/t5.py`, Causal DiT, KV cache and camera
Plücker conditioning are unchanged.

## Usage

The VLM-only smoke test writes `world_condition.json`, `world_prompt.txt`,
and `run_info.json` without constructing LingBot:

```bash
python scripts/g0_smoke.py --image examples/00/image.jpg \
  --prompt "Keep the main subject and move the camera forward" \
  --output-dir g0-smoke
```

```bash
# B: VLM observation -> composed prompt -> existing LingBot path
torchrun --nproc_per_node=4 generate.py \
  --task i2v-1.3B --ckpt_dir /models/lingbot-1.3b \
  --image examples/04/image.jpg --prompt "Move the camera left while keeping the red car" \
  --vlm_world_prompt --vlm_model openbmb/MiniCPM-V-4.6 --vlm_device auto \
  --dump_world_prompt --save_dir output/vlm

# Re-run LingBot without loading MiniCPM-V.
python generate.py ... --world_condition_file output/vlm/world_condition.json \
  --dump_world_prompt --save_dir output/vlm-replay
```

MiniCPM-V 4.6's official Transformers path currently uses
`AutoModelForImageTextToText`/`AutoProcessor` and a recent Transformers
release. The upstream LingBot environment is intentionally not upgraded by
this change; use an isolated VLM environment if its pinned Transformers
version is too old. A load/API failure is logged and generation continues with
the original user prompt.

## A/B protocol

Run the same command twice with the same `--base_seed`, `--image`,
`--action_path`, `--frame_num`, `--size`, `--infer_mode`, and LingBot
checkpoint. A omits `--vlm_world_prompt` and uses
`--save_file baseline.mp4`; B enables it and uses
`--save_file vlm_conditioned.mp4` while writing `world_condition.json` and
`world_prompt.txt`. Compare the two videos using the same frame samples. Score
identity, layout, color/attributes, camera-motion continuity, hallucinated
objects, and whether the explicit user request remains dominant. This is a
G0 human-review experiment, not evidence of real-world or device validation.

## Verification boundary

The included unit tests cover schema parsing, malformed-output fallback,
disabled-path identity, cache round-trip, user-prompt precedence, and VLM
load-failure fallback. Unity/DiT inference, MiniCPM-V weights, GPU memory
usage, generated videos, and any quality improvement are **NOT RUN** here.
Do not proceed to a hidden-state adapter or G1 unless paired A/B samples show a
repeatable world-consistency improvement.
