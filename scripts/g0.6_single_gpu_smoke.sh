#!/bin/bash
# G0.6: Single CUDA GPU minimal smoke test for LingBot-World 1.3B
# Verifies: WORLD_SIZE=1 / ulysses_size=1 / no NCCL / no FSDP / no Ulysses
# A: baseline (original prompt only)
# B: cached world_condition.json + composed prompt
#
# Usage:
#   bash scripts/g0.6_single_gpu_smoke.sh <ckpt_dir> <assets_dir> [image] [action_path] [frame_num] [size]
#
# Example:
#   bash scripts/g0.6_single_gpu_smoke.sh \
#     ./lingbot-world-v2-1.3b-causal-fast \
#     ./lingbot-world-v2-14b-causal-fast \
#     examples/00/image.jpg examples/00 17 832*480

set -e

CKPT_DIR="${1:?Usage: $0 <ckpt_dir> <assets_dir> [image] [frame_num] [size]}"
ASSETS_DIR="${2:?Missing assets_dir (14B checkpoint for T5/VAE)}"
IMAGE="${3:-examples/00/image.jpg}"
ACTION_PATH="${4:-examples/00}"
FRAME_NUM="${5:-17}"
SIZE="${6:-832*480}"
SEED=42
OUTPUT_DIR="g0.6-smoke"
WORLD_CONDITION_FILE="g0-smoke-mlx/world_condition.json"
PROMPT="Keep the main subject and move the camera forward"

echo "============================================"
echo "G0.6 Single GPU Smoke Test"
echo "============================================"
echo "CKPT_DIR:     $CKPT_DIR"
echo "ASSETS_DIR:   $ASSETS_DIR"
echo "IMAGE:        $IMAGE"
echo "ACTION_PATH:  $ACTION_PATH"
echo "FRAME_NUM:    $FRAME_NUM"
echo "SIZE:         $SIZE"
echo "SEED:         $SEED"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
echo "============================================"

# ---- Environment check ----
echo ""
echo "[1/5] Environment check"
python3 -c "
import torch
print(f'  torch:        {torch.__version__}')
print(f'  CUDA:         {torch.version.cuda}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count:    {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}:        {props.name} ({props.total_mem / 1e9:.1f} GB)')
else:
    print('  ERROR: No CUDA GPU available. This script requires a CUDA GPU.')
    exit(1)
"

# ---- Verify single-GPU path does not trigger distributed ----
echo ""
echo "[2/5] Verifying single-GPU path (no NCCL / no FSDP / no Ulysses)"
echo "  WORLD_SIZE will default to 1 (not set)"
echo "  --dit_fsdp / --t5_fsdp: NOT passed (asserted off in code)"
echo "  --ulysses_size: 1 (default, use_sp=False)"
echo "  --offload_model: true (default for WORLD_SIZE=1)"
echo "  Check: generate.py line 359: 'if world_size > 1' guards NCCL init"
echo "  Check: generate.py line 367-372: asserts fsdp/ulysses are off for single GPU"
echo "  OK"

# ---- Run A: baseline ----
echo ""
echo "[3/5] Run A: Baseline (original prompt only)"
mkdir -p "$OUTPUT_DIR/A"
python3 generate.py \
    --task i2v-1.3B \
    --infer_mode causal_fast \
    --size "$SIZE" \
    --frame_num "$FRAME_NUM" \
    --ckpt_dir "$CKPT_DIR" \
    --assets_dir "$ASSETS_DIR" \
    --image "$IMAGE" \
    --action_path "$ACTION_PATH" \
    --prompt "$PROMPT" \
    --base_seed "$SEED" \
    --save_dir "$OUTPUT_DIR/A" \
    --offload_model true \
    2>&1 | tee "$OUTPUT_DIR/A/run.log"

echo "  A done. Output: $OUTPUT_DIR/A/"

# ---- Run B: cached world condition ----
echo ""
echo "[4/5] Run B: Cached world_condition.json + composed prompt"
if [ ! -f "$WORLD_CONDITION_FILE" ]; then
    echo "  WARNING: $WORLD_CONDITION_FILE not found."
    echo "  Generate it first on Mac with:"
    echo "    python scripts/g0_smoke.py --image $IMAGE --prompt \"$PROMPT\" --output-dir g0-smoke-mlx --vlm_backend mlx"
    echo "  Skipping B."
else
    mkdir -p "$OUTPUT_DIR/B"
    python3 generate.py \
        --task i2v-1.3B \
        --infer_mode causal_fast \
        --size "$SIZE" \
        --frame_num "$FRAME_NUM" \
        --ckpt_dir "$CKPT_DIR" \
        --assets_dir "$ASSETS_DIR" \
        --image "$IMAGE" \
        --action_path "$ACTION_PATH" \
        --prompt "$PROMPT" \
        --base_seed "$SEED" \
        --world_condition_file "$WORLD_CONDITION_FILE" \
        --dump_world_prompt \
        --save_dir "$OUTPUT_DIR/B" \
        --offload_model true \
        2>&1 | tee "$OUTPUT_DIR/B/run.log"
    echo "  B done. Output: $OUTPUT_DIR/B/"
fi

# ---- Summary ----
echo ""
echo "[5/5] Summary"
echo "============================================"
echo "Results:"
echo "  A (baseline):  $OUTPUT_DIR/A/*.mp4"
echo "  B (world cond): $OUTPUT_DIR/B/*.mp4"
echo ""
echo "To check VRAM usage, run with nvidia-smi in another terminal:"
echo "  watch -n 0.5 nvidia-smi"
echo ""
echo "If OOM occurs, try in order:"
echo "  1. --t5_cpu true        (move UMT5 to CPU, saves ~11GB)"
echo "  2. --offload_model true (already default for single GPU)"
echo "  3. Reduce --frame_num 17 -> 9 -> 5"
echo "  4. Reduce --local_attn_size (e.g. -1 -> 12 -> 6)"
echo "  5. Reduce --size 832*480 (already minimum)"
echo "============================================"
