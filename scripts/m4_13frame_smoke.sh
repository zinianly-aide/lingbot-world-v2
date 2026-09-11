#!/bin/bash
# M4 13-frame smoke test harness for LingBot-World 2.0 1.3B on Apple Silicon
#
# This script runs the full pipeline in 3 independent stages (M3.6):
#   Stage A: encode-image  (VAE only)
#   Stage B: generate-latents (DiT only, uses cached image condition)
#   Stage C: decode         (VAE only, uses cached latents)
#
# Each stage records memory usage (RSS + MPS allocated/driver memory).
#
# Usage:
#   bash scripts/m4_13frame_smoke.sh [full|encode-image|generate-latents|decode]
#
# Prerequisites:
#   - LingBot 1.3B checkpoint downloaded
#   - T5 prompt embedding pre-computed (scripts/encode_prompt.py)
#   - VAE checkpoint available
#   - examples/00/ action_path with poses.npy, intrinsics.npy, image.jpg

set -euo pipefail

# ==================== Configuration ====================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Paths
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/snapshots/*/}"
ASSETS_DIR="${ASSETS_DIR:-/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-14b-causal-fast/snapshots/*/}"
ACTION_PATH="${ACTION_PATH:-examples/00}"
IMAGE_PATH="${IMAGE_PATH:-examples/00/image.jpg}"
PROMPT="${PROMPT:-Keep the main subject and move the camera forward}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-m4-smoke-13frame}"
mkdir -p "$OUTPUT_DIR"

PROMPT_EMBEDS_FILE="${PROMPT_EMBEDS_FILE:-$OUTPUT_DIR/prompt_embeds.safetensors}"
IMAGE_CONDITION_FILE="${IMAGE_CONDITION_FILE:-$OUTPUT_DIR/image_condition.safetensors}"
LATENTS_FILE="${LATENTS_FILE:-$OUTPUT_DIR/generated_latents.safetensors}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-$OUTPUT_DIR/output.mp4}"

# Generation params
DEVICE="${DEVICE:-mps}"
FRAME_NUM="${FRAME_NUM:-13}"
SIZE="${SIZE:-832*480}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
SEED="${SEED:-42}"
INFER_MODE="${INFER_MODE:-causal_fast}"

# Memory log
MEMORY_LOG="$OUTPUT_DIR/memory_log.txt"

# ==================== Helper Functions ====================

log_memory() {
    local label="$1"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # RSS (in MB)
    local rss_mb
    rss_mb=$(ps -o rss= -p $$ 2>/dev/null | awk '{printf "%.1f", $1/1024}' || echo "N/A")

    # MPS memory (if available)
    local mps_allocated="N/A"
    local mps_driver="N/A"
    if python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        mps_allocated=$(python -c "
import torch
try:
    print(f'{torch.mps.current_allocated_memory()/1024/1024:.1f}')
except:
    print('N/A')
" 2>/dev/null || echo "N/A")
        mps_driver=$(python -c "
import torch
try:
    print(f'{torch.mps.driver_allocated_memory()/1024/1024:.1f}')
except:
    print('N/A')
" 2>/dev/null || echo "N/A")
    fi

    echo "[$timestamp] $label | RSS=${rss_mb}MB | MPS_alloc=${mps_allocated}MB | MPS_driver=${mps_driver}MB" | tee -a "$MEMORY_LOG"
}

run_stage() {
    local stage="$1"
    echo ""
    echo "========================================"
    echo "Stage: $stage"
    echo "========================================"
    log_memory "before_${stage}"

    local start_time
    start_time=$(date +%s)

    case "$stage" in
        encode-image)
            python generate.py \
                --checkpoint_dir "$CHECKPOINT_DIR" \
                --assets_dir "$ASSETS_DIR" \
                --device "$DEVICE" \
                --infer_mode "$INFER_MODE" \
                --image "$IMAGE_PATH" \
                --prompt "$PROMPT" \
                --action_path "$ACTION_PATH" \
                --frame_num "$FRAME_NUM" \
                --size "$SIZE" \
                --chunk_size "$CHUNK_SIZE" \
                --seed "$SEED" \
                --stage encode-image \
                --dump_image_condition "$IMAGE_CONDITION_FILE" \
                --sequential_load \
                2>&1 | tee -a "$OUTPUT_DIR/stage_encode.log"
            ;;

        generate-latents)
            if [ ! -f "$PROMPT_EMBEDS_FILE" ]; then
                echo "ERROR: Prompt embeds not found: $PROMPT_EMBEDS_FILE"
                echo "Run: python scripts/encode_prompt.py --prompt \"$PROMPT\" --assets_dir \"$ASSETS_DIR\" --output \"$PROMPT_EMBEDS_FILE\""
                exit 1
            fi
            if [ ! -f "$IMAGE_CONDITION_FILE" ]; then
                echo "ERROR: Image condition not found: $IMAGE_CONDITION_FILE"
                echo "Run stage encode-image first."
                exit 1
            fi

            python generate.py \
                --checkpoint_dir "$CHECKPOINT_DIR" \
                --assets_dir "$ASSETS_DIR" \
                --device "$DEVICE" \
                --infer_mode "$INFER_MODE" \
                --image "$IMAGE_PATH" \
                --prompt "$PROMPT" \
                --action_path "$ACTION_PATH" \
                --frame_num "$FRAME_NUM" \
                --size "$SIZE" \
                --chunk_size "$CHUNK_SIZE" \
                --seed "$SEED" \
                --stage generate-latents \
                --prompt_embeds_file "$PROMPT_EMBEDS_FILE" \
                --image_condition_file "$IMAGE_CONDITION_FILE" \
                --output_latents_file "$LATENTS_FILE" \
                --sequential_load \
                2>&1 | tee -a "$OUTPUT_DIR/stage_generate.log"
            ;;

        decode)
            if [ ! -f "$LATENTS_FILE" ]; then
                echo "ERROR: Latents not found: $LATENTS_FILE"
                echo "Run stage generate-latents first."
                exit 1
            fi

            python generate.py \
                --checkpoint_dir "$CHECKPOINT_DIR" \
                --assets_dir "$ASSETS_DIR" \
                --device "$DEVICE" \
                --infer_mode "$INFER_MODE" \
                --image "$IMAGE_PATH" \
                --prompt "$PROMPT" \
                --action_path "$ACTION_PATH" \
                --frame_num "$FRAME_NUM" \
                --size "$SIZE" \
                --chunk_size "$CHUNK_SIZE" \
                --seed "$SEED" \
                --stage decode \
                --latents_file "$LATENTS_FILE" \
                --output "$OUTPUT_VIDEO" \
                --sequential_load \
                2>&1 | tee -a "$OUTPUT_DIR/stage_decode.log"
            ;;

        full)
            # Full pipeline in one process (for comparison)
            python generate.py \
                --checkpoint_dir "$CHECKPOINT_DIR" \
                --assets_dir "$ASSETS_DIR" \
                --device "$DEVICE" \
                --infer_mode "$INFER_MODE" \
                --image "$IMAGE_PATH" \
                --prompt "$PROMPT" \
                --action_path "$ACTION_PATH" \
                --frame_num "$FRAME_NUM" \
                --size "$SIZE" \
                --chunk_size "$CHUNK_SIZE" \
                --seed "$SEED" \
                --stage full \
                --prompt_embeds_file "$PROMPT_EMBEDS_FILE" \
                --output "$OUTPUT_VIDEO" \
                --sequential_load \
                2>&1 | tee -a "$OUTPUT_DIR/stage_full.log"
            ;;

        *)
            echo "Unknown stage: $stage"
            echo "Usage: $0 [full|encode-image|generate-latents|decode]"
            exit 1
            ;;
    esac

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_memory "after_${stage} (duration=${duration}s)"
}

# ==================== Main ====================

echo "LingBot-World 2.0 1.3B M4 13-frame Smoke Test"
echo "================================================"
echo "Repo: $REPO_ROOT"
echo "Device: $DEVICE"
echo "Frame num: $FRAME_NUM"
echo "Size: $SIZE"
echo "Chunk size: $CHUNK_SIZE"
echo "Seed: $SEED"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Initialize memory log
echo "M4 13-frame Smoke Test Memory Log" > "$MEMORY_LOG"
echo "=================================" >> "$MEMORY_LOG"
echo "Device: $DEVICE" >> "$MEMORY_LOG"
echo "Frame num: $FRAME_NUM" >> "$MEMORY_LOG"
echo "Size: $SIZE" >> "$MEMORY_LOG"
echo "" >> "$MEMORY_LOG"

log_memory "process_start"

# Determine which stage(s) to run
STAGE="${1:-all}"

case "$STAGE" in
    all)
        run_stage "encode-image"
        run_stage "generate-latents"
        run_stage "decode"
        ;;
    full|encode-image|generate-latents|decode)
        run_stage "$STAGE"
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: $0 [all|full|encode-image|generate-latents|decode]"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "M4 13-frame Smoke Test Complete"
echo "================================================"
echo "Memory log: $MEMORY_LOG"
if [ -f "$OUTPUT_VIDEO" ]; then
    echo "Output video: $OUTPUT_VIDEO"
fi
echo ""
echo "To view memory log:"
echo "  cat $MEMORY_LOG"
