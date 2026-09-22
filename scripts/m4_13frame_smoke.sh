#!/bin/bash
# M4 13-frame smoke/equivalence harness for LingBot-World 2.0 1.3B on Apple Silicon.
#
# Usage:
#   bash scripts/m4_13frame_smoke.sh prepare-prompt
#   bash scripts/m4_13frame_smoke.sh all
#   bash scripts/m4_13frame_smoke.sh equivalence
#   bash scripts/m4_13frame_smoke.sh [encode-image|generate-latents|decode|full]
#
# ``equivalence`` encodes the image condition once, then feeds that exact cache
# to both staged and full DiT paths before comparing generated latents bitwise.
# This removes MPS VAE encode run-to-run nondeterminism from the comparison.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -d "venv" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

CHECKPOINT_DIR_RAW="${CHECKPOINT_DIR:-/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-1.3b-causal-fast/snapshots/*}"
ASSETS_DIR_RAW="${ASSETS_DIR:-/Volumes/ssd/huggingface/hub/models--robbyant--lingbot-world-v2-14b-causal-fast/snapshots/*}"
ACTION_PATH="${ACTION_PATH:-examples/00}"
IMAGE_PATH="${IMAGE_PATH:-examples/00/image.jpg}"
PROMPT="${PROMPT:-Keep the main subject and move the camera forward}"

OUTPUT_DIR="${OUTPUT_DIR:-m4-smoke-13frame}"
mkdir -p "$OUTPUT_DIR"

PROMPT_EMBEDS_FILE="${PROMPT_EMBEDS_FILE:-$OUTPUT_DIR/prompt_embeds.safetensors}"
IMAGE_CONDITION_FILE="${IMAGE_CONDITION_FILE:-$OUTPUT_DIR/image_condition.safetensors}"
STAGED_LATENTS_FILE="${STAGED_LATENTS_FILE:-$OUTPUT_DIR/staged_latents.safetensors}"
FULL_LATENTS_FILE="${FULL_LATENTS_FILE:-$OUTPUT_DIR/full_latents.safetensors}"
STAGED_OUTPUT_VIDEO="${STAGED_OUTPUT_VIDEO:-$OUTPUT_DIR/staged.mp4}"
FULL_OUTPUT_VIDEO="${FULL_OUTPUT_VIDEO:-$OUTPUT_DIR/full.mp4}"

DEVICE="${DEVICE:-mps}"
FRAME_NUM="${FRAME_NUM:-13}"
SIZE="${SIZE:-832*480}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
SEED="${SEED:-42}"
INFER_MODE="${INFER_MODE:-causal_fast}"
TASK="${TASK:-i2v-1.3B}"

resolve_glob_dir() {
    local value="$1"
    if [[ "$value" == *"*"* || "$value" == *"?"* || "$value" == *"["* ]]; then
        local resolved
        resolved="$(compgen -G "$value" | head -n 1 || true)"
        if [ -z "$resolved" ]; then
            echo "ERROR: no directory matched: $value" >&2
            return 1
        fi
        printf '%s\n' "$resolved"
    else
        printf '%s\n' "$value"
    fi
}

CHECKPOINT_DIR="$(resolve_glob_dir "$CHECKPOINT_DIR_RAW")"
ASSETS_DIR="$(resolve_glob_dir "$ASSETS_DIR_RAW")"

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required file not found: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [ ! -d "$1" ]; then
        echo "ERROR: required directory not found: $1" >&2
        exit 1
    fi
}

COMMON_ARGS=(
    --task "$TASK"
    --ckpt_dir "$CHECKPOINT_DIR"
    --assets_dir "$ASSETS_DIR"
    --device "$DEVICE"
    --infer_mode "$INFER_MODE"
    --image "$IMAGE_PATH"
    --prompt "$PROMPT"
    --action_path "$ACTION_PATH"
    --frame_num "$FRAME_NUM"
    --size "$SIZE"
    --chunk_size "$CHUNK_SIZE"
    --base_seed "$SEED"
    --save_dir "$OUTPUT_DIR"
)

run_generate() {
    local log_file="$1"
    shift
    python generate.py "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "$log_file"
}

prepare_prompt() {
    require_dir "$ASSETS_DIR"
    echo "Encoding prompt once: $PROMPT_EMBEDS_FILE"
    python scripts/encode_prompt.py \
        --prompt "$PROMPT" \
        --assets_dir "$ASSETS_DIR" \
        --output "$PROMPT_EMBEDS_FILE" \
        --device cpu \
        2>&1 | tee "$OUTPUT_DIR/prepare_prompt.log"
}

run_stage() {
    local stage="$1"
    echo ""
    echo "===== M4 stage: $stage ====="

    case "$stage" in
        encode-image)
            run_generate "$OUTPUT_DIR/stage_encode.log" \
                --stage encode-image \
                --dump_image_condition "$IMAGE_CONDITION_FILE"
            ;;

        generate-latents)
            require_file "$PROMPT_EMBEDS_FILE"
            require_file "$IMAGE_CONDITION_FILE"
            run_generate "$OUTPUT_DIR/stage_generate.log" \
                --stage generate-latents \
                --prompt_embeds_file "$PROMPT_EMBEDS_FILE" \
                --image_condition_file "$IMAGE_CONDITION_FILE" \
                --output_latents_file "$STAGED_LATENTS_FILE"
            ;;

        decode)
            require_file "$STAGED_LATENTS_FILE"
            run_generate "$OUTPUT_DIR/stage_decode.log" \
                --stage decode \
                --latents_file "$STAGED_LATENTS_FILE" \
                --save_file "$STAGED_OUTPUT_VIDEO"
            ;;

        full)
            require_file "$PROMPT_EMBEDS_FILE"
            run_generate "$OUTPUT_DIR/stage_full.log" \
                --stage full \
                --prompt_embeds_file "$PROMPT_EMBEDS_FILE" \
                --save_file "$FULL_OUTPUT_VIDEO"
            ;;

        full-cached-condition)
            require_file "$PROMPT_EMBEDS_FILE"
            require_file "$IMAGE_CONDITION_FILE"
            run_generate "$OUTPUT_DIR/stage_full_cached_condition.log" \
                --stage full \
                --prompt_embeds_file "$PROMPT_EMBEDS_FILE" \
                --image_condition_file "$IMAGE_CONDITION_FILE" \
                --output_latents_file "$FULL_LATENTS_FILE" \
                --save_file "$FULL_OUTPUT_VIDEO"
            ;;

        *)
            echo "ERROR: unknown stage: $stage" >&2
            exit 1
            ;;
    esac
}

require_dir "$CHECKPOINT_DIR"
require_dir "$ASSETS_DIR"
require_dir "$ACTION_PATH"
require_file "$IMAGE_PATH"
require_file "$ACTION_PATH/poses.npy"
require_file "$ACTION_PATH/intrinsics.npy"

MODE="${1:-all}"

echo "LingBot-World M4 13-frame validation"
echo "  task=$TASK device=$DEVICE frames=$FRAME_NUM size=$SIZE chunk=$CHUNK_SIZE seed=$SEED"
echo "  checkpoint=$CHECKPOINT_DIR"
echo "  assets=$ASSETS_DIR"
echo "  output=$OUTPUT_DIR"

case "$MODE" in
    prepare-prompt)
        prepare_prompt
        ;;
    all)
        require_file "$PROMPT_EMBEDS_FILE"
        run_stage encode-image
        run_stage generate-latents
        run_stage decode
        ;;
    equivalence)
        require_file "$PROMPT_EMBEDS_FILE"
        # Produce y once. Both DiT paths must consume this same cache.
        run_stage encode-image
        run_stage generate-latents
        run_stage decode
        run_stage full-cached-condition
        python scripts/compare_m4_latents.py \
            "$STAGED_LATENTS_FILE" \
            "$FULL_LATENTS_FILE"
        echo "M4 equivalence artifacts:"
        echo "  staged latents: $STAGED_LATENTS_FILE"
        echo "  full latents:   $FULL_LATENTS_FILE"
        echo "  staged video:   $STAGED_OUTPUT_VIDEO"
        echo "  full video:     $FULL_OUTPUT_VIDEO"
        ;;
    full|encode-image|generate-latents|decode)
        run_stage "$MODE"
        ;;
    *)
        echo "Usage: $0 [prepare-prompt|all|equivalence|full|encode-image|generate-latents|decode]" >&2
        exit 1
        ;;
esac
