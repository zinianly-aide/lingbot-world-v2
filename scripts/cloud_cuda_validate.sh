#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PYTHON="${PYTHON:-python}"
CKPT_DIR="${CKPT_DIR:-}"
ASSETS_DIR="${ASSETS_DIR:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/eval/cloud-5090/$STAMP}"
PROMPT="Move the camera slowly forward while keeping the lone tree stable and centered."
IMAGE="$REPO_ROOT/examples/03/image.jpg"
ACTION_PATH="$REPO_ROOT/examples/03"
PROMPT_EMBEDS="$REPO_ROOT/eval/e1.2/embeddings/single_subject_minicpm.safetensors"

mkdir -p "$OUT_ROOT"

require_gpu_assets() {
  if [[ -z "$CKPT_DIR" || -z "$ASSETS_DIR" ]]; then
    echo "Set CKPT_DIR and ASSETS_DIR before GPU validation." >&2
    exit 2
  fi
  [[ -d "$CKPT_DIR" ]] || { echo "Missing CKPT_DIR: $CKPT_DIR" >&2; exit 2; }
  [[ -d "$ASSETS_DIR" ]] || { echo "Missing ASSETS_DIR: $ASSETS_DIR" >&2; exit 2; }
  [[ -f "$PROMPT_EMBEDS" ]] || { echo "Missing prompt embedding: $PROMPT_EMBEDS" >&2; exit 2; }
}

start_gpu_log() {
  local label="$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits -lms 200 > "$OUT_ROOT/$label.gpu.csv" &
    GPU_MON_PID=$!
  else
    GPU_MON_PID=""
  fi
}

stop_gpu_log() {
  if [[ -n "${GPU_MON_PID:-}" ]]; then
    kill "$GPU_MON_PID" 2>/dev/null || true
    wait "$GPU_MON_PID" 2>/dev/null || true
  fi
  GPU_MON_PID=""
}

run_static() {
  echo "== static =="
  cd "$REPO_ROOT"
  "$PYTHON" -m py_compile \
    wan/image2video.py \
    scripts/q2_live_quest_poc.py \
    scripts/quest_frame_bridge.py \
    scripts/quest_stream_replay.py
  "$PYTHON" -m unittest -v \
    tests.test_m36_staged \
    tests.test_streaming_contract \
    tests.test_quest_streaming \
    tests.test_replay_loop
}

run_smoke13() {
  require_gpu_assets
  echo "== smoke13 full pipeline =="
  local out="$OUT_ROOT/smoke13"
  mkdir -p "$out"
  start_gpu_log "smoke13"
  set +e
  ( cd "$REPO_ROOT" && \
    "$PYTHON" generate.py \
      --task i2v-1.3B \
      --infer_mode causal_fast \
      --size "832*480" \
      --frame_num 13 \
      --chunk_size 4 \
      --ckpt_dir "$CKPT_DIR" \
      --assets_dir "$ASSETS_DIR" \
      --image "$IMAGE" \
      --action_path "$ACTION_PATH" \
      --prompt "$PROMPT" \
      --prompt_embeds_file "$PROMPT_EMBEDS" \
      --device cuda \
      --ulysses_size 1 \
      --offload_model false \
      --base_seed 42 \
      --save_file "$out/out.mp4" \
      --save_dir "$out" ) 2>&1 | tee "$out/run.log"
  local rc=${PIPESTATUS[0]}
  set -e
  stop_gpu_log
  [[ $rc -eq 0 ]] || return $rc
  [[ -s "$out/out.mp4" ]] || { echo "smoke13 did not produce out.mp4" >&2; return 3; }
  if command -v ffprobe >/dev/null 2>&1; then
    ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=width,height,r_frame_rate,duration,nb_read_frames \
      -of default=noprint_wrappers=1 "$out/out.mp4" | tee "$out/ffprobe.txt"
  fi
}

run_q2_exact() {
  require_gpu_assets
  echo "== q2 exact M4-parity profile =="
  local out="$OUT_ROOT/q2-exact"
  mkdir -p "$out"
  start_gpu_log "q2-exact"
  set +e
  ( cd "$REPO_ROOT" && \
    "$PYTHON" scripts/q2_live_quest_poc.py \
      --device cuda \
      --vae-dtype bf16 \
      --ckpt-dir "$CKPT_DIR" \
      --assets-dir "$ASSETS_DIR" \
      --image "$IMAGE" \
      --action-path "$ACTION_PATH" \
      --prompt "$PROMPT" \
      --prompt-embeds "$PROMPT_EMBEDS" \
      --frame-num 33 \
      --chunk-size 2 \
      --max-area 258048 \
      --seed 123 \
      --bridge-host 127.0.0.1 \
      --bridge-port 8765 \
      --work-dir "$out/work" ) 2>&1 | tee "$out/run.log"
  local rc=${PIPESTATUS[0]}
  set -e
  stop_gpu_log
  return $rc
}

run_q2_480p() {
  require_gpu_assets
  echo "== q2 480x832 target profile =="
  local out="$OUT_ROOT/q2-480p"
  mkdir -p "$out"
  start_gpu_log "q2-480p"
  set +e
  ( cd "$REPO_ROOT" && \
    "$PYTHON" scripts/q2_live_quest_poc.py \
      --device cuda \
      --vae-dtype bf16 \
      --ckpt-dir "$CKPT_DIR" \
      --assets-dir "$ASSETS_DIR" \
      --image "$IMAGE" \
      --action-path "$ACTION_PATH" \
      --prompt "$PROMPT" \
      --prompt-embeds "$PROMPT_EMBEDS" \
      --frame-num 33 \
      --chunk-size 4 \
      --max-area 399360 \
      --seed 42 \
      --bridge-host 127.0.0.1 \
      --bridge-port 8765 \
      --work-dir "$out/work" ) 2>&1 | tee "$out/run.log"
  local rc=${PIPESTATUS[0]}
  set -e
  stop_gpu_log
  return $rc
}

case "$MODE" in
  static) run_static ;;
  smoke13) run_smoke13 ;;
  q2-exact) run_q2_exact ;;
  q2-480p) run_q2_480p ;;
  all)
    run_static
    run_smoke13
    run_q2_exact
    run_q2_480p
    ;;
  *)
    echo "Usage: $0 {static|smoke13|q2-exact|q2-480p|all}" >&2
    exit 2
    ;;
esac

echo "PASS: $MODE"
echo "Artifacts: $OUT_ROOT"
