#!/bin/bash
# Generate G0.7 A/B/C side-by-side montage videos and frame comparison grids
set -e

REPO=/Users/anshi/clawd/lingbot-world-v2
G07=$REPO/eval/g0.7
OUT_MONTAGE=$G07/blind/montage
OUT_FRAMES=$G07/blind/frames

mkdir -p "$OUT_MONTAGE" "$OUT_FRAMES"

# Scene mapping (anonymous)
SCENES="single_subject:scene_01 spatial:scene_02 indoor:scene_03 outdoor:scene_04 camera_motion:scene_05"
SEEDS="42 123 2026"
FRAMES="1 7 13"

count=0
for mapping in $SCENES; do
    scene="${mapping%%:*}"
    anon="${mapping##*:}"
    VDIR=$G07/$scene/videos

    for seed in $SEEDS; do
        A=$VDIR/A_seed${seed}.mp4
        B=$VDIR/B_seed${seed}.mp4
        C=$VDIR/C_seed${seed}.mp4

        if [ ! -f "$A" ] || [ ! -f "$B" ] || [ ! -f "$C" ]; then
            echo "SKIP $scene seed=$seed (missing video)"
            continue
        fi

        # --- Montage video: A | B | C with labels ---
        MONTAGE_OUT=$OUT_MONTAGE/${anon}_seed${seed}.mp4
        ffmpeg -y -i "$A" -i "$B" -i "$C" \
            -filter_complex "\
[0:v]scale=416:232,drawtext=text='A':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[a];\
[1:v]scale=416:232,drawtext=text='B':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[b];\
[2:v]scale=416:232,drawtext=text='C':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[c];\
[a][b][c]hstack=inputs=3[out]" \
            -map "[out]" -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
            "$MONTAGE_OUT" 2>/dev/null
        echo "MONTAGE: $anon seed=$seed -> $(basename $MONTAGE_OUT)"

        # --- Frame comparison grids: 3x3 (rows A/B/C, cols frame1/7/13) ---
        for f in $FRAMES; do
            GRID_OUT=$OUT_FRAMES/${anon}_seed${seed}_frame${f}.png
            # frame index is 0-based in ffmpeg, so frame 1 = n=0, frame 7 = n=6, frame 13 = n=12
            fidx=$((f - 1))
            ffmpeg -y -i "$A" -i "$B" -i "$C" \
                -filter_complex "\
[0:v]select=eq(n\\,${fidx}),scale=277:155[a];\
[1:v]select=eq(n\\,${fidx}),scale=277:155[b];\
[2:v]select=eq(n\\,${fidx}),scale=277:155[c];\
[a][b][c]vstack=inputs=3[col];\
[col]drawtext=text='A':x=5:y=5:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5;\
drawtext=text='B':x=5:y=160:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5;\
drawtext=text='C':x=5:y=315:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5[out]" \
                -map "[out]" -frames:v 1 \
                "$GRID_OUT" 2>/dev/null
            echo "  GRID: frame $f -> $(basename $GRID_OUT)"
        done

        count=$((count + 1))
    done
done

echo ""
echo "=== Done: $count montage videos + frame grids ==="
echo "Montages: $OUT_MONTAGE/"
echo "Frames: $OUT_FRAMES/"
