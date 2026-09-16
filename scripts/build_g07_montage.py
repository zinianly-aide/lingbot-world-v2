#!/usr/bin/env python3
"""Generate G0.7 A/B/C montage videos and frame grids using ffmpeg pipe + Pillow."""
import subprocess
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = "/Users/anshi/clawd/lingbot-world-v2"
G07 = os.path.join(REPO, "eval/g0.7")
OUT_MONTAGE = os.path.join(G07, "blind/montage")
OUT_FRAMES = os.path.join(G07, "blind/frames")

SCENES = [
    ("single_subject", "scene_01"),
    ("spatial", "scene_02"),
    ("indoor", "scene_03"),
    ("outdoor", "scene_04"),
    ("camera_motion", "scene_05"),
]
SEEDS = [42, 123, 2026]
FRAMES = [1, 7, 13]  # 1-based

# Scale each panel to WxH, montage = 3W x H, grid = W x 3H
PANEL_W = 416
PANEL_H = 232
FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"

os.makedirs(OUT_MONTAGE, exist_ok=True)
os.makedirs(OUT_FRAMES, exist_ok=True)


def read_frames(video_path, target_w, target_h):
    """Read all frames from video as numpy array [N,H,W,3] via ffmpeg pipe."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"scale={target_w}:{target_h}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-loglevel", "error",
        "pipe:1"
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg read failed: {proc.stderr[-200:]}")
    raw = proc.stdout
    frame_size = target_w * target_h * 3
    n_frames = len(raw) // frame_size
    frames = np.frombuffer(raw[:n_frames * frame_size], dtype=np.uint8)
    frames = frames.reshape(n_frames, target_h, target_w, 3)
    return frames


def add_label(img_array, text):
    """Add label text to top-left of an RGB numpy image."""
    pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(FONT_PATH, 22)
    except Exception:
        font = ImageFont.load_default()
    # Background box
    bbox = draw.textbbox((4, 4), text, font=font)
    draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill=(0, 0, 0, 180))
    draw.text((6, 6), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


def write_video(frames, out_path, fps=13):
    """Write [N,H,W,3] uint8 frames to mp4 via ffmpeg pipe."""
    n, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        out_path
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg write failed: {proc.stderr[-200:]}")


def make_montage(a_path, b_path, c_path, out_path):
    """Create A|B|C side-by-side montage video with labels."""
    fa = read_frames(a_path, PANEL_W, PANEL_H)
    fb = read_frames(b_path, PANEL_W, PANEL_H)
    fc = read_frames(c_path, PANEL_W, PANEL_H)
    n = min(len(fa), len(fb), len(fc))

    stacked = []
    for i in range(n):
        a = add_label(fa[i], "A")
        b = add_label(fb[i], "B")
        c = add_label(fc[i], "C")
        stacked.append(np.hstack([a, b, c]))
    stacked = np.array(stacked, dtype=np.uint8)
    write_video(stacked, out_path)
    return True


def make_frame_grid(a_path, b_path, c_path, frame_idx, out_path):
    """Create 3-row (A/B/C) grid for a specific frame (0-based)."""
    fa = read_frames(a_path, PANEL_W, PANEL_H)
    fb = read_frames(b_path, PANEL_W, PANEL_H)
    fc = read_frames(c_path, PANEL_W, PANEL_H)
    idx = min(frame_idx, len(fa) - 1, len(fb) - 1, len(fc) - 1)

    a = add_label(fa[idx], "A")
    b = add_label(fb[idx], "B")
    c = add_label(fc[idx], "C")
    grid = np.vstack([a, b, c])
    Image.fromarray(grid).save(out_path)
    return True


def main():
    count_m = 0
    count_f = 0
    for scene, anon in SCENES:
        vdir = os.path.join(G07, scene, "videos")
        for seed in SEEDS:
            a = os.path.join(vdir, f"A_seed{seed}.mp4")
            b = os.path.join(vdir, f"B_seed{seed}.mp4")
            c = os.path.join(vdir, f"C_seed{seed}.mp4")
            if not all(os.path.exists(p) for p in (a, b, c)):
                print(f"SKIP {scene} seed={seed}")
                continue

            try:
                montage_out = os.path.join(OUT_MONTAGE, f"{anon}_seed{seed}.mp4")
                make_montage(a, b, c, montage_out)
                count_m += 1
                print(f"OK montage: {anon} seed={seed}")
            except Exception as e:
                print(f"FAIL montage {anon} seed={seed}: {e}")

            for f in FRAMES:
                try:
                    grid_out = os.path.join(OUT_FRAMES, f"{anon}_seed{seed}_frame{f}.png")
                    make_frame_grid(a, b, c, f - 1, grid_out)
                    count_f += 1
                except Exception as e:
                    print(f"FAIL grid {anon} seed={seed} frame={f}: {e}")

    print(f"\n=== Done: {count_m} montages, {count_f} frame grids ===")


if __name__ == "__main__":
    main()
