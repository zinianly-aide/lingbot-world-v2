#!/usr/bin/env python3
"""Extract frame 1/7/13 from blind eval videos and build compare sheets."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
E2_BLIND = REPO / "eval/e2_new/blind"
G07_BLIND = REPO / "eval/g0.7/blind"
OUT_E2 = REPO / "eval/e2_new/blind/frames_xy"
OUT_G07 = REPO / "eval/g0.7/blind/frames_xy"
FRAMES = [1, 7, 13]


def extract_frames(video: Path, out_dir: Path) -> dict[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for f in FRAMES:
        out = out_dir / f"f{f:02d}.png"
        # 1-indexed frame select via select filter
        cmd = [
            "ffmpeg", "-y", "-i", str(video),
            "-vf", f"select=eq(n\\,{f - 1})",
            "-vsync", "0",
            "-frames:v", "1",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError(f"failed extract {video} frame {f}")
        paths[f] = out
    return paths


def hstack(images: list[Image.Image], pad: int = 8, bg=(24, 24, 24)) -> Image.Image:
    if not images:
        raise ValueError("no images")
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    x = 0
    for im in images:
        canvas.paste(im, (x, (h - im.height) // 2))
        x += im.width + pad
    return canvas


def vstack(images: list[Image.Image], pad: int = 10, bg=(24, 24, 24)) -> Image.Image:
    if not images:
        raise ValueError("no images")
    w = max(im.width for im in images)
    h = sum(im.height for im in images) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    y = 0
    for im in images:
        canvas.paste(im, ((w - im.width) // 2, y))
        y += im.height + pad
    return canvas


def label_row(im: Image.Image, text: str, height: int = 28) -> Image.Image:
    bar = Image.new("RGB", (im.width, height), (18, 18, 18))
    draw = ImageDraw.Draw(bar)
    draw.text((10, 6), text, fill=(230, 230, 230))
    canvas = Image.new("RGB", (im.width, im.height + height), (18, 18, 18))
    canvas.paste(bar, (0, 0))
    canvas.paste(im, (0, height))
    return canvas


def main():
    # --- E2 pairs ---
    bmap = json.loads((E2_BLIND / "blind_map.json").read_text())
    pairs = bmap["pairs"]
    OUT_E2.mkdir(parents=True, exist_ok=True)
    for p in pairs:
        pid = p["pair_id"]
        left = E2_BLIND / "videos" / f"{p['left']}.mp4"
        right = E2_BLIND / "videos" / f"{p['right']}.mp4"
        ldir = OUT_E2 / pid / "X"
        rdir = OUT_E2 / pid / "Y"
        lframes = extract_frames(left, ldir)
        rframes = extract_frames(right, rdir)
        # contact sheet: row X frames1-7-13, row Y frames1-7-13
        xrow = hstack([Image.open(lframes[f]) for f in FRAMES])
        yrow = hstack([Image.open(rframes[f]) for f in FRAMES])
        sheet = vstack([label_row(xrow, f"{pid}  video_X={p['left']}  frames 1/7/13"),
                        label_row(yrow, f"{pid}  video_Y={p['right']}  frames 1/7/13")])
        sheet_path = OUT_E2 / f"{pid}_sheet.png"
        sheet.save(sheet_path, optimize=True)
        print(f"E2 {pid}: {sheet_path}")

    # --- G0.7 single videos ---
    gmap = json.loads((REPO / "eval/g0.7/blind_map.json").read_text())["mapping"]
    scene_inputs = {
        "camera_motion": REPO / "examples/00/image.jpg",
        "indoor": REPO / "eval/g0.7/eval_assets/indoor/image.jpg",
        "outdoor": REPO / "examples/04/image.jpg",
        "single_subject": REPO / "examples/03/image.jpg",
        "spatial": REPO / "examples/01/image.jpg",
    }
    prompts = {
        scene: (REPO / f"eval/g0.7/{scene}/prompt_A.txt").read_text().strip()
        for scene in scene_inputs
    }
    OUT_G07.mkdir(parents=True, exist_ok=True)
    for bid in sorted(gmap):
        info = gmap[bid]
        video = G07_BLIND / "videos" / f"{bid}.mp4"
        vdir = OUT_G07 / bid
        frames = extract_frames(video, vdir)
        scene = info["scene_id"]
        inp = Image.open(scene_inputs[scene])
        # scale input to video frame height
        v0 = Image.open(frames[1])
        if inp.height != v0.height:
            nw = int(inp.width * v0.height / inp.height)
            inp = inp.resize((nw, v0.height), Image.Resampling.LANCZOS)
        row = hstack([inp] + [Image.open(frames[f]) for f in FRAMES])
        sheet = label_row(row, f"{bid}  scene={scene}  prompt: {prompts[scene][:80]}")
        sheet_path = OUT_G07 / f"{bid}_sheet.png"
        sheet.save(sheet_path, optimize=True)
        print(f"G07 {bid}: {sheet_path}")

    # save prompt dump for reference
    (REPO / "eval/g0.7/blind/frames_xy/prompts.json").write_text(json.dumps(prompts, indent=2))
    print("done")


if __name__ == "__main__":
    main()
