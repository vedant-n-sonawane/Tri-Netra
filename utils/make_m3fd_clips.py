#!/usr/bin/env python3
"""
Build short RGB/thermal clips from runs of consecutive M3FD frames.

M3FD ships as still image pairs, but neighbouring file numbers often come from
the same scene (boxes overlap heavily between frame i and i+1). This groups
such runs ("bursts") and writes the longest ones as MP4 pairs under
datasets/RGB and datasets/thermal, named like the clips the tracking UI and
inference/video.py already expect (sequence_NN_1min_{RGB,Thermal}.mp4).

Usage:
    python utils/make_m3fd_clips.py --num 14 --min-len 10 --fps 10
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def _iou(p, q):
    x1, y1 = max(p[0], q[0]), max(p[1], q[1])
    x2, y2 = min(p[0] + p[2], q[0] + q[2]), min(p[1] + p[3], q[1] + q[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (p[2] * p[3] + q[2] * q[3] - inter + 1e-9)


def _matched_fraction(a, b, thr=0.5):
    if not a or not b:
        return 0.0
    return sum(max(_iou(p, q) for q in b) > thr for p in a) / len(a)


def find_bursts(coco: dict, link_thr: float = 0.5):
    boxes = defaultdict(list)
    for a in coco["annotations"]:
        boxes[a["image_id"]].append(a["bbox"])
    images = sorted(coco["images"], key=lambda i: i["file_name"])
    bursts, cur = [], [images[0]]
    for prev, nxt in zip(images, images[1:]):
        if _matched_fraction(boxes[prev["id"]], boxes[nxt["id"]]) >= link_thr:
            cur.append(nxt)
        else:
            bursts.append(cur)
            cur = [nxt]
    bursts.append(cur)
    return bursts, boxes


def out_stats(args):
    return Path(args.out) / "m3fd_scene_stats.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m3fd", default=str(ROOT / "datasets" / "M3FD"))
    ap.add_argument("--annotations", default=str(ROOT / "datasets" / "annotations" / "instances_all.json"))
    ap.add_argument("--out", default=str(ROOT / "datasets"))
    ap.add_argument("--num", type=int, default=14)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--night", type=int, default=0,
                    help="reserve this many slots for the darkest bursts (needs datasets/m3fd_scene_stats.csv "
                         "from utils/scan_m3fd_trc.py); the rest are the longest bursts, shown first as sequences")
    ap.add_argument("--night-rgb-mean", type=float, default=70.0)
    args = ap.parse_args()

    coco = json.load(open(args.annotations, encoding="utf-8"))
    bursts, _ = find_bursts(coco)
    lens = sorted(len(b) for b in bursts)
    print(f"{len(bursts)} bursts; longest {lens[-5:]}; >= {args.min_len}: "
          f"{sum(l >= args.min_len for l in lens)}")
    eligible = [b for b in bursts if len(b) >= args.min_len]
    chosen = []
    if args.night:
        import csv
        stats = {r["file_name"]: float(r["rgb_mean"])
                 for r in csv.DictReader(open(out_stats(args), encoding="utf-8"))}
        mean_rgb = lambda b: sum(stats[i["file_name"]] for i in b) / len(b)
        night = sorted([b for b in eligible if mean_rgb(b) < args.night_rgb_mean], key=len, reverse=True)
        chosen = sorted(night[: args.night], key=lambda b: b[0]["file_name"])
        print(f"night bursts available: {len(night)}; using {len(chosen)}")
        eligible = [b for b in eligible if b not in night]
    rest = sorted(eligible, key=len, reverse=True)[: args.num - len(chosen)]
    chosen += sorted(rest, key=lambda b: b[0]["file_name"])

    m3fd, out = Path(args.m3fd), Path(args.out)
    (out / "RGB").mkdir(parents=True, exist_ok=True)
    (out / "thermal").mkdir(parents=True, exist_ok=True)
    index = {}
    for n, burst in enumerate(chosen, 1):
        w, h = burst[0]["width"], burst[0]["height"]
        vw = cv2.VideoWriter(str(out / "RGB" / f"sequence_{n:02d}_1min_RGB.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        tw = cv2.VideoWriter(str(out / "thermal" / f"sequence_{n:02d}_1min_Thermal.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for im in burst:
            for writer, sub in ((vw, "Vis"), (tw, "Ir")):
                img = cv2.imread(str(m3fd / sub / im["file_name"]), cv2.IMREAD_COLOR)
                if img.shape[1] != w or img.shape[0] != h:
                    img = cv2.resize(img, (w, h))
                writer.write(img)
        vw.release(); tw.release()
        index[n] = {"first": burst[0]["file_name"], "last": burst[-1]["file_name"], "frames": len(burst)}
        print(f"sequence_{n:02d}: {burst[0]['file_name']}..{burst[-1]['file_name']} ({len(burst)} frames)")
    (out / "m3fd_clips.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
