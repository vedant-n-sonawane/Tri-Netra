#!/usr/bin/env python3
"""
Side-by-side demo video from runs/fusion_vs_rgb/results.json (run
utils/compare_fusion_vs_rgb.py first): the dull night RGB frame with RGB-only
detections (left) versus the same frame with fusion detections (right), plus the
thermal image underneath and a running person count against ground truth.
Only held-out M3FD test images are used.

Usage:
    python utils/make_fusion_demo_video.py --fps 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COL = {1: (60, 60, 255), 2: (255, 160, 40), 3: (255, 90, 200), 4: (60, 220, 255), 5: (60, 255, 120), 7: (200, 200, 200)}
NAME = {1: "person", 2: "car", 3: "bus", 4: "truck", 5: "motorcycle", 7: "lamp"}


def draw(img, dets):
    out = img.copy()
    for x1, y1, x2, y2, conf, cls in dets:
        c = COL.get(int(cls), (255, 255, 255))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.putText(out, f"{NAME.get(int(cls), '?')} {conf:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return out


def label(img, text, sub, color):
    cv2.rectangle(img, (0, 0), (img.shape[1], 54), (20, 20, 20), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(img, sub, (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "runs/fusion_vs_rgb/results.json"))
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--out", default=str(ROOT / "runs/fusion_vs_rgb/night_rgb_only_vs_fusion.mp4"))
    args = ap.parse_args()

    R = json.load(open(args.results))
    gt = json.load(open(ROOT / "datasets/annotations/test.json"))
    gt_people = {}
    for a in gt["annotations"]:
        if a["category_id"] == 1:
            gt_people[a["image_id"]] = gt_people.get(a["image_id"], 0) + 1
    # most people first: that is where RGB-only misses matter
    ids = sorted(R["night_ids"], key=lambda i: -gt_people.get(i, 0))
    W, H = 640, 480
    wr = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * W, H + H // 2))
    for i in ids:
        fn = R["files"][str(i)]
        vis = cv2.imread(str(ROOT / "datasets/M3FD/Vis" / fn))
        thr = cv2.imread(str(ROOT / "datasets/M3FD/Ir" / fn))
        sx, sy = W / vis.shape[1], H / vis.shape[0]
        scale = lambda d: [[r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy, r[4], r[5]] for r in d]
        v, t = cv2.resize(vis, (W, H)), cv2.resize(thr, (W, H))
        n_gt = gt_people.get(i, 0)
        n_rgb = sum(int(r[5]) == 1 for r in R["dets"]["rgb"][str(i)])
        n_fus = sum(int(r[5]) == 1 for r in R["dets"]["fusion"][str(i)])
        left = label(draw(v, scale(R["dets"]["rgb"][str(i)])), "RGB ONLY", f"people found {n_rgb} / {n_gt} real", (80, 80, 255))
        right = label(draw(v, scale(R["dets"]["fusion"][str(i)])), "FUSED (RGB + thermal, TRC-gated)",
                      f"people found {n_fus} / {n_gt} real", (80, 230, 80))
        thermal = cv2.resize(t, (2 * W, H // 2))
        cv2.putText(thermal, f"thermal input   {fn}   TRC {R['trc'][str(i)]:.2f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        wr.write(cv2.vconcat([cv2.hconcat([left, right]), thermal]))
    wr.release()
    print("wrote", args.out, len(ids), "frames")


if __name__ == "__main__":
    main()
