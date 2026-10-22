#!/usr/bin/env python3
"""
Build a test video from M3FD: many short bursts of consecutive frames, night and
day interleaved, at a watchable frame rate.

Writes to datasets/test_video/:
    m3fd_test_RGB.mp4, m3fd_test_Thermal.mp4   input pair for inference/video.py and the tracking UI
    m3fd_test_preview.mp4                      RGB | thermal side by side, ground-truth boxes and the
                                               Phase 7 threat level (needs runs/threats/m3fd_threat_scores.csv)
    m3fd_test_index.json                       frame range -> source files for every burst

Bursts are separated by hard scene cuts (the source is stills), so trackers should
be expected to reset at each cut.

Usage:
    python utils/make_test_video.py --night 15 --day 10 --fps 10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.make_m3fd_clips import find_bursts  # noqa: E402

COLORS = {1: (60, 60, 255), 2: (255, 160, 40), 3: (255, 90, 200), 4: (60, 220, 255), 5: (60, 255, 120), 6: (255, 255, 60), 7: (200, 200, 200)}
NAMES = {1: "person", 2: "car", 3: "bus", 4: "truck", 5: "motorcycle", 6: "cyclist", 7: "lamp"}
LEVEL_COL = {1: (80, 160, 80), 2: (60, 190, 200), 3: (40, 160, 240), 4: (30, 100, 240), 5: (40, 40, 220)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--night", type=int, default=15)
    ap.add_argument("--day", type=int, default=10)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--night-rgb-mean", type=float, default=70.0)
    args = ap.parse_args()

    coco = json.load(open(ROOT / "datasets/annotations/instances_all.json", encoding="utf-8"))
    stats = pd.read_csv(ROOT / "datasets/m3fd_scene_stats.csv").set_index("file_name")
    scores = pd.read_csv(ROOT / "runs/threats/m3fd_threat_scores.csv").set_index("file_name")
    boxes = defaultdict(list)
    name_of = {i["id"]: i["file_name"] for i in coco["images"]}
    for a in coco["annotations"]:
        boxes[name_of[a["image_id"]]].append((a["bbox"], a["category_id"]))

    bursts, _ = find_bursts(coco)
    bursts = [b[: args.max_len] for b in bursts if len(b) >= args.min_len]
    mean = lambda b, col: float(stats.loc[[i["file_name"] for i in b], col].mean())
    peak = lambda b: float(scores.loc[[i["file_name"] for i in b], "points"].mean())
    night = sorted([b for b in bursts if mean(b, "rgb_mean") < args.night_rgb_mean], key=peak, reverse=True)[: args.night]
    day = sorted([b for b in bursts if mean(b, "rgb_mean") >= args.night_rgb_mean], key=peak, reverse=True)[: args.day]
    order = []
    for k in range(max(len(night), len(day))):          # interleave so the demo alternates night / day
        if k < len(night):
            order.append(("night", night[k]))
        if k < len(day):
            order.append(("day", day[k]))
    print(f"{len(night)} night + {len(day)} day bursts, {sum(len(b) for _, b in order)} frames, "
          f"{sum(len(b) for _, b in order) / args.fps:.0f}s at {args.fps:g} fps")

    out = ROOT / "datasets" / "test_video"
    out.mkdir(parents=True, exist_ok=True)
    W, H = 1024, 768
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = {k: cv2.VideoWriter(str(out / f"m3fd_test_{k}.mp4"), fourcc, args.fps, (W, H)) for k in ("RGB", "Thermal")}
    pv = cv2.VideoWriter(str(out / "m3fd_test_preview.mp4"), fourcc, args.fps, (W, H // 2 + 56))
    index, frame_no = [], 0

    for kind, burst in order:
        first = frame_no
        for im in burst:
            fn = im["file_name"]
            vis = cv2.imread(str(ROOT / "datasets/M3FD/Vis" / fn))
            thr = cv2.imread(str(ROOT / "datasets/M3FD/Ir" / fn))
            vis, thr = cv2.resize(vis, (W, H)), cv2.resize(thr, (W, H))
            wr["RGB"].write(vis)
            wr["Thermal"].write(thr)

            panels = []
            for img in (vis, thr):
                p = img.copy()
                for (x, y, w, h), c in boxes[fn]:
                    cv2.rectangle(p, (int(x), int(y)), (int(x + w), int(y + h)), COLORS.get(c, (255, 255, 255)), 2)
                panels.append(cv2.resize(p, (W // 2, H // 2)))
            canvas = cv2.vconcat([cv2.hconcat(panels), 255 * __import__("numpy").ones((56, W, 3), dtype="uint8") * 0])
            sc = scores.loc[fn]
            cv2.rectangle(canvas, (0, H // 2), (W, H // 2 + 56), (25, 25, 25), -1)
            cv2.rectangle(canvas, (0, H // 2), (170, H // 2 + 56), LEVEL_COL[int(sc.level)], -1)
            cv2.putText(canvas, f"THREAT {int(sc.level)}/5", (10, H // 2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, f"{kind.upper()}  {fn}  RGB {stats.loc[fn, 'rgb_mean']:.0f}  TRC {stats.loc[fn, 'trc']:.2f}",
                        (180, H // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
            cv2.putText(canvas, str(sc.reasons)[:120], (180, H // 2 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(canvas, "RGB", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas, "THERMAL", (W // 2 + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            pv.write(canvas)
            frame_no += 1
        index.append({"kind": kind, "frames": [first, frame_no - 1], "first_file": burst[0]["file_name"],
                      "last_file": burst[-1]["file_name"], "mean_rgb": round(mean(burst, "rgb_mean"), 1)})
    for w in (*wr.values(), pv):
        w.release()
    (out / "m3fd_test_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("wrote", out, "total frames", frame_no)


if __name__ == "__main__":
    main()
