#!/usr/bin/env python3
"""
Phase 7 demo: score M3FD frames with models/risk_model/threat_scorer.py.

Uses the ground-truth boxes by default (M3FD has no threat labels, so this
shows what the rules do, not detector accuracy). Writes
runs/threats/m3fd_threat_scores.csv and prints the level distribution and
examples.

Usage:
    python utils/score_threats.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.risk_model.threat_scorer import find_thermal_only, score_frame  # noqa: E402


def main():
    coco = json.load(open(ROOT / "datasets/annotations/instances_all.json", encoding="utf-8"))
    stats = pd.read_csv(ROOT / "datasets/m3fd_scene_stats.csv").set_index("file_name")
    by = defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by[a["image_id"]].append((x, y, x + w, y + h, a["category_id"], 1.0))
    out = ROOT / "runs/threats"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for im in sorted(coco["images"], key=lambda i: i["file_name"]):
        fn, dets = im["file_name"], by[im["id"]]
        s = stats.loc[fn]
        hidden = []
        if s.rgb_mean < 70 and any(int(d[4]) == 1 for d in dets):
            v = cv2.imread(str(ROOT / "datasets/M3FD/Vis" / fn), 0).astype(float)
            t = cv2.imread(str(ROOT / "datasets/M3FD/Ir" / fn), 0).astype(float)
            hidden = find_thermal_only(dets, v, t)
        r = score_frame(dets, rgb_mean=s.rgb_mean, trc=s.trc, thermal_only=hidden)
        rows.append({"file_name": fn, "level": r["level"], "points": r["points"], "night": r["factors"]["night"],
                     "persons": r["factors"]["persons"], "vehicles": r["factors"]["vehicles"],
                     "thermal_only": r["factors"]["thermal_only"], "reasons": "; ".join(r["reasons"])})
    df = pd.DataFrame(rows)
    df.to_csv(out / "m3fd_threat_scores.csv", index=False)
    print("level distribution (all):  ", dict(sorted(Counter(df.level).items())))
    print("level distribution (night):", dict(sorted(Counter(df[df.night].level).items())))
    print("level distribution (day):  ", dict(sorted(Counter(df[~df.night].level).items())))
    print("\ntop examples:")
    for _, r in df.sort_values(["level", "points"], ascending=False).head(6).iterrows():
        print(f"  {r.file_name} L{r.level}: {r.reasons}")


if __name__ == "__main__":
    main()
