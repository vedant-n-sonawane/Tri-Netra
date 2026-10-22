#!/usr/bin/env python3
"""
Score every M3FD pair with the Phase 3 TRC pipeline plus RGB brightness stats,
writing datasets/m3fd_scene_stats.csv. Used to pick demo scenes where fusion
matters (low thermal reliability, or dark / unreliable RGB).

Usage:
    python utils/scan_m3fd_trc.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.pipeline import build_preprocess_pipeline  # noqa: E402


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml", encoding="utf-8"))
    pre = build_preprocess_pipeline(cfg)
    m3fd = ROOT / "datasets" / "M3FD"
    names = sorted(p.name for p in (m3fd / "Vis").glob("*.png"))
    out = ROOT / "datasets" / "m3fd_scene_stats.csv"
    cols = ["file_name", "trc", "rgb_mean", "rgb_p95", "rgb_dark_frac", "thermal_mean"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, n in enumerate(names):
            vis = cv2.cvtColor(cv2.imread(str(m3fd / "Vis" / n)), cv2.COLOR_BGR2RGB)
            thr_bgr = cv2.imread(str(m3fd / "Ir" / n))
            thr = cv2.cvtColor(thr_bgr, cv2.COLOR_BGR2GRAY)[:, :, None]
            trc = float(pre(vis, thr)[2].get("trc", 1.0)) if pre is not None else 1.0
            g = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY)
            w.writerow([n, f"{trc:.4f}", f"{g.mean():.2f}", f"{np.percentile(g, 95):.1f}",
                        f"{(g < 40).mean():.3f}", f"{thr.mean():.2f}"])
            if i % 500 == 0:
                print(f"{i}/{len(names)}", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
