#!/usr/bin/env python3
"""
TriNetra-AMRF — Unit tests for utils/mot_metrics.py
====================================================

Every expected value below is derived by hand from the metric definitions on a
tiny scenario, so a passing run shows the implementation matches the
mathematics (not merely that it runs).

Usage:
    python -m utils.check_mot_metrics
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mot_metrics import evaluate_mot  # noqa: E402

_ok = []


def check(name, cond, detail=""):
    _ok.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def near(a, b, tol=1e-6):
    return abs(a - b) <= tol


def box_track(tid, x0, y0, frames, dx=0.0, w=40.0, h=100.0):
    return {f: [tid, x0 + dx, y0, x0 + dx + w, y0 + h] for f in frames}


def merge(*tracks):
    out = {}
    for tr in tracks:
        for f, row in tr.items():
            out.setdefault(f, []).append(row)
    return {f: np.array(rows, dtype=float) for f, rows in out.items()}


F = range(100)
A_gt, B_gt = box_track(1, 100, 100, F), box_track(2, 500, 300, F)
GT = merge(A_gt, B_gt)


def main() -> int:
    print("=== check_mot_metrics ===")

    print("\n[1] perfect tracking")
    r = evaluate_mot(GT, merge(box_track(11, 100, 100, F), box_track(12, 500, 300, F)))
    check("MOTA = IDF1 = HOTA = 1, no errors", near(r["MOTA"], 1) and near(r["IDF1"], 1) and near(r["HOTA"], 1)
          and r["IDSW"] == 0 and r["FP"] == 0 and r["FN"] == 0, f"HOTA={r['HOTA']:.4f}")
    check("both trajectories mostly tracked, MOTP = 1", r["MT"] == 2 and near(r["MOTP"], 1))

    print("\n[2] 20 missed detections on one track (frames 40-59)")
    pred = merge(box_track(11, 100, 100, F), box_track(12, 500, 300, [f for f in F if not 40 <= f < 60]))
    r = evaluate_mot(GT, pred)
    check("FN = 20, MOTA = 1 - 20/200 = 0.9", r["FN"] == 20 and near(r["MOTA"], 0.9), f"MOTA={r['MOTA']:.4f}")
    check("one fragmentation event, no ID switch", r["Frag"] == 1 and r["IDSW"] == 0)
    check("IDF1 = 2*180/(2*180+0+20) = 0.9474", near(r["IDF1"], 360 / 380), f"IDF1={r['IDF1']:.4f}")
    # DetA = 180/200; AssA = (100*1 + 80*0.8)/180; HOTA = sqrt(DetA*AssA)
    ass = (100 * 1.0 + 80 * 0.8) / 180
    check("HOTA = sqrt(0.9 * 0.9111) = 0.9055", near(r["HOTA"], np.sqrt(0.9 * ass)) and near(r["DetA"], 0.9)
          and near(r["AssA"], ass), f"HOTA={r['HOTA']:.4f}")

    print("\n[3] two tracks swap identities at frame 50")
    a_swap = {f: [11 if f < 50 else 12, *A_gt[f][1:]] for f in F}
    b_swap = {f: [12 if f < 50 else 11, *B_gt[f][1:]] for f in F}
    r = evaluate_mot(GT, merge(a_swap, b_swap))
    check("IDSW = 2, MOTA = 1 - 2/200 = 0.99", r["IDSW"] == 2 and near(r["MOTA"], 0.99), f"IDSW={r['IDSW']}")
    check("IDF1 = 0.5 (identity metric punishes the swap far harder than MOTA)", near(r["IDF1"], 0.5))
    check("HOTA = sqrt(1 * 1/3) = 0.5774", near(r["HOTA"], np.sqrt(1 / 3)) and near(r["AssA"], 1 / 3),
          f"HOTA={r['HOTA']:.4f}, AssA={r['AssA']:.4f}")

    print("\n[4] 30 false-positive boxes")
    fp_track = box_track(99, 800, 600, range(30))
    r = evaluate_mot(GT, merge(box_track(11, 100, 100, F), box_track(12, 500, 300, F), fp_track))
    check("FP = 30, MOTA = 1 - 30/200 = 0.85", r["FP"] == 30 and near(r["MOTA"], 0.85))
    check("IDF1 = 400/430", near(r["IDF1"], 400 / 430), f"IDF1={r['IDF1']:.4f}")
    check("HOTA = sqrt(DetA=200/230, AssA=1)", near(r["HOTA"], np.sqrt(200 / 230)) and near(r["AssA"], 1))

    print("\n[5] one ground-truth track split into two IDs at frame 50 (no gap)")
    split = merge({f: [21 if f < 50 else 22, *A_gt[f][1:]] for f in F})
    r = evaluate_mot(merge(A_gt), split)
    check("IDSW = 1, MOTA = 0.99", r["IDSW"] == 1 and near(r["MOTA"], 0.99))
    check("IDF1 = 0.5 and HOTA = sqrt(0.5) = 0.7071", near(r["IDF1"], 0.5) and near(r["HOTA"], np.sqrt(0.5)))

    print("\n[6] boxes shifted to IoU = 0.72 with the ground truth")
    dx = 40 * (1 - 0.72) / (1 + 0.72)
    shifted = merge(box_track(11, 100, 100, F, dx=dx), box_track(12, 500, 300, F, dx=dx))
    r = evaluate_mot(GT, shifted)
    check("still fully matched at IoU 0.5: MOTA = 1, MOTP = 0.72",
          near(r["MOTA"], 1) and near(r["MOTP"], 0.72, 1e-3), f"MOTP={r['MOTP']:.4f}")
    check("HOTA = 14/19 (perfect for the 14 IoU thresholds <= 0.70, zero above)",
          near(r["HOTA"], 14 / 19, 1e-6), f"HOTA={r['HOTA']:.4f}")

    print("\n[7] empty predictions")
    r = evaluate_mot(GT, {})
    check("MOTA = 0 (all missed), IDF1 = 0, HOTA = 0, ML = 2",
          near(r["MOTA"], 0) and near(r["IDF1"], 0) and near(r["HOTA"], 0, 1e-6) and r["ML"] == 2)

    n = sum(_ok)
    print(f"\n{n}/{len(_ok)} checks passed")
    return 0 if all(_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
