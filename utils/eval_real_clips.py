#!/usr/bin/env python3
"""
TriNetra-AMRF — Real-Clip Tracking Evaluation (label-free)
==========================================================

Replays every tracker variant on the cached detections of each real clip
(``inference/video.py --save-detections``) and reports the label-free quality
metrics of ``utils/tracking_quality.py``, plus the clip's TRC statistics and the
behaviour flags the tracker raised. No GPU needed: detections are cached, so
every tracker sees exactly the same input.

Usage:
    python -m utils.eval_real_clips --dets runs/tracking_eval/clean
"""

from __future__ import annotations

import argparse
import json
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

from utils.eval_synthetic import TRACKERS, make_tracker, run_tracker  # noqa: E402
from utils.tracking_quality import load_cache, quality_metrics, replay_bytetrack  # noqa: E402

METRICS = ["n_tracks", "len_median", "len_mean", "short_ratio", "coverage", "frag_rate",
           "jump_rate", "recoveries", "mean_concurrent", "tracker_ms_per_frame"]


def evaluate_clip(npz: Path, high_thresh: float = 0.6, trackers=TRACKERS) -> dict:
    seq = load_cache(npz)
    trc = np.asarray(seq["trc_obs"], dtype=float)
    res = {"clip": npz.name.replace(".dets.npz", ""), "n_frames": len(trc),
           "trc": {"mean": float(trc.mean()), "min": float(trc.min()), "p5": float(np.percentile(trc, 5)),
                   "frac_below_0.6": float((trc < 0.6).mean())},
           "trackers": {}, "flags": {}}
    for name in trackers:
        if name.startswith("DeepSORT"):
            pred = run_tracker(name, seq)
            res["trackers"][name] = quality_metrics(pred, seq, high_thresh)
        else:
            r = replay_bytetrack(seq, make_tracker(name))
            res["trackers"][name] = quality_metrics(r["pred"], seq, high_thresh, r["recoveries"], r["ms_per_frame"])
            if name.endswith("(ours)"):
                counts = {}
                for flags in r["flags"].values():
                    for fl in flags:
                        counts[fl] = counts.get(fl, 0) + 1
                res["flags"] = {"tracks_with_flags": counts, "n_tracks": len(r["flags"])}
    return res


def aggregate(clips: list, trackers=TRACKERS) -> dict:
    out = {}
    for name in trackers:
        rows = [c["trackers"][name] for c in clips]
        agg = {}
        for m in METRICS:
            vals = [r[m] for r in rows if r.get(m) is not None]
            agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                      "sum": float(np.sum(vals))} if vals else None
        out[name] = agg
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default=str(PROJECT_ROOT / "runs" / "tracking_eval" / "clean"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dets_dir = Path(args.dets)
    files = sorted(dets_dir.glob("sequence_*.dets.npz"))
    if not files:
        raise SystemExit(f"no cached detections in {dets_dir}")

    clips = []
    for f in files:
        c = evaluate_clip(f)
        clips.append(c)
        o = c["trackers"]["ByteTrack + TRC gate (ours)"]
        print(f"{c['clip']}: {c['n_frames']} frames | TRC mean {c['trc']['mean']:.2f} min {c['trc']['min']:.2f} | "
              f"ours: {o['n_tracks']} tracks, median length {o['len_median']:.0f}, "
              f"coverage {o['coverage']:.2f}, frag {o['frag_rate']:.2f}", flush=True)
    agg = aggregate(clips)
    out = Path(args.out) if args.out else dets_dir.parent / "real_clips_results.json"
    out.write_text(json.dumps({"clips": clips, "aggregate": agg}, indent=1), encoding="utf-8")
    print("\nsaved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
