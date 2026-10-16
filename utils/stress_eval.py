#!/usr/bin/env python3
"""
TriNetra-AMRF — Thermal-Degradation Stress Test on Real Footage
===============================================================

For a real clip, the thermal stream is damaged inside a segment (blur, or
saturation — the corruptions used in the Phase 3 / Phase 5 stress tests) while
the RGB stream is untouched, and the detector is re-run. Two questions:

 1. COUPLING  Does the detector actually get worse when TRC falls? The TRC gate
              assumes low TRC <-> unreliable boxes. Measured here on real
              detections: for the confident detections of the clean run, how
              often is the same object still detected (recall @ IoU 0.5), how
              far do boxes drift (mean IoU), and how much do scores drop —
              binned by the frame's TRC.

 2. STABILITY How well do each tracker's identities survive the damage?
              Each tracker's tracks on the clean run are the reference; its
              tracks on the degraded run are scored against them (IDF1, HOTA,
              ID switches) inside the degraded segment. This measures
              self-consistency, NOT accuracy — the reference is a tracker
              output, not human annotation.

Files (produced by ``inference/video.py --save-detections``):
    <clean_dir>/sequence_NN.dets.npz              full clean clip
    <stress_dir>/sequence_NN_<kind>.dets.npz      frames START..END, damaged

Usage:
    python -m utils.stress_eval --clean runs/tracking_eval/clean --stress runs/tracking_eval/stress
"""

from __future__ import annotations

import argparse
import json
import re
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
from utils.mot_metrics import _iou, evaluate_mot  # noqa: E402
from utils.tracking_quality import load_cache, replay_bytetrack  # noqa: E402

DEG_START, DEG_END = 400, 700          # absolute frames damaged by the stress runs
TRC_BINS = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.75), (0.75, 1.01)]


def sub_sequence(seq: dict, frames) -> dict:
    keep = set(int(f) for f in frames)
    idx = [i for i, f in enumerate(seq["frames"]) if f in keep]
    out = {"frames": [seq["frames"][i] for i in idx], "dets": [seq["dets"][i] for i in idx],
          "trc_obs": np.asarray(seq["trc_obs"])[idx], "trc_true": np.asarray(seq["trc_true"])[idx],
          "meta": {"T": len(idx)}}
    if "appearance" in seq:
        out["appearance"] = [seq["appearance"][i] for i in idx]
    return out


def coupling_stats(clean: dict, stressed: dict, high: float = 0.6) -> dict:
    """Per-TRC-bin detector behaviour of the stressed run relative to the clean run."""
    clean_by = dict(zip(clean["frames"], clean["dets"]))
    rows = []   # (trc, n_ref, n_hit, sum_iou, sum_score_ratio, n_dets_stressed, n_dets_clean)
    for f, d_s, trc in zip(stressed["frames"], stressed["dets"], stressed["trc_obs"]):
        d_c = clean_by.get(f)
        if d_c is None:
            continue
        ref = d_c[d_c[:, 4] >= high]
        n_hit = 0; iou_sum = 0.0; ratio_sum = 0.0
        if len(ref) and len(d_s):
            iou = _iou(ref[:, :4], d_s[:, :4])
            j = iou.argmax(1)
            best = iou.max(1)
            ok = best >= 0.5
            n_hit = int(ok.sum())
            iou_sum = float(best[ok].sum())
            ratio_sum = float((d_s[j[ok], 4] / ref[ok, 4]).sum())
        rows.append((float(trc), len(ref), n_hit, iou_sum, ratio_sum, len(d_s), len(d_c)))
    a = np.array(rows, dtype=float).reshape(-1, 7)
    out = {}
    for lo, hi in TRC_BINS:
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.sum() == 0:
            continue
        ref_n, hit = a[m, 1].sum(), a[m, 2].sum()
        out[f"{lo:.2f}-{min(hi, 1.0):.2f}"] = {
            "frames": int(m.sum()),
            "recall_vs_clean": float(hit / ref_n) if ref_n else None,
            "mean_iou_of_matched": float(a[m, 3].sum() / hit) if hit else None,
            "score_ratio_of_matched": float(a[m, 4].sum() / hit) if hit else None,
            "dets_per_frame_ratio": float(a[m, 5].sum() / max(a[m, 6].sum(), 1)),
        }
    return out


def run_pred(name: str, seq: dict) -> dict:
    if name.startswith("DeepSORT"):
        return run_tracker(name, seq)
    return replay_bytetrack(seq, make_tracker(name))["pred"]


def stability(clean: dict, stressed: dict, deg: tuple, trackers=TRACKERS) -> dict:
    frames = stressed["frames"]
    clean_seg = sub_sequence(clean, frames)
    seg_frames = [f for f in frames if deg[0] <= f < deg[1]]
    out = {}
    for name in trackers:
        ref = run_pred(name, clean_seg)
        pred = run_pred(name, stressed)
        m = evaluate_mot(ref, pred, frames=seg_frames)
        out[name] = {k: m[k] for k in ("IDF1", "HOTA", "AssA", "DetA", "IDSW", "Frag", "FP", "FN", "MOTA")}
        out[name]["n_ref_tracks_in_segment"] = m["num_gt_tracks"]
    return out


def find_pairs(clean_dir: Path, stress_dir: Path):
    for p in sorted(stress_dir.glob("sequence_*_*.dets.npz")):
        m = re.match(r"sequence_(\d+)_(\w+)\.dets\.npz", p.name)
        c = clean_dir / f"sequence_{m.group(1)}.dets.npz"
        if m and c.exists():
            yield m.group(1), m.group(2), c, p


def paired(results, a="ByteTrack + TRC gate (ours)", b="ByteTrack", metrics=("IDF1", "HOTA", "IDSW", "AssA"),
           n_boot=10000):
    rng = np.random.default_rng(0)
    out = {}
    for m in metrics:
        d = np.array([r["stability"][a][m] - r["stability"][b][m] for r in results], dtype=float)
        boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        out[m] = {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                         float(np.percentile(boots, 97.5))],
                  "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": len(d)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(PROJECT_ROOT / "runs" / "tracking_eval" / "clean"))
    ap.add_argument("--stress", default=str(PROJECT_ROOT / "runs" / "tracking_eval" / "stress"))
    ap.add_argument("--deg-start", type=int, default=DEG_START)
    ap.add_argument("--deg-end", type=int, default=DEG_END)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    deg = (args.deg_start, args.deg_end)

    results = []
    for seq_id, kind, cpath, spath in find_pairs(Path(args.clean), Path(args.stress)):
        clean, stressed = load_cache(cpath), load_cache(spath)
        clean_seg = sub_sequence(clean, stressed["frames"])
        seg_mask = [deg[0] <= f < deg[1] for f in stressed["frames"]]
        seg_stressed = sub_sequence(stressed, [f for f, m in zip(stressed["frames"], seg_mask) if m])
        r = {"clip": seq_id, "kind": kind,
             "trc_clean_mean_in_segment": float(np.mean(sub_sequence(clean, seg_stressed["frames"])["trc_obs"])),
             "trc_stressed_mean_in_segment": float(np.mean(seg_stressed["trc_obs"])),
             "coupling": coupling_stats(clean_seg, seg_stressed),
             "stability": stability(clean, stressed, deg)}
        results.append(r)
        s = r["stability"]
        print(f"seq {seq_id} {kind:<9} TRC {r['trc_clean_mean_in_segment']:.2f} -> {r['trc_stressed_mean_in_segment']:.2f} | "
              f"IDF1 vs clean: " + "  ".join(f"{n.split(' ')[0] if 'ours' not in n else 'ours'} {s[n]['IDF1']:.3f}" for n in TRACKERS),
              flush=True)
    if not results:
        raise SystemExit("no clean/stress pairs found")

    summary = {n: {m: {"mean": float(np.mean([r["stability"][n][m] for r in results])),
                       "std": float(np.std([r["stability"][n][m] for r in results], ddof=1)) if len(results) > 1 else 0.0}
                   for m in ("IDF1", "HOTA", "AssA", "IDSW", "Frag")} for n in TRACKERS}
    payload = {"deg_frames": list(deg), "runs": results, "summary": summary,
               "paired_ours_vs_bytetrack": paired(results)}
    out = Path(args.out) if args.out else Path(args.stress).parent / "stress_results.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("\nsaved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
