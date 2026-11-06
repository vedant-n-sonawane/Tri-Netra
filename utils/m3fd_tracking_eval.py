#!/usr/bin/env python3
"""
TriNetra: tracking statistics on the M3FD clips
================================================

Reads the detections cached by ``utils/run_m3fd_tracking.py`` and replays the four
trackers on identical input. Everything is reported per clip and aggregated with
clip-level uncertainty (the clip is the unit of analysis):

  1. clean clips   label-free quality proxies per tracker, paired differences with
                   95% bootstrap CIs and Wilcoxon signed-rank tests
  2. stress test   thermal blurred / saturated in the middle third of every clip:
                   detector coupling to the TRC (per TRC bin and as a rank
                   correlation) and identity stability of each tracker against its
                   own clean-run tracks

No labels are used, so these are consistency and plausibility statistics, not accuracy.

    python -m utils.m3fd_tracking_eval
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.eval_real_clips import evaluate_clip                      # noqa: E402
from utils.eval_synthetic import TRACKERS                            # noqa: E402
from utils.mot_metrics import _iou                                   # noqa: E402
from utils.stress_eval import TRC_BINS, stability, sub_sequence       # noqa: E402
from utils.tracking_quality import load_cache                        # noqa: E402

OURS = "ByteTrack + TRC gate (ours)"
BASE = "ByteTrack"
SORTS = "SORT-style"
DEEP = "DeepSORT (motion)"
RNG = np.random.default_rng(0)
N_BOOT = 10000


def paired_stats(diffs) -> dict:
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    boots = RNG.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
    try:
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"n": int(len(d)), "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wilcoxon_p": p, "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}


def mean_ci(values) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    boots = RNG.choice(v, size=(N_BOOT, len(v)), replace=True).mean(axis=1)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n": int(len(v))}


# ── clean clips ──────────────────────────────────────────────────────────────
def clean_stats(clean_dir: Path, segments: dict) -> dict:
    clips = []
    for cid in sorted(segments):
        f = clean_dir / f"sequence_{cid}.dets.npz"
        c = evaluate_clip(f)
        n_frames = c["n_frames"]
        for name, m in c["trackers"].items():
            m["persistence"] = m["len_mean"] / n_frames                    # mean track length / clip length
            m["tracks_per_target"] = m["n_tracks"] / max(m["mean_concurrent"], 1e-9)
        clips.append(c)
    metrics = ["n_tracks", "persistence", "tracks_per_target", "coverage", "jump_rate", "frag_rate",
               "mean_concurrent", "tracker_ms_per_frame"]
    agg = {name: {m: mean_ci([c["trackers"][name][m] for c in clips]) for m in metrics} for name in TRACKERS}
    totals = {name: {"tracks": int(sum(c["trackers"][name]["n_tracks"] for c in clips))} for name in TRACKERS}
    pair = {}
    for a, b in ((OURS, BASE), (BASE, SORTS), (BASE, DEEP)):
        pair[f"{a} vs {b}"] = {m: paired_stats([c["trackers"][a][m] - c["trackers"][b][m] for c in clips])
                               for m in ("n_tracks", "persistence", "tracks_per_target", "coverage", "jump_rate")}
    flags = sum(sum(c["flags"].get("tracks_with_flags", {}).values()) for c in clips if c.get("flags"))
    trc = [c["trc"]["mean"] for c in clips]
    return {"n_clips": len(clips), "n_frames": int(sum(c["n_frames"] for c in clips)),
            "trc_mean_over_clips": mean_ci(trc), "trc_min": float(min(c["trc"]["min"] for c in clips)),
            "behaviour_flags_raised": int(flags), "aggregate": agg, "totals": totals, "paired": pair,
            "per_clip": [{"clip": c["clip"], "n_frames": c["n_frames"],
                          **{n: {m: c["trackers"][n][m] for m in metrics} for n in TRACKERS}} for c in clips]}


# ── stress ───────────────────────────────────────────────────────────────────
def coupling_rows(clean: dict, stressed: dict, high: float = 0.6):
    clean_by = dict(zip(clean["frames"], clean["dets"]))
    rows = []   # trc, n_ref, n_hit, sum_iou, sum_score_ratio, n_dets_stressed, n_dets_clean
    for f, d_s, trc in zip(stressed["frames"], stressed["dets"], stressed["trc_obs"]):
        d_c = clean_by.get(f)
        if d_c is None:
            continue
        ref = d_c[d_c[:, 4] >= high]
        n_hit = 0
        iou_sum = ratio_sum = 0.0
        if len(ref) and len(d_s):
            iou = _iou(ref[:, :4], d_s[:, :4])
            j, best = iou.argmax(1), iou.max(1)
            ok = best >= 0.5
            n_hit, iou_sum = int(ok.sum()), float(best[ok].sum())
            ratio_sum = float((d_s[j[ok], 4] / ref[ok, 4]).sum())
        rows.append((float(trc), len(ref), n_hit, iou_sum, ratio_sum, len(d_s), len(d_c)))
    return rows


def stress_stats(clean_dir: Path, stress_dir: Path, segments: dict) -> dict:
    runs, rows_all, per_frame = [], [], []
    clip_ids = sorted(segments)
    for cid in clip_ids:
        seg = tuple(segments[cid]["segment"])
        clean = load_cache(clean_dir / f"sequence_{cid}.dets.npz")
        for kind in ("blur", "saturate"):
            sp = stress_dir / f"sequence_{cid}_{kind}.dets.npz"
            if not sp.exists():
                continue
            stressed = load_cache(sp)
            seg_frames = [f for f in stressed["frames"] if seg[0] <= f < seg[1]]
            if len(seg_frames) < 2:
                continue
            seg_s = sub_sequence(stressed, seg_frames)
            clean_seg = sub_sequence(clean, seg_frames)
            rows = coupling_rows(clean_seg, seg_s)
            rows_all.append((kind, rows))
            for r in rows:
                if r[1] > 0:
                    per_frame.append((r[0], r[2] / r[1], kind))
            st = stability(clean, stressed, seg)
            runs.append({"clip": cid, "kind": kind, "seg": list(seg),
                         "trc_clean": float(np.mean(clean_seg["trc_obs"])), "trc_stressed": float(np.mean(seg_s["trc_obs"])),
                         "stability": st})
    # coupling by TRC bin (pooled over frames)
    out_bins = {}
    allrows = np.array([r for _, rows in rows_all for r in rows], dtype=float).reshape(-1, 7)
    for lo, hi in TRC_BINS:
        m = (allrows[:, 0] >= lo) & (allrows[:, 0] < hi)
        if m.sum() == 0:
            continue
        ref_n, hit = allrows[m, 1].sum(), allrows[m, 2].sum()
        out_bins[f"{lo:.2f}-{min(hi, 1.0):.2f}"] = {
            "frames": int(m.sum()), "confident_refs": int(ref_n),
            "recall_vs_clean": float(hit / ref_n) if ref_n else None,
            "mean_iou_of_matched": float(allrows[m, 3].sum() / hit) if hit else None,
            "score_ratio_of_matched": float(allrows[m, 4].sum() / hit) if hit else None,
            "dets_per_frame_ratio": float(allrows[m, 5].sum() / max(allrows[m, 6].sum(), 1))}
    pf = np.array([(a, b) for a, b, _ in per_frame])
    rho, p = stats.spearmanr(pf[:, 0], pf[:, 1])
    # clip-level means of the two kinds (clip = unit of analysis)
    by_clip = {}
    for r in runs:
        by_clip.setdefault(r["clip"], []).append(r)
    metrics = ("IDF1", "HOTA", "AssA", "IDSW")
    clip_level = {t: {m: [np.nanmean([x["stability"][t][m] for x in rs]) for rs in by_clip.values()]
                      for m in metrics} for t in TRACKERS}
    summary = {t: {m: mean_ci(clip_level[t][m]) for m in metrics} for t in TRACKERS}
    pair = {}
    for a, b in ((OURS, BASE), (BASE, SORTS), (BASE, DEEP)):
        pair[f"{a} vs {b}"] = {m: paired_stats(np.array(clip_level[a][m]) - np.array(clip_level[b][m])) for m in metrics}
    per_kind = {}
    for kind in ("blur", "saturate"):
        rs = [r for r in runs if r["kind"] == kind]
        per_kind[kind] = {
            "n": len(rs), "trc_clean": float(np.mean([r["trc_clean"] for r in rs])),
            "trc_stressed": float(np.mean([r["trc_stressed"] for r in rs])),
            "share_below_0.5": float(np.mean([r["trc_stressed"] < 0.5 for r in rs])),
            "ours_vs_bytetrack_IDF1": paired_stats([r["stability"][OURS]["IDF1"] - r["stability"][BASE]["IDF1"] for r in rs])}
    return {"n_clips": len(by_clip), "n_runs": len(runs), "coupling_by_trc_bin": out_bins,
            "spearman_trc_vs_frame_recall": {"rho": float(rho), "p": float(p), "n_frames": int(len(pf))},
            "clip_level_summary": summary, "paired_clip_level": pair, "per_kind": per_kind}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT / "runs" / "tracking_eval_m3fd"))
    args = ap.parse_args()
    root = Path(args.root)
    segments = json.loads((root / "segments.json").read_text(encoding="utf-8"))
    res = {"clean": clean_stats(root / "clean", segments), "stress": stress_stats(root / "clean", root / "stress", segments)}
    (root / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    c, s = res["clean"], res["stress"]
    print(f"clean: {c['n_clips']} clips, {c['n_frames']} frames, TRC mean {c['trc_mean_over_clips']['mean']:.3f}, "
          f"behaviour flags {c['behaviour_flags_raised']}")
    for n in TRACKERS:
        a = c["aggregate"][n]
        print(f"  {n:<30} tracks/clip {a['n_tracks']['mean']:.1f}  persistence {a['persistence']['mean']:.2f}  "
              f"cov {a['coverage']['mean']:.3f}  jump {a['jump_rate']['mean']:.1f}")
    print("paired ours vs ByteTrack:", json.dumps(c["paired"][f"{OURS} vs {BASE}"], indent=0)[:600])
    print("stress:", s["n_clips"], "clips,", s["n_runs"], "runs; rho", s["spearman_trc_vs_frame_recall"])
    for n in TRACKERS:
        print(f"  {n:<30} IDF1 {s['clip_level_summary'][n]['IDF1']['mean']:.3f}  IDSW {s['clip_level_summary'][n]['IDSW']['mean']:.2f}")
    print("saved", root / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
