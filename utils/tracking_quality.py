#!/usr/bin/env python3
"""
TriNetra-AMRF — Label-Free Tracking Quality Metrics (real footage)
==================================================================

The real clips carry no track annotations, so MOTA / IDF1 / HOTA cannot be
computed on them. These metrics need only the detections and the tracker's
output, and quantify the failure modes that matter for tracking quality. They
are *proxies*, not accuracy: they cannot tell a correct long track from a
confidently wrong one, and they say nothing about detection accuracy.

  n_tracks, tracks_per_1k_frames      identity budget (fewer = more consolidated)
  len_mean / len_median / len_p90     track lifetimes in frames
  short_ratio                         share of tracks shorter than 1 s (25 frames):
                                      flicker / spurious tracks
  coverage                            share of confident detections (score >= high
                                      threshold) explained by a reported track
                                      (IoU >= 0.5)
  frag_suspects, frag_rate            a track ends, and a new track of the same
                                      class starts within 2 s and within one box
                                      height of where it ended: probable identity
                                      fragmentation (one object, two IDs)
  jump_rate                           implausible frame-to-frame centroid jumps
                                      (> 0.75 box heights) per 1000 track-frames:
                                      a proxy for identity switches
  recoveries, mean_recovery_gap       lost tracks re-acquired with the same ID,
                                      and how long they had been lost
  mean_concurrent                     average tracks visible per frame
  tracker_ms_per_frame                tracker-only cost

All computed identically for every tracker variant on identical detections.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from utils.mot_metrics import _iou

SHORT_FRAMES = 25
FRAG_WINDOW = 50
JUMP_HEIGHTS = 0.75


def load_cache(npz_path) -> dict:
    """Cached raw detections -> the same ``seq`` layout the synthetic benchmark uses.

    If a matching ``<name>.embeds.npz`` exists (see ``utils/compute_reid_cache.py``), its
    rows (same order as ``dets``) are split per frame the same way and returned as
    ``seq["appearance"]``, so the ReID tracker variants can be replayed on real footage too.
    """
    from pathlib import Path
    z = np.load(npz_path)
    dets, trc = z["dets"], z["trc"]
    frames = trc[:, 0].astype(int)
    embeds_path = Path(npz_path).with_suffix("").with_suffix(".embeds.npz")
    embeds_all = np.load(embeds_path)["embeds"] if embeds_path.exists() else None
    by_frame, app_by_frame = {}, {}
    if len(dets):
        order = np.argsort(dets[:, 0], kind="stable")
        dets = dets[order]
        embeds_all = embeds_all[order] if embeds_all is not None else None
        cuts = np.searchsorted(dets[:, 0], frames)
        ends = np.searchsorted(dets[:, 0], frames, side="right")
        for f, a, b in zip(frames, cuts, ends):
            by_frame[int(f)] = dets[a:b, 1:7]
            if embeds_all is not None:
                app_by_frame[int(f)] = embeds_all[a:b]
    empty = np.zeros((0, 6))
    out = {"frames": frames.tolist(),
          "dets": [by_frame.get(int(f), empty) for f in frames],
          "trc_obs": trc[:, 1], "trc_true": trc[:, 1],
          "meta": {"T": len(frames)}}
    if embeds_all is not None:
        out["appearance"] = [app_by_frame.get(int(f), np.zeros((0, 1280))) for f in frames]
    return out


def replay_bytetrack(seq: dict, tracker) -> dict:
    """Run a ByteTracker over cached detections, recording lifecycle events."""
    frames = seq.get("frames", list(range(len(seq["dets"]))))
    appearance = seq.get("appearance")
    out, recoveries, times, flags = {}, [], [], {}
    for i, (f, dets, trc) in enumerate(zip(frames, seq["dets"], seq["trc_obs"])):
        before = {t.track_id: (t.state, t.time_since_update) for t in tracker.tracks}
        t0 = time.perf_counter()
        embeds = appearance[i] if (appearance is not None and getattr(tracker, "use_appearance", False)) else None
        res = tracker.update(dets, trc=float(trc), frame_id=int(f), embeds=embeds)
        times.append(time.perf_counter() - t0)
        rows = []
        for r in res:
            rows.append([r["track_id"], *r["bbox_xyxy"], r["cls"]])
            flags.setdefault(r["track_id"], set()).update(r["behaviour"].get("anomaly_flags", []))
            prev = before.get(r["track_id"])
            if prev and prev[0] == "lost":
                recoveries.append(prev[1] + 1)
        if rows:
            out[int(f)] = np.array(rows, dtype=float)
    return {"pred": out, "recoveries": recoveries, "ms_per_frame": 1000 * float(np.mean(times)),
            "flags": {k: sorted(v) for k, v in flags.items()}}


def track_records(pred: Dict[int, np.ndarray]) -> List[dict]:
    per: Dict[int, list] = {}
    for f in sorted(pred):
        for row in pred[f]:
            per.setdefault(int(row[0]), []).append((f, row[1:5], int(row[5]) if len(row) > 5 else 0))
    recs = []
    for tid, obs in per.items():
        frames = np.array([o[0] for o in obs])
        boxes = np.stack([o[1] for o in obs])
        recs.append({"id": tid, "frames": frames, "boxes": boxes, "cls": obs[0][2],
                     "n": len(obs), "first": int(frames[0]), "last": int(frames[-1])})
    return recs


def _center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def quality_metrics(pred: Dict[int, np.ndarray], seq: dict, high_thresh: float = 0.6,
                    recoveries: Optional[List[int]] = None, ms_per_frame: Optional[float] = None) -> dict:
    recs = track_records(pred)
    n_frames = len(seq["dets"])
    lengths = np.array([r["n"] for r in recs], dtype=float) if recs else np.zeros(0)

    # coverage of confident detections
    hit = tot = 0
    for f, dets in zip(seq.get("frames", range(n_frames)), seq["dets"]):
        conf = dets[dets[:, 4] >= high_thresh] if len(dets) else dets
        tot += len(conf)
        p = pred.get(int(f))
        if len(conf) and p is not None and len(p):
            hit += int((_iou(conf[:, :4], p[:, 1:5]).max(1) >= 0.5).sum())

    # fragmentation suspects: end -> start of a same-class track nearby in time and space
    last_frame = max((r["last"] for r in recs), default=0)
    suspects = 0
    for a in recs:
        if a["last"] >= last_frame - 5:
            continue
        end_c = _center(a["boxes"][-1])
        ha = a["boxes"][-1][3] - a["boxes"][-1][1]
        for b in recs:
            if b is a or b["cls"] != a["cls"]:
                continue
            gap = b["first"] - a["last"]
            if 0 < gap <= FRAG_WINDOW:
                hb = b["boxes"][0][3] - b["boxes"][0][1]
                if np.linalg.norm(_center(b["boxes"][0]) - end_c) <= max(ha, hb):
                    suspects += 1
                    break

    # implausible jumps
    jumps = track_frames = 0
    for r in recs:
        c = np.stack([_center(b) for b in r["boxes"]])
        h = (r["boxes"][:, 3] - r["boxes"][:, 1])
        for i in range(1, len(c)):
            if r["frames"][i] - r["frames"][i - 1] <= 5:
                track_frames += 1
                if np.linalg.norm(c[i] - c[i - 1]) > JUMP_HEIGHTS * max((h[i] + h[i - 1]) / 2, 1e-6):
                    jumps += 1

    concurrent = np.mean([len(pred.get(int(f), [])) for f in seq.get("frames", range(n_frames))]) if n_frames else 0.0
    n = len(recs)
    return {
        "n_tracks": n,
        "tracks_per_1k_frames": 1000.0 * n / max(n_frames, 1),
        "len_mean": float(lengths.mean()) if n else 0.0,
        "len_median": float(np.median(lengths)) if n else 0.0,
        "len_p90": float(np.percentile(lengths, 90)) if n else 0.0,
        "short_ratio": float((lengths < SHORT_FRAMES).mean()) if n else 0.0,
        "coverage": hit / max(tot, 1),
        "frag_suspects": suspects,
        "frag_rate": suspects / max(n, 1),
        "jump_rate": 1000.0 * jumps / max(track_frames, 1),
        "recoveries": len(recoveries) if recoveries is not None else None,
        "mean_recovery_gap": float(np.mean(recoveries)) if recoveries else 0.0,
        "mean_concurrent": float(concurrent),
        "tracker_ms_per_frame": ms_per_frame,
        "n_frames": n_frames, "n_conf_dets": int(tot),
    }
