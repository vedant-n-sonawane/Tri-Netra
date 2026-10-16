#!/usr/bin/env python3
"""
TriNetra-AMRF — Trajectory Behaviour Features (Phase 6, v1 rule-based)
======================================================================

Turns a track's history buffer into behaviour features and threshold-based
anomaly flags for Phase 7. v1 is deliberately deterministic and interpretable
(every threshold lives in ``tracking.behaviour`` of the config); an
unsupervised trajectory model is the planned v2.

Features (all from *matched* observations only):
  avg_speed / avg_speed_norm  windowed speed, px/frame and box-heights/frame
  loitering_frames            length of the trailing streak of near-zero speed
  path_efficiency             displacement / path length on the smoothed path
  heading_variance            variance of heading change between moving samples
  direction_reversals         consecutive moving samples heading > 90 deg apart
                              ("moving" = above move_speed_thresh, stricter than "still",
                              so detector jitter on a near-stationary object is ignored)
  in_restricted_zone, zone_dwell_frames
  mean_trc                    reliability of the evidence behind this track

Speeds are normalised by box height so thresholds do not depend on camera
resolution or distance. Positions are smoothed before computing path length so
detector jitter on a stationary object does not masquerade as wandering.

``anomaly_flags`` are only raised for classes in ``flag_classes`` (unified
category ids); a low-TRC flag is raised for every class so weak evidence is
surfaced to Phase 7 instead of silently hidden.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

_DEFAULTS = {
    "history_len": 300,
    "speed_window": 5,          # observations spanned when measuring speed
    "smooth_window": 5,         # moving-average width for the path
    "loiter_speed_thresh": 0.01,   # box-heights / frame below which an object is "still"
    "move_speed_thresh": 0.02,     # box-heights / frame above which a sample counts as real movement
    "loitering_frame_threshold": 150,
    "path_efficiency_threshold": 0.4,
    "reversal_window": 20,      # trailing velocity samples inspected for reversals
    "reversal_count_threshold": 2,
    "min_obs": 50,              # observations needed before path-shape flags
    "zone_dwell_threshold": 25,
    "low_trc_threshold": 0.5,
    "restricted_zones": [],
    "flag_classes": [1, 5],     # person, motorcycle
}


def _smooth(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(a) < w:
        return a
    padded = np.pad(a, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(padded, np.ones(w) / w, mode="valid")


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _inside(point, polygons) -> bool:
    import cv2
    for poly in polygons:
        contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
        if len(contour) >= 3 and cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0:
            return True
    return False


def compute_behaviour(track, cfg: dict) -> Dict:
    """Behaviour features + anomaly flags for one track (see module docstring)."""
    c = {**_DEFAULTS, **(cfg or {})}
    hist = list(track.history)
    n = len(hist)
    mean_trc = float(np.mean([h["trc"] for h in hist])) if n else 1.0

    out = {
        "n_obs": n, "avg_speed": 0.0, "avg_speed_norm": 0.0, "loitering_frames": 0,
        "in_restricted_zone": False, "zone_dwell_frames": 0, "path_efficiency": 1.0,
        "heading_variance": 0.0, "direction_reversals": 0, "mean_trc": mean_trc,
        "anomaly_flags": [],
    }
    flags: List[str] = []
    if mean_trc < float(c["low_trc_threshold"]) and n >= 3:
        flags.append("low_trc_track")
    if n < 2:
        out["anomaly_flags"] = flags
        return out

    frames = np.array([h["frame_id"] for h in hist], dtype=np.float64)
    cx = np.array([h["cx"] for h in hist])
    cy = np.array([h["cy"] for h in hist])
    heights = np.array([max(h["bbox"][3] - h["bbox"][1], 1e-6) for h in hist])
    med_h = float(np.median(heights))
    sx, sy = _smooth(cx, int(c["smooth_window"])), _smooth(cy, int(c["smooth_window"]))

    # windowed speed per observation
    sw = max(1, int(c["speed_window"]))
    spd = np.zeros(n)
    for i in range(n):
        j = max(0, i - sw)
        df = frames[i] - frames[j]
        if df > 0:
            spd[i] = math.hypot(sx[i] - sx[j], sy[i] - sy[j]) / df
    spd_norm = spd / heights
    out["avg_speed"] = float(np.mean(spd[-sw:]))
    out["avg_speed_norm"] = float(np.mean(spd_norm[-sw:]))

    # trailing near-zero-speed streak (frames)
    thr = float(c["loiter_speed_thresh"])
    idx = n - 1
    while idx >= 0 and spd_norm[idx] < thr:
        idx -= 1
    out["loitering_frames"] = int(frames[-1] - frames[idx + 1] + 1) if idx + 1 <= n - 1 else 0

    # path shape
    steps = np.hypot(np.diff(sx), np.diff(sy))
    path_len = float(steps.sum())
    disp = math.hypot(sx[-1] - sx[0], sy[-1] - sy[0])
    out["path_efficiency"] = float(disp / path_len) if path_len > 1e-6 else 1.0

    # heading changes / reversals from velocity samples taken every `sw` observations
    samples = list(range(0, n, sw))
    vecs = []
    for a, b in zip(samples[:-1], samples[1:]):
        df = frames[b] - frames[a]
        if df <= 0:
            continue
        vx, vy = (sx[b] - sx[a]) / df, (sy[b] - sy[a]) / df
        if math.hypot(vx, vy) / med_h >= float(c["move_speed_thresh"]):
            vecs.append((vx, vy))
    vecs = vecs[-int(c["reversal_window"]):]
    if len(vecs) >= 2:
        ang = [math.atan2(vy, vx) for vx, vy in vecs]
        out["heading_variance"] = float(np.var([_wrap(b - a) for a, b in zip(ang[:-1], ang[1:])]))
        out["direction_reversals"] = int(sum(
            1 for (ax, ay), (bx, by) in zip(vecs[:-1], vecs[1:]) if ax * bx + ay * by < 0))

    # restricted-zone dwell (trailing streak inside any polygon)
    zones = c["restricted_zones"] or []
    if zones:
        inside_now = _inside((cx[-1], cy[-1]), zones)
        out["in_restricted_zone"] = bool(inside_now)
        if inside_now:
            k = n - 1
            while k >= 0 and _inside((cx[k], cy[k]), zones):
                k -= 1
            out["zone_dwell_frames"] = int(frames[-1] - frames[k + 1] + 1)

    # rule-based flags (only for the configured classes)
    flag_classes = c["flag_classes"] or []
    if (not flag_classes) or track.class_id in flag_classes:
        if out["loitering_frames"] >= int(c["loitering_frame_threshold"]):
            flags.append("loitering")
        if (n >= int(c["min_obs"]) and path_len >= 2 * med_h
                and out["path_efficiency"] < float(c["path_efficiency_threshold"])):
            flags.append("erratic_path")
        if out["direction_reversals"] >= int(c["reversal_count_threshold"]):
            flags.append("pacing")
        if out["in_restricted_zone"] and out["zone_dwell_frames"] >= int(c["zone_dwell_threshold"]):
            flags.append("restricted_zone")

    out["anomaly_flags"] = flags
    return out
