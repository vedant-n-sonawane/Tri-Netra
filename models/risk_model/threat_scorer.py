"""
TriNetra-AMRF — Phase 7 frame-level threat scorer
=================================================

Rule-based, explainable 1-5 threat level for one frame (or one short burst)
from detections plus scene context. M3FD has no threat labels and only short
bursts of consecutive frames (longest 42), so the rules are frame-level;
loitering / pacing style rules from Phase 6 cannot fire on it.

Input detections are (x1, y1, x2, y2, class_id, conf) with unified class ids
(1 person, 2 car, 3 bus, 4 truck, 5 motorcycle, 6 cyclist, 7 lamp).
Scene context: ``rgb_mean`` (0-255 mean luminance of the visible frame, low =
night) and ``trc`` (thermal reliability, 0-1). ``thermal_only`` lists indices
of person detections visible in thermal but not in RGB (see ``find_thermal_only``).

Points -> level:  0 -> 1, 1 -> 2, 2-3 -> 3, 4-5 -> 4, >=6 -> 5.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

PERSON, BUS, TRUCK, MOTORCYCLE = 1, 3, 4, 5
VEHICLES = (2, 3, 4, 5)
HEAVY = (BUS, TRUCK, MOTORCYCLE)
LEVEL_NAMES = {1: "benign", 2: "low", 3: "elevated", 4: "high", 5: "critical"}

DEFAULTS = {
    "night_rgb_mean": 70.0,       # mean luminance below this = night / RGB unreliable
    "crowd_medium": 5,
    "crowd_large": 10,
    "congestion_vehicles": 10,
    "near_margin": 0.5,           # person counts as "near" within this x max(vehicle w, h) of its box
    "hidden_medium": 1,           # thermal-only persons: >= this -> +2
    "hidden_large": 3,            # >= this -> +3
    "low_trc": 0.5,
    "contrast_rgb_max": 10.0,     # thermal-only person: RGB contrast below this ...
    "contrast_thermal_min": 25.0, # ... and thermal contrast above this
}


def modality_contrast(gray: np.ndarray, box) -> float:
    """|mean inside box - mean of the surrounding ring| on a grayscale image."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return 0.0
    X1, Y1, X2, Y2 = max(0, x1 - bw), max(0, y1 - bh // 2), min(w, x2 + bw), min(h, y2 + bh // 2)
    inner = gray[y1:y2, x1:x2]
    ring_n = (Y2 - Y1) * (X2 - X1) - bw * bh
    if ring_n <= 0:
        return 0.0
    ring = (gray[Y1:Y2, X1:X2].sum(dtype=np.float64) - inner.sum(dtype=np.float64)) / ring_n
    return float(abs(inner.mean() - ring))


def find_thermal_only(dets: Sequence, vis_gray: np.ndarray, thr_gray: np.ndarray, cfg: dict | None = None) -> List[int]:
    """Indices of person detections that are low-contrast in RGB but clear in thermal."""
    c = {**DEFAULTS, **(cfg or {})}
    out = []
    for i, d in enumerate(dets):
        if int(d[4]) == PERSON and modality_contrast(vis_gray, d[:4]) < c["contrast_rgb_max"] \
                and modality_contrast(thr_gray, d[:4]) > c["contrast_thermal_min"]:
            out.append(i)
    return out


def _near(person, veh, margin: float) -> bool:
    cx, cy = (person[0] + person[2]) / 2, (person[1] + person[3]) / 2
    m = margin * max(veh[2] - veh[0], veh[3] - veh[1])
    return veh[0] - m <= cx <= veh[2] + m and veh[1] - m <= cy <= veh[3] + m


def score_frame(dets: Sequence, rgb_mean: float = 128.0, trc: float = 1.0,
                thermal_only: Sequence[int] = (), cfg: dict | None = None) -> Dict:
    """Return {"level", "label", "points", "reasons": [str], "factors": {...}}."""
    c = {**DEFAULTS, **(cfg or {})}
    dets = [tuple(d) for d in dets]
    persons = [d for d in dets if int(d[4]) == PERSON]
    vehicles = [d for d in dets if int(d[4]) in VEHICLES]
    heavy = [d for d in vehicles if int(d[4]) in HEAVY]
    night = rgb_mean < c["night_rgb_mean"]
    pts, reasons = 0, []

    n = len(persons)
    if n >= c["crowd_large"]:
        pts += 2
        reasons.append(f"large crowd ({n} people)")
    elif n >= c["crowd_medium"]:
        pts += 1
        reasons.append(f"crowd ({n} people)")

    near_any = [p for p in persons if any(_near(p, v, c["near_margin"]) for v in vehicles)]
    near_heavy = [p for p in persons if any(_near(p, v, c["near_margin"]) for v in heavy)]
    if near_any:
        pts += 1
        reasons.append(f"{len(near_any)} {'person' if len(near_any) == 1 else 'people'} close to vehicles")
    if near_heavy:
        pts += 1
        reasons.append(f"{len(near_heavy)} close to a bus, truck or motorcycle")

    if len(vehicles) >= c["congestion_vehicles"]:
        pts += 1
        reasons.append(f"congestion ({len(vehicles)} vehicles)")

    if night and n:
        pts += 1
        reasons.append(f"people present in low light (RGB brightness {rgb_mean:.0f})")

    h = len(thermal_only)
    if h >= c["hidden_large"]:
        pts += 3
        reasons.append(f"{h} people visible only in thermal (hidden from RGB)")
    elif h >= c["hidden_medium"]:
        pts += 2
        reasons.append(f"{h} {'person' if h == 1 else 'people'} visible only in thermal (hidden from RGB)")

    if trc < c["low_trc"] and n:
        pts += 1
        reasons.append(f"thermal reliability low (TRC {trc:.2f}): detections less certain")

    level = 1 if pts == 0 else 2 if pts == 1 else 3 if pts <= 3 else 4 if pts <= 5 else 5
    return {"level": level, "label": LEVEL_NAMES[level], "points": pts,
            "reasons": reasons or ["no threat indicators"],
            "factors": {"persons": n, "vehicles": len(vehicles), "night": night, "thermal_only": h}}
