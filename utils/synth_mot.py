#!/usr/bin/env python3
"""
TriNetra-AMRF — Synthetic Multi-Object Tracking Benchmark (exact ground truth)
==============================================================================

The real clips have no track annotations, so tracker accuracy cannot be scored
on them directly. This module generates sequences whose ground truth is known
exactly, and simulates a detector whose quality is *coupled to the frame's
Thermal Reliability Confidence (TRC)*:

    lower TRC  ->  more missed detections, noisier boxes, lower scores,
                   more false positives

That coupling is the modelling assumption behind the TRC gate (it is measured
on real footage separately, in the thermal-degradation stress test). To keep
the benchmark honest, ``condition`` also provides controls in which the gate
has no reason to help:

  "coupled"        detector degrades when TRC dips AND the tracker sees TRC
  "uninformative"  detector degrades on the same episodes, but the tracker is
                   given a constant TRC (the signal carries no information)
  "no_degradation" TRC dips, but the detector is unaffected (the gate should,
                   if anything, cost a little)

Scene model: people-like boxes with smooth wandering motion (some loiter),
staggered entry/exit, crossing paths, static occluders that hide objects for
stretches, and fixed clutter sites that fire false positives. All randomness
is seeded, so every tracker sees identical input.

Simulated appearance (for the ReID ablation): each real object gets a fixed
random unit vector ("what it looks like") at creation; every detection of it
carries that vector plus small noise, independent of the TRC-coupled
detection-quality degradation — modelling that appearance comes from the
(undamaged) RGB stream while ``coupling`` models thermal-driven detection
failures. False positives get a fresh random vector each time (clutter has no
consistent appearance), so a tracker that actually uses appearance should
reject them as a match, not just one that uses motion.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

W, H = 1280.0, 1024.0
CONDITIONS = ("coupled", "uninformative", "no_degradation")
APPEARANCE_DIM = 32
APPEARANCE_NOISE = 0.25    # std of per-observation noise added to an object's base vector


def _unit_vec(rng, d=APPEARANCE_DIM) -> np.ndarray:
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def _noisy_vec(rng, base: np.ndarray, noise: float = APPEARANCE_NOISE) -> np.ndarray:
    v = base + rng.normal(scale=noise, size=base.shape)
    return v / np.linalg.norm(v)


def _trc_series(rng, T: int, base: float = 0.78) -> np.ndarray:
    trc = base + np.convolve(rng.normal(0, 0.02, T + 20), np.ones(10) / 10, mode="same")[10:T + 10]
    for _ in range(rng.integers(2, 5)):
        start, length = int(rng.integers(30, max(T - 150, 40))), int(rng.integers(40, 121))
        level = rng.uniform(0.2, 0.45)
        ramp = 10
        env = np.zeros(T)
        env[start:start + length] = 1.0
        env = np.convolve(env, np.hanning(2 * ramp + 1) / np.hanning(2 * ramp + 1).sum(), mode="same")
        trc = trc * (1 - env) + level * env
    return np.clip(trc, 0.05, 0.98)


def _visibility(box, occluders) -> float:
    x1, y1, x2, y2 = box
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    hidden = 0.0
    for ox1, oy1, ox2, oy2 in occluders:
        iw = max(0.0, min(x2, ox2) - max(x1, ox1))
        ih = max(0.0, min(y2, oy2) - max(y1, oy1))
        hidden = max(hidden, iw * ih / area)
    return 1.0 - hidden


def make_sequence(seed: int, T: int = 600, condition: str = "coupled", severity: float = 1.0) -> dict:
    """Returns {gt, dets, trc_true, trc_obs, meta}.

    gt:   {frame: ndarray [n,5]}  (id, x1, y1, x2, y2), only frames where the
          object is inside the image and >=30% visible
    dets: list (len T) of ndarray [n,6]  (x1, y1, x2, y2, score, cls=1)
    """
    assert condition in CONDITIONS
    rng = np.random.default_rng(seed)
    trc_true = _trc_series(rng, T)
    coupling = trc_true if condition != "no_degradation" else np.full(T, 0.78)
    trc_obs = trc_true if condition != "uninformative" else np.full(T, 0.78) + rng.normal(0, 0.01, T)
    sev = float(severity)

    occluders = []
    for _ in range(int(rng.integers(2, 4))):
        w, h = rng.uniform(110, 200), rng.uniform(300, 480)
        x, y = rng.uniform(150, W - 350), rng.uniform(80, H - 520)
        occluders.append((x, y, x + w, y + h))
    clutter = [(rng.uniform(100, W - 100), rng.uniform(150, H - 250), rng.uniform(70, 110), rng.uniform(150, 230))
               for _ in range(int(rng.integers(3, 6)))]

    n_obj = int(rng.integers(8, 15))
    objs = []
    for oid in range(1, n_obj + 1):
        h = rng.uniform(140, 260)
        loiter = rng.random() < 0.25
        speed = rng.uniform(0.002, 0.006) * h if loiter else rng.uniform(0.015, 0.045) * h
        ang = rng.uniform(0, 2 * np.pi)
        objs.append({
            "id": oid, "h": h, "w": 0.42 * h,
            "t0": int(rng.integers(0, T // 2)), "life": int(rng.integers(150, T + 1)),
            "p": np.array([rng.uniform(150, W - 150), rng.uniform(150, H - 150)]),
            "v": np.array([np.cos(ang), np.sin(ang)]) * speed, "speed": speed,
            "feat": _unit_vec(rng),
        })

    gt: Dict[int, List[list]] = {}
    dets: List[np.ndarray] = []
    appearance: List[np.ndarray] = []
    for t in range(T):
        rows = []
        feats = []
        for ob in objs:
            if not (ob["t0"] <= t < ob["t0"] + ob["life"]):
                continue
            ob["v"] = 0.98 * ob["v"] + rng.normal(0, 0.0025 * ob["h"], 2)
            sp = np.linalg.norm(ob["v"])
            if sp > 1.6 * ob["speed"] + 1e-6:
                ob["v"] *= 1.6 * ob["speed"] / sp
            ob["p"] = ob["p"] + ob["v"]
            for k, lim in enumerate((W, H)):          # bounce off the image border
                if not 60 < ob["p"][k] < lim - 60:
                    ob["v"][k] *= -1
                    ob["p"][k] = np.clip(ob["p"][k], 61, lim - 61)
            cx, cy, w, h = ob["p"][0], ob["p"][1], ob["w"], ob["h"]
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            vis = _visibility(box, occluders)
            if vis < 0.3:
                continue
            gt.setdefault(t, []).append([ob["id"], *box])

            tq = coupling[t]
            bad = (1 - tq) * sev
            if rng.random() < 0.95 * (vis ** 0.5) * (1 - 0.55 * bad):
                sc = 0.03 * (1 + 4.0 * bad)
                ncx, ncy = cx + rng.normal(0, sc * h), cy + rng.normal(0, sc * h)
                nw, nh = w * np.exp(rng.normal(0, 0.5 * sc)), h * np.exp(rng.normal(0, 0.5 * sc))
                score = float(np.clip(rng.normal(0.9 - 0.5 * bad - 0.25 * (1 - vis), 0.07), 0.12, 0.99))
                rows.append([ncx - nw / 2, ncy - nh / 2, ncx + nw / 2, ncy + nh / 2, score, 1])
                feats.append(_noisy_vec(rng, ob["feat"]))

        bad = (1 - coupling[t]) * sev
        for (fx, fy, fw, fh) in clutter:                # clutter sites firing false positives
            if rng.random() < 0.20 + 0.5 * bad:
                rows.append([fx - fw / 2, fy - fh / 2, fx + fw / 2, fy + fh / 2,
                             float(rng.uniform(0.15, 0.55 + 0.2 * bad)), 1])
                feats.append(_unit_vec(rng))
        for _ in range(rng.poisson(0.15 + 1.5 * bad)):  # random false positives
            fw = rng.uniform(50, 110); fh = fw / 0.42
            fx, fy = rng.uniform(100, W - 100), rng.uniform(150, H - 150)
            rows.append([fx - fw / 2, fy - fh / 2, fx + fw / 2, fy + fh / 2,
                         float(rng.uniform(0.1, 0.45 + 0.3 * bad)), 1])
            feats.append(_unit_vec(rng))
        dets.append(np.clip(np.array(rows, dtype=float).reshape(-1, 6), [0, 0, 0, 0, 0, 0], [W, H, W, H, 1, 9]))
        appearance.append(np.array(feats, dtype=float).reshape(-1, APPEARANCE_DIM))

    return {
        "gt": {f: np.array(v, dtype=float) for f, v in gt.items()},
        "dets": dets, "trc_true": trc_true, "trc_obs": trc_obs, "appearance": appearance,
        "meta": {"seed": seed, "T": T, "condition": condition, "severity": sev, "n_objects": n_obj,
                 "mean_trc": float(trc_true.mean()),
                 "frac_degraded": float((trc_true < 0.6).mean())},
    }
