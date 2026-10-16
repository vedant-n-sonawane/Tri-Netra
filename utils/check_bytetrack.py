#!/usr/bin/env python3
"""
TriNetra-AMRF — Phase 6 ByteTrack / TRC-Gate / Behaviour Self-Test
==================================================================

Runs the v1 tracker on synthetic sequences whose correct answer is known — no
dataset, detector or GPU needed:

  1. Kalman filter        constant-velocity motion is predicted accurately
  2. Matching             IoU matrix values and threshold-respecting assignment
  3. Lifecycle            tentative -> confirmed at n_init; tentative dropped on a
                          miss; lost track recovers its ID inside max_age and gets
                          a new ID after it
  4. Identity             two objects keep separate, stable IDs
  5. Low-conf recovery    ByteTrack keeps a track alive through low-confidence
                          frames that a confidence-discarding tracker loses
  6. TRC gate (key test)  the SAME shaky low-confidence box is accepted at
                          trc=1, rejected at trc=0.2, and accepted again when the
                          gate is disabled -> the gate is wired and effective
  7. Behaviour            loitering / straight path / pacing / restricted zone /
                          class filtering / low-TRC flag

Usage:
    python -m utils.check_bytetrack
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tracker import ByteTracker, build_tracker  # noqa: E402
from models.tracker.kalman_filter import KalmanFilter  # noqa: E402
from models.tracker.matching import iou_matrix, linear_assignment  # noqa: E402
from models.tracker.track import xyxy_to_xyah  # noqa: E402
from models.tracker.trc_gate import trc_adjusted_cost  # noqa: E402

_results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def det(x, y, w=40.0, h=100.0, score=0.9, cls=1):
    """Detection row from a top-left corner + size."""
    return [x, y, x + w, y + h, score, cls]


BASE_CFG = {"tracking": {
    "matching": {"high_conf_threshold": 0.6, "low_conf_threshold": 0.1,
                 "iou_threshold_stage1": 0.3, "iou_threshold_stage2": 0.5},
    "trc_gate": {"enabled": True, "beta": 0.5},
    "lifecycle": {"n_init": 3, "max_age": 30},
    "behaviour": {"enabled": True},
}}


def cfg_with(**over):
    c = copy.deepcopy(BASE_CFG)
    for path, val in over.items():
        node = c["tracking"]
        *parents, leaf = path.split("__")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = val
    return c


def ids(out):
    return sorted(t["track_id"] for t in out)


# ── 1. Kalman ────────────────────────────────────────────────────────────────
def test_kalman():
    print("\n[1] Kalman filter")
    kf = KalmanFilter()
    box = [100.0, 100.0, 140.0, 200.0]
    mean, cov = kf.initiate(xyxy_to_xyah(box))
    for f in range(1, 21):
        mean, cov = kf.predict(mean, cov)
        box = [100.0 + 5 * f, 100.0, 140.0 + 5 * f, 200.0]
        mean, cov = kf.update(mean, cov, xyxy_to_xyah(box))
    mean, cov = kf.predict(mean, cov)
    err = abs(mean[0] - (120.0 + 5 * 21))
    check("predicts next centre of a 5 px/frame mover", err < 1.0, f"error {err:.2f}px")


# ── 2. Matching ──────────────────────────────────────────────────────────────
def test_matching():
    print("\n[2] Matching")
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [50, 50, 60, 60]], float)
    iou = iou_matrix(a, b)[0]
    check("IoU identical/half-shifted/disjoint = 1 / 1/3 / 0",
          abs(iou[0] - 1) < 1e-9 and abs(iou[1] - 1 / 3) < 1e-9 and iou[2] == 0, str(np.round(iou, 3)))
    cost = np.array([[0.2, 0.9], [0.8, 0.95]])
    m, ur, uc = linear_assignment(cost, thresh=0.5)
    check("assignment rejects pairs above the cost ceiling",
          m == [(0, 0)] and ur == [1] and uc == [1], f"matches={m}")
    check("TRC cost formula: trc=1 unchanged, trc=0 inflated by (1+beta)",
          abs(trc_adjusted_cost(0.4, 1.0, 0.5) - 0.4) < 1e-12
          and abs(trc_adjusted_cost(0.4, 0.0, 0.5) - 0.6) < 1e-12)


# ── 3. Lifecycle ─────────────────────────────────────────────────────────────
def test_lifecycle():
    print("\n[3] Track lifecycle")
    tr = ByteTracker(cfg_with())
    reported = [bool(tr.update([det(100, 100)])) for _ in range(5)]
    check("reported only once confirmed (n_init=3)", reported == [False, False, True, True, True],
          str(reported))

    tr = ByteTracker(cfg_with())
    tr.update([det(100, 100)])
    tr.update([])
    check("tentative track dropped on its first miss", len(tr.tracks) == 0)

    tr = ByteTracker(cfg_with())
    for f in range(10):
        out = tr.update([det(100 + 3 * f, 100)])
    tid = out[0]["track_id"]
    for _ in range(10):
        tr.update([])
    out = tr.update([det(100 + 3 * 20, 100)])
    check("lost track recovers its ID inside max_age",
          len(out) == 1 and out[0]["track_id"] == tid, f"id {tid} -> {ids(out)}")

    tr = ByteTracker(cfg_with())
    for f in range(10):
        out = tr.update([det(100, 100)])
    tid = out[0]["track_id"]
    for _ in range(40):
        tr.update([])
    new = []
    for _ in range(4):
        new = tr.update([det(100, 100)])
    check("track removed after max_age; same spot gets a NEW ID",
          len(new) == 1 and new[0]["track_id"] != tid, f"old {tid}, new {ids(new)}")


# ── 4. Identity ──────────────────────────────────────────────────────────────
def test_identity():
    print("\n[4] Identity continuity / separation")
    tr = ByteTracker(cfg_with())
    seen_a, seen_b = set(), set()
    for f in range(60):
        out = tr.update([det(50 + 4 * f, 100, cls=1), det(700 - 4 * f, 400, cls=2)])
        for t in out:
            (seen_a if t["cls"] == 1 else seen_b).add(t["track_id"])
    check("each object keeps one ID for 60 frames", len(seen_a) == 1 and len(seen_b) == 1,
          f"person ids {seen_a}, car ids {seen_b}")
    check("the two objects never share an ID", not (seen_a & seen_b))


# ── 5. Low-confidence recovery ───────────────────────────────────────────────
def _run_low_conf(cfg):
    tr = ByteTracker(cfg)
    n_reported = 0
    ids_seen = set()
    for f in range(60):
        score = 0.3 if 20 <= f < 31 else 0.9
        out = tr.update([det(100 + 3 * f, 100, score=score)])
        n_reported += bool(out)
        ids_seen.update(t["track_id"] for t in out)
    return n_reported, ids_seen


def test_low_conf():
    print("\n[5] Low-confidence recovery (ByteTrack's core idea)")
    kept, ids1 = _run_low_conf(cfg_with())
    dropped, ids2 = _run_low_conf(cfg_with(matching__low_conf_threshold=0.35))
    check("keeps reporting the track through 11 low-confidence frames", kept - dropped >= 10,
          f"reported {kept} frames vs {dropped} when low-conf boxes are discarded")
    check("no ID switch either way", len(ids1) == 1 and len(ids2) == 1)


# ── 6. TRC gate ──────────────────────────────────────────────────────────────
def _shaky_frame(gate_enabled: bool, trc_at_shaky: float):
    """30 clean frames, then one low-confidence box shifted to IoU ~0.55 with the
    prediction. Returns (track_reported_at_shaky_frame, same_id_as_before)."""
    tr = ByteTracker(cfg_with(trc_gate__enabled=gate_enabled))
    for f in range(30):
        out = tr.update([det(100 + 5 * f, 100)], trc=1.0)
    tid = out[0]["track_id"]
    dx = 11.6  # IoU = (40-dx)/(40+dx) = 0.55 for equal-sized boxes
    out = tr.update([det(100 + 5 * 30 + dx, 100, score=0.3)], trc=trc_at_shaky)
    return (len(out) == 1 and out[0]["track_id"] == tid)


def test_trc_gate():
    print("\n[6] TRC gate — the TriNetra-specific behaviour")
    accepted_full = _shaky_frame(True, 1.0)
    accepted_low = _shaky_frame(True, 0.2)
    accepted_off = _shaky_frame(False, 0.2)
    check("gate on,  trc=1.0 : shaky box ACCEPTED (behaves like ByteTrack)", accepted_full)
    check("gate on,  trc=0.2 : same box REJECTED (needs tighter motion agreement)", not accepted_low)
    check("gate off, trc=0.2 : same box ACCEPTED (stock ByteTrack)", accepted_off)
    check("outcome differs between real TRC and trc forced to 1.0",
          accepted_full != accepted_low, "gate is wired and effective")

    tr = ByteTracker(cfg_with())
    for f in range(5):
        out = tr.update([det(100, 100)], trc=None)
    check("missing TRC falls back to full trust (no crash)", len(out) == 1)


# ── 7. Behaviour ─────────────────────────────────────────────────────────────
def _track_sequence(positions, cls=1, trc=1.0, cfg=None):
    tr = ByteTracker(cfg or cfg_with())
    out = []
    for f, (x, y) in enumerate(positions):
        out = tr.update([det(x, y, cls=cls)], trc=trc)
    return out[0]["behaviour"] if out else {}


def test_behaviour():
    print("\n[7] Behaviour analysis")
    rng = np.random.default_rng(0)

    still = [(300 + rng.normal(0, 0.5), 300 + rng.normal(0, 0.5)) for _ in range(200)]
    b = _track_sequence(still)
    check("stationary person -> 'loitering' flag", "loitering" in b["anomaly_flags"],
          f"loitering_frames={b['loitering_frames']}")
    check("jitter on a stationary person is NOT flagged as erratic path",
          "erratic_path" not in b["anomaly_flags"], f"efficiency={b['path_efficiency']:.2f}")

    walk = [(100 + 3 * f, 300) for f in range(100)]
    b = _track_sequence(walk)
    check("straight walker -> path efficiency ~1, no flags",
          b["path_efficiency"] > 0.95 and not b["anomaly_flags"] and b["loitering_frames"] == 0,
          f"eff={b['path_efficiency']:.2f}, speed={b['avg_speed']:.1f}px/f")

    pace = []
    for _ in range(4):
        pace += [(200 + 3 * k, 300) for k in range(40)]
        pace += [(200 + 3 * (39 - k), 300) for k in range(40)]
    b = _track_sequence(pace)
    check("back-and-forth pacing -> reversals and 'pacing' flag",
          b["direction_reversals"] >= 2 and "pacing" in b["anomaly_flags"],
          f"reversals={b['direction_reversals']}")

    uturn = [(200 + 3 * k, 300) for k in range(60)] + [(200 + 3 * (59 - k), 300) for k in range(60)]
    b = _track_sequence(uturn)
    check("a single U-turn is NOT flagged as pacing",
          b["direction_reversals"] <= 1 and "pacing" not in b["anomaly_flags"],
          f"reversals={b['direction_reversals']}")

    zone = [[[250, 250], [400, 250], [400, 450], [250, 450]]]
    inside = [(300 + rng.normal(0, 0.5), 300) for _ in range(60)]
    b = _track_sequence(inside, cfg=cfg_with(behaviour__restricted_zones=zone))
    check("dwell inside a restricted zone -> 'restricted_zone' flag",
          b["in_restricted_zone"] and "restricted_zone" in b["anomaly_flags"],
          f"dwell={b['zone_dwell_frames']} frames")

    b = _track_sequence(still, cls=2)
    check("parked car (class not in flag_classes) is not flagged for loitering",
          "loitering" not in b["anomaly_flags"])

    b = _track_sequence(walk, trc=0.3)
    check("low mean TRC -> 'low_trc_track' surfaced to Phase 7",
          "low_trc_track" in b["anomaly_flags"], f"mean_trc={b['mean_trc']:.2f}")


# ── 7b. Real-data regression ─────────────────────────────────────────────────
def _replay_fixture(move_speed_thresh=None):
    """Replay cached raw detections (sequence 7, first 100 frames) through the tracker."""
    z = np.load(PROJECT_ROOT / "utils" / "fixtures" / "seq07_first100_dets.npz")
    dets, trcs = z["dets"], z["trc"]
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "local_6gb.yaml", encoding="utf-8"))
    if move_speed_thresh is not None:
        cfg["tracking"]["behaviour"]["move_speed_thresh"] = move_speed_thresh
    tr = ByteTracker(cfg)
    flagged, max_rev = set(), 0
    for fi, trc in trcs:
        for t in tr.update(dets[dets[:, 0] == fi][:, 1:], trc=trc, frame_id=int(fi)):
            b = t["behaviour"]
            max_rev = max(max_rev, b["direction_reversals"])
            if "pacing" in b["anomaly_flags"]:
                flagged.add(t["track_id"])
    return flagged, max_rev


def test_real_regression():
    print("\n[7b] Real-data regression (sequence 7, cached detections)")
    old_flags, old_rev = _replay_fixture(move_speed_thresh=0.01)
    new_flags, new_rev = _replay_fixture()
    check("fixture reproduces the false 'pacing' flag under the old jitter-level threshold",
          bool(old_flags), f"flagged tracks {sorted(old_flags)}, {old_rev} reversals")
    check("configured threshold removes it (a near-still person is not 'pacing')",
          not new_flags, f"flagged tracks {sorted(new_flags)}, {new_rev} reversals")


# ── 8. Real config wiring ────────────────────────────────────────────────────
def test_config():
    print("\n[8] Real config wiring")
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "local_6gb.yaml", encoding="utf-8"))
    tr = build_tracker(cfg)
    out = []
    for f in range(6):
        out = tr.update([det(100 + 2 * f, 100)], trc=0.8)
    check("build_tracker(configs/local_6gb.yaml) yields a working ByteTracker",
          isinstance(tr, ByteTracker) and len(out) == 1)


def main() -> int:
    print("=== check_bytetrack (Phase 6 v1) ===")
    for fn in (test_kalman, test_matching, test_lifecycle, test_identity,
               test_low_conf, test_trc_gate, test_behaviour, test_real_regression, test_config):
        fn()
    n_ok = sum(_results)
    print(f"\n{n_ok}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
