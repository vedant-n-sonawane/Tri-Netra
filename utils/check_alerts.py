#!/usr/bin/env python3
"""
Self-checks for the Phase 7 threat scorer and the Phase 8 alert manager
(no model, no data needed).

Usage:
    python -m utils.check_alerts
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.risk_model.alerts import AlertManager  # noqa: E402
from models.risk_model.threat_scorer import score_frame  # noqa: E402

PASSED = 0


def check(cond: bool, name: str) -> None:
    global PASSED
    if not cond:
        raise AssertionError(name)
    PASSED += 1
    print(f"  [OK] {name}")


def T(level: int) -> dict:
    return {"level": level, "reasons": [f"level {level}"]}


def main() -> int:
    # threat scorer
    check(score_frame([])["level"] == 1, "empty frame is benign")
    crowd = [(10 * i, 0, 10 * i + 8, 20, 1, 0.9) for i in range(10)]
    check(score_frame(crowd)["points"] == 2, "10 people = large crowd (+2)")
    bus = [(0, 0, 100, 60, 3, 0.9), (50, 30, 60, 50, 1, 0.9)]
    r = score_frame(bus)
    check(r["points"] == 2 and r["level"] == 3, "person near bus: +1 near vehicle, +1 heavy vehicle")
    r = score_frame(crowd, rgb_mean=40, thermal_only=[0, 1, 2])
    check(r["level"] == 5, "night crowd with 3 hidden people is critical")
    check(score_frame(bus, trc=0.3)["points"] == 3, "low TRC with people adds a point")

    # alert manager
    m = AlertManager({"min_level": 4, "persist_frames": 2, "cooldown_frames": 10, "low_trc": 0.5})
    check(m.update(0, T(4), 0.8) == [], "one high frame is not enough")
    a = m.update(1, T(4), 0.8)
    check(len(a) == 1 and a[0]["kind"] == "threat", "second consecutive high frame raises an alert")
    check(m.update(2, T(4), 0.8) == [], "same level inside cooldown is suppressed")
    a = m.update(3, T(5), 0.8)
    check(len(a) == 1 and a[0]["level"] == 5, "escalation to critical alerts at once")
    check(m.update(14, T(5), 0.8) != [], "same level re-alerts after the cooldown")
    check(m.update(15, T(2), 0.8) == [], "low level never alerts")

    s = AlertManager({"persist_frames": 2, "low_trc": 0.5})
    s.update(0, T(1), 0.4)
    a = s.update(1, T(1), 0.4)
    check(len(a) == 1 and a[0]["kind"] == "sensor", "two low-TRC frames raise a sensor alert")
    check(s.update(2, T(1), 0.4) == [], "one sensor alert per low-TRC episode")
    s.update(3, T(1), 0.8)
    s.update(4, T(1), 0.4)
    check(len(s.update(5, T(1), 0.4)) == 1, "a new low-TRC episode alerts again")

    b = AlertManager()
    trk = {"track_id": 7, "cls_name": "person", "behaviour": {"anomaly_flags": ["loitering", "low_trc_track"]}}
    a = b.update(0, T(1), 0.8, [trk])
    check(len(a) == 1 and a[0]["kind"] == "behaviour" and a[0]["track_id"] == 7,
          "new behaviour flag raises one alert (low_trc_track ignored)")
    check(b.update(1, T(1), 0.8, [trk]) == [], "the same flag does not alert twice")
    check([x["alert_id"] for x in b.alerts] == [1], "alert ids are sequential")

    print(f"\n  all {PASSED} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
