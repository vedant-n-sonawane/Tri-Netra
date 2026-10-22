"""
TriNetra-AMRF — Phase 8 explainable alert generation
====================================================

Turns the per-frame output of ``threat_scorer.score_frame`` (plus the frame TRC
and the tracker's behaviour flags) into a short list of alerts a human operator
can act on, instead of one message per frame.

Three alert kinds:
  * ``threat``     the frame threat level stays at or above ``min_level`` for
                   ``persist_frames`` consecutive frames. A higher level than
                   the last alert escalates at once; the same or a lower level
                   is held back for ``cooldown_frames``.
  * ``sensor``     the frame TRC stays below ``low_trc`` for ``persist_frames``
                   frames: the thermal stream has become unreliable, so
                   detections are less certain (one alert per episode).
  * ``behaviour``  a confirmed track receives a behaviour flag it did not have
                   before (loitering, erratic path, pacing, zone dwell).

Every alert carries the reasons that produced it, so it explains itself.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

LEVEL_NAMES = {1: "benign", 2: "low", 3: "elevated", 4: "high", 5: "critical"}
BEHAVIOUR_TEXT = {
    "loitering": "loitering",
    "erratic_path": "erratic path",
    "pacing": "pacing back and forth",
    "zone_dwell": "dwelling in a restricted zone",
}

DEFAULTS = {
    "min_level": 4,          # threat alerts start at "high"
    "persist_frames": 2,     # consecutive frames needed before an alert fires
    "cooldown_frames": 25,   # same-or-lower level re-alerts no sooner than this
    "low_trc": 0.5,          # sensor alert threshold
}


class AlertManager:
    def __init__(self, cfg: Optional[dict] = None):
        c = {**DEFAULTS, **(cfg or {})}
        self.min_level = int(c["min_level"])
        self.persist = max(1, int(c["persist_frames"]))
        self.cooldown = int(c["cooldown_frames"])
        self.low_trc = float(c["low_trc"])
        self.alerts: List[Dict] = []
        self._streak = 0
        self._last_level = 0
        self._last_frame = None
        self._low_streak = 0
        self._sensor_open = False
        self._seen_flags: Dict[int, set] = {}

    def _emit(self, **alert) -> Dict:
        alert["alert_id"] = len(self.alerts) + 1
        self.alerts.append(alert)
        return alert

    def update(self, frame: int, threat: Dict, trc: float,
               tracks: Iterable[Dict] = ()) -> List[Dict]:
        """Feed one frame; return the alerts raised on it (often none)."""
        new: List[Dict] = []
        level = int(threat["level"])

        # threat level alerts
        if level >= self.min_level:
            self._streak += 1
        else:
            self._streak = 0
            if self._last_frame is not None and frame - self._last_frame >= self.cooldown:
                self._last_level = 0          # calm long enough: the next episode starts fresh
        if self._streak >= self.persist:
            escalate = level > self._last_level
            cooled = self._last_frame is None or frame - self._last_frame >= self.cooldown
            if escalate or cooled:
                new.append(self._emit(
                    kind="threat", frame=frame, level=level, label=LEVEL_NAMES[level],
                    title=f"Threat level {level} ({LEVEL_NAMES[level]})",
                    reasons=list(threat["reasons"]), trc=round(float(trc), 3)))
                self._last_level, self._last_frame = level, frame

        # sensor health alerts
        if trc < self.low_trc:
            self._low_streak += 1
            if self._low_streak >= self.persist and not self._sensor_open:
                self._sensor_open = True
                new.append(self._emit(
                    kind="sensor", frame=frame, level=3, label=LEVEL_NAMES[3],
                    title="Thermal stream unreliable",
                    reasons=[f"TRC {trc:.2f} below {self.low_trc:.2f} for {self._low_streak} frames",
                             "fusion is leaning on RGB; detections are less certain"],
                    trc=round(float(trc), 3)))
        else:
            self._low_streak = 0
            self._sensor_open = False

        # behaviour alerts (a flag a track did not have before)
        for t in tracks:
            flags = {f for f in t.get("behaviour", {}).get("anomaly_flags", []) if f in BEHAVIOUR_TEXT}
            seen = self._seen_flags.setdefault(int(t["track_id"]), set())
            for f in sorted(flags - seen):
                new.append(self._emit(
                    kind="behaviour", frame=frame, level=4, label=LEVEL_NAMES[4],
                    title=f"Track #{t['track_id']} ({t.get('cls_name', t.get('cls'))}): {BEHAVIOUR_TEXT[f]}",
                    reasons=[f"behaviour rule '{f}' fired on a confirmed track"],
                    trc=round(float(trc), 3), track_id=int(t["track_id"])))
            seen |= flags
        return new
