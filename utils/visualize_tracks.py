#!/usr/bin/env python3
"""
TriNetra-AMRF — Track Visualization (Phase 6 QA)
================================================

Draws tracker output on a frame for eyeballing ID stability: a persistent
colour per track ID, the box with ``ID / class / confidence``, a trailing path,
a per-track mean-TRC badge, and any behaviour anomaly flags in red. A small HUD
shows the frame number, the frame's TRC, and the live track count.

Usage:
    viz = TrackVisualizer()
    frame = viz.draw(frame_bgr, tracks, frame_idx=i, frame_trc=0.74)
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional

import cv2
import numpy as np


def _color_for(track_id: int):
    """Deterministic, well-separated BGR colour per ID (golden-ratio hue walk)."""
    hue = int((track_id * 0.61803398875 % 1.0) * 179)
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class TrackVisualizer:
    def __init__(self, max_trail: int = 60, thickness: Optional[int] = None):
        self.max_trail = int(max_trail)
        self.thickness = thickness
        self._trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_trail))

    def reset(self) -> None:
        self._trails.clear()

    def draw(self, frame: np.ndarray, tracks: List[dict], frame_idx: Optional[int] = None,
             frame_trc: Optional[float] = None, extra_text: Optional[str] = None) -> np.ndarray:
        img = frame.copy()
        h, w = img.shape[:2]
        scale = max(w, h) / 1280.0
        th = self.thickness or max(1, int(round(2 * scale)))
        fs = 0.55 * scale + 0.15

        for t in tracks:
            tid = t["track_id"]
            x1, y1, x2, y2 = (int(round(v)) for v in t["bbox_xyxy"])
            col = _color_for(tid)
            self._trails[tid].append((int((x1 + x2) / 2), int(y2)))   # foot point

            trail = list(self._trails[tid])
            for a, b in zip(trail[:-1], trail[1:]):
                cv2.line(img, a, b, col, max(1, th - 1), cv2.LINE_AA)

            cv2.rectangle(img, (x1, y1), (x2, y2), col, th)

            name = t.get("cls_name") or f"c{t['cls']}"
            label = f"#{tid} {name} {t['conf']:.2f}"
            beh = t.get("behaviour") or {}
            flags = beh.get("anomaly_flags", [])
            alert = [f for f in flags if f != "low_trc_track"]

            (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.rectangle(img, (x1, max(0, y1 - tht - 6)), (x1 + tw + 6, y1), col, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 0, 0), 1, cv2.LINE_AA)

            # per-track TRC badge (mean reliability of the evidence behind this track)
            if "mean_trc" in beh:
                trc = beh["mean_trc"]
                badge = f"TRC {trc:.2f}"
                bcol = (60, 200, 60) if trc >= 0.7 else (0, 200, 255) if trc >= 0.5 else (60, 60, 255)
                cv2.putText(img, badge, (x1 + 3, y2 + int(16 * scale) + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 0.9, bcol, 1, cv2.LINE_AA)

            if alert:
                cv2.putText(img, " ".join(a.upper() for a in alert),
                            (x1 + 3, y2 + int(34 * scale) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (40, 40, 255), max(1, th - 1), cv2.LINE_AA)

        hud = []
        if frame_idx is not None:
            hud.append(f"frame {frame_idx}")
        if frame_trc is not None:
            hud.append(f"frame TRC {frame_trc:.2f}")
        hud.append(f"tracks {len(tracks)}")
        if extra_text:
            hud.append(extra_text)
        cv2.rectangle(img, (0, 0), (int(12 + 12 * len("   ".join(hud)) * scale * 0.62), int(30 * scale) + 4),
                      (0, 0, 0), -1)
        cv2.putText(img, "   ".join(hud), (8, int(22 * scale) + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), 1, cv2.LINE_AA)
        return img
