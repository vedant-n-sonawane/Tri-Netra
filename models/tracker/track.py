#!/usr/bin/env python3
"""
TriNetra-AMRF — Track Object and Lifecycle (Phase 6)
====================================================

One tracked object across time: a Kalman state, a rolling observation history
(consumed by ``utils.behaviour``), and the SORT-family lifecycle

    tentative --(n_init matches)--> confirmed --(missed)--> lost
        |                                                    |
     (missed)                       (re-matched)-> confirmed |
        v                                                    v
     removed <----------------(time_since_update > max_age)--+

The history stores one entry per *matched* frame:
``{frame_id, bbox (xyxy), cx, cy, trc, score}``.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from .kalman_filter import KalmanFilter

TENTATIVE, CONFIRMED, LOST, REMOVED = "tentative", "confirmed", "lost", "removed"


def xyxy_to_xyah(box) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / max(h, 1e-6), h])


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = xyah[:4]
    w = a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


class Track:
    def __init__(self, track_id: int, box_xyxy, score: float, cls: int,
                 frame_id: int, trc: float, kf: KalmanFilter,
                 n_init: int = 3, history_len: int = 300, feat=None):
        self.track_id = int(track_id)
        self.n_init = int(n_init)
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(box_xyxy))
        self.state = CONFIRMED if self.n_init <= 1 else TENTATIVE

        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.start_frame = int(frame_id)
        self.last_frame = int(frame_id)

        self.score = float(score)
        self.trc = float(trc)
        self._cls_votes: Dict[int, float] = {int(cls): float(score)}
        self.class_id = int(cls)

        self.history: Deque[dict] = deque(maxlen=int(history_len))
        self.behaviour: dict = {}
        # Appearance re-identification (v1.5): a running average of the
        # matched detections' appearance embeddings, used alongside IoU so a
        # track can be re-acquired after occlusion even when its Kalman
        # prediction has drifted away from the object's new position. None
        # when no embedder is configured (appearance stays opt-in).
        self.feat: Optional[np.ndarray] = (
            feat / max(np.linalg.norm(feat), 1e-8) if feat is not None else None)
        self._push(box_xyxy, frame_id, trc, score)

    # ── geometry ─────────────────────────────────────────────────────────────
    @property
    def xyxy(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def predict(self, kf: KalmanFilter) -> None:
        mean = self.mean.copy()
        if self.state != CONFIRMED:
            mean[7] = 0.0                  # don't extrapolate height growth blindly
        self.mean, self.cov = kf.predict(mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update_feat(self, feat, alpha: float = 0.9) -> None:
        """Exponential moving average of the appearance embedding (DeepSORT-style)."""
        feat = feat / max(np.linalg.norm(feat), 1e-8)
        self.feat = feat if self.feat is None else alpha * self.feat + (1 - alpha) * feat
        self.feat = self.feat / max(np.linalg.norm(self.feat), 1e-8)

    def update(self, box_xyxy, score: float, cls: int, frame_id: int,
               trc: float, kf: KalmanFilter, feat=None, feat_alpha: float = 0.9) -> None:
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_xyah(box_xyxy))
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = int(frame_id)
        self.score = float(score)
        self.trc = float(trc)

        # Score-weighted class vote keeps the label stable when the detector
        # flickers between similar classes (e.g. person / motorcycle riders).
        self._cls_votes[int(cls)] = self._cls_votes.get(int(cls), 0.0) + float(score)
        self.class_id = max(self._cls_votes, key=self._cls_votes.get)

        if self.state == TENTATIVE and self.hits >= self.n_init:
            self.state = CONFIRMED
        elif self.state == LOST:
            self.state = CONFIRMED
        if feat is not None:
            self.update_feat(feat, feat_alpha)
        self._push(box_xyxy, frame_id, trc, score)

    def mark_missed(self) -> None:
        if self.state == TENTATIVE:
            self.state = REMOVED
        elif self.state == CONFIRMED:
            self.state = LOST

    def mark_removed(self) -> None:
        self.state = REMOVED

    # ── history ──────────────────────────────────────────────────────────────
    def _push(self, box_xyxy, frame_id: int, trc: float, score: float) -> None:
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        self.history.append({
            "frame_id": int(frame_id),
            "bbox": [x1, y1, x2, y2],
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "trc": float(trc),
            "score": float(score),
        })

    def summary(self, class_name: Optional[str] = None) -> dict:
        return {
            "track_id": self.track_id,
            "state": self.state,
            "bbox_xyxy": [float(v) for v in self.xyxy],
            "cls": self.class_id,
            "cls_name": class_name,
            "conf": self.score,
            "trc": self.trc,
            "age": self.age,
            "hits": self.hits,
            "behaviour": dict(self.behaviour),
        }
