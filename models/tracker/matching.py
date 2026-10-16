#!/usr/bin/env python3
"""
TriNetra-AMRF — Association Helpers (Phase 6)
=============================================

IoU cost matrix and Hungarian assignment with a cost ceiling, shared by
ByteTrack's association stages. Boxes are ``xyxy`` in pixels.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``a`` [N,4] and ``b`` [M,4] (xyxy) -> [N,M]."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def iou_distance(track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    """``1 - IoU`` cost matrix [n_tracks, n_dets]."""
    return 1.0 - iou_matrix(track_boxes, det_boxes)


def linear_assignment(
    cost: np.ndarray, thresh: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian matching that never accepts a pair with ``cost > thresh``.

    Over-threshold entries are priced prohibitively so the solver prefers any
    valid alternative, then dropped from the result.

    Returns:
        matches:            list of (row, col)
        unmatched_rows:     rows with no accepted pair
        unmatched_cols:     columns with no accepted pair
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    big = 1e6
    priced = np.where(cost > thresh, big, cost)
    rows, cols = linear_sum_assignment(priced)

    matches, used_r, used_c = [], set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= thresh:
            matches.append((int(r), int(c)))
            used_r.add(r)
            used_c.add(c)
    unmatched_rows = [r for r in range(n_rows) if r not in used_r]
    unmatched_cols = [c for c in range(n_cols) if c not in used_c]
    return matches, unmatched_rows, unmatched_cols
