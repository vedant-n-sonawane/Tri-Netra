#!/usr/bin/env python3
"""
TriNetra-AMRF — TRC-Gated Association Cost (Phase 6, the novel piece)
=====================================================================

Stock ByteTrack matches low-confidence detections with a fixed, generous IoU
threshold because low confidence usually just means "partly occluded". In a
dual-modality system a low-confidence box can also mean "the sensor feeding
this detection was unreliable" (low Thermal Reliability Confidence, TRC), in
which case the box position itself is shakier and should be trusted less.

The gate inflates the stage-2 association cost by the detection's TRC:

    cost' = cost * (1 + beta * (1 - trc))

  trc = 1  -> cost unchanged (behaves exactly like ByteTrack)
  trc -> 0 -> cost up to (1 + beta)x, so the match is only accepted when the
              motion prediction agrees much more tightly

Missing or non-finite TRC is treated as 1.0 (full trust), so the tracker still
works, without the adaptive benefit, when TRC is not wired through upstream.
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def sanitize_trc(trc: ArrayLike, n: int) -> np.ndarray:
    """Return a length-``n`` TRC vector in [0, 1]; scalar broadcast, bad -> 1.0."""
    if trc is None:
        return np.ones(n)
    arr = np.asarray(trc, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.full(n, float(arr[0]))
    if arr.size != n:
        return np.ones(n)
    arr = np.where(np.isfinite(arr), arr, 1.0)
    return np.clip(arr, 0.0, 1.0)


def trc_adjusted_cost(iou_cost: ArrayLike, trc: ArrayLike, beta: float = 0.5) -> ArrayLike:
    """``cost * (1 + beta * (1 - trc))`` (scalar or broadcastable arrays)."""
    return np.asarray(iou_cost) * (1.0 + beta * (1.0 - np.asarray(trc)))


def gate_cost_matrix(cost: np.ndarray, det_trc: np.ndarray, beta: float = 0.5,
                     enabled: bool = True) -> np.ndarray:
    """Apply the gate column-wise: column ``j`` (detection ``j``) uses ``det_trc[j]``."""
    if not enabled or cost.size == 0:
        return cost
    return trc_adjusted_cost(cost, det_trc[None, :], beta)
