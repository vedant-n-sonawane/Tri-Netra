#!/usr/bin/env python3
"""
TriNetra-AMRF — Constant-Velocity Kalman Filter for Box Tracking (Phase 6)
==========================================================================

State is 8-D  ``[cx, cy, a, h, vx, vy, va, vh]``  (box centre, aspect ratio
w/h, height, and their velocities); the measurement is the 4-D
``[cx, cy, a, h]``. Noise scales with the box height so the filter behaves the
same for near and far objects. This is the standard SORT-family motion model.

Pure NumPy/SciPy, no learned parameters.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg


class KalmanFilter:
    def __init__(self, std_weight_position: float = 1.0 / 20,
                 std_weight_velocity: float = 1.0 / 160):
        self._wp = std_weight_position
        self._wv = std_weight_velocity
        self._motion_mat = np.eye(8)
        for i in range(4):
            self._motion_mat[i, 4 + i] = 1.0          # dt = 1 frame
        self._update_mat = np.eye(4, 8)

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create a track state from an unassociated ``[cx, cy, a, h]``."""
        mean = np.r_[measurement, np.zeros(4)]
        h = measurement[3]
        std = [2 * self._wp * h, 2 * self._wp * h, 1e-2, 2 * self._wp * h,
               10 * self._wv * h, 10 * self._wv * h, 1e-5, 10 * self._wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std_pos = [self._wp * h, self._wp * h, 1e-2, self._wp * h]
        std_vel = [self._wv * h, self._wv * h, 1e-5, self._wv * h]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        cov = self._motion_mat @ cov @ self._motion_mat.T + motion_cov
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._wp * h, self._wp * h, 1e-1, self._wp * h]
        innovation_cov = np.diag(np.square(std))
        return (self._update_mat @ mean,
                self._update_mat @ cov @ self._update_mat.T + innovation_cov)

    def update(self, mean: np.ndarray, cov: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov)
        chol, lower = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        gain = scipy.linalg.cho_solve(
            (chol, lower), (cov @ self._update_mat.T).T, check_finite=False).T
        new_mean = mean + (measurement - proj_mean) @ gain.T
        new_cov = cov - gain @ proj_cov @ gain.T
        return new_mean, new_cov
