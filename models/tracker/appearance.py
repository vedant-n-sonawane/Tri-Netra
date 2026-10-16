#!/usr/bin/env python3
"""
TriNetra-AMRF — Appearance Embedder for Re-Identification (Phase 6 v1.5)
==========================================================================

Wraps ``deep_sort_realtime``'s pretrained MobileNetv2 person-appearance
embedder (the same model your DeepSORT baseline already ships with, normally
switched off for the motion-only comparison) so ByteTracker can use it too.

No training happens here on the base weights: the network was pretrained by its
authors on public person re-identification data. M3FD has no identity labels
(no "this crop and that crop are the same person" annotations), so training a
new embedder from scratch was out of scope.

``utils/train_reid_embedder.py`` adds a v1.5 self-supervised fine-tune
(tracklets as pseudo-identities, no manual labels) on top of the stock
weights — pass its output path explicitly via ``weights_path`` to use it. It
is NOT used automatically: measured on the real-clip stress test, after
re-tuning its own weight/threshold fairly, it was a statistical tie with the
stock weights (IDF1 0.809 vs 0.811, IDSW 4.33 vs 3.83 — within noise at n=6),
not an improvement. Tracklet-derived pseudo-identities from ~5 short clips
(63 identities, same handful of real people, narrow appearance variation per
identity) likely aren't diverse enough to beat a model already pretrained on
thousands of real, varied identities. Kept available for experimentation, not
promoted to the default — see utils/train_reid_embedder.py's docstring.

Usage:
    embedder = AppearanceEmbedder()
    feats = embedder.extract(frame_bgr_uint8, boxes_xyxy)   # [n, D], L2-normalized
"""

from __future__ import annotations

from pathlib import Path

import warnings
from typing import Optional

import numpy as np


class AppearanceEmbedder:
    """Crops boxes out of a frame and returns an L2-normalized appearance vector per box."""

    def __init__(self, gpu: Optional[bool] = None, weights_path: Optional[str] = None):
        """
        Args:
            weights_path: Path to alternative MobileNetv2 weights, e.g. the
                self-supervised fine-tune from ``utils/train_reid_embedder.py``
                (``weights/tracker/person_reid_finetuned.pt``). ``None``
                (default) uses the stock pretrained weights — see this
                module's docstring for why the fine-tune isn't the default.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder
            import torch
            use_gpu = torch.cuda.is_available() if gpu is None else gpu
            self._model = MobileNetv2_Embedder(
                model_wts_path=weights_path, bgr=True, gpu=use_gpu)

    def extract(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """``frame_bgr``: HxWx3 uint8. ``boxes_xyxy``: [n,4] pixel coords (clipped to the frame).

        Returns [n, D] float32, each row L2-normalized (zero rows for degenerate boxes).
        """
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, 1280), dtype=np.float32)
        h, w = frame_bgr.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((2, 2, 3), dtype=np.uint8))
            else:
                crops.append(frame_bgr[y1:y2, x1:x2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.asarray(self._model.predict(crops), dtype=np.float32)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return np.divide(out, norm, out=np.zeros_like(out), where=norm > 1e-8)
