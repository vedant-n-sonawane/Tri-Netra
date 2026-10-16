#!/usr/bin/env python3
"""
TriNetra-AMRF — ByteTrack with a TRC-Gated Second Stage (Phase 6 v1 core)
=========================================================================

ByteTrack associates *every* detection, including low-confidence ones, using
only motion (Kalman prediction + IoU) — no appearance network, no training.
TriNetra's addition is the TRC gate (``trc_gate.py``): the stage-2 cost for
low-confidence boxes is inflated when the frame's Thermal Reliability
Confidence is low, so an unreliable box must agree more tightly with the motion
prediction before it is accepted.

Optional appearance re-identification (v1.5, ``tracking.reid.enabled``): when
each detection is given an appearance embedding (``update(..., embeds=...)``),
stage-1 and stage-3 costs blend IoU with cosine distance to the track's
running-average embedding (``reid.weight``, 0 = IoU only). The embedder
itself is not trained here (no identity-labelled data exists for this
project) — see ``models/tracker/appearance.py``.

Root-cause-driven design (measured on the synthetic benchmark with exact
ground truth, ``utils/eval_synthetic.py``): ID switches split roughly into two
unrelated failure modes, and a single blended IoU+appearance stage only fixes
one of them.
  (a) ~46%: a track is correctly lost during occlusion, survives the
      ``max_age`` grace window, but the object doesn't reappear for a median
      of ~2x that window — so it's destroyed before it can be reclaimed.
      Raising ``max_age`` alone does *not* fix this: the longer a track
      coasts on its Kalman prediction with no new evidence, the further that
      prediction drifts, so by the time the object is back, IoU is near zero
      regardless of how long the track was kept alive — and the blended
      stage-1 cost ceiling (tuned for IoU-plausible matches) rejects a
      near-zero-IoU pairing even when appearance says it's the same object.
      Stage 1.5 below is the fix: a *separate*, appearance-only pass for
      tracks lost longer than ``reid.reacquire_gap`` frames, with its own
      threshold, that never requires IoU at all.
  (b) ~54%: two simultaneously-visible, continuously-tracked objects pass
      within ~30px of each other and swap identities on a single frame. This
      is a live disambiguation problem, not a re-acquisition problem, and
      stage 1.5 does not address it (future work: adaptive, proximity-aware
      appearance weighting in stage 1 itself).

Per frame:
    1. PREDICT   every track with the Kalman filter
    2. SPLIT     detections into high / low confidence (below ``low`` dropped)
    3. STAGE 1   high dets  <-> confirmed + lost tracks      (IoU [+appearance])
    4. STAGE 1.5 high dets  <-> long-lost tracks             (appearance only)
    5. STAGE 2   low dets   <-> still-unmatched confirmed    (IoU, TRC-gated)
    6. STAGE 3   remaining high dets <-> tentative tracks
    7. LIFECYCLE new tracks from unmatched high dets; unmatched tentative
                 removed; unmatched confirmed -> lost; lost older than
                 ``max_age`` removed
    8. BEHAVIOUR history buffers updated and behaviour features recomputed

Input per frame: ``[n, 6]`` array/tensor ``(x1, y1, x2, y2, score, cls)`` plus
the frame's TRC (scalar, or one value per detection).

Config: ``tracking`` block of configs/default.yaml (see ``ByteTracker.__init__``).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .kalman_filter import KalmanFilter
from .matching import iou_distance, linear_assignment
from .track import CONFIRMED, LOST, TENTATIVE, Track
from .trc_gate import gate_cost_matrix, sanitize_trc


def _to_numpy(dets) -> np.ndarray:
    if dets is None:
        return np.zeros((0, 6))
    if hasattr(dets, "detach"):
        dets = dets.detach().cpu().numpy()
    dets = np.asarray(dets, dtype=np.float64)
    return dets.reshape(-1, 6) if dets.size else np.zeros((0, 6))


def _appearance_distance(track_feats, det_feats) -> np.ndarray:
    """1 - cosine similarity, [n_tracks, n_dets]. A track with no feat yet (feat=None)
    gets the neutral distance 0.5 against every detection (appearance abstains, falls
    back to whatever IoU says) rather than forcing a bad match."""
    n_t, n_d = len(track_feats), len(det_feats)
    out = np.full((n_t, n_d), 0.5)
    if n_t == 0 or n_d == 0:
        return out
    det = np.stack(det_feats)
    for i, f in enumerate(track_feats):
        if f is not None:
            out[i] = 1.0 - det @ f
    return np.clip(out, 0.0, 2.0)


class ByteTracker:
    def __init__(self, cfg: dict, behaviour_fn: Optional[Callable] = None,
                 class_names: Optional[Dict[int, str]] = None):
        t = cfg.get("tracking", {}) if "tracking" in cfg else cfg
        m = t.get("matching", {})
        life = t.get("lifecycle", {})
        gate = t.get("trc_gate", {})

        self.high_thresh = float(m.get("high_conf_threshold", 0.6))
        self.low_thresh = float(m.get("low_conf_threshold", 0.1))
        self.iou_stage1 = float(m.get("iou_threshold_stage1", 0.3))
        self.iou_stage2 = float(m.get("iou_threshold_stage2", 0.5))
        self.class_aware = bool(m.get("class_aware", False))

        self.gate_enabled = bool(gate.get("enabled", True))
        self.beta = float(gate.get("beta", 0.5))

        reid = t.get("reid", {})
        self.use_appearance = bool(reid.get("enabled", False))
        self.reid_weight = float(reid.get("weight", 0.5))     # 0 = IoU only, 1 = appearance only
        self.reid_alpha = float(reid.get("ema_alpha", 0.9))   # track-embedding smoothing
        # Long-gap re-acquisition (root-cause fix, see module docstring): after
        # this many missed frames a lost track's Kalman prediction has drifted
        # too far for IoU to ever match it again, so it is matched on
        # appearance alone instead, with its own (tighter) threshold — not
        # blended with (by-then-meaningless) IoU. 0 disables this stage.
        self.reacquire_gap = int(reid.get("reacquire_gap", 15))
        self.reacquire_thresh = float(reid.get("reacquire_thresh", 0.3))
        # The pretrained embedder (models/tracker/appearance.py) is a *person*
        # re-identification model; applying it to vehicle crops is out-of-domain
        # noise, not signal (measured: it roughly tripled ID switches on real
        # multi-class footage). Restrict appearance blending to these classes;
        # every other class stays IoU-only. Default: person (unified id 1).
        self.reid_classes = set(reid.get("classes", [1]))
        # Crowding fix (root cause B, see module docstring): ~54% of switches
        # are two simultaneously-visible, continuously-tracked objects passing
        # close together, with IoU genuinely unable to tell them apart (the
        # correct and the wrong detection have near-identical cost). Boosting
        # the appearance weight everywhere to fix this was already tried and
        # made easy, unambiguous matches worse elsewhere (see `reid.weight`'s
        # history). Instead, weight is raised only per *track* and only when
        # that track actually has a close competitor this frame: the margin
        # between its best and second-best IoU cost is below
        # `ambiguity_margin` -> weight ramps from `weight` up to
        # `weight_ambiguous`; no competitor (the common case) -> unchanged.
        self.weight_ambiguous = float(reid.get("weight_ambiguous", self.reid_weight))
        self.ambiguity_margin = float(reid.get("ambiguity_margin", 0.0))

        # Second attempt at root cause B, independent of appearance/the
        # embedder entirely: does the direction from a track's last position
        # to this candidate detection match the direction the track has
        # actually been moving in? Two crossing/parallel objects can have
        # near-identical IoU at the moment of closest approach, but a wrong
        # (swapped) assignment usually implies an implausible sudden turn,
        # which this catches regardless of what either object looks like —
        # so, unlike appearance, it works for every class, not just person.
        head = t.get("heading", {})
        self.use_heading = bool(head.get("enabled", False))
        self.heading_weight = float(head.get("weight", 0.5))   # max contribution when ambiguous
        self.heading_margin = float(head.get("margin", 0.0))   # same ambiguity-gating idea as reid
        self.heading_min_hist = int(head.get("min_hist", 3))   # matched observations before trusted
        self.heading_window = int(head.get("window", 5))       # frames spanned by the heading estimate
        self.heading_min_disp_frac = float(head.get("min_disp_frac", 0.05))  # min net displacement
                                                                 # over `window`, as a fraction of box
                                                                 # height, before a heading is trusted
                                                                 # at all (excludes loitering/near-
                                                                 # stationary objects, whose apparent
                                                                 # direction is mostly position noise)

        self.n_init = int(life.get("n_init", 3))
        self.max_age = int(life.get("max_age", 30))

        self._bcfg = dict(t.get("behaviour", {}))
        self.history_len = int(self._bcfg.get("history_len", 300))
        self.class_names = class_names or {}
        if behaviour_fn is None and self._bcfg.get("enabled", True):
            from utils.behaviour import compute_behaviour
            behaviour_fn = compute_behaviour
        self._behaviour_fn = behaviour_fn

        self.kf = KalmanFilter()
        self.reset()

    # ── state ────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self.tracks: List[Track] = []
        self.frame_id = 0
        self._next_id = 1
        self.removed_count = 0

    def _new_track(self, det: np.ndarray, trc: float, feat=None) -> Track:
        tr = Track(self._next_id, det[:4], det[4], int(det[5]), self.frame_id,
                   trc, self.kf, n_init=self.n_init, history_len=self.history_len, feat=feat)
        self._next_id += 1
        return tr

    @staticmethod
    def _ambiguity_fraction(iou_cost: np.ndarray, margin: float) -> np.ndarray:
        """Per-row ``[0,1]``: 0 = this track has a close IoU competitor this frame
        (ambiguous), 1 = a single clear best candidate (or none at all)."""
        n_t, n_d = iou_cost.shape
        if n_d < 2 or margin <= 0:
            return np.ones(n_t)
        s = np.sort(iou_cost, axis=1)
        best, second = s[:, 0], s[:, 1]
        has_competitor = second < 0.999        # second-best candidate overlaps at all
        frac = np.clip(second - best, 0.0, margin) / margin
        return np.where(has_competitor, frac, 1.0)

    def _ambiguity_weight(self, iou_cost: np.ndarray) -> np.ndarray:
        """Per-track appearance weight: ``reid_weight`` normally, ramping up to
        ``weight_ambiguous`` when this track's best and second-best IoU
        candidate this frame are nearly tied (a close competitor — the
        crowding scenario). A track with a single clear candidate, or no
        competitor at all, is left at the base weight."""
        frac = self._ambiguity_fraction(iou_cost, self.ambiguity_margin)
        return self.weight_ambiguous - frac * (self.weight_ambiguous - self.reid_weight)

    def _heading_vec(self, track: Track) -> Optional[np.ndarray]:
        hist = track.history
        if len(hist) < self.heading_min_hist:
            return None
        w = min(self.heading_window, len(hist) - 1)
        if w < 1:
            return None
        a, b = hist[-1 - w], hist[-1]
        # Heading quality and match quality both degrade from the same cause
        # (missed detections — low TRC, occlusion): a track that was missed
        # partway through this window has a sparser, noisier position history,
        # and its IoU match is *already* the most fragile right when that
        # happens. Trusting heading there was measured to reject otherwise-
        # correct matches (an already-marginal IoU cost pushed over the
        # acceptance threshold). Require a gap-free run of matched frames.
        if b["frame_id"] - a["frame_id"] != w:
            return None
        dx, dy = b["cx"] - a["cx"], b["cy"] - a["cy"]
        norm = float(np.hypot(dx, dy))
        # A near-stationary or loitering object's apparent direction over a few
        # frames is dominated by detection/position noise, not true travel
        # intent. Require real net displacement, scaled by the track's own
        # box height so this works at any object size / distance from camera.
        box_h = max(b["bbox"][3] - b["bbox"][1], 1.0)
        return None if norm < self.heading_min_disp_frac * box_h else np.array([dx / norm, dy / norm])

    def _heading_cost(self, tracks: List[Track], dets: np.ndarray):
        """``(1 - cos(angle between track heading and track->detection direction)) / 2``,
        in ``[0, 1]``; 0 = detection lies in the direction the track has been moving,
        1 = directly behind it (an implausible sudden reversal).

        Returns ``(cost, available)``: ``available[i]`` is False when track ``i`` has no
        reliable heading yet (new, or recently stationary) — those rows are neutral
        0.5 *and* the caller must weight them 0, the same way appearance abstains for a
        track with no embedding yet, so a track with no opinion can't still drag a
        good IoU/appearance match toward an uninformative 0.5.
        """
        n_t, n_d = len(tracks), len(dets)
        out = np.full((n_t, n_d), 0.5)
        available = np.zeros(n_t, dtype=bool)
        if n_t == 0 or n_d == 0:
            return out, available
        cx = (dets[:, 0] + dets[:, 2]) / 2.0
        cy = (dets[:, 1] + dets[:, 3]) / 2.0
        for i, tr in enumerate(tracks):
            hv = self._heading_vec(tr)
            if hv is None:
                continue
            last = tr.history[-1]
            dx, dy = cx - last["cx"], cy - last["cy"]
            norm = np.hypot(dx, dy)
            valid = norm > 1e-6
            cos = np.zeros(n_d)
            cos[valid] = (dx[valid] * hv[0] + dy[valid] * hv[1]) / norm[valid]
            out[i] = np.where(valid, (1.0 - cos) / 2.0, 0.5)
            available[i] = True
        return out, available

    def _cost(self, tracks: List[Track], dets: np.ndarray, embeds=None) -> np.ndarray:
        if not tracks or len(dets) == 0:
            return np.zeros((len(tracks), len(dets)))
        iou_cost = iou_distance(np.stack([t.xyxy for t in tracks]), dets[:, :4])
        cost = iou_cost
        if self.use_appearance and embeds is not None and len(embeds):
            app = _appearance_distance([t.feat for t in tracks], embeds)
            # Only blend appearance for (track, detection) pairs where both sides
            # are a class the embedder was actually trained for (person, by
            # default) — everything else stays IoU-only.
            track_ok = np.array([t.class_id in self.reid_classes for t in tracks])[:, None]
            det_ok = np.array([int(c) in self.reid_classes for c in dets[:, 5]])[None, :]
            allowed = track_ok & det_ok
            w = self._ambiguity_weight(iou_cost)[:, None]
            blended = (1 - w) * cost + w * app
            cost = np.where(allowed, blended, cost)
        if self.use_heading:
            # Ambiguity computed from the ORIGINAL IoU cost (not the
            # appearance-blended one above), so this is an independent
            # disambiguator, not a secondary correction of the first.
            head, head_available = self._heading_cost(tracks, dets)
            wh = self.heading_weight * (1.0 - self._ambiguity_fraction(iou_cost, self.heading_margin))
            wh = np.where(head_available, wh, 0.0)[:, None]
            cost = (1 - wh) * cost + wh * head
        if self.class_aware:
            t_cls = np.array([t.class_id for t in tracks])[:, None]
            d_cls = dets[:, 5].astype(int)[None, :]
            cost = np.where(t_cls != d_cls, 1e3, cost)
        return cost

    # ── main entry ───────────────────────────────────────────────────────────
    def update(self, detections, trc=1.0, frame_id: Optional[int] = None,
               frame=None, embeds=None) -> List[dict]:
        """Advance one frame; return one dict per confirmed, currently matched track.

        ``frame`` is accepted for signature parity with the appearance-based
        tracker and ignored. ``embeds``: optional ``[n, D]`` appearance vector
        per row of ``detections`` (same order), used for re-identification
        when ``tracking.reid.enabled``; ``None`` keeps ByteTrack motion-only.
        """
        self.frame_id = self.frame_id + 1 if frame_id is None else int(frame_id)

        dets = _to_numpy(detections)
        det_trc = sanitize_trc(trc, len(dets))
        embeds = np.asarray(embeds) if embeds is not None and len(embeds) else None
        if embeds is not None and len(embeds) != len(dets):
            embeds = None  # malformed input: fall back to motion-only rather than crash
        keep = dets[:, 4] >= self.low_thresh
        dets, det_trc = dets[keep], det_trc[keep]
        embeds_k = embeds[keep] if embeds is not None else None

        is_high = dets[:, 4] >= self.high_thresh
        high, high_trc = dets[is_high], det_trc[is_high]
        low, low_trc = dets[~is_high], det_trc[~is_high]
        high_emb = embeds_k[is_high] if embeds_k is not None else None
        low_emb = embeds_k[~is_high] if embeds_k is not None else None

        for t in self.tracks:
            t.predict(self.kf)

        confirmed = [t for t in self.tracks if t.state == CONFIRMED]
        tentative = [t for t in self.tracks if t.state == TENTATIVE]
        lost = [t for t in self.tracks if t.state == LOST]
        pool = confirmed + lost

        # stage 1 — high-confidence detections vs confirmed + lost tracks.
        # This is where appearance matters most: a lost track's Kalman
        # prediction may have drifted off the object during occlusion, so IoU
        # alone can fail to re-acquire it even though the object still looks
        # the same.
        matches, u_pool, u_high = linear_assignment(
            self._cost(pool, high, high_emb), 1.0 - self.iou_stage1)
        for ti, di in matches:
            pool[ti].update(high[di, :4], high[di, 4], int(high[di, 5]),
                            self.frame_id, high_trc[di], self.kf,
                            feat=(high_emb[di] if high_emb is not None else None),
                            feat_alpha=self.reid_alpha)

        # stage 1.5 — long-lost tracks vs leftover high-confidence detections,
        # matched on appearance ALONE (no IoU: by this point the Kalman
        # prediction has coasted too long to be meaningful). Only tracks lost
        # for at least ``reacquire_gap`` frames are eligible, so short,
        # IoU-recoverable gaps are left to stage 1 untouched.
        reacquired_pool_idx = set()
        if self.use_appearance and self.reacquire_gap > 0 and u_high:
            eligible = [i for i in u_pool
                       if pool[i].state == LOST
                       and pool[i].time_since_update >= self.reacquire_gap
                       and pool[i].feat is not None
                       and pool[i].class_id in self.reid_classes]
            cand = [i for i in u_high if int(high[i, 5]) in self.reid_classes]
            if eligible and cand:
                cand_emb = high_emb[cand] if high_emb is not None else None
                app = _appearance_distance([pool[i].feat for i in eligible], cand_emb) \
                    if cand_emb is not None else np.ones((len(eligible), len(cand)))
                matches15, _, _ = linear_assignment(app, self.reacquire_thresh)
                for ti, di in matches15:
                    pi, hi = eligible[ti], cand[di]
                    pool[pi].update(high[hi, :4], high[hi, 4], int(high[hi, 5]),
                                    self.frame_id, high_trc[hi], self.kf,
                                    feat=(high_emb[hi] if high_emb is not None else None),
                                    feat_alpha=self.reid_alpha)
                    reacquired_pool_idx.add(pi)
                claimed = {cand[di] for _, di in matches15}
                u_high = [i for i in u_high if i not in claimed]

        # stage 2 — low-confidence detections vs still-unmatched confirmed
        # tracks, association cost gated by TRC (the TriNetra-specific step).
        # The TRC gate is applied on top of the (possibly appearance-blended)
        # cost, so low reliability still demands a tighter match regardless.
        # Tracks reacquired in stage 1.5 are already updated this frame and
        # excluded here even though their state is now CONFIRMED.
        rem_tracks = [pool[i] for i in u_pool
                     if pool[i].state == CONFIRMED and i not in reacquired_pool_idx]
        cost2 = gate_cost_matrix(self._cost(rem_tracks, low, low_emb), low_trc,
                                 self.beta, self.gate_enabled)
        matches2, u_rem, _ = linear_assignment(cost2, 1.0 - self.iou_stage2)
        for ti, di in matches2:
            rem_tracks[ti].update(low[di, :4], low[di, 4], int(low[di, 5]),
                                  self.frame_id, low_trc[di], self.kf,
                                  feat=(low_emb[di] if low_emb is not None else None),
                                  feat_alpha=self.reid_alpha)
        for ti in u_rem:
            rem_tracks[ti].mark_missed()

        # stage 3 — tentative tracks vs leftover high-confidence detections
        left = [high[i] for i in u_high]
        left_trc = [high_trc[i] for i in u_high]
        left_emb = [high_emb[i] for i in u_high] if high_emb is not None else None
        left_arr = np.stack(left) if left else np.zeros((0, 6))
        matches3, u_tent, u_left = linear_assignment(
            self._cost(tentative, left_arr, left_emb), 1.0 - self.iou_stage1)
        for ti, di in matches3:
            tentative[ti].update(left_arr[di, :4], left_arr[di, 4], int(left_arr[di, 5]),
                                 self.frame_id, left_trc[di], self.kf,
                                 feat=(left_emb[di] if left_emb is not None else None),
                                 feat_alpha=self.reid_alpha)
        for ti in u_tent:
            tentative[ti].mark_missed()

        # new tracks from detections nothing claimed
        for di in u_left:
            self.tracks.append(self._new_track(
                left_arr[di], left_trc[di],
                feat=(left_emb[di] if left_emb is not None else None)))

        # lost tracks that outlived the grace window are dropped
        for t in self.tracks:
            if t.state == LOST and t.time_since_update > self.max_age:
                t.mark_removed()
        self.removed_count += sum(1 for t in self.tracks if t.state == "removed")
        self.tracks = [t for t in self.tracks if t.state != "removed"]

        # report + behaviour
        out = []
        for t in self.tracks:
            if t.state != CONFIRMED or t.time_since_update != 0:
                continue
            if self._behaviour_fn is not None:
                t.behaviour = self._behaviour_fn(t, self._bcfg)
            out.append(t.summary(self.class_names.get(t.class_id)))
        return out
