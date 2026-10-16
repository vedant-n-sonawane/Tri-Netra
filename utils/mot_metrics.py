#!/usr/bin/env python3
"""
TriNetra-AMRF — Multi-Object Tracking Metrics (Phase 6 validation)
==================================================================

Self-contained implementations of the standard tracking metrics, so results in
the paper come from code that is inspectable and unit-tested
(``utils/check_mot_metrics.py``), not from an opaque dependency:

  CLEAR-MOT  (Bernardin & Stiefelhagen 2008): MOTA, MOTP, FP, FN, IDSW,
             MT / PT / ML (mostly tracked / partially / mostly lost), Frag
  Identity   (Ristani et al. 2016): IDF1, IDP, IDR via a global optimal
             one-to-one assignment of ground-truth and predicted trajectories
  HOTA       (Luiten et al. 2021): HOTA, DetA, AssA, LocA averaged over
             IoU thresholds 0.05..0.95, following the TrackEval reference
             algorithm

Input format: ``{frame_index: ndarray [n, 5]}`` with columns
``(track_id, x1, y1, x2, y2)``; frames absent from a dict have no boxes.
Matching is class-agnostic. IoU threshold for CLEAR / IDF1 is 0.5.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

Tracks = Dict[int, np.ndarray]
_EPS = np.finfo(float).eps


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _frames(gt: Tracks, pred: Tracks, frames=None):
    if frames is not None:
        return sorted(frames)
    return sorted(set(gt) | set(pred))


def _get(d: Tracks, f: int) -> np.ndarray:
    a = d.get(f)
    if a is None or len(a) == 0:
        return np.zeros((0, 5))
    return np.atleast_2d(np.asarray(a, dtype=np.float64))[:, :5]      # extra columns (e.g. class) ignored


# ── CLEAR-MOT ────────────────────────────────────────────────────────────────
def clear_mot(gt: Tracks, pred: Tracks, iou_thr: float = 0.5, frames=None) -> dict:
    fp = fn = idsw = 0
    n_gt = 0
    iou_sum, n_match = 0.0, 0
    last_hyp: Dict[int, int] = {}          # gt id -> hyp id it was last matched to
    prev_map: Dict[int, int] = {}          # correspondences carried over from the last frame
    gt_len: Dict[int, int] = {}
    gt_tracked: Dict[int, int] = {}
    gt_state: Dict[int, bool] = {}         # was the gt matched in its previous frame?
    frag = 0

    for f in _frames(gt, pred, frames):
        g, h = _get(gt, f), _get(pred, f)
        n_gt += len(g)
        iou = _iou(g[:, 1:], h[:, 1:])
        g_ids, h_ids = g[:, 0].astype(int), h[:, 0].astype(int)
        for gi in g_ids:
            gt_len[gi] = gt_len.get(gi, 0) + 1

        matched_g, matched_h, pairs = set(), set(), []
        # 1) keep previous correspondences that still hold
        for i, gi in enumerate(g_ids):
            hj_id = prev_map.get(gi)
            if hj_id is None:
                continue
            js = np.where(h_ids == hj_id)[0]
            if len(js) and iou[i, js[0]] >= iou_thr:
                pairs.append((i, js[0])); matched_g.add(i); matched_h.add(js[0])
        # 2) optimal assignment of the rest
        rem_g = [i for i in range(len(g)) if i not in matched_g]
        rem_h = [j for j in range(len(h)) if j not in matched_h]
        if rem_g and rem_h:
            sub = iou[np.ix_(rem_g, rem_h)]
            cost = np.where(sub >= iou_thr, 1.0 - sub, 1e6)
            r, c = linear_sum_assignment(cost)
            for ri, ci in zip(r, c):
                if sub[ri, ci] >= iou_thr:
                    pairs.append((rem_g[ri], rem_h[ci])); matched_g.add(rem_g[ri]); matched_h.add(rem_h[ci])

        new_map = {}
        for i, j in pairs:
            gi, hj = g_ids[i], h_ids[j]
            if gi in last_hyp and last_hyp[gi] != hj:
                idsw += 1
            last_hyp[gi] = hj
            new_map[gi] = hj
            iou_sum += iou[i, j]; n_match += 1
            gt_tracked[gi] = gt_tracked.get(gi, 0) + 1
        prev_map = new_map
        fn += len(g) - len(pairs)
        fp += len(h) - len(pairs)

        matched_now = {g_ids[i] for i, _ in pairs}
        for gi in g_ids:
            was = gt_state.get(gi)
            now = gi in matched_now
            if was is False and now:        # interrupted, then picked up again
                frag += 1
            gt_state[gi] = now

    mt = pt = ml = 0
    for gi, L in gt_len.items():
        ratio = gt_tracked.get(gi, 0) / max(L, 1)
        mt += ratio >= 0.8
        ml += ratio <= 0.2
        pt += 0.2 < ratio < 0.8
    return {
        "MOTA": 1.0 - (fn + fp + idsw) / max(n_gt, 1),
        "MOTP": iou_sum / max(n_match, 1),
        "FP": fp, "FN": fn, "IDSW": idsw, "MT": int(mt), "PT": int(pt), "ML": int(ml),
        "Frag": frag, "num_gt_dets": n_gt, "num_gt_tracks": len(gt_len),
    }


# ── Identity metrics ─────────────────────────────────────────────────────────
def identity_metrics(gt: Tracks, pred: Tracks, iou_thr: float = 0.5, frames=None) -> dict:
    gt_ids_all, pr_ids_all = {}, {}
    n_gt = n_pr = 0
    per_frame = []
    for f in _frames(gt, pred, frames):
        g, h = _get(gt, f), _get(pred, f)
        n_gt += len(g); n_pr += len(h)
        for i in g[:, 0].astype(int):
            gt_ids_all.setdefault(i, len(gt_ids_all))
        for j in h[:, 0].astype(int):
            pr_ids_all.setdefault(j, len(pr_ids_all))
        per_frame.append((g, h))
    tp = np.zeros((len(gt_ids_all), len(pr_ids_all)))
    for g, h in per_frame:
        if len(g) == 0 or len(h) == 0:
            continue
        iou = _iou(g[:, 1:], h[:, 1:])
        gi = np.array([gt_ids_all[int(x)] for x in g[:, 0]])
        hj = np.array([pr_ids_all[int(x)] for x in h[:, 0]])
        ii, jj = np.where(iou >= iou_thr)
        np.add.at(tp, (gi[ii], hj[jj]), 1)
    idtp = 0.0
    if tp.size:
        r, c = linear_sum_assignment(-tp)
        idtp = float(tp[r, c].sum())
    idfn, idfp = n_gt - idtp, n_pr - idtp
    return {
        "IDF1": 2 * idtp / max(2 * idtp + idfp + idfn, 1),
        "IDP": idtp / max(idtp + idfp, 1), "IDR": idtp / max(idtp + idfn, 1),
        "IDTP": int(idtp), "IDFP": int(idfp), "IDFN": int(idfn), "num_pred_dets": n_pr,
        "num_pred_tracks": len(pr_ids_all),
    }


# ── HOTA (TrackEval reference algorithm) ─────────────────────────────────────
def hota(gt: Tracks, pred: Tracks, frames=None) -> dict:
    alphas = np.arange(0.05, 0.99, 0.05)
    gt_map, pr_map = {}, {}
    seq = []
    for f in _frames(gt, pred, frames):
        g, h = _get(gt, f), _get(pred, f)
        gi = np.array([gt_map.setdefault(int(x), len(gt_map)) for x in g[:, 0]], dtype=int)
        hj = np.array([pr_map.setdefault(int(x), len(pr_map)) for x in h[:, 0]], dtype=int)
        seq.append((gi, hj, _iou(g[:, 1:], h[:, 1:])))
    n_g, n_p = len(gt_map), len(pr_map)
    pot = np.zeros((n_g, n_p)); gt_cnt = np.zeros((n_g, 1)); pr_cnt = np.zeros((1, n_p))

    for gi, hj, sim in seq:
        if len(gi) and len(hj):
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.where(denom > _EPS, sim / np.maximum(denom, _EPS), 0.0)
            pot[gi[:, None], hj[None, :]] += sim_iou
        gt_cnt[gi, 0] += 1
        pr_cnt[0, hj] += 1
    glob = pot / np.maximum(gt_cnt + pr_cnt - pot, _EPS)

    A = len(alphas)
    tp = np.zeros(A); fn = np.zeros(A); fp = np.zeros(A); loc = np.zeros(A)
    mcount = [np.zeros((n_g, n_p)) for _ in range(A)]
    for gi, hj, sim in seq:
        if len(gi) == 0:
            fp += len(hj); continue
        if len(hj) == 0:
            fn += len(gi); continue
        score = glob[gi[:, None], hj[None, :]] * sim
        r, c = linear_sum_assignment(-score)
        for a, alpha in enumerate(alphas):
            ok = sim[r, c] >= alpha - _EPS
            rr, cc = r[ok], c[ok]
            tp[a] += len(rr); fn[a] += len(gi) - len(rr); fp[a] += len(hj) - len(rr)
            if len(rr):
                loc[a] += sim[rr, cc].sum()
                mcount[a][gi[rr], hj[cc]] += 1

    ass_a = np.zeros(A)
    for a in range(A):
        m = mcount[a]
        ass_a[a] = np.sum(m * (m / np.maximum(1, gt_cnt + pr_cnt - m))) / max(1.0, tp[a])
    det_a = tp / np.maximum(1, tp + fn + fp)
    loc_a = np.maximum(1e-10, loc) / np.maximum(1e-10, tp)
    hota_a = np.sqrt(det_a * ass_a)
    return {"HOTA": float(hota_a.mean()), "DetA": float(det_a.mean()),
            "AssA": float(ass_a.mean()), "LocA": float(loc_a.mean())}


def evaluate_mot(gt: Tracks, pred: Tracks, iou_thr: float = 0.5, frames=None) -> dict:
    """All metrics in one dict."""
    out = {}
    out.update(clear_mot(gt, pred, iou_thr, frames))
    out.update(identity_metrics(gt, pred, iou_thr, frames))
    out.update(hota(gt, pred, frames))
    return out
