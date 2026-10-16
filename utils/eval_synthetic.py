#!/usr/bin/env python3
"""
TriNetra-AMRF — Synthetic Tracking Benchmark Runner
===================================================

Runs the tracker variants on seeded synthetic sequences with exact ground truth
(``utils/synth_mot.py``) and scores them with ``utils/mot_metrics.py``.

Trackers compared (identical detections and TRC values for all):
  SORT-style        ByteTrack association without the low-confidence stage
                    (detections below the high threshold are discarded)
  DeepSORT (motion) deep_sort_realtime, appearance embedder disabled,
                    detections below 0.4 discarded
  ByteTrack         low-confidence second stage, TRC gate off
  ByteTrack + TRC   low-confidence second stage, TRC-gated cost (TriNetra)

Experiments:
  main      the "coupled" condition (detector degrades when TRC dips)
  controls  "uninformative" TRC and "no_degradation" (gate has no reason to help)
  beta      sensitivity of the gate strength beta
  severity  gain as degradation severity grows

Usage:
    python -m utils.eval_synthetic --seeds 30
    python -m utils.eval_synthetic --seeds 30 --experiments main controls beta severity
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tracker import ByteTracker, FusionTracker  # noqa: E402
from utils.mot_metrics import evaluate_mot  # noqa: E402
from utils.synth_mot import CONDITIONS, make_sequence  # noqa: E402

TRACKERS = ["SORT-style", "DeepSORT (motion)", "ByteTrack", "ByteTrack + TRC gate (ours)",
           "ByteTrack + ReID", "ByteTrack + TRC + ReID (ours)"]
COLUMNS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Frag"]
LOWER_IS_BETTER = {"IDSW", "FP", "FN", "Frag"}


def _base_cfg(beta=0.5, gate=True, low=0.1, reid=False) -> dict:
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "local_6gb.yaml", encoding="utf-8"))
    t = cfg["tracking"]
    t["behaviour"]["enabled"] = False
    t["matching"]["low_conf_threshold"] = low
    t["trc_gate"].update(enabled=gate, beta=beta)
    t["reid"]["enabled"] = reid
    if reid:
        # max_age=30 (the non-ReID default) is deliberately left alone for the
        # plain/TRC-gate trackers above — raising it only helps once stage 1.5
        # (appearance-only re-acquisition) exists to use the extra lifetime
        # safely; without it, a longer-lived lost track just has more chances
        # to steal a nearby detection (measured: plain ByteTrack's IDSW gets
        # *worse* as max_age rises). 90 is the value validated for ReID
        # trackers specifically (see configs/*.yaml reid.reacquire_thresh note).
        t["lifecycle"]["max_age"] = 90
    return cfg


def make_tracker(name: str, beta: float = 0.5):
    if name == "SORT-style":
        cfg = _base_cfg(gate=False, low=_base_cfg()["tracking"]["matching"]["high_conf_threshold"])
        return ByteTracker(cfg)
    if name == "ByteTrack":
        return ByteTracker(_base_cfg(gate=False))
    if name == "ByteTrack + TRC gate (ours)":
        return ByteTracker(_base_cfg(gate=True, beta=beta))
    if name == "ByteTrack + ReID":
        return ByteTracker(_base_cfg(gate=False, reid=True))
    if name == "ByteTrack + TRC + ReID (ours)":
        return ByteTracker(_base_cfg(gate=True, beta=beta, reid=True))
    if name == "DeepSORT (motion)":
        cfg = _base_cfg()
        cfg["tracking"].update(algorithm="deepsort", embedder=None)
        return FusionTracker(cfg)
    raise KeyError(name)


def run_tracker(name: str, seq: dict, beta: float = 0.5) -> dict:
    """Frame -> ndarray [n,5] (id, x1, y1, x2, y2) of the tracker's reported tracks."""
    tr = make_tracker(name, beta)
    is_ds = name.startswith("DeepSORT")
    uses_reid = "ReID" in name
    appearance = seq.get("appearance")
    id_map, out = {}, {}
    frames = seq.get("frames", range(len(seq["dets"])))
    for i, (t, dets, trc) in enumerate(zip(frames, seq["dets"], seq["trc_obs"])):
        t = int(t)
        if is_ds:
            d = dets[dets[:, 4] >= 0.4] if len(dets) else dets
            raw = [([float(a), float(b), float(c - a), float(e - b)], float(sc), int(k))
                   for a, b, c, e, sc, k in d.tolist()]
            tracks = tr.tracker.update_tracks(raw, embeds=[[1.0]] * len(raw))
            # only tracks matched this frame; coasting predictions are not observations
            rows = [[id_map.setdefault(t.track_id, len(id_map) + 1), *t.to_ltrb()]
                    for t in tracks if t.is_confirmed() and t.time_since_update == 0]
        else:
            embeds = appearance[i] if (uses_reid and appearance is not None) else None
            res = tr.update(dets, trc=float(trc), frame_id=t, embeds=embeds)
            rows = [[r["track_id"], *r["bbox_xyxy"]] for r in res]
        if rows:
            out[t] = np.array(rows, dtype=float)
    return out


def score(seq: dict, pred: dict) -> dict:
    m = evaluate_mot(seq["gt"], pred, frames=range(seq["meta"]["T"]))
    return {k: m[k] for k in COLUMNS}


def run_condition(condition: str, seeds, trackers=TRACKERS, beta=0.5, severity=1.0, T=600, verbose=True):
    per_seed = {n: [] for n in trackers}
    t0 = time.time()
    for i, s in enumerate(seeds):
        seq = make_sequence(int(s), T=T, condition=condition, severity=severity)
        for n in trackers:
            per_seed[n].append(score(seq, run_tracker(n, seq, beta)))
        if verbose and (i + 1) % 5 == 0:
            print(f"    {condition:<15} seed {i + 1}/{len(seeds)}  {time.time() - t0:.0f}s", flush=True)
    return per_seed


def summarize(per_seed: dict) -> dict:
    out = {}
    for n, rows in per_seed.items():
        arr = {c: np.array([r[c] for r in rows], dtype=float) for c in COLUMNS}
        out[n] = {c: {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}
                  for c, a in arr.items()}
    return out


def paired_stats(per_seed: dict, a: str, b: str, metrics=("HOTA", "IDF1", "MOTA", "IDSW", "AssA"), n_boot=10000):
    """Paired difference a - b across seeds: mean, bootstrap 95% CI, Wilcoxon p."""
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(0)
    res = {}
    for m in metrics:
        d = np.array([x[m] for x in per_seed[a]]) - np.array([x[m] for x in per_seed[b]])
        boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        try:
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        res[m] = {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                          float(np.percentile(boots, 97.5))],
                  "wilcoxon_p": p, "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": len(d)}
    return res


def fmt_table(summary: dict, title: str) -> str:
    lines = [title, "| Tracker | " + " | ".join(COLUMNS) + " |", "|" + "---|" * (len(COLUMNS) + 1)]
    best = {c: (min if c in LOWER_IS_BETTER else max)(summary[n][c]["mean"] for n in summary) for c in COLUMNS}
    for n, row in summary.items():
        cells = []
        for c in COLUMNS:
            v, sd = row[c]["mean"], row[c]["std"]
            txt = f"{v:.3f} ± {sd:.3f}" if c in ("HOTA", "DetA", "AssA", "MOTA", "IDF1") else f"{v:.1f} ± {sd:.1f}"
            cells.append(f"**{txt}**" if abs(v - best[c]) < 1e-9 else txt)
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--severity", type=float, default=1.0, help="degradation strength for main/controls")
    ap.add_argument("--experiments", nargs="+", default=["main", "controls", "beta", "severity"],
                    choices=["main", "controls", "beta", "severity"])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "runs" / "tracking_eval" / "synthetic"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1000, 1000 + args.seeds))
    results = {"n_seeds": args.seeds, "frames": args.frames}
    ours = "ByteTrack + TRC gate (ours)"

    if "main" in args.experiments or "controls" in args.experiments:
        conds = [c for c, e in (("coupled", "main"), ("uninformative", "controls"), ("no_degradation", "controls"))
                 if e in args.experiments]
        for c in conds:
            print(f"[{c}]", flush=True)
            ps = run_condition(c, seeds, T=args.frames, severity=args.severity)
            results[c] = {"summary": summarize(ps), "per_seed": ps,
                          "paired_ours_vs_bytetrack": paired_stats(ps, ours, "ByteTrack"),
                          "paired_ours_vs_sort": paired_stats(ps, ours, "SORT-style"),
                          "paired_ours_vs_deepsort": paired_stats(ps, ours, "DeepSORT (motion)")}
            print(fmt_table(results[c]["summary"], f"\n{c}"), flush=True)

    if "beta" in args.experiments:
        print("[beta sweep, coupled]", flush=True)
        results["beta_sweep"] = {}
        for beta in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            ps = run_condition("coupled", seeds, trackers=[ours], beta=beta, severity=args.severity,
                               T=args.frames, verbose=False)
            results["beta_sweep"][str(beta)] = summarize(ps)[ours]
            r = results["beta_sweep"][str(beta)]
            print(f"    beta={beta:<4} HOTA {r['HOTA']['mean']:.3f}  IDF1 {r['IDF1']['mean']:.3f}  "
                  f"IDSW {r['IDSW']['mean']:.1f}", flush=True)

    if "severity" in args.experiments:
        print("[severity sweep, coupled]", flush=True)
        results["severity_sweep"] = {}
        for sev in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
            ps = run_condition("coupled", seeds, trackers=TRACKERS, severity=sev,
                               T=args.frames, verbose=False)
            results["severity_sweep"][str(sev)] = {"summary": summarize(ps),
                                                   "paired": paired_stats(ps, ours, "ByteTrack")}
            s = results["severity_sweep"][str(sev)]["summary"]
            print(f"    severity={sev:<4} HOTA  ByteTrack {s['ByteTrack']['HOTA']['mean']:.3f}  "
                  f"ours {s[ours]['HOTA']['mean']:.3f}   IDSW  {s['ByteTrack']['IDSW']['mean']:.1f} -> "
                  f"{s[ours]['IDSW']['mean']:.1f}", flush=True)
            results["severity_sweep"][str(sev)]["frac_degraded_frames"] = float(np.mean(
                [make_sequence(int(sd), T=args.frames, condition="coupled")["meta"]["frac_degraded"]
                 for sd in seeds[:10]]))

    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nsaved {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
