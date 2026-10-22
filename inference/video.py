#!/usr/bin/env python3
"""
TriNetra-AMRF — Offline Video Pipeline: Fusion Detection -> Tracking -> Behaviour
=================================================================================

Runs a synchronized RGB + thermal video pair through the trained fusion
detector, tracks the detections with the Phase 6 ByteTracker (TRC-gated), and
writes an annotated video plus per-track analytics.

Per frame:  read RGB + thermal -> preprocess (Phase 3; yields the frame TRC)
            -> FusionDetector -> NMS -> boxes scaled to the original frame
            -> ByteTracker.update(dets, trc)  -> behaviour features
            -> annotated frame

Outputs (next to ``--output``):
    <name>.mp4          annotated video
    <name>.tracks.json  one summary per track (class, span, mean TRC, flags)
    <name>.frames.jsonl one line per frame (frame TRC, tracks, threat level, alert ids)
    <name>.alerts.json  every alert raised (kind, frame, level, reasons, snapshot)
    <name>.dets.npz     raw per-frame detections + TRC (with --save-detections),
                        so tracker variants can be compared without re-running the model

Each frame also gets a rule-based threat level (Phase 7, models/risk_model/
threat_scorer.py) drawn as a banner, and the alert manager (Phase 8,
models/risk_model/alerts.py) turns those levels into alerts written to
<name>.alerts.json with a snapshot per alert in <name>_alerts/. Live camera
capture (Phase 9, ``realtime.py``) is not part of this path.

Usage:
    python inference/video.py --sequence 5 --max-frames 300
    python inference/video.py --visible-video a_rgb.mp4 --thermal-video a_thermal.mp4 \\
        --output runs/tracking/a.mp4 --layout both

    # ablations: disable the TRC gate, or damage the thermal stream mid-clip
    python inference/video.py --sequence 5 --no-trc-gate
    python inference/video.py --sequence 5 --stress saturate --stress-frames 300:600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "local_6gb.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "fusion" / "best.pt"


def clip_paths(sequence: int, root: Path = PROJECT_ROOT / "datasets"):
    return (root / "RGB" / f"sequence_{sequence:02d}_1min_RGB.mp4",
            root / "thermal" / f"sequence_{sequence:02d}_1min_Thermal.mp4")


# BGR banner colour per threat level (1 benign .. 5 critical)
LEVEL_BGR = {1: (80, 160, 80), 2: (60, 170, 190), 3: (40, 150, 235), 4: (30, 90, 235), 5: (40, 40, 210)}


def _draw_threat_banner(canvas: np.ndarray, threat: dict, alerted: bool) -> np.ndarray:
    """Threat level strip across the top of the output frame; red border on an alert frame."""
    h, w = canvas.shape[:2]
    bar = max(28, h // 14)
    lvl = int(threat["level"])
    cv2.rectangle(canvas, (0, 0), (w, bar), (25, 25, 25), -1)
    cv2.rectangle(canvas, (0, 0), (min(w, 190), bar), LEVEL_BGR[lvl], -1)
    scale = bar / 40
    cv2.putText(canvas, f"THREAT {lvl}/5", (8, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                0.75 * scale, (255, 255, 255), 2)
    reason = "; ".join(threat["reasons"])
    max_chars = max(10, int((w - 200) / (11 * scale)))
    cv2.putText(canvas, reason[:max_chars], (200, int(bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * scale, (235, 235, 235), 1)
    if alerted:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
    return canvas


def process_video(
    visible_video: str,
    thermal_video: str,
    output: str,
    checkpoint: str = str(DEFAULT_CHECKPOINT),
    config_path: str = str(DEFAULT_CONFIG),
    device: str | None = None,
    start: int = 0,
    max_frames: int | None = None,
    stride: int = 1,
    layout: str = "rgb",
    out_scale: float = 1.0,
    trc_gate: bool = True,
    stress: str = "none",
    stress_frames: tuple[int, int] | None = None,
    save_detections: bool = False,
    log_every: int = 50,
    write_video: bool = True,
    progress_cb=None,
):
    from datasets.augmentations import build_transforms
    from models.risk_model.alerts import AlertManager
    from models.risk_model.threat_scorer import find_thermal_only, score_frame
    from models.tracker import build_tracker
    from preprocessing.pipeline import build_preprocess_pipeline
    from training.evaluate import decode_predictions, load_checkpoint_model
    from training.loss_adapter import get_class_id_map
    from utils.visualize_tracks import TrackVisualizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    if not trc_gate:
        run_cfg.setdefault("tracking", {}).setdefault("trc_gate", {})["enabled"] = False

    model, cfg = load_checkpoint_model(checkpoint, run_cfg, device)   # cfg = checkpoint snapshot
    model.eval()
    # threat scoring uses the deployment threshold; the tracker gets the low-confidence boxes too
    threat_conf = float(cfg.get("detection", {}).get("confidence_threshold", 0.25))
    risk_cfg = run_cfg.get("risk_model", {}) or {}
    threat_cfg = risk_cfg.get("threat", {}) or {}
    night_rgb = float(threat_cfg.get("night_rgb_mean", 70.0))
    alert_mgr = AlertManager(risk_cfg.get("alerts", {}))
    # ByteTrack's second stage needs the low-confidence detections too
    low = float(run_cfg["tracking"].get("matching", {}).get("low_conf_threshold", 0.1))
    cfg.setdefault("detection", {})["confidence_threshold"] = low

    class_map = get_class_id_map(cfg)                    # unified -> head index
    head_classes = sorted(set(class_map.values()))
    unified_from_head = {v: k for k, v in class_map.items()}
    id_to_name = {c["id"]: c["name"] for c in cfg["datasets"]["categories"]}

    preprocess = build_preprocess_pipeline(cfg)
    transform = build_transforms(cfg, "val")
    tracker = build_tracker(run_cfg, class_names=id_to_name)
    viz = TrackVisualizer()

    cap_v, cap_t = cv2.VideoCapture(str(visible_video)), cv2.VideoCapture(str(thermal_video))
    if not cap_v.isOpened() or not cap_t.isOpened():
        raise FileNotFoundError(f"cannot open {visible_video} / {thermal_video}")
    fps = cap_v.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(min(cap_v.get(cv2.CAP_PROP_FRAME_COUNT), cap_t.get(cv2.CAP_PROP_FRAME_COUNT)))
    width, height = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end = n_total if max_frames is None else min(n_total, start + max_frames)
    if start:
        cap_v.set(cv2.CAP_PROP_POS_FRAMES, start)
        cap_t.set(cv2.CAP_PROP_POS_FRAMES, start)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_w, panel_h = int(width * out_scale), int(height * out_scale)
    canvas_w = panel_w * (2 if layout == "both" else 1)
    writer = (cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps / max(1, stride), (canvas_w, panel_h))
              if write_video else None)

    frames_log = open(out_path.with_suffix(".frames.jsonl"), "w", encoding="utf-8")
    snap_dir = out_path.parent / (out_path.stem + "_alerts")
    track_info: dict[int, dict] = {}
    det_rows, trc_rows = [], []
    n_dets_total = 0
    t0 = time.time()
    processed = 0

    for fi in range(start, end):
        ok_v, frame_v = cap_v.read()
        ok_t, frame_t = cap_t.read()
        if not (ok_v and ok_t):
            break
        if (fi - start) % stride:
            continue

        vis_rgb = cv2.cvtColor(frame_v, cv2.COLOR_BGR2RGB)
        thr = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)[:, :, None]
        if stress != "none" and stress_frames and stress_frames[0] <= fi < stress_frames[1]:
            from utils.visualize_preproc import _degrade
            thr = _degrade(thr, stress)

        trc = 1.0
        vis_p, thr_p = vis_rgb, thr
        if preprocess is not None:
            vis_p, thr_p, extras = preprocess(vis_rgb, thr)
            trc = float(extras.get("trc", 1.0))

        vis_t, thr_t, _, _ = transform(vis_p, thr_p, [], [])
        with torch.no_grad():
            out = model(vis_t.unsqueeze(0).to(device), thr_t.unsqueeze(0).to(device),
                        torch.tensor([trc], dtype=torch.float32, device=device))
        decoded = out[0] if isinstance(out, (list, tuple)) else out
        det = decode_predictions(decoded, cfg, keep_classes=head_classes)[0]

        rh, rw = vis_t.shape[-2:]
        sx, sy = width / rw, height / rh
        rows = []
        if det is not None and len(det):
            for x1, y1, x2, y2, conf, cls in det.cpu().tolist():
                uid = unified_from_head.get(int(cls))
                if uid is not None:
                    rows.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy, conf, uid])
        dets = np.array(rows, dtype=np.float64).reshape(-1, 6)
        n_dets_total += len(dets)

        tracks = tracker.update(dets, trc=trc, frame_id=fi)

        for t in tracks:
            info = track_info.setdefault(t["track_id"], {
                "track_id": t["track_id"], "cls": t["cls"], "cls_name": t["cls_name"],
                "first_frame": fi, "last_frame": fi, "n_frames": 0, "flags": set(),
                "trc_sum": 0.0})
            info.update(cls=t["cls"], cls_name=t["cls_name"], last_frame=fi)
            info["n_frames"] += 1
            info["trc_sum"] += t["trc"]
            info["flags"].update(t["behaviour"].get("anomaly_flags", []))

        # Phase 7/8: frame threat level from the confident detections, then alerts
        boxes = [(d[0], d[1], d[2], d[3], int(d[5]), d[4]) for d in dets if d[4] >= threat_conf]
        vis_gray = cv2.cvtColor(frame_v, cv2.COLOR_BGR2GRAY)
        rgb_mean = float(vis_gray.mean())
        hidden = []
        if rgb_mean < night_rgb and any(b[4] == 1 for b in boxes):
            hidden = find_thermal_only(boxes, vis_gray.astype(np.float64),
                                       thr[:, :, 0].astype(np.float64), threat_cfg)
        threat = score_frame(boxes, rgb_mean=rgb_mean, trc=trc, thermal_only=hidden, cfg=threat_cfg)
        new_alerts = alert_mgr.update(fi, threat, trc, tracks)

        frames_log.write(json.dumps({
            "frame": fi, "trc": round(trc, 4), "n_dets": len(dets),
            "tracks": [[t["track_id"], *[round(v, 1) for v in t["bbox_xyxy"]], t["cls"],
                        round(t["conf"], 3)] for t in tracks],
            "threat": {"level": threat["level"], "label": threat["label"], "reasons": threat["reasons"],
                       "rgb_mean": round(rgb_mean, 1), "thermal_only": len(hidden)},
            "alerts": [a["alert_id"] for a in new_alerts]}) + "\n")
        if save_detections:
            for r in dets:
                det_rows.append([fi, *r.tolist()])
            trc_rows.append([fi, trc])

        if writer is not None:
            panel = viz.draw(frame_v, tracks, frame_idx=fi, frame_trc=trc)
            panel = cv2.resize(panel, (panel_w, panel_h)) if out_scale != 1.0 else panel
            if layout == "both":
                thr_vis = cv2.cvtColor(cv2.resize(thr[:, :, 0], (panel_w, panel_h)), cv2.COLOR_GRAY2BGR)
                sc = out_scale
                for t in tracks:
                    x1, y1, x2, y2 = (int(v * sc) for v in t["bbox_xyxy"])
                    cv2.rectangle(thr_vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
                panel = np.hstack([panel, thr_vis])
            panel = _draw_threat_banner(panel, threat, alerted=bool(new_alerts))
            writer.write(panel)
            if new_alerts:
                snap_dir.mkdir(parents=True, exist_ok=True)
                for a in new_alerts:
                    snap = snap_dir / f"alert_{a['alert_id']:03d}_f{fi}.jpg"
                    cv2.imwrite(str(snap), panel)
                    a["snapshot"] = str(snap)

        processed += 1
        if progress_cb is not None:
            progress_cb(processed, max(1, (end - start) // max(1, stride)))
        if processed % log_every == 0:
            el = time.time() - t0
            eta = el / processed * ((end - start) // max(1, stride) - processed)
            print(f"  frame {fi}/{end}  processed {processed}  dets {len(dets)}  "
                  f"tracks {len(tracks)}  TRC {trc:.2f}  {processed / el:.1f} fps  ETA {eta:.0f}s",
                  flush=True)

    cap_v.release(); cap_t.release(); frames_log.close()
    if writer is not None:
        writer.release()

    summaries = []
    for info in sorted(track_info.values(), key=lambda d: d["track_id"]):
        summaries.append({**{k: v for k, v in info.items() if k not in ("flags", "trc_sum")},
                          "mean_trc": round(info["trc_sum"] / max(1, info["n_frames"]), 3),
                          "flags": sorted(info["flags"])})
    out_path.with_suffix(".tracks.json").write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    out_path.with_suffix(".alerts.json").write_text(json.dumps(alert_mgr.alerts, indent=1), encoding="utf-8")
    if save_detections:
        np.savez_compressed(out_path.with_suffix(".dets.npz"),
                            dets=np.array(det_rows, dtype=np.float64).reshape(-1, 7),
                            trc=np.array(trc_rows, dtype=np.float64).reshape(-1, 2))

    elapsed = time.time() - t0
    return {
        "frames": processed, "elapsed_s": round(elapsed, 1), "fps": round(processed / max(elapsed, 1e-9), 2),
        "detections": n_dets_total, "tracks_created": tracker._next_id - 1,
        "tracks_reported": len(summaries), "alerts": len(alert_mgr.alerts), "output": str(out_path),
        "track_summaries": summaries, "alert_list": alert_mgr.alerts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="TriNetra detect -> track -> annotate on a paired video")
    ap.add_argument("--sequence", type=int, help="1-14: use datasets/RGB + datasets/thermal clip pair")
    ap.add_argument("--visible-video")
    ap.add_argument("--thermal-video")
    ap.add_argument("--output", help="annotated .mp4 (default: runs/tracking/sequence_NN.mp4)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--device", default=None)
    ap.add_argument("--start", type=int, default=0, help="first frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--layout", choices=["rgb", "both"], default="rgb")
    ap.add_argument("--out-scale", type=float, default=1.0)
    ap.add_argument("--no-trc-gate", action="store_true", help="stock ByteTrack (ablation)")
    ap.add_argument("--stress", choices=["none", "blur", "saturate", "flat"], default="none",
                    help="damage the thermal stream to force low TRC (ablation)")
    ap.add_argument("--stress-frames", default=None, help="START:END frame range for --stress")
    ap.add_argument("--save-detections", action="store_true")
    ap.add_argument("--log-every", type=int, default=50, help="progress line every N processed frames")
    ap.add_argument("--no-video", action="store_true",
                    help="skip drawing/writing the annotated video (faster; detections + tracks still saved)")
    args = ap.parse_args()

    if args.sequence is not None:
        vis, thr = clip_paths(args.sequence)
        out = args.output or str(PROJECT_ROOT / "runs" / "tracking" / f"sequence_{args.sequence:02d}.mp4")
    elif args.visible_video and args.thermal_video:
        vis, thr = Path(args.visible_video), Path(args.thermal_video)
        out = args.output or str(PROJECT_ROOT / "runs" / "tracking" / (vis.stem + "_tracked.mp4"))
    else:
        ap.error("give --sequence N, or both --visible-video and --thermal-video")

    sf = tuple(int(v) for v in args.stress_frames.split(":")) if args.stress_frames else None
    res = process_video(vis, thr, out, checkpoint=args.checkpoint, config_path=args.config,
                        device=args.device, start=args.start, max_frames=args.max_frames,
                        stride=args.stride, layout=args.layout, out_scale=args.out_scale,
                        trc_gate=not args.no_trc_gate, stress=args.stress, stress_frames=sf,
                        save_detections=args.save_detections, log_every=args.log_every,
                        write_video=not args.no_video)
    print("\n=== summary ===")
    for k in ("frames", "elapsed_s", "fps", "detections", "tracks_created", "tracks_reported", "alerts", "output"):
        print(f"  {k:<16}{res[k]}")
    for a in res["alert_list"]:
        print(f"  ALERT #{a['alert_id']} frame {a['frame']} [{a['kind']}] {a['title']}: {'; '.join(a['reasons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
