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
    <name>.frames.jsonl one line per frame (frame TRC + the reported tracks)
    <name>.dets.npz     raw per-frame detections + TRC (with --save-detections),
                        so tracker variants can be compared without re-running the model

Scope: this is the detect->track->annotate path used by Phase 6. Risk scoring
(Phase 7) and live camera capture (Phase 9, ``realtime.py``) are not part of it.

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
    # the tracker wants low-confidence detections too (ByteTrack's second stage)
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

        frames_log.write(json.dumps({
            "frame": fi, "trc": round(trc, 4), "n_dets": len(dets),
            "tracks": [[t["track_id"], *[round(v, 1) for v in t["bbox_xyxy"]], t["cls"],
                        round(t["conf"], 3)] for t in tracks]}) + "\n")
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
            writer.write(panel)

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
    if save_detections:
        np.savez_compressed(out_path.with_suffix(".dets.npz"),
                            dets=np.array(det_rows, dtype=np.float64).reshape(-1, 7),
                            trc=np.array(trc_rows, dtype=np.float64).reshape(-1, 2))

    elapsed = time.time() - t0
    return {
        "frames": processed, "elapsed_s": round(elapsed, 1), "fps": round(processed / max(elapsed, 1e-9), 2),
        "detections": n_dets_total, "tracks_created": tracker._next_id - 1,
        "tracks_reported": len(summaries), "output": str(out_path),
        "track_summaries": summaries,
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
    for k in ("frames", "elapsed_s", "fps", "detections", "tracks_created", "tracks_reported", "output"):
        print(f"  {k:<16}{res[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
