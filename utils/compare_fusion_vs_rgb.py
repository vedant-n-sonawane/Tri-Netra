#!/usr/bin/env python3
"""
Fusion vs RGB-only on dull / night RGB, on held-out M3FD test images.

RGB-only baseline = the COCO-pretrained YOLOv8m (weights/yolov8m.pt) on the
visible image alone; this is the same network the fusion model's visible branch
is initialised from and keeps frozen, so the gap comes from adding the thermal
branch with TRC-gated fusion. ``fusion_no_thermal`` is the same fusion
weights with the thermal stream blanked and TRC forced to 0 - the like-for-like
"not fused" case, free of any M3FD-training advantage the baseline lacks. Both run at the config's confidence/NMS thresholds
and are scored with the same COCO evaluation as training/evaluate.py.

Splits scored (test.json only, never seen in training):
    night  - RGB mean luminance < 70 (dull / dark visible stream)
    day    - a random control sample of brighter images (fusion should not hurt)

Writes runs/fusion_vs_rgb/results.json (metrics + per-image detections for the
side-by-side video) and prints a table.

Usage:
    python utils/compare_fusion_vs_rgb.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.augmentations import build_transforms  # noqa: E402
from preprocessing.pipeline import build_preprocess_pipeline  # noqa: E402
from training.evaluate import (_coco_map, _precision_recall_f1, decode_predictions,  # noqa: E402
                               load_checkpoint_model)
from training.loss_adapter import get_class_id_map  # noqa: E402


def _fusion_dets(model, cfg, pre, tf, vis_bgr, thr_gray, device, head_classes, unified_from_head, no_thermal=False):
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    thr = thr_gray[:, :, None]
    trc = 1.0
    vis_p, thr_p = vis_rgb, thr
    if pre is not None:
        vis_p, thr_p, extras = pre(vis_rgb, thr)
        trc = float(extras.get("trc", 1.0))
    v, t, _, _ = tf(vis_p, thr_p, [], [])
    if no_thermal:                       # same weights, thermal stream switched off (blank + TRC 0)
        t, trc = torch.zeros_like(t), 0.0
    with torch.no_grad():
        out = model(v.unsqueeze(0).to(device), t.unsqueeze(0).to(device),
                    torch.tensor([trc], dtype=torch.float32, device=device))
    decoded = out[0] if isinstance(out, (list, tuple)) else out
    det = decode_predictions(decoded, cfg, keep_classes=head_classes)[0]
    h, w = vis_bgr.shape[:2]
    rh, rw = v.shape[-2:]
    rows = []
    if det is not None and len(det):
        for x1, y1, x2, y2, conf, cls in det.cpu().tolist():
            uid = unified_from_head.get(int(cls))
            if uid is not None:
                rows.append([x1 * w / rw, y1 * h / rh, x2 * w / rw, y2 * h / rh, conf, uid])
    return rows, trc


def _rgb_dets(yolo, cfg, vis_bgr, device, head_classes, unified_from_head):
    d = cfg.get("detection", {})
    r = yolo.predict(vis_bgr, imgsz=640, conf=float(d.get("confidence_threshold", 0.25)),
                     iou=float(d.get("iou_threshold", 0.45)), classes=head_classes,
                     device=device, verbose=False)[0]
    rows = []
    for (x1, y1, x2, y2), conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        uid = unified_from_head.get(int(cls))
        if uid is not None:
            rows.append([x1, y1, x2, y2, conf, uid])
    return rows


def _score(coco_gt, dets_by_img, ids):
    res = [{"image_id": i, "category_id": int(r[5]), "bbox": [r[0], r[1], r[2] - r[0], r[3] - r[1]],
            "score": float(r[4])} for i in ids for r in dets_by_img[i]]
    m = _coco_map(coco_gt, res, ids)
    m.update(_precision_recall_f1(coco_gt, res, ids, sorted(coco_gt.getCatIds())))
    person = [r for r in res if r["category_id"] == 1]
    mp = _coco_map_person(coco_gt, person, ids)
    return {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "person_AP50": mp}


def _coco_map_person(coco_gt, res, ids):
    from pycocotools.cocoeval import COCOeval
    import contextlib, io
    if not res:
        return float("nan")
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco_gt, coco_gt.loadRes(res), "bbox")
        ev.params.imgIds, ev.params.catIds = sorted(ids), [1]
        ev.evaluate(); ev.accumulate()
    p = ev.eval["precision"][0, :, 0, 0, 2]
    p = p[p > -1]
    return float(p.mean()) if p.size else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(ROOT / "weights/fusion/best.pt"))
    ap.add_argument("--rgb-weights", default=str(ROOT / "weights/yolov8m.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--day-n", type=int, default=60)
    ap.add_argument("--night-rgb-mean", type=float, default=70.0)
    ap.add_argument("--out", default=str(ROOT / "runs/fusion_vs_rgb"))
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from ultralytics import YOLO

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
    model, cfg = load_checkpoint_model(args.checkpoint, cfg, args.device)
    model.eval()
    class_map = get_class_id_map(cfg)
    head_classes = sorted(set(class_map.values()))
    unified_from_head = {v: k for k, v in class_map.items()}
    pre, tf = build_preprocess_pipeline(cfg), build_transforms(cfg, "val")
    yolo = YOLO(args.rgb_weights)

    coco_gt = COCO(str(ROOT / "datasets/annotations/test.json"))
    stats = pd.read_csv(ROOT / "datasets/m3fd_scene_stats.csv").set_index("file_name")
    imgs = {i["id"]: i["file_name"] for i in coco_gt.dataset["images"]}
    night = sorted(i for i, f in imgs.items() if stats.loc[f, "rgb_mean"] < args.night_rgb_mean)
    day_pool = sorted(i for i, f in imgs.items() if stats.loc[f, "rgb_mean"] >= args.night_rgb_mean)
    random.Random(0).shuffle(day_pool)
    day = sorted(day_pool[: args.day_n])
    print(f"night test images: {len(night)}, day control: {len(day)}")

    fus, rgb, nth, trcs = {}, {}, {}, {}
    for k, i in enumerate(night + day):
        fn = imgs[i]
        vis = cv2.imread(str(ROOT / "datasets/M3FD/Vis" / fn))
        thr = cv2.imread(str(ROOT / "datasets/M3FD/Ir" / fn), cv2.IMREAD_GRAYSCALE)
        fus[i], trcs[i] = _fusion_dets(model, cfg, pre, tf, vis, thr, args.device, head_classes, unified_from_head)
        nth[i], _ = _fusion_dets(model, cfg, pre, tf, vis, thr, args.device, head_classes, unified_from_head, no_thermal=True)
        rgb[i] = _rgb_dets(yolo, cfg, vis, args.device, head_classes, unified_from_head)
        if k % 20 == 0:
            print(f"  {k}/{len(night) + len(day)}", flush=True)

    table = {}
    for name, ids in (("night", night), ("day", day)):
        table[name] = {"n_images": len(ids),
                       "rgb_only": _score(coco_gt, rgb, ids),
                       "fusion_no_thermal": _score(coco_gt, nth, ids),
                       "fusion": _score(coco_gt, fus, ids)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"metrics": table, "night_ids": night, "day_ids": day,
               "files": {str(i): imgs[i] for i in night + day},
               "dets": {"fusion": {str(i): fus[i] for i in fus}, "no_thermal": {str(i): nth[i] for i in nth}, "rgb": {str(i): rgb[i] for i in rgb}},
               "trc": {str(i): trcs[i] for i in trcs}}, open(out / "results.json", "w"))
    cols = ["mAP50", "mAP50_95", "precision", "recall", "f1", "person_AP50"]
    for name in ("night", "day"):
        print(f"\n=== {name} ({table[name]['n_images']} test images) ===")
        print(f"{'':10s}" + "".join(f"{c:>12s}" for c in cols))
        for m in ("rgb_only", "fusion_no_thermal", "fusion"):
            print(f"{m:10s}" + "".join(f"{table[name][m][c]:12.3f}" for c in cols))


if __name__ == "__main__":
    main()
