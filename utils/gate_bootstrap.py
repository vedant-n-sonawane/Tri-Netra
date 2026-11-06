#!/usr/bin/env python3
"""
TriNetra: gate ablation with bootstrap confidence intervals
===========================================================

Re-runs the inference-time gate ablation (adaptive TRC, fixed TRC, full thermal,
thermal off) on the M3FD test split, clean and with the thermal stream corrupted,
and attaches 95% bootstrap confidence intervals over test images.

The backbone is run once per batch and shared by the four gate modes, because the
reliability value only enters after the backbone (in the fusion blocks).

Stage 1 (slow, resumable): detections for every (condition, mode) are written to
    runs/gate_bootstrap/<condition>_<mode>.json
Stage 2 (fast): COCO matching is done once, then images are resampled (paired across
    modes) to give CIs for mAP@50, mAP@50:95 and for paired differences.

Usage:
    python -m utils.gate_bootstrap --device cpu
    python -m utils.gate_bootstrap --stage2-only
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs" / "gate_bootstrap"
FIXED_TRC = 0.782
MODES = {"adaptive": None, "fixed": FIXED_TRC, "full": 1.0, "off": 0.0}
CONDITIONS = ("clean", "corrupted")


class CachedBackbone(nn.Module):
    """Runs the wrapped backbone once per input tensor and reuses the result."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self._key = None
        self._val = None

    def forward(self, vis, thr):
        if self._key is not vis:
            self._val = self.inner(vis, thr)
            self._key = vis
        return self._val


def get_class_map(cfg):
    from training.evaluate import get_class_id_map
    return get_class_id_map(cfg)


def run_condition(model, cfg, condition: str, device: str, limit, batch_size: int):
    from datasets.dataloader import create_dataloaders
    from models.detector.fusion_detector import FusionDetector
    from training.evaluate import decode_predictions, detections_to_coco, install_stress_corruption

    class_map = get_class_map(cfg)
    head_classes = sorted(set(class_map.values()))
    coco_from_head = {v: k for k, v in class_map.items()}

    import datasets.dataloader as dl
    from torch.utils.data import DataLoader
    m3fd = PROJECT_ROOT / "datasets" / "M3FD"
    ds = dl.DualModalDataset(
        visible_dir=str(m3fd / "Vis"), thermal_dir=str(m3fd / "Ir"),
        annotations_json=str(PROJECT_ROOT / "datasets" / "annotations" / "test.json"),
        transform=dl.build_transforms(cfg, "test"), preprocess=dl.build_preprocess_pipeline(cfg))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        collate_fn=dl.dual_modal_collate_fn)
    if condition == "corrupted":
        install_stress_corruption(ds)

    results = {m: [] for m in MODES}
    meta = {}
    n_done = 0
    t0 = time.time()
    with torch.no_grad():
        for i, (visible, thermal, targets) in enumerate(loader):
            vis = visible.to(device)
            thr = thermal.to(device)
            trc_real = FusionDetector.trc_from_targets(targets, device=device)
            img_hw = tuple(visible.shape[-2:])
            for mode, val in MODES.items():
                trc = trc_real if val is None else torch.full_like(trc_real, float(val))
                out = model(vis, thr, trc)
                decoded = out[0] if isinstance(out, (list, tuple)) else out
                dets = decode_predictions(decoded, cfg, keep_classes=head_classes)
                results[mode] += detections_to_coco(dets, targets, img_hw, coco_from_head)
            for t in targets:
                meta[int(t["image_id"])] = {"trc": float(t.get("trc", 1.0)),
                                            "file": str(t.get("file_name", t.get("fname", "")))}
            n_done += len(targets)
            if (i + 1) % 5 == 0:
                el = time.time() - t0
                print(f"  [{condition}] {n_done} images  {el:.0f}s  ({n_done / el:.2f} img/s)", flush=True)
            if limit and n_done >= limit:
                break
    return results, meta, loader.dataset


def stage1(args) -> None:
    from training.evaluate import load_checkpoint_model, load_config
    cfg = load_config()
    torch.manual_seed(int(cfg["datasets"].get("seed", 42)))
    torch.set_num_threads(args.threads)
    model, cfg = load_checkpoint_model(args.checkpoint, cfg, args.device)
    model.eval()
    model.backbone = CachedBackbone(model.backbone)
    OUT.mkdir(parents=True, exist_ok=True)
    for cond in CONDITIONS:
        done = all((OUT / f"{cond}_{m}.json").exists() for m in MODES)
        if done and not args.force:
            print(f"[skip] {cond}: already done")
            continue
        results, meta, ds = run_condition(model, cfg, cond, args.device, args.limit, args.batch_size)
        for m, r in results.items():
            (OUT / f"{cond}_{m}.json").write_text(json.dumps(r), encoding="utf-8")
        (OUT / f"{cond}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        print(f"[done] {cond}: {len(meta)} images")


# ── stage 2: bootstrap ───────────────────────────────────────────────────────
class Matched:
    """COCOeval matching flattened for fast weighted AP at all ten IoU thresholds."""

    def __init__(self, coco_gt, results, img_ids, cat_ids):
        from pycocotools.cocoeval import COCOeval
        with contextlib.redirect_stdout(io.StringIO()):
            dt = coco_gt.loadRes(results) if results else None
            ev = COCOeval(coco_gt, dt, iouType="bbox") if dt is not None else None
            ev.params.imgIds = sorted(img_ids)
            ev.params.catIds = list(cat_ids)
            ev.params.areaRng = [ev.params.areaRng[0]]
            ev.params.areaRngLbl = [ev.params.areaRngLbl[0]]
            ev.evaluate()
        self.img_ids = list(ev.params.imgIds)
        self.index = {iid: j for j, iid in enumerate(self.img_ids)}
        self.cat_ids = list(ev.params.catIds)
        n_img = len(self.img_ids)
        n_t = len(ev.params.iouThrs)
        self.n_img = n_img
        self.per_cat = []
        for k, _ in enumerate(self.cat_ids):
            sc, tp, fp, im = [], [], [], []
            ngt = np.zeros(n_img)
            for i in range(n_img):
                e = ev.evalImgs[k * n_img + i]
                if e is None:
                    continue
                ngt[i] = int((~e["gtIgnore"].astype(bool)).sum())
                d = len(e["dtScores"])
                if d:
                    ig = e["dtIgnore"].astype(bool)
                    m = e["dtMatches"] > 0
                    sc.append(np.asarray(e["dtScores"], float))
                    tp.append(m & ~ig)
                    fp.append(~m & ~ig)
                    im.append(np.full(d, i))
            if sc:
                sc = np.concatenate(sc)
                tp = np.concatenate(tp, axis=1)
                fp = np.concatenate(fp, axis=1)
                im = np.concatenate(im)
                order = np.argsort(-sc, kind="mergesort")
                sc, tp, fp, im = sc[order], tp[:, order], fp[:, order], im[order]
            else:
                sc, tp, fp, im = np.zeros(0), np.zeros((n_t, 0), bool), np.zeros((n_t, 0), bool), np.zeros(0, int)
            self.per_cat.append((tp.astype(float), fp.astype(float), im, ngt))
        self.rec_thrs = np.linspace(0.0, 1.0, 101)

    def ap(self, w: np.ndarray, cats=None):
        """Mean AP per IoU threshold, weighted by image multiplicities ``w`` (len n_img)."""
        aps, used = [], []
        for k, (tp, fp, im, ngt) in enumerate(self.per_cat):
            if cats is not None and self.cat_ids[k] not in cats:
                continue
            npig = float((ngt * w).sum())
            if npig == 0:
                continue
            wd = w[im]
            ctp = np.cumsum(tp * wd, axis=1)
            cfp = np.cumsum(fp * wd, axis=1)
            q = np.zeros((tp.shape[0], len(self.rec_thrs)))
            for t in range(tp.shape[0]):
                rc = ctp[t] / npig
                pr = ctp[t] / np.maximum(ctp[t] + cfp[t], np.spacing(1))
                pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
                inds = np.searchsorted(rc, self.rec_thrs, side="left")
                ok = inds < len(pr)
                q[t, ok] = pr[inds[ok]]
            aps.append(q.mean(axis=1))
        if not aps:
            return np.full(10, np.nan)
        return np.mean(aps, axis=0)


def stage2(args) -> None:
    from training.evaluate import load_config
    from pycocotools.coco import COCO
    cfg = load_config()
    ann = PROJECT_ROOT / cfg["datasets"]["annotations_dir"] / "instances_test.json"
    if not ann.exists():
        cands = sorted((PROJECT_ROOT / "datasets" / "annotations").glob("*test*.json"))
        ann = cands[0]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(ann))
    cat_ids = sorted(coco_gt.getCatIds())
    rng = np.random.default_rng(args.seed)

    summary = {"n_boot": args.n_boot, "seed": args.seed, "conditions": {}}
    for cond in CONDITIONS:
        meta = json.loads((OUT / f"{cond}_meta.json").read_text(encoding="utf-8"))
        ids = sorted(int(k) for k in meta)
        mt = {m: Matched(coco_gt, json.loads((OUT / f"{cond}_{m}.json").read_text(encoding="utf-8")), ids, cat_ids)
              for m in MODES}
        n = len(ids)
        # subsets (by position in mt[...].img_ids)
        index = mt["adaptive"].index
        trc = np.array([meta[str(i)]["trc"] for i in mt["adaptive"].img_ids])
        subsets = {"all": np.ones(n, bool)}
        if cond == "corrupted":
            subsets["trc<0.6"] = trc < 0.6
        base = np.ones(n)
        point = {m: mt[m].ap(base) for m in MODES}
        boots = {m: np.zeros((args.n_boot, 2)) for m in MODES}
        for b in range(args.n_boot):
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            for m in MODES:
                a = mt[m].ap(w)
                boots[m][b] = (a[0], a.mean())
        cond_out = {"n_images": n, "modes": {}, "paired": {}}
        for m in MODES:
            lo50, hi50 = np.percentile(boots[m][:, 0], [2.5, 97.5])
            lo95, hi95 = np.percentile(boots[m][:, 1], [2.5, 97.5])
            cond_out["modes"][m] = {
                "mAP50": float(point[m][0]) * 100, "mAP50_ci": [float(lo50) * 100, float(hi50) * 100],
                "mAP50_95": float(point[m].mean()) * 100, "mAP50_95_ci": [float(lo95) * 100, float(hi95) * 100]}
        for other in ("fixed", "full", "off"):
            d50 = boots["adaptive"][:, 0] - boots[other][:, 0]
            d95 = boots["adaptive"][:, 1] - boots[other][:, 1]
            cond_out["paired"][f"adaptive-{other}"] = {
                "d_mAP50": float(point["adaptive"][0] - point[other][0]) * 100,
                "d_mAP50_ci": [float(np.percentile(d50, 2.5)) * 100, float(np.percentile(d50, 97.5)) * 100],
                "p_le0_mAP50": float((d50 <= 0).mean()),
                "d_mAP50_95": float(point["adaptive"].mean() - point[other].mean()) * 100,
                "d_mAP50_95_ci": [float(np.percentile(d95, 2.5)) * 100, float(np.percentile(d95, 97.5)) * 100]}
        # per-class AP50 (adaptive vs fixed), point + CI
        cond_out["per_class_adaptive"] = {}
        names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
        for cid in cat_ids:
            pts = mt["adaptive"].ap(base, cats={cid})[0] * 100
            bs = []
            r2 = np.random.default_rng(args.seed + cid)
            for b in range(args.n_boot):
                w = np.bincount(r2.integers(0, n, n), minlength=n).astype(float)
                bs.append(mt["adaptive"].ap(w, cats={cid})[0] * 100)
            cond_out["per_class_adaptive"][names[cid]] = {
                "AP50": float(pts), "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}
        # subsets
        cond_out["subsets"] = {}
        for sname, mask in subsets.items():
            if sname == "all":
                continue
            w0 = mask.astype(float)
            sub = {}
            for m in ("adaptive", "fixed"):
                sub[m] = float(mt[m].ap(w0)[0]) * 100
            ds = []
            for b in range(args.n_boot):
                pick = rng.choice(np.where(mask)[0], size=int(mask.sum()))
                w = np.bincount(pick, minlength=n).astype(float)
                ds.append(mt["adaptive"].ap(w)[0] - mt["fixed"].ap(w)[0])
            sub["d_mAP50"] = sub["adaptive"] - sub["fixed"]
            sub["d_ci"] = [float(np.percentile(ds, 2.5)) * 100, float(np.percentile(ds, 97.5)) * 100]
            sub["n"] = int(mask.sum())
            cond_out["subsets"][sname] = sub
        summary["conditions"][cond] = cond_out
        print(f"[{cond}]", json.dumps(cond_out["modes"], indent=1))
        print("  paired:", json.dumps(cond_out["paired"], indent=1))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT / "summary.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "weights" / "fusion" / "best.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stage2-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.stage2_only:
        stage1(args)
    stage2(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
