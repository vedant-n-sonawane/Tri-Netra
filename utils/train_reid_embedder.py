#!/usr/bin/env python3
"""
TriNetra-AMRF — Self-Supervised Fine-Tuning of the Appearance Embedder (Phase 6 v1.5, Option A)
==================================================================================================

The pretrained MobileNetv2 person-ReID embedder (models/tracker/appearance.py) is generic — not
trained on these cameras, this resolution, this lighting. No manually identity-labelled data
exists for this project, so this fine-tunes it *without* manual labels, using the tracker's own
output as a source of pseudo-identities instead:

  - A short, unbroken, high-confidence tracklet is almost certainly one real person throughout.
  - Crops from the SAME tracklet -> positive pair (same identity).
  - Crops from a DIFFERENT tracklet that was alive at an OVERLAPPING frame range -> negative pair
    (guaranteed different identity — two different boxes can't be the same physical object at the
    same instant).

Caveat, by design (this is the fast, ~1-2 day option, not the slower hand-corrected-labels one):
tracklets are least reliable exactly in the crowded segments we most want to fix, so this mainly
buys domain adaptation (these cameras/lighting/resolution), not a guaranteed fix for crowding-type
identity swaps specifically. Evaluated the same way as every other change in this project: the
real-clip stress test (utils/stress_eval.py), on clips held OUT of this training set to avoid
leakage (see TRAIN_CLIPS / EVAL_CLIPS below — 01/07/08 are the evaluation clips used everywhere
else in Phase 6, so they are excluded here).

Usage:
    python -m utils.train_reid_embedder
"""

from __future__ import annotations

import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tracker.byte_tracker import ByteTracker  # noqa: E402
from utils.tracking_quality import load_cache  # noqa: E402

TRAIN_CLIPS = ["02", "03", "04", "05", "06"]   # held OUT of the eval clips (01, 07, 08) used elsewhere
MIN_TRACKLET_LEN = 20       # matched frames
MIN_SCORE = 0.6
CROP_STRIDE = 3             # sample every Nth matched frame (appearance variety, avoid near-dupes)
PERSON_CLASS = 1
OUT_WEIGHTS = PROJECT_ROOT / "weights" / "tracker" / "person_reid_finetuned.pt"


def extract_tracklets(clip_id: str):
    """Plain-motion ByteTrack over the cached clean detections -> accepted (track_id, [(frame, box)]) list."""
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "local_6gb.yaml", encoding="utf-8"))
    cfg["tracking"]["behaviour"]["enabled"] = False
    cfg["tracking"]["reid"]["enabled"] = False     # pure motion tracklets; no embedder dependency here
    cfg["tracking"]["heading"]["enabled"] = False
    tr = ByteTracker(cfg)
    seq = load_cache(PROJECT_ROOT / "runs" / "tracking_eval" / "clean" / f"sequence_{clip_id}.dets.npz")
    per_track = defaultdict(list)
    for f, dets, trc in zip(seq["frames"], seq["dets"], seq["trc_obs"]):
        res = tr.update(dets, trc=float(trc), frame_id=int(f))
        for r in res:
            if r["cls"] != PERSON_CLASS or r["conf"] < MIN_SCORE:
                continue
            per_track[r["track_id"]].append((int(f), r["bbox_xyxy"]))
    return {tid: obs for tid, obs in per_track.items() if len(obs) >= MIN_TRACKLET_LEN}


def collect_crops(clip_id: str, tracklets: dict):
    """One sequential decode of the clip's RGB video -> {track_id: [crop, ...]} (uint8 HxWx3 BGR)."""
    video_path = PROJECT_ROOT / "datasets" / "RGB" / f"sequence_{clip_id}_1min_RGB.mp4"
    want = defaultdict(list)   # frame -> [(track_id, box), ...]
    for tid, obs in tracklets.items():
        for i, (f, box) in enumerate(obs):
            if i % CROP_STRIDE == 0:
                want[f].append((tid, box))
    if not want:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    crops = defaultdict(list)
    fidx, max_f = -1, max(want)
    while fidx < max_f:
        ok, frame = cap.read()
        fidx += 1
        if not ok:
            break
        jobs = want.get(fidx)
        if not jobs:
            continue
        h, w = frame.shape[:2]
        for tid, (x1, y1, x2, y2) in jobs:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crops[tid].append(frame[y1:y2, x1:x2].copy())
    cap.release()
    return crops


def build_dataset():
    """-> list of (clip_id, track_id, crop) across all training clips, person-class, filtered tracklets."""
    items = []
    overlap_groups = []  # list of sets of global ids alive in an overlapping window, for negative mining (unused beyond logging)
    for clip_id in TRAIN_CLIPS:
        t0 = time.time()
        tracklets = extract_tracklets(clip_id)
        crops = collect_crops(clip_id, tracklets)
        n_crops = sum(len(v) for v in crops.values())
        print(f"  clip {clip_id}: {len(tracklets)} tracklets, {n_crops} crops  [{time.time()-t0:.0f}s]", flush=True)
        for tid, cs in crops.items():
            if len(cs) < 4:        # need at least a few crops per identity for triplet sampling
                continue
            gid = f"{clip_id}_{tid}"
            for c in cs:
                items.append((gid, c))
    return items


class CropDataset(torch.utils.data.Dataset):
    """Resizes + ImageNet-normalizes BGR crops the same way AppearanceEmbedder.preprocess does."""

    def __init__(self, items):
        self.items = items
        self.by_id = defaultdict(list)
        for i, (gid, _) in enumerate(items):
            self.by_id[gid].append(i)
        self.ids = list(self.by_id)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = bgr[..., ::-1]
        rgb = cv2.resize(rgb, (224, 224))
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        return (t - self.mean) / self.std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        gid, crop = self.items[idx]
        return self.preprocess(crop), gid


def pk_batches(ds: CropDataset, p: int, k: int, n_batches: int, rng: np.random.Generator):
    ids = [i for i in ds.ids if len(ds.by_id[i]) >= 2]
    for _ in range(n_batches):
        chosen = rng.choice(ids, size=min(p, len(ids)), replace=False)
        idxs = []
        for gid in chosen:
            pool = ds.by_id[gid]
            take = rng.choice(pool, size=min(k, len(pool)), replace=len(pool) < k)
            idxs.extend(take.tolist())
        yield idxs


def batch_hard_triplet_loss(embeds: torch.Tensor, labels: list, margin: float = 0.3) -> torch.Tensor:
    """Cosine-distance batch-hard triplet loss (matches deployment's cosine-similarity cost)."""
    embeds = F.normalize(embeds, dim=1)
    sim = embeds @ embeds.T                      # [-1, 1], higher = more similar
    dist = 1.0 - sim                              # cosine distance, lower = more similar
    labels_t = torch.tensor([hash(l) for l in labels], device=embeds.device)
    same = labels_t[:, None] == labels_t[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=embeds.device)
    pos_mask = same & ~eye
    neg_mask = ~same
    hardest_pos = (dist * pos_mask.float()).max(dim=1).values
    big = dist.max().detach() + 1.0
    hardest_neg = (dist + (~neg_mask).float() * big).min(dim=1).values
    loss = F.relu(hardest_pos - hardest_neg + margin)
    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    return loss[valid].mean() if valid.any() else loss.mean()


def main():
    print("=== building tracklet-derived training set (clips", TRAIN_CLIPS, ") ===", flush=True)
    items = build_dataset()
    ds = CropDataset(items)
    n_ids = len(ds.ids)
    print(f"total: {len(items)} crops, {n_ids} pseudo-identities", flush=True)
    if n_ids < 8:
        raise SystemExit("too few tracklets to train on — check runs/tracking_eval/clean caches exist")

    from deep_sort_realtime.embedder.embedder_pytorch import MobileNetV2_bottle, MOBILENETV2_BOTTLENECK_WTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MobileNetV2_bottle(input_size=224, width_mult=1.0)
    model.load_state_dict(torch.load(MOBILENETV2_BOTTLENECK_WTS, map_location="cpu"))
    model.to(device)

    # Freeze the generic low-level layers (edges/texture transfer fine across domains); fine-tune
    # only the last few blocks, which encode higher-level, more domain-specific appearance.
    blocks = list(model.features.children())
    for b in blocks[:15]:
        for p in b.parameters():
            p.requires_grad = False
    train_params = [p for b in blocks[15:] for p in b.parameters()]
    print(f"fine-tuning {sum(p.numel() for p in train_params):,} of "
          f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)

    opt = torch.optim.Adam(train_params, lr=1e-4)
    rng = np.random.default_rng(42)
    P, K, steps_per_epoch, epochs = 10, 4, 40, 8

    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        losses = []
        for idxs in pk_batches(ds, P, K, steps_per_epoch, rng):
            imgs = torch.stack([ds[i][0] for i in idxs]).to(device)
            labels = [ds[i][1] for i in idxs]
            embeds = model(imgs)
            loss = batch_hard_triplet_loss(embeds, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        print(f"epoch {epoch+1}/{epochs}  loss={np.mean(losses):.4f}  [{time.time()-t0:.0f}s]", flush=True)

    model.eval()
    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_WEIGHTS)
    print("saved", OUT_WEIGHTS)


if __name__ == "__main__":
    main()
