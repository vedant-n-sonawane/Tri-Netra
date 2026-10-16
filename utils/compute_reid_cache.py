#!/usr/bin/env python3
"""
TriNetra-AMRF — Appearance Embedding Cache for the Real-Clip Stress Test (Phase 6 v1.5)
=========================================================================================

``inference/video.py --save-detections`` caches boxes/scores/TRC (``<name>.dets.npz``) but
not image crops, so the ReID ablation can't run on real clips without this extra pass.

The stress test damages only the *thermal* stream (``utils/stress_eval.py``'s docstring),
so the RGB video for a clip is identical between its clean and stress(blur/saturate) runs —
only the detected boxes differ (because the fused detector reacts to the damaged thermal
input). This script therefore decodes each clip's RGB video **once**, sequentially, and at
every frame embeds whatever boxes any of that clip's cached ``.dets.npz`` files (clean +
stress variants) need at that frame — instead of re-opening the video per cache file.

Output: ``<name>.embeds.npz`` next to every ``<name>.dets.npz``, with one row of embeddings
per row of that file's ``dets`` array (same order), so ``utils/tracking_quality.load_cache``
can load it in lock-step.

Usage:
    python -m utils.compute_reid_cache --clean runs/tracking_eval/clean --stress runs/tracking_eval/stress
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tracker.appearance import AppearanceEmbedder  # noqa: E402


def clip_id_of(path: Path) -> str:
    m = re.match(r"sequence_(\d+)(?:_\w+)?\.dets\.npz", path.name)
    if not m:
        raise ValueError(f"unrecognised cache filename: {path.name}")
    return m.group(1)


def build_cache(clean_dir: Path, stress_dir: Path, batch: int = 64) -> None:
    stress_files = sorted(stress_dir.glob("sequence_*_*.dets.npz"))
    clip_ids = sorted({clip_id_of(p) for p in stress_files})
    by_clip: dict[str, list[Path]] = defaultdict(list)
    for p in stress_files:
        by_clip[clip_id_of(p)].append(p)
    for cid in clip_ids:
        clean_p = clean_dir / f"sequence_{cid}.dets.npz"
        if clean_p.exists():
            by_clip[cid].append(clean_p)
        else:
            print(f"  [WARN] no clean cache for clip {cid}")

    embedder = AppearanceEmbedder()
    for clip_id, paths in sorted(by_clip.items()):
        video_path = PROJECT_ROOT / "datasets" / "RGB" / f"sequence_{clip_id}_1min_RGB.mp4"
        if not video_path.exists():
            print(f"  [SKIP] clip {clip_id}: no RGB video at {video_path}")
            continue
        # frame_id -> list of (array_index_into_file, box) per source file
        needs: dict[Path, dict] = {}
        want_frames: dict[int, list] = defaultdict(list)
        for p in paths:
            z = np.load(p)
            dets = z["dets"]
            embeds = np.zeros((len(dets), 1280), dtype=np.float32)
            needs[p] = {"dets": dets, "embeds": embeds}
            for i, row in enumerate(dets):
                want_frames[int(row[0])].append((p, i, row[1:5]))

        if not want_frames:
            continue
        max_frame = max(want_frames)
        t0 = time.time()
        cap = cv2.VideoCapture(str(video_path))
        fidx = -1
        n_done = 0
        while fidx < max_frame:
            ok, frame = cap.read()
            fidx += 1
            if not ok:
                break
            jobs = want_frames.get(fidx)
            if not jobs:
                continue
            boxes = np.stack([j[2] for j in jobs]).astype(np.float64)
            feats = embedder.extract(frame, boxes)
            for (p, i, _), f in zip(jobs, feats):
                needs[p]["embeds"][i] = f
            n_done += len(jobs)
        cap.release()
        for p, d in needs.items():
            out = p.with_suffix("").with_suffix(".embeds.npz")
            np.savez_compressed(out, embeds=d["embeds"])
            print(f"  [OK] {out.relative_to(PROJECT_ROOT)}  ({len(d['dets'])} boxes)")
        print(f"  clip {clip_id}: {n_done} boxes embedded over {fidx + 1} decoded frames, "
              f"{time.time() - t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(PROJECT_ROOT / "runs" / "tracking_eval" / "clean"))
    ap.add_argument("--stress", default=str(PROJECT_ROOT / "runs" / "tracking_eval" / "stress"))
    args = ap.parse_args()
    build_cache(Path(args.clean), Path(args.stress))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
