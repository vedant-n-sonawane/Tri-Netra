#!/usr/bin/env python3
"""
TriNetra: tracking evaluation on real M3FD clips (batch driver)
================================================================

Runs detection + tracking over the M3FD clips in ``datasets/m3fd_eval`` (built by
``utils/make_m3fd_clips.py --out datasets/m3fd_eval``) and caches the raw detections so
every tracker variant can be replayed on identical input.

For each clip:
    clean run                      -> runs/tracking_eval_m3fd/clean/sequence_NN.dets.npz
    thermal blurred in the middle  -> runs/tracking_eval_m3fd/stress/sequence_NN_blur.dets.npz
    thermal saturated in the middle-> runs/tracking_eval_m3fd/stress/sequence_NN_saturate.dets.npz

The damaged segment is fixed in advance for every clip: frames [n//3, 2n//3) of an
n-frame clip. All clips are used (no selection). The model is loaded once.

Resumable: a run whose .dets.npz exists is skipped.

    python -m utils.run_m3fd_tracking --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLIPS_DIR = PROJECT_ROOT / "datasets" / "m3fd_eval"
OUT = PROJECT_ROOT / "runs" / "tracking_eval_m3fd"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--kinds", nargs="*", default=["blur", "saturate"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (testing)")
    ap.add_argument("--min-frames", type=int, default=18, help="use only clips with at least this many frames (fixed in advance)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    import training.evaluate as ev
    from inference.video import process_video

    cache = {}
    orig = ev.load_checkpoint_model

    def cached(ckpt_path, cfg, device):
        key = (str(ckpt_path), device)
        if key not in cache:
            cache[key] = orig(ckpt_path, cfg, device)
        return cache[key]

    ev.load_checkpoint_model = cached           # model is loaded once, not once per clip

    index = json.loads((CLIPS_DIR / "m3fd_clips.json").read_text(encoding="utf-8"))
    clips = sorted(int(k) for k in index if int(index[k]["frames"]) >= args.min_frames)
    if args.limit:
        clips = clips[: args.limit]
    (OUT / "clean").mkdir(parents=True, exist_ok=True)
    (OUT / "stress").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for n in clips:
        frames = int(index[str(n)]["frames"])
        vis = CLIPS_DIR / "RGB" / f"sequence_{n:02d}_1min_RGB.mp4"
        thr = CLIPS_DIR / "thermal" / f"sequence_{n:02d}_1min_Thermal.mp4"
        seg = (frames // 3, 2 * frames // 3)
        jobs = [("clean", OUT / "clean" / f"sequence_{n:02d}", "none", None)]
        jobs += [(k, OUT / "stress" / f"sequence_{n:02d}_{k}", k, seg) for k in args.kinds]
        for label, stem, stress, window in jobs:
            if stem.with_suffix(".dets.npz").exists():
                continue
            r = process_video(str(vis), str(thr), str(stem.with_suffix(".mp4")), device=args.device,
                              stress=stress, stress_frames=window, save_detections=True,
                              write_video=False, log_every=10 ** 6)
            print(f"seq {n:02d} {label:<9} {r['frames']:>3} frames  {r['fps']:.2f} fps  "
                  f"tracks {r['tracks_reported']:>3}   total {time.time() - t0:.0f}s", flush=True)
    (OUT / "segments.json").write_text(json.dumps(
        {f"{n:02d}": {"frames": int(index[str(n)]["frames"]),
                      "segment": [int(index[str(n)]["frames"]) // 3, 2 * int(index[str(n)]["frames"]) // 3]}
         for n in clips}, indent=1), encoding="utf-8")
    print("ALL CLIPS FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
