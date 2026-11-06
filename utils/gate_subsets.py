#!/usr/bin/env python3
"""Bootstrap CIs for the blur / saturation subsets of the corrupted-thermal gate ablation.

Uses the detections saved by ``utils.gate_bootstrap`` (no model run). The corruption kind of
each test image is the one ``training.evaluate.StressThermalLoader`` applied (byte-sum of the
file name), so the subsets are exactly those in the paper (blur 333, saturation 297).

    python -m utils.gate_subsets
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pycocotools.coco import COCO                                            # noqa: E402
from utils.gate_bootstrap import OUT, Matched                                # noqa: E402

N_BOOT = 1000


def main() -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(PROJECT_ROOT / "datasets" / "annotations" / "test.json"))
    cats = sorted(gt.getCatIds())
    meta = json.loads((OUT / "corrupted_meta.json").read_text(encoding="utf-8"))
    ids = sorted(int(k) for k in meta)
    kinds = ("blur", "saturate")
    m = {mode: Matched(gt, json.loads((OUT / f"corrupted_{mode}.json").read_text(encoding="utf-8")), ids, cats)
         for mode in ("adaptive", "fixed", "full", "off")}
    order = m["adaptive"].img_ids
    kind = np.array([kinds[sum(gt.imgs[i]["file_name"].encode()) % 2] for i in order])
    rng = np.random.default_rng(42)
    out = {}
    for k in kinds:
        mask = kind == k
        n = len(order)
        pts = {mode: float(m[mode].ap(mask.astype(float))[0]) * 100 for mode in m}
        d = {"fixed": [], "full": [], "off": []}
        for _ in range(N_BOOT):
            pick = rng.choice(np.where(mask)[0], size=int(mask.sum()))
            w = np.bincount(pick, minlength=n).astype(float)
            a = m["adaptive"].ap(w)[0]
            for o in d:
                d[o].append(a - m[o].ap(w)[0])
        out[k] = {"n": int(mask.sum()), "mAP50": pts,
                  "adaptive_minus": {o: {"d": pts["adaptive"] - pts[o],
                                         "ci": [float(np.percentile(v, 2.5)) * 100, float(np.percentile(v, 97.5)) * 100]}
                                     for o, v in d.items()}}
        print(k, json.dumps(out[k], indent=1))
    (OUT / "subsets.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
