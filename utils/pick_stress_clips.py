#!/usr/bin/env python3
"""Pick the clips for the thermal-degradation stress test by a fixed, pre-declared
rule: the K clips with the most concurrently tracked objects inside the stress
window (mean tracks per frame over frames 200-800 of the clean run). The rule
uses only the clean run, never the gate's results, so the choice cannot favour it.

    python -m utils.pick_stress_clips --k 3      # prints e.g. "05 07 10"
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(PROJECT_ROOT / "runs" / "tracking_eval" / "clean"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lo", type=int, default=200)
    ap.add_argument("--hi", type=int, default=800)
    args = ap.parse_args()
    scores = {}
    for p in sorted(Path(args.clean).glob("sequence_*.frames.jsonl")):
        seq_id = re.match(r"sequence_(\d+)", p.name).group(1)
        counts = []
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if args.lo <= r["frame"] < args.hi:
                counts.append(len(r["tracks"]))
        if counts:
            scores[seq_id] = sum(counts) / len(counts)
    ranked = sorted(scores, key=lambda s: -scores[s])[:args.k]
    print(" ".join(ranked))
    print("mean concurrent tracks:", {k: round(v, 2) for k, v in sorted(scores.items())}, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
