#!/bin/bash
# Detection + tracking over all 14 paired clips (no video output), caching raw
# detections so tracker variants can be replayed without the GPU. Resumable:
# a clip whose .dets.npz exists is skipped; a failed clip is retried.
#   usage: bash utils/run_tracking_clips.sh [outdir]
cd /d/TRI-NETRA/Tri-Netra
OUT="${1:-runs/tracking_eval/clean}"
LOG=/d/TRI-NETRA/tracking_clips.log
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
for i in $(seq 1 14); do
  n=$(printf "%02d" $i)
  if [ -f "$OUT/sequence_$n.dets.npz" ]; then echo "=== seq $n already done, skipping ===" >> $LOG; continue; fi
  for attempt in 1 2 3; do
    echo "=== seq $n attempt $attempt  $(date +%T) ===" >> $LOG
    "/d/condaenvs/trinetra/python.exe" inference/video.py --sequence $i --no-video --save-detections \
        --output "$OUT/sequence_$n.mp4" --log-every 250 >> $LOG 2>&1
    [ -f "$OUT/sequence_$n.dets.npz" ] && { echo "=== seq $n DONE $(date +%T) ===" >> $LOG; break; }
    echo "=== seq $n FAILED (attempt $attempt) ===" >> $LOG
    sleep 20
  done
done
echo "=== ALL CLIPS FINISHED $(date +%T) ===" >> $LOG
