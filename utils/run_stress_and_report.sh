#!/bin/bash
# Waits for the clean all-clips run, then: pick stress clips -> run thermal-damage
# variants -> evaluate everything -> build the report. Log: /d/TRI-NETRA/tracking_final.log
cd /d/TRI-NETRA/Tri-Netra
PY="/d/condaenvs/trinetra/python.exe"
LOG=/d/TRI-NETRA/tracking_final.log
export PYTHONUNBUFFERED=1
echo "=== waiting for the clean run $(date +%T) ===" > $LOG
until grep -q "ALL CLIPS FINISHED" /d/TRI-NETRA/tracking_clips.log 2>/dev/null; do sleep 30; done
echo "=== clean run finished $(date +%T) ===" >> $LOG

CLIPS=$($PY -m utils.pick_stress_clips --k 3 2>>$LOG)
echo "=== stress clips (most concurrent tracks): $CLIPS ===" >> $LOG
mkdir -p runs/tracking_eval/stress
for n in $CLIPS; do
  for kind in saturate blur; do
    out="runs/tracking_eval/stress/sequence_${n}_${kind}"
    [ -f "$out.dets.npz" ] && { echo "=== $n $kind already done ===" >> $LOG; continue; }
    for attempt in 1 2 3; do
      echo "=== stress seq $n $kind attempt $attempt $(date +%T) ===" >> $LOG
      $PY inference/video.py --sequence $((10#$n)) --start 200 --max-frames 600 --stress $kind \
          --stress-frames 400:700 --no-video --save-detections --output "$out.mp4" --log-every 300 >> $LOG 2>&1
      [ -f "$out.dets.npz" ] && break
      sleep 20
    done
  done
done
echo "=== stress runs finished $(date +%T) ===" >> $LOG

export CUDA_VISIBLE_DEVICES=""
$PY -m utils.eval_real_clips --dets runs/tracking_eval/clean >> $LOG 2>&1
$PY -m utils.stress_eval >> $LOG 2>&1
$PY -m utils.build_tracking_report >> $LOG 2>&1
echo "=== PIPELINE COMPLETE $(date +%T) ===" >> $LOG
