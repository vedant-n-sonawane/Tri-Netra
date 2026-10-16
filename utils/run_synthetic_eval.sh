#!/bin/bash
cd /d/TRI-NETRA/Tri-Netra
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY="/d/condaenvs/trinetra/python.exe"
LOG=/d/TRI-NETRA/synthetic_eval.log
echo "=== severity 0.5 $(date +%T) ===" > $LOG
$PY -m utils.eval_synthetic --seeds 30 --severity 0.5 --experiments main controls beta severity --out runs/tracking_eval/synthetic/sev0.5 >> $LOG 2>&1
echo "=== severity 1.0 $(date +%T) ===" >> $LOG
$PY -m utils.eval_synthetic --seeds 30 --severity 1.0 --experiments main controls beta --out runs/tracking_eval/synthetic/sev1.0 >> $LOG 2>&1
echo "=== SYNTHETIC EVAL DONE $(date +%T) ===" >> $LOG
