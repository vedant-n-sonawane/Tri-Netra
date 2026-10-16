#!/bin/bash
# Stop the clean all-clips run once sequence 08 is done (scope trimmed by request),
# discard the partial next clip, and signal the follow-on pipeline to proceed.
LOG=/d/TRI-NETRA/tracking_clips.log
until grep -a -q "=== seq 08 DONE" $LOG; do sleep 5; done
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'run_tracking_clip[s].sh|inference/vide[o].py' } | ForEach-Object { taskkill /PID \$_.ProcessId /F /T | Out-Null }"
sleep 3
rm -f /d/TRI-NETRA/Tri-Netra/runs/tracking_eval/clean/sequence_09.*
echo "=== ALL CLIPS FINISHED (trimmed to sequences 01-08 by request) $(date +%T) ===" >> $LOG
