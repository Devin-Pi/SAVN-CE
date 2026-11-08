#!/usr/bin/env bash
# Collector supervisor only; never sends a signal to the training process.
set -u
monitor_root=/media/4tbssd/av_seld/workspaces/avn_swanlab_monitor_v1
cd "$monitor_root"
exec 9>supervisor.lock
flock -n 9 || { echo 'Collector supervisor already active' >&2; exit 2; }
child_pid=''
stop_collector() {
  if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid"
    wait "$child_pid" || true
  fi
  exit 0
}
trap stop_collector TERM INT
printf '%s\n' "$$" > supervisor.pid
while true; do
  "$monitor_root/venv/bin/python" -u "$monitor_root/swanlab_monitor.py" \
    --run-dir /media/4tbssd/av_seld/workspaces/avn_foa_full_finetune_v1/run_seed7_full100 \
    --state-dir "$monitor_root/state" --training-pid 978563 \
    >> "$monitor_root/monitor.log" 2>&1 &
  child_pid=$!
  printf '%s\n' "$child_pid" > collector.pid
  wait "$child_pid"
  outcome=$?
  child_pid=''
  if [ "$outcome" -eq 0 ] || [ "$outcome" -eq 2 ]; then
    exit "$outcome"
  fi
  echo 'Collector exited unexpectedly; retrying the same SwanLab run in 30 seconds' >&2
  sleep 30
done
