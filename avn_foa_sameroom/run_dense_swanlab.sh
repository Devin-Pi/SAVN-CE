#!/usr/bin/env bash
# Independent host process; never signals training.
set -u
experiment=/media/4tbssd/av_seld/workspaces/avn_foa_dense15h_finetune_v02
monitor="$experiment/monitor"
python=/media/4tbssd/av_seld/workspaces/avn_swanlab_monitor_v1/venv/bin/python
training_pid=$1
cd "$monitor" || exit 2
exec 9>supervisor.lock
flock -n 9 || exit 2
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
  "$python" -u "$experiment/code_snapshot/avn_foa_sameroom/swanlab_monitor.py" \
    --run-dir "$experiment/run_seed7_full100" --state-dir "$monitor/state" \
    --training-pid "$training_pid" --experiment-name FOA21-Dense15h-full100-seed7 \
    --id-prefix avnfoa-dense15h-seed7- \
    --source-file "$experiment/code_snapshot/avn_foa_sameroom/finetune.py" >> monitor.log 2>&1 &
  child_pid=$!
  printf '%s\n' "$child_pid" > collector.pid
  wait "$child_pid"
  outcome=$?
  child_pid=''
  if [ "$outcome" -eq 0 ] || [ "$outcome" -eq 2 ]; then exit "$outcome"; fi
  echo 'Collector disconnected; retrying same run after 30 seconds' >&2
  sleep 30
done
