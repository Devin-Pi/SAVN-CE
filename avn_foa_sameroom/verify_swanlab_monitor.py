"""Read SwanLab back and compare its series to source JSONL, without analysis."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

try:
    from .swanlab_monitor import atomic_json, complete_history, epoch_metrics, process_identity
except ImportError:
    from swanlab_monitor import atomic_json, complete_history, epoch_metrics, process_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor-root', type=Path, required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--key-file', type=Path, default=Path.home() / '.config/avn-swanlab/api_key')
    args = parser.parse_args()
    root = args.monitor_root
    state = json.loads((root / 'state/status.json').read_text())
    assert state['status'] in ('following', 'completed')
    run_dir = Path(state['source_run'])
    rows = complete_history(run_dir / 'history.jsonl')
    assert rows
    source = {row['epoch']: epoch_metrics(row) for row in rows}
    keys = list(source[1])
    import swanlab
    key = args.key_file.read_text().strip()
    experiment = swanlab.Api(api_key=key).run(f"{args.owner}/{state['project']}/{state['run_id']}")
    remote = experiment.metrics(keys=keys, ignore_timestamp=True)
    comparisons = []
    covered = []
    for series in remote['list']:
        name = series['key']
        points = {int(point['index']): float(point['data']) for point in series['metrics']}
        assert len(points) == len(series['metrics']), f'Duplicate remote steps: {name}'
        epoch_end = min(max(source), max(points))
        assert sorted(e for e in points if e <= epoch_end) == list(range(1, epoch_end + 1)), name
        errors = []
        for epoch in range(1, epoch_end + 1):
            expected = source[epoch][name]
            assert math.isclose(points[epoch], expected, rel_tol=1e-6, abs_tol=1e-8), (name, epoch)
            errors.append(abs(points[epoch] - expected))
        comparisons.append(dict(key=name, matched_epochs=epoch_end, max_absolute_difference=max(errors)))
        covered.append(epoch_end)
    assert {s['key'] for s in remote['list']} == set(keys)
    minimum = min(covered)
    initial = json.loads((root / 'initial_attach_status.json').read_text())
    assert minimum > initial['historical_rows_at_start'], 'Wait for a new epoch after initial attachment'
    before = json.loads((root / 'training_before.json').read_text())
    identity = process_identity(before['training_pid'])
    assert identity is not None and identity['start_ticks'] == before['process_start_ticks']
    command = Path(f"/proc/{before['training_pid']}/cmdline").read_bytes()
    assert hashlib.sha256(command).hexdigest() == before['command_sha256']
    assert hashlib.sha256((run_dir / 'config.json').read_bytes()).hexdigest() == before['config_sha256']
    source_file = run_dir.parent / 'code/avn_foa_sameroom/finetune.py'
    assert hashlib.sha256(source_file.read_bytes()).hexdigest() == before['script_sha256']
    report = dict(status='passed', checked_at_unix=time.time(), url=state['url'],
                  source_epochs=len(rows), remote_verified_epochs=minimum,
                  initial_backfilled_epochs=initial['historical_rows_at_start'],
                  new_epochs_verified=minimum-initial['historical_rows_at_start'],
                  metric_series_verified=len(comparisons), comparisons=comparisons,
                  original_training_pid=before['training_pid'], training_process_identity_unchanged=True,
                  training_command_unchanged=True, training_config_unchanged=True, training_code_unchanged=True,
                  no_training_restart=True, swanlab_version=swanlab.__version__)
    atomic_json(root / 'verification.json', report)
    print(json.dumps({k: report[k] for k in ('status','url','remote_verified_epochs','initial_backfilled_epochs','new_epochs_verified','metric_series_verified','no_training_restart')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
