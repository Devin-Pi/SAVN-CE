"""Read-only JSONL -> SwanLab bridge, separate from the training interpreter.

No torch import, checkpoint loading, training signals, or writes to run_dir.
Restart replays the same immutable epoch records to the same SwanLab run ID.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import time


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def complete_history(path):
    """Ignore an unfinished trailing write; reject gaps and conflicting duplicates."""
    if not path.exists():
        return []
    payload = path.read_bytes()
    end = payload.rfind(b'\n')
    if end < 0:
        return []
    rows = {}
    for line in payload[:end].splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        epoch = row['epoch']
        if type(epoch) is not int or epoch < 1:
            raise ValueError('Invalid epoch index')
        if epoch in rows and rows[epoch] != row:
            raise ValueError(f'Conflicting history at epoch {epoch}')
        rows[epoch] = row
    if sorted(rows) != list(range(1, len(rows) + 1)):
        raise ValueError('History epochs are not contiguous from epoch 1')
    return [rows[epoch] for epoch in sorted(rows)]


def epoch_metrics(row):
    values = {
        'train/loss': row['train_loss'],
        'training/epoch': row['epoch'],
        'training/global_step': row['global_step'],
        'lr/encoder': row['learning_rates'][0],
        'lr/head': row['learning_rates'][1],
        'timing/epoch_seconds': row['epoch_wall_s'],
        'timing/training_elapsed_seconds': row['elapsed_s'],
    }
    values.update({f'val/{key}': row['validation'][key] for key in ('ER', 'F20', 'LE', 'LR', 'SELD', 'loss')})
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite or nonnumeric history metric')
    return values


def process_identity(pid):
    try:
        p = Path('/proc') / str(pid)
        start = (p / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        return dict(pid=pid, start_ticks=start, command=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


class RedactedStream:
    """Keep the supplied credential out of collector stdout/stderr."""
    def __init__(self, stream, secret):
        self.stream, self.secret = stream, secret
    def write(self, value):
        return self.stream.write(value.replace(self.secret, '[REDACTED]'))
    def flush(self):
        return self.stream.flush()
    def __getattr__(self, name):
        return getattr(self.stream, name)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, default=Path.home() / '.config/avn-swanlab/api_key')
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--project', default='AVN-Localization')
    parser.add_argument('--poll-seconds', type=float, default=10)
    args = parser.parse_args()
    run_dir, state_dir = args.run_dir.resolve(), args.state_dir.resolve()
    if state_dir == run_dir or run_dir in state_dir.parents:
        raise ValueError('Collector state must be outside the training output directory')
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / 'collector.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    key_stat = args.key_file.stat()
    if stat.S_IMODE(key_stat.st_mode) & 0o077 or key_stat.st_uid != os.getuid():
        raise ValueError('Credential must be owned by this user with mode 0600 or stricter')
    key = args.key_file.read_text().strip()
    if not key:
        raise ValueError('Empty credential file')
    sys.stdout = RedactedStream(sys.stdout, key)
    sys.stderr = RedactedStream(sys.stderr, key)
    os.environ['SWANLAB_API_KEY'] = key
    os.environ['SWANLAB_DISABLE_GIT'] = 'True'
    os.environ['SWANLAB_ROOT'] = str(state_dir / '.swanlab')
    config_payload = (run_dir / 'config.json').read_bytes()
    config = json.loads(config_payload)
    config_sha = hashlib.sha256(config_payload).hexdigest()
    source_file = Path('/media/4tbssd/av_seld/workspaces/avn_foa_full_finetune_v1/code/avn_foa_sameroom/finetune.py')
    source_sha = hashlib.sha256(source_file.read_bytes()).hexdigest() if source_file.exists() else None
    identity = process_identity(args.training_pid)
    state_path = state_dir / 'status.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous.get('training_process') and identity is not None and identity != previous['training_process']:
        raise ValueError('Training process identity changed; inspect before reattaching')
    identity = identity or previous.get('training_process')
    identifier = 'avnfoa-full100-seed7-' + hashlib.sha256((str(run_dir) + config_sha).encode()).hexdigest()[:12]
    tracking_config = {k: config[k] for k in ('epochs', 'batch', 'seed', 'encoder_lr', 'head_lr', 'optimizer', 'loss', 'patience', 'precision')}
    tracking_config.update(
        sample_rate_hz=16000, num_classes=len(config['classes']),
        source_config_sha256=config_sha, source_script_sha256=source_sha,
        initial_checkpoint_sha256=config['initial_checkpoint_sha256'],
        dataset_seal_sha256=config['dataset_seal_sha256'],
        collector='independent read-only JSONL follower',
        augmentation=config['augmentation'],
        hardware_history='from collector start only; no historical GPU backfill',
        cpu_memory_scope='collector process, not the training interpreter',
        primary_training_gpu='physical GPU2; automatic GPU metrics cover host devices',
    )
    state = dict(status='connecting', run_id=identifier, project=args.project,
                 source_run=str(run_dir), training_process=identity,
                 source_config_sha256=config_sha, source_script_sha256=source_sha,
                 last_enqueued_epoch=0, collector_pid=os.getpid(),
                 started_at_unix=time.time(), historical_rows_at_start=len(complete_history(run_dir / 'history.jsonl')))
    atomic_json(state_path, state)
    import swanlab
    settings = swanlab.Settings(
        interactive=False,
        terminal={'proxy_type': 'none'},
        probe={'git': False, 'runtime': False, 'requirements': False, 'conda': False,
               'swanlab': False, 'hardware': True, 'monitor': True, 'monitor_interval': 30},
    )
    run = swanlab.init(
        project=args.project, experiment_name='FOA21-full100-seed7',
        id=identifier, resume='allow', mode='online', on_online_error='raise',
        public=False, logdir=str(state_dir / 'swanlog'), config=tracking_config, settings=settings,
        description='Backfilled and followed from the original fixed-100-epoch training history. Training process/config untouched.',
    )
    state.update(status='following', swanlab_version=swanlab.__version__)
    # SDK versions expose public URLs differently; console prints the canonical link.
    for name in ('url', 'project_url'):
        value = getattr(run, name, None)
        if isinstance(value, str):
            state[name] = value
    atomic_json(state_path, state)
    print(json.dumps(dict(event='collector_connected', run_id=identifier)), flush=True)
    sent, missing_since = {}, None
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        while True:
            rows = complete_history(run_dir / 'history.jsonl')
            if len(rows) < len(sent):
                raise ValueError('Training history was truncated')
            for row in rows:
                epoch = row['epoch']
                if epoch in sent:
                    if sent[epoch] != row:
                        raise ValueError('Previously submitted history changed')
                    continue
                swanlab.log(epoch_metrics(row), step=epoch)
                sent[epoch] = row
                state.update(last_enqueued_epoch=epoch, updated_at_unix=time.time())
                atomic_json(state_path, state)
            result_file = run_dir / 'results.json'
            if result_file.exists():
                try:
                    result = json.loads(result_file.read_text())
                except json.JSONDecodeError:
                    time.sleep(args.poll_seconds)
                    continue
                if result.get('status') != 'completed' or result['completed_epochs'] != config['epochs'] or len(sent) != config['epochs']:
                    raise ValueError('Results do not describe the complete requested training')
                final = {f'test/{k}': result['test'][k] for k in ('ER', 'F20', 'LE', 'LR', 'SELD', 'loss')}
                final.update({'best/epoch': result['best_epoch'], 'best/val_SELD': result['best_validation']['SELD']})
                swanlab.log(final, step=config['epochs'] + 1)
                swanlab.finish(state='success')
                state.update(status='completed', final_test_enqueued=True, completed_at_unix=time.time())
                atomic_json(state_path, state)
                print('COLLECTOR_COMPLETED', flush=True)
                return
            alive = identity is not None and process_identity(args.training_pid) == identity
            if not alive:
                missing_since = missing_since or time.monotonic()
                if time.monotonic() - missing_since > 120:
                    raise RuntimeError('Training process disappeared without complete results; collector did not signal it')
            else:
                missing_since = None
            state.update(updated_at_unix=time.time(), training_process_alive=alive)
            atomic_json(state_path, state)
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        swanlab.finish(state='aborted', error='Collector stopped; training process untouched')
        state.update(status='collector_stopped', updated_at_unix=time.time())
        atomic_json(state_path, state)
    except Exception as exc:
        safe = str(exc).replace(key, '[REDACTED]')
        swanlab.finish(state='crashed', error=safe)
        state.update(status='collector_error', error=safe, updated_at_unix=time.time())
        atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        print(f'COLLECTOR_STOPPED: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(2)
