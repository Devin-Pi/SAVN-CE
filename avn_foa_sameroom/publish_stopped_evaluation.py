"""Publish final early-stop metrics from saved evaluation artifacts, CPU only."""
import argparse
import json
import os
from pathlib import Path
import stat
import sys
import time

try:
    from .swanlab_monitor import RedactedStream, atomic_json
except ImportError:
    from swanlab_monitor import RedactedStream, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--monitor-root', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, default=Path.home()/'.config/avn-swanlab/api_key')
    args = parser.parse_args()
    result = json.loads((args.evaluation_dir/'results.json').read_text())
    assert result['status'] == 'completed' and result['termination'] == 'user_requested_early_stop'
    assert result['validation_checkpoint_reproduced'] and not result['test_used_for_selection']
    assert result['test_dataset_evaluation_passes'] == 1
    state_file = args.monitor_root/'state/status.json'
    state = json.loads(state_file.read_text())
    publication = args.monitor_root/'state/final_evaluation_publication.json'
    if publication.exists():
        previous = json.loads(publication.read_text())
        assert previous['checkpoint_sha256'] == result['best_checkpoint_sha256']
        print('Final metrics already published',previous['url'])
        return
    info = args.key_file.stat()
    assert stat.S_IMODE(info.st_mode) & 0o077 == 0 and info.st_uid == os.getuid()
    key = args.key_file.read_text().strip()
    sys.stdout = RedactedStream(sys.stdout,key);sys.stderr = RedactedStream(sys.stderr,key)
    os.environ['SWANLAB_API_KEY'] = key
    os.environ['SWANLAB_ROOT'] = str(args.monitor_root/'state/.swanlab')
    os.environ['SWANLAB_DISABLE_GIT'] = 'True'
    import swanlab
    settings = swanlab.Settings(interactive=False, terminal={'proxy_type':'none'},
        probe={'git':False,'runtime':False,'requirements':False,'conda':False,'swanlab':False,'hardware':False,'monitor':False})
    run = swanlab.init(project=state['project'],id=state['run_id'],resume='must',mode='online',on_online_error='raise',
        public=False,logdir=str(args.monitor_root/'state/swanlog'),settings=settings,
        description=f"User-requested stop after {result['completed_epochs']} complete epochs; original plan {result['requested_epochs']}. Final validation and test completed.",
        config=dict(original_planned_epochs=result['requested_epochs'],actual_completed_epochs=result['completed_epochs'],
                    termination='user_requested_early_stop',all_planned_epochs_completed=False,
                    final_best_epoch=result['best_epoch'],test_used_for_selection=False,
                    effective_float32_matmul_precision=result['effective_float32_matmul_precision']))
    final={f'test/{k}':result['test'][k] for k in ('ER','F20','LE','LR','SELD','loss')}
    final.update({'best/epoch':result['best_epoch'],'best/val_SELD':result['saved_validation']['SELD'],
                  'final/completed_epochs':result['completed_epochs'],'final/planned_epochs':result['requested_epochs'],
                  'final/stopped_by_user':1})
    final.update({f'validation_recheck/{k}':result['validation'][k] for k in ('ER','F20','LE','LR','SELD')})
    step=result['completed_epochs']+1
    swanlab.log(final,step=step)
    swanlab.log({'report/final_validation_and_test':swanlab.Text((args.evaluation_dir/'README_zh-CN.md').read_text())},step=step)
    url=run.url
    swanlab.finish(state='success')
    record=dict(status='published',published_at_unix=time.time(),url=url,run_id=state['run_id'],step=step,
                completed_epochs=result['completed_epochs'],planned_epochs=result['requested_epochs'],
                best_epoch=result['best_epoch'],checkpoint_sha256=result['best_checkpoint_sha256'],
                source_results=str(args.evaluation_dir/'results.json'),metric_keys=list(final),
                termination='user_requested_early_stop',complete_100_epochs=False)
    atomic_json(publication,record)
    state.update(status='completed_user_requested_early_stop',last_enqueued_epoch=result['completed_epochs'],
                 actual_completed_epochs=result['completed_epochs'],original_planned_epochs=result['requested_epochs'],
                 training_process_alive=False,final_test_enqueued=True,evaluation_directory=str(args.evaluation_dir),
                 termination='user_requested_early_stop',completed_at_unix=time.time())
    atomic_json(state_file,state)
    print('FINAL_EVALUATION_PUBLISHED',json.dumps(record),flush=True)


if __name__=='__main__':
    main()
