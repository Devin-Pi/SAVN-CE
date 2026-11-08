"""Explicit prepare/test/smoke/train/monitor phases for three isolated GPU jobs."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PROJECT=Path('/media/4tbssd/00_PROJECT_CODE/SAVN-CE')
CODE=PROJECT/'outputs/avn_candidate_shadow_worktree'
WORK=Path('/media/4tbssd/av_seld/workspaces')
EXPERIMENT=WORK/'avn_short_windows_v1_20261108'
IMAGE='rdpi/avseld_final:latest'
JOBS=[('w025',.25,0),('w100',1.,1),('w150',1.5,2)]
PRETRAIN='/workspaces/av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):(EXPERIMENT/name).write_text(json.dumps(value,indent=2)+'\n')
def inspect(name):return json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]


def command(name,gpu,output):
    uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    output.mkdir(exist_ok=True,parents=True)
    return ['docker','run','--name',name,'--runtime=nvidia','--gpus','device='+uuid,
        '--shm-size=4g','--init','--network','none',
        '-e','CUDA_VISIBLE_DEVICES=0','-e','PYTHONPATH=/training_source','-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2',
        '-e','OPENBLAS_NUM_THREADS=1','-e','NUMBA_NUM_THREADS=1',
        '-v',str(WORK)+':/workspaces:ro',
        '-v',str(PROJECT/'research_outputs/AVN-FOA-SameRoom-v0.2-Dense15h')+':/datasets/main:ro',
        '-v',str(EXPERIMENT/'labels50')+':/labels50:ro',
        '-v',str(EXPERIMENT/'code_snapshot')+':/training_source:ro',
        '-v',str(output)+':/experiment:rw','--entrypoint','python',IMAGE]


def training_args(window,gpu,output,smoke=False):
    args=['-u','-m','avn_foa_sameroom.train_short','--data','/datasets/main','--labels','/labels50',
        '--runtime','/workspaces/dsf_localization_runtime_v1','--checkpoint',PRETRAIN,
        '--output',output,'--window-s',str(window),'--physical-gpu',str(gpu),
        '--epochs','100','--batch','16','--workers','4','--seed','7','--encoder-lr','1e-5','--head-lr','1e-4']
    return args+(['--smoke'] if smoke else [])


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','test','smoke','train','monitor','status']);a=p.parse_args()
    if a.phase=='prepare':
        assert json.loads((EXPERIMENT/'labels50/validation.json').read_text())['status']=='completed'
        snap=EXPERIMENT/'code_snapshot';snap.mkdir()
        for name in ('avn_foa_sameroom','spatial_audio'):
            shutil.copytree(CODE/name,snap/name,ignore=shutil.ignore_patterns('__pycache__','._*'))
        (snap/'scripts').mkdir()
        for name in ('launch_short_windows.py','short_swanlab_supervisor.py'):
            shutil.copy2(CODE/'scripts'/name,snap/'scripts'/name)
        write('source_manifest.json',dict(source_sha256={str(f.relative_to(snap)):sha(f) for f in snap.rglob('*.py')},
            image_id=subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip(),
            label_cache_validation_sha256=sha(EXPERIMENT/'labels50/validation.json'),
            diagnosis_completed_before_training=True,jobs=JOBS))
        print('SHORT_SOURCE_FROZEN');return
    manifest=json.loads((EXPERIMENT/'source_manifest.json').read_text())
    for name,expected in manifest['source_sha256'].items():assert sha(EXPERIMENT/'code_snapshot'/name)==expected,name
    if a.phase=='test':
        cmd=command('avn_short_contract_v1',3,EXPERIMENT/'verification')+['-m','avn_foa_sameroom.test_short_window',
            '--data','/datasets/main','--labels','/labels50','--runtime','/workspaces/dsf_localization_runtime_v1',
            '--output','/experiment/contract_tests.json']
        write('contract_test_launch.json',dict(command=cmd));subprocess.run(cmd,check=True);return
    if a.phase in ('smoke','train'):
        assert json.loads((EXPERIMENT/'verification/contract_tests.json').read_text())['status']=='passed'
        if a.phase=='train':
            tests=[json.loads((EXPERIMENT/'runs'/label/'smoke_batch16/smoke_result.json').read_text()) for label,_,_ in JOBS]
            assert all(r['status']=='passed' for r in tests)
            assert len({r['initial_model_state_sha256'] for r in tests})==len({r['initial_head_sha256'] for r in tests})==1
            for label,_,_ in JOBS:
                info=inspect(f'avn_short_{label}_smoke_v1');assert not info['State']['Running'] and info['State']['ExitCode']==0
            write('smoke_comparison.json',dict(status='passed',all_three_initial_models_identical=True,
                all_three_initial_heads_identical=True,all_three_encoder_gradients_and_updates_passed=True,results=tests))
        launched=[]
        for label,window,gpu in JOBS:
            used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
            assert used<1024,f'GPU{gpu} busy; do not launch onto another task'
            phase=a.phase
            run=EXPERIMENT/'runs'/label
            target='smoke_batch16' if phase=='smoke' else 'run_seed7_full100'
            assert not (run/target).exists()
            cmd=command(f'avn_short_{label}_{phase}_v1',gpu,run)
            cmd.insert(2,'-d')
            cmd+=training_args(window,gpu,'/experiment/'+target,phase=='smoke')
            identifier=subprocess.check_output(cmd,text=True).strip()
            launched.append(dict(label=label,window_s=window,physical_gpu=gpu,container=identifier,command=cmd))
            print('LAUNCHED',phase,label,identifier,flush=True)
        write(a.phase+'_launch.json',dict(jobs=launched));return
    if a.phase=='monitor':
        launched=[]
        for label,window,gpu in JOBS:
            info=inspect(f'avn_short_{label}_train_v1');assert info['State']['Running']
            run=EXPERIMENT/'runs'/label/'run_seed7_full100';assert (run/'config.json').exists()
            monitor=EXPERIMENT/'monitors'/label;monitor.mkdir(parents=True,exist_ok=False)
            cmd=['python3','-u',str(EXPERIMENT/'code_snapshot/scripts/short_swanlab_supervisor.py'),
                '--python',str(WORK/'avn_swanlab_monitor_v1/venv/bin/python'),
                '--collector',str(EXPERIMENT/'code_snapshot/avn_foa_sameroom/swanlab_monitor.py'),
                '--run-dir',str(run),'--state-dir',str(monitor/'state'),
                '--training-pid',str(info['State']['Pid']),'--label',label,
                '--source',str(EXPERIMENT/'code_snapshot/avn_foa_sameroom/train_short.py')]
            log=(monitor/'monitor.log').open('w')
            process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            launched.append(dict(label=label,pid=process.pid,command=cmd))
            print('MONITOR_STARTED',label,process.pid,flush=True)
        write('monitor_launch.json',dict(jobs=launched));return
    for label,window,gpu in JOBS:
        info=inspect(f'avn_short_{label}_train_v1')
        path=EXPERIMENT/'runs'/label/'run_seed7_full100/history.jsonl'
        history=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        state=EXPERIMENT/'monitors'/label/'state/status.json'
        monitor=json.loads(state.read_text()) if state.exists() else {}
        print(json.dumps(dict(label=label,window_s=window,physical_gpu=gpu,running=info['State']['Running'],
            completed_epochs=len(history),swanlab_status=monitor.get('status'),url=monitor.get('url'),
            last_enqueued_epoch=monitor.get('last_enqueued_epoch'))),flush=True)


if __name__=='__main__':main()
