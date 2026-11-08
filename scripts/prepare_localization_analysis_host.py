"""Copy only diagnostic observations and selected immutable offline controls to model mount."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

PROJECT = Path('/media/4tbssd/00_PROJECT_CODE/SAVN-CE')
CODE = PROJECT/'outputs/avn_candidate_shadow_worktree'
NAME = 'localization_diagnosis_20261108'
DATA = PROJECT/'research_outputs/AVN-FOA-SameRoom-v0.2-Dense15h'
WORK = Path('/media/4tbssd/av_seld/workspaces')


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default=NAME);a=p.parse_args()
    if Path(a.name).name!=a.name:raise ValueError('Use a single output folder name')
    src = CODE/'outputs'/a.name
    assert json.loads((src/'collection_complete.json').read_text())['cases']==16
    out = WORK/a.name
    shutil.copytree(src,out)
    code = out/'code'
    code.mkdir()
    for name in ('spatial_audio','agent_episode','scripts'):
        (code/name).mkdir()
    for name in ('spatial_audio/__init__.py','spatial_audio/foa.py','agent_episode/candidate_worker.py',
                 'agent_episode/spatial_frames.py','scripts/analyze_localization_controls.py',
                 'scripts/analyze_localization_source_bridge.py','scripts/analyze_localization_window_edges.py'):
        dest = code/name
        shutil.copy2(CODE/name,dest)
    offline=out/'offline_controls';offline.mkdir()
    rows=[json.loads(l) for l in (DATA/'metadata.jsonl').read_text().splitlines()]
    selected=[r for r in rows if r['split']=='test' and r['official_category']=='chair' and not r['is_no_event']]
    selected += [r for r in rows if r['split']=='test' and r['is_no_event']][:8]
    selected += [r for r in rows if r['split']=='train' and r['official_category']=='chair'
                 and r['scene_id']=='17DRP5sb8fy' and not r['is_no_event']][:4]
    assert sum(r['split']=='test' and not r['is_no_event'] for r in selected)==24
    for row in selected:
        for key in ('audio','labels'):
            path=DATA/row[key]
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row[key+'_sha256']
            dest=offline/row[key];dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(path,dest)
    (offline/'metadata.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected))
    shutil.copy2(WORK/'avn_foa_dense15h_finetune_v02/run_seed7_full100/test_predictions.npz',offline/'stored_predictions.npz')
    shutil.copy2(PROJECT/'data/sounds/semantic_splits/train/chair.wav',out/'original_chair.wav')
    (out/'preparation.json').write_text(json.dumps(dict(status='prepared',offline_records=len(selected),
        all_selected_dataset_hashes_verified=True,original_data_modified=False,
        source_package_sha256={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*.py')}),indent=2)+'\n')
    print('PREPARED',out,'offline controls',len(selected),flush=True)


if __name__=='__main__':
    main()
