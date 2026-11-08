"""Host-side native collection orchestration, no model training or navigation policy."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path('/media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree')
SIMROOT = '/workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree'


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='localization_diagnosis_20261108');a=p.parse_args()
    if Path(a.name).name!=a.name:raise ValueError('Use a single output folder name')
    rel='outputs/'+a.name
    output = ROOT / rel
    output.mkdir(exist_ok=True)
    conditions = [(seed, motion, schedule, True) for seed in (7,8,9)
                  for motion in ('static','motion') for schedule in ('continuous','intermittent')]
    conditions += [(7,m,s,False) for m in ('static','motion') for s in ('continuous','intermittent')]
    manifest = dict(episode='1000000008', seeds=[7,8,9],
        factorial='2x2 static/motion x continuous/intermittent, 3 paired source-offset seeds',
        temporal_sensitivity='seed7 all four conditions, native true vs explicitly false',
        waveform='native16k, 64x0.25s, windows[0,10),[3,13),[6,16); no padding',
        inference='frozen epoch97 candidate; activity threshold0.5; no optimizer/no policy',
        controls='native waveform vs a uniform -27.5dBFS W-level; first fixed RIR whole vs chunked; dense crop same-source/same-RIR bridge; stored offline chair positives',
        seeds_do_not_guarantee='opaque RLR Monte Carlo bitwise repeatability; overlapping windows and same source are not independent samples',
        cases=[], navigation_control=False, checkpoint_selection_or_threshold_tuning=False)
    for seed,motion,schedule,temporal in conditions:
        name=f'{motion}_{schedule}_seed{seed}'+('' if temporal else '_temporalfalse')
        cmd=['docker','exec','-e','HABITAT_SIM_LOG=quiet','-e','MAGNUM_LOG=quiet','-w',SIMROOT,
             'savnce_dev','python','scripts/collect_localization_controls.py',
             '--output',f'{rel}/{name}','--actions',
             'outputs/avn_candidate_shadow_1000000008_codex_seed7_64_20261108/navigation/actions.jsonl',
             '--motion',motion,'--schedule',schedule,'--seed',str(seed),'--temporal',str(temporal).lower()]
        manifest['cases'].append(dict(case=name,command=shlex.join(cmd)))
    (output/'design.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for case in manifest['cases']:
        name=case['case']
        if not (output/name/'collection.json').exists():
            with (output/f'{name}.log').open('w') as log:
                subprocess.run(shlex.split(case['command']),stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COLLECTED',name,flush=True)
    (output/'collection_complete.json').write_text(json.dumps(dict(status='passed',cases=len(conditions)),indent=2)+'\n')


if __name__=='__main__':
    main()
