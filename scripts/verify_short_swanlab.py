"""Check actual SwanLab metric delivery, never analyze or signal training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.experiment/'code_snapshot'))
    from avn_foa_sameroom.swanlab_monitor import complete_history,epoch_metrics,RedactedStream
    key=(Path.home()/'.config/avn-swanlab/api_key').read_text().strip()
    sys.stdout=RedactedStream(sys.stdout,key);sys.stderr=RedactedStream(sys.stderr,key)
    import swanlab
    api=swanlab.Api(api_key=key);results=[]
    for label in ('w025','w100','w150'):
        state=json.loads((a.experiment/'monitors'/label/'state/status.json').read_text())
        assert state['status'] in ('following','completed'),state['status']
        history=complete_history(a.experiment/'runs'/label/'run_seed7_full100/history.jsonl')
        assert history,'No completed epoch yet; retry after history is written'
        source={r['epoch']:epoch_metrics(r) for r in history};keys=list(source[1])
        run=api.run(f"rdpi1995/{state['project']}/{state['run_id']}")
        series=run.metrics(keys=keys,ignore_timestamp=True)['list'];assert {s['key'] for s in series}==set(keys)
        covered=[]
        for s in series:
            points={int(r['index']):float(r['data']) for r in s['metrics']}
            assert 1 in points,(label,s['key'],'first epoch not uploaded yet')
            end=min(max(source),max(points))
            for i in range(1,end+1):assert math.isclose(points[i],source[i][s['key']],rel_tol=1e-6,abs_tol=1e-8),(label,s['key'],i)
            covered.append(end)
        results.append(dict(label=label,status='passed',url=state.get('url'),
            remotely_verified_epochs=min(covered),metric_series_verified=len(keys)))
    manifest=json.loads((a.experiment/'source_manifest.json').read_text())
    for path,expected in manifest['source_sha256'].items():
        assert hashlib.sha256((a.experiment/'code_snapshot'/path).read_bytes()).hexdigest()==expected,path
    report=dict(status='passed',no_training_process_modified=True,source_snapshot_unchanged=True,
        checked_at_unix=time.time(),runs=results)
    (a.experiment/'monitor_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
