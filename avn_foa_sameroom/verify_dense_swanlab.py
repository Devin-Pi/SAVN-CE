"""Check real online dense/sparse metric delivery without judging model quality."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
try:
    from .swanlab_monitor import complete_history,epoch_metrics,process_identity,RedactedStream,atomic_json
except ImportError:
    from swanlab_monitor import complete_history,epoch_metrics,process_identity,RedactedStream,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args()
    root=a.experiment;state=json.loads((root/'monitor/state/status.json').read_text())
    assert state['status'] in ('following','completed')
    rows=complete_history(root/'run_seed7_full100/history.jsonl');assert rows,'No completed epoch yet'
    source={r['epoch']:epoch_metrics(r) for r in rows};keys=list(source[1])
    key=(Path.home()/'.config/avn-swanlab/api_key').read_text().strip()
    sys.stdout=RedactedStream(sys.stdout,key);sys.stderr=RedactedStream(sys.stderr,key)
    import swanlab
    remote=swanlab.Api(api_key=key).run(f"rdpi1995/{state['project']}/{state['run_id']}")
    series=remote.metrics(keys=keys,ignore_timestamp=True)['list'];assert {s['key'] for s in series}==set(keys)
    covered=[]
    for s in series:
        points={int(p['index']):float(p['data']) for p in s['metrics']};assert points
        end=min(max(source),max(points));assert end>=1
        assert sorted(k for k in points if k<=end)==list(range(1,end+1))
        for i in range(1,end+1):assert math.isclose(points[i],source[i][s['key']],rel_tol=1e-6,abs_tol=1e-8),(s['key'],i)
        covered.append(end)
    manifest=json.loads((root/'launch_manifest.json').read_text())
    for name,expected in manifest['source_sha256'].items():
        assert hashlib.sha256((root/'code_snapshot'/name).read_bytes()).hexdigest()==expected,name
    assert process_identity(state['training_process']['pid'])==state['training_process']
    report=dict(status='passed',url=state['url'],remote_verified_epochs=min(covered),metric_series_verified=len(keys),
                source_code_unchanged=True,training_process_identity_unchanged=True,checked_at_unix=time.time())
    atomic_json(root/'monitor/verification.json',report);print(json.dumps(report))

if __name__=='__main__':main()
