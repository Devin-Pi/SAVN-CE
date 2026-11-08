"""Archive the completed dense candidate without changing Agent configuration."""
import argparse,fcntl,hashlib,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--card-dir',type=Path,required=True);a=p.parse_args()
    card=json.loads((a.card_dir/'candidate.json').read_text());source=Path(card['source_training_run'])
    destination=Path(card['checkpoint_host']).parent;destination.mkdir(parents=True,exist_ok=True)
    lock=(destination/'.archive.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result=json.loads((source/'results.json').read_text())
    assert result['completed_epochs']==100 and result['all_requested_epochs_completed']
    assert result['best_epoch']==card['selected_epoch']==97
    expected=card['checkpoint_sha256'];assert sha(source/'best.pt')==expected==result['best_checkpoint_sha256']
    checkpoint=destination/'best.pt'
    if checkpoint.exists():assert sha(checkpoint)==expected,'Existing candidate differs; refuse overwrite'
    else:
        staged=destination/'.best.pt.copying';shutil.copyfile(source/'best.pt',staged)
        assert sha(staged)==expected
        os.replace(staged,checkpoint)
    checkpoint.chmod(0o444)
    (destination/'candidate.json').write_text(json.dumps(card,indent=2,ensure_ascii=False)+'\n')
    readme=(a.card_dir/'README_zh-CN.md').read_text().replace('../../reports/avn_foa_dense15h_full100/ANALYSIS_zh-CN.md','reports/ANALYSIS_zh-CN.md')
    (destination/'README_zh-CN.md').write_text(readme)
    reports=destination/'reports';reports.mkdir(exist_ok=True)
    for name in ('results.json','completed_run_validation.json','posthoc_test_metric_check.json','test_per_class.csv','reference_test_per_class.csv'):
        shutil.copyfile(source/name,reports/name)
    analysis=source.parent/'analysis_full100_20261108'
    for name in ('comparison.json','no_event_audit.json','final_verification.json','ANALYSIS_zh-CN.md','class_comparison.csv'):
        shutil.copyfile(analysis/name,reports/name)
    assert sha(checkpoint)==sha(source/'best.pt')==expected
    report=dict(status='passed',candidate_id=card['candidate_id'],archived_utc=datetime.now(timezone.utc).isoformat(),
        source_checkpoint=str(source/'best.pt'),archived_checkpoint=str(checkpoint),checkpoint_sha256=expected,
        checkpoint_bytes=checkpoint.stat().st_size,independent_copy=True,source_unchanged=True,
        agent_configuration_modified=False,activated=False)
    assert checkpoint.stat().st_ino!=(source/'best.pt').stat().st_ino
    (destination/'archive_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    files=[f for f in sorted(destination.rglob('*')) if f.is_file() and f.name not in ('.archive.lock','SHA256SUMS')]
    (destination/'SHA256SUMS').write_text(''.join(sha(f)+'  '+str(f.relative_to(destination))+'\n' for f in files))
    print(json.dumps(report))

if __name__=='__main__':main()
