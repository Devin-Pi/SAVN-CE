"""Compare the exact pilot implementation with the canonical module on recorded observations."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from spatial_audio import foa


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    old=module(a.run/'source_snapshot/spatial_frames.py','pilot_frames');old_native=module(a.run/'source_snapshot/foa.py','pilot_native')
    fixture=json.loads(Path('tests/fixtures/foa_28_calibrated_build.json').read_text())
    for case in fixture['cases']:
        native=np.asarray([case['native_w_normalized']],np.float32)
        np.testing.assert_array_equal(old_native.to_ambix(native),foa.native_to_world(native))
    results=[]
    for record in map(json.loads,(a.run/'predictions.jsonl').read_text().splitlines()):
        if record['status']!='ok':continue
        chunks=[]
        for ident in record['provenance']['observation_ids']:
            with np.load(a.run/'foa_shadow/observations'/f'{ident}.npz') as obs:chunks.append(obs['foa_audio'])
        wave=np.concatenate(chunks);q=record['pose_refs'][-1]['robot_pose']['rotation_xyzw']
        prior=old.world_foa_to_listener(wave,q);current=foa.world_foa_to_listener(wave,q)
        np.testing.assert_array_equal(prior,current)
        results.append(dict(anchor=record['at'],max_abs_error=float(np.max(np.abs(prior-current))),bitwise_identical=True))
    assert len(results)==8
    report=dict(status='passed',conversion_contract_version=foa.CONTRACT_VERSION,canonical_module_sha256=hashlib.sha256(Path(foa.__file__).read_bytes()).hexdigest(),
        pilot_native_to_world_cases=28,windows=results,scope='Recorded pilot input conversion equivalence, not a new autonomous run or an inference-quality certificate')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
