"""Derive real 50ms supervision from saved arrival intervals, never repeat 100ms labels."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf

VERSION='avn-arrival-labels50-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def activity(occupancy,power,floor):
    audible=np.sqrt(power)>floor
    ignore=((occupancy>0)&(occupancy<.8))|((occupancy==0)&audible)|((occupancy>=.8)&(~audible))
    return (occupancy>=.8)&(~ignore),~ignore


def rebuild(row,data,out):
    path=out/(row['sample_id']+'.npz')
    wave,rate=sf.read(data/row['audio'],dtype='float32',always_2d=True)
    assert rate==16000 and wave.shape==(160000,4)
    arriving=np.zeros(160000,bool)
    for lo,hi in row['received_direct_activity_intervals_s']:
        arriving[max(0,round(lo*16000)):min(160000,round(hi*16000))]=True
    occupancy=arriving.reshape(200,800).mean(axis=1)
    power=np.mean(wave.reshape(200,800,4).astype(np.float64)**2,axis=(1,2))
    floor=float(row.get('wet_audibility_rms_floor',max(float(np.sqrt(power.reshape(100,2).mean(1)).max())*1e-4,1e-9)))
    active,valid=activity(occupancy,power,floor)
    a100,v100=activity(occupancy.reshape(100,2).mean(1),power.reshape(100,2).mean(1),floor)
    with np.load(data/row['labels'],allow_pickle=False) as old:
        np.testing.assert_array_equal(a100,old['activity'].astype(bool))
        np.testing.assert_array_equal(v100,~old['ignore_mask'].astype(bool))
        if a100.any():
            np.testing.assert_allclose(old['direction_xyz'][a100],np.repeat(np.asarray(row['audio_direction_xyz'])[None],int(a100.sum()),axis=0),atol=1e-6)
    temporary=path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,occupancy=occupancy,power=power,audibility_floor=floor,
        activity=active,valid=valid,unit=np.asarray(row['audio_direction_xyz'],np.float32),class_id=row['class_id'],
        original_metadata_sha256=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest())
    temporary.replace(path)
    return dict(sample_id=row['sample_id'],split=row['split'],active_frames50=int(active.sum()),
        valid_frames50=int(valid.sum()),reconstruction100_passed=True,sha256=sha(path))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in (a.data/'metadata.jsonl').read_text().splitlines()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        results=[]
        for row in pool.map(lambda r:rebuild(r,a.data,a.output),rows):
            results.append(row)
            if len(results)%500==0:print('LABELS50',len(results),flush=True)
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    manifest=dict(status='completed',version=VERSION,rows=len(rows),rate_hz=20,
        source='saved sample-accurate received_direct_activity_intervals_s and original FOA waveform power; not repeated old labels',
        activity_rule='>=80% causal arrival occupancy; partial/tail-only/inaudible ignored; original fixed audibility floor',
        all_original_100ms_activity_and_ignore_masks_reconstructed=True,
        original_dataset_unmodified=True,dataset_seal_file_sha256=sha(a.data/'SHA256SUMS'),
        source_metadata_sha256=sha(a.data/'metadata.jsonl'),code_sha256=sha(Path(__file__)),
        manifest_sha256=sha(a.output/'manifest.jsonl'))
    (a.output/'validation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('LABELS50_COMPLETE',json.dumps(manifest),flush=True)


if __name__=='__main__':main()
