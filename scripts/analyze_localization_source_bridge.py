"""Post-factorial source/RIR ablations, frozen model and no production changes."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_localization_controls import Analysis,to_training_level,diagnostic_masks,truth_in_anchor,RATE
from spatial_audio.foa import native_to_world


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    out=a.data/'source_bridge';out.mkdir(exist_ok=False)
    run=a.data/'static_continuous_seed7'
    meta=json.loads((run/'collection.json').read_text())
    logs=[json.loads(s) for s in (run/'actions.jsonl').read_text().splitlines()]
    with np.load(run/'initial_rir.npz') as z:ir=z['native']
    support=np.flatnonzero(np.max(np.abs(ir),axis=1)>max(float(np.abs(ir).max())*1e-5,1e-12))
    for row in logs:row.setdefault('rir_onset_samples',int(support[0]))
    dry,_=sf.read(run/'emitted_source.wav',dtype='float32')
    frozen,_=sf.read(run/'frozen_initial_rir_full.wav',dtype='float32',always_2d=True)
    native_source,_=sf.read(run/'native_normalized_source.wav',dtype='float32')
    analysis=Analysis(a.data)
    from avn_foa_sameroom.synthesis import read_source
    from avn_foa_sameroom.dense_synthesis import active_regions,synthesize_dense,reconstruct_timeline,supervision
    source=read_source(a.data/'original_chair.wav')
    regions,profile=active_regions(source)
    selected=np.zeros(len(source),bool)
    for start,end in regions:selected[start:end]=True
    source_indices=(meta['initial_source_offset_samples']+np.arange(len(dry)))%len(native_source)
    selection=selected[source_indices]
    active,valid,quiet=diagnostic_masks(dry,frozen,logs)
    irworld=native_to_world(ir)
    source_windows=[]
    for start in (0,3,6):
        wave=frozen[start*RATE:(start+10)*RATE].copy();q=logs[(start+10)*4-1]['rotation_xyzw']
        truth=truth_in_anchor(logs[start*4:(start+10)*4],meta['evaluation_only']['source_position'],q)
        mask=slice(start*10,(start+10)*10)
        scaled,gain=to_training_level(wave)
        analysis.predict(f'frozen_native_stream__{start}',scaled,q,active[mask],valid[mask],quiet[mask],truth,
            dict(view='fixed_rir_native_stream',start_s=start,global_gain=gain))
        stream=dry[start*RATE:(start+10)*RATE]
        source_windows.append(dict(start_s=start,selected_region_sample_fraction=float(selection[start*RATE:(start+10)*RATE].mean()),
            source_rms=float(np.sqrt(np.mean(stream.astype(float)**2))),
            source_100ms_rms=np.sqrt(np.mean(stream.reshape(100,1600).astype(float)**2,axis=1)).tolist()))
        if start in (3,6):
            for db in (-35.,-20.):
                gain=10**(db/20)/max(float(np.sqrt(np.mean(wave[:,0].astype(float)**2))),1e-12)
                analysis.predict(f'frozen_native_stream__{start}__level{db}',(wave*gain).astype(np.float32),q,
                    active[mask],valid[mask],quiet[mask],truth,
                    dict(view='fixed_rir_gain_endpoint',start_s=start,fixed_w_dbfs=db,global_gain=gain))
    q=logs[39]['rotation_xyzw'];truth=truth_in_anchor(logs[:40],meta['evaluation_only']['source_position'],q)
    for seed in (7,8,9):
        dense,labels,details=synthesize_dense(source,irworld,np.asarray([1.,0,0]),0,21,np.random.default_rng(seed))
        scaled,gain=to_training_level(dense)
        active=labels['activity'].astype(bool);valid=~labels['ignore_mask'].astype(bool)
        analysis.predict(f'dense_crops__polyphase__seed{seed}',scaled,q,active,valid,np.zeros(100,bool),truth,
            dict(view='fixed_rir_dense_crops_polyphase',seed=seed,global_gain=gain))
        # Identical event times, source indices, fades and RIR; substitute only
        # the native resampler/normalizer version of that very same source WAV.
        stream=reconstruct_timeline(native_source,details['events'])
        wet=fftconvolve(stream[:,None],irworld,axes=0)[:160000].astype(np.float32)
        wet,gain=to_training_level(wet)
        label,_=supervision(stream,wet,irworld,np.asarray([1.,0,0]),0,21)
        analysis.predict(f'dense_crops__native_source__seed{seed}',wet,q,label['activity'].astype(bool),
            ~label['ignore_mask'].astype(bool),np.zeros(100,bool),truth,
            dict(view='fixed_rir_dense_crops_native_source',seed=seed,global_gain=gain))
    (out/'diagnostics.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in analysis.rows))
    np.savez_compressed(out/'raw_predictions.npz',**analysis.raw)
    (out/'summary.json').write_text(json.dumps(dict(status='completed',source_active_region_profile=profile,
        source_windows=source_windows,views=len(analysis.rows),
        scope='Same scene, source, first actual RIR, orientation and frozen model. Fixed gain endpoints are sensitivity checks, not production selection.',
        new_training=False,threshold_changed=False,navigation_control=False),indent=2)+'\n')
    print('SOURCE_BRIDGE_COMPLETE',len(analysis.rows),flush=True)


if __name__=='__main__':main()
