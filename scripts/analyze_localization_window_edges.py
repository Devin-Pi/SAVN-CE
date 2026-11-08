"""Separate ongoing-window history from source content on one unchanged fixed RIR."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_localization_controls import Analysis,to_training_level,truth_in_anchor,RATE
from spatial_audio.foa import native_to_world,world_foa_to_listener


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    out=a.data/'window_edges';out.mkdir(exist_ok=False)
    root=a.data/'static_continuous_seed7'
    logs=[json.loads(l) for l in (root/'actions.jsonl').read_text().splitlines()]
    meta=json.loads((root/'collection.json').read_text())
    dry,_=sf.read(root/'emitted_source.wav',dtype='float32')
    with np.load(root/'initial_rir.npz') as z:ir=native_to_world(z['native'])
    analysis=Analysis(a.data)
    from avn_foa_sameroom.dense_synthesis import synthesize_dense,supervision
    for start in (0,3,6):
        cut=dry[start*RATE:(start+10)*RATE].copy()
        q=logs[(start+10)*4-1]['rotation_xyzw']
        truth=truth_in_anchor(logs[start*4:(start+10)*4],meta['evaluation_only']['source_position'],q)
        for view in ('fresh_convolution','fresh_50ms_onset','densified_same_cut'):
            signal=cut.copy()
            if view=='fresh_50ms_onset':
                signal=np.r_[np.zeros(800,np.float32),cut[:-800]].astype(np.float32)
                signal[800:880]*=np.linspace(0,1,80,dtype=np.float32)
            if view=='densified_same_cut':
                wave,label,details=synthesize_dense(cut,ir,np.array([1.,0,0]),0,21,np.random.default_rng(7))
            else:
                wave=fftconvolve(signal[:,None],ir,axes=0)[:160000].astype(np.float32)
                label,_=supervision(signal,wave,ir,np.array([1.,0,0]),0,21)
            wave,gain=to_training_level(wave)
            active=label['activity'].astype(bool);valid=~label['ignore_mask'].astype(bool)
            analysis.predict(f'{view}__source_start{start}',wave,q,active,valid,np.zeros(100,bool),truth,
                dict(view=view,source_window_start_s=start,global_gain=gain))
            with torch.inference_mode():
                local=world_foa_to_listener(wave,q)
                f=analysis.candidate.front(torch.from_numpy(local.T.copy())[None].cuda())
                analysis.rows[-1]['iv_temporal_std_median_by_channel']=f[:,4:].std(dim=2).median(dim=-1).values.cpu()[0].tolist()
                analysis.rows[-1]['iv_temporal_mean_abs_median_by_channel']=f[:,4:].mean(dim=2).abs().median(dim=-1).values.cpu()[0].tolist()
    (out/'diagnostics.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in analysis.rows))
    np.savez_compressed(out/'raw_predictions.npz',**analysis.raw)
    (out/'summary.json').write_text(json.dumps(dict(status='completed',views=len(analysis.rows),
        scope='diagnostic resynthesis on original first RIR; fresh convolution removes pre-window tail; 50ms delay changes onset alignment; densification changes crop/retrigger schedule. None is used to alter a running Agent observation.',
        no_new_training=True,no_threshold_tuning=True),indent=2)+'\n')
    print('WINDOW_EDGE_COMPLETE',len(analysis.rows),flush=True)


if __name__=='__main__':main()
