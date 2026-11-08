"""Frozen-candidate diagnostic inference. Truth is consumed only after prediction."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'agent_episode'))
from candidate_worker import Candidate
from spatial_audio.foa import native_to_world, world_foa_to_listener, listener_basis

RATE = 16000


def stats(x):
    x = np.asarray(x)
    return None if not len(x) else dict(n=len(x), median=float(np.median(x)),
        q90=float(np.quantile(x,.9)), maximum=float(x.max()), above_threshold_fraction=float((x>.5).mean()))


def level(wave):
    rms = np.sqrt(np.mean(wave.astype(np.float64)**2,axis=0))
    return dict(channel_rms=rms.tolist(), w_dbfs=float(20*np.log10(max(rms[0],1e-30))),
        peak=float(np.abs(wave).max()), finite=bool(np.isfinite(wave).all()))


def to_training_level(wave):
    # Fixed middle of declared training W-level range; no labels or score-based tuning.
    gain = 10**(-27.5/20)/max(float(np.sqrt(np.mean(wave[:,0].astype(np.float64)**2))),1e-12)
    return (wave*gain).astype(np.float32),float(gain)


def diagnostic_masks(dry,wave,logs):
    p=np.sqrt(np.mean(dry.reshape(-1,160).astype(np.float64)**2,axis=1))
    micro=p>max(float(p.max())*.00316227766,1e-8)
    source_mask=np.repeat(micro,160)
    arriving=np.zeros(len(dry),bool)
    for i,row in enumerate(logs):
        delay=int(row['rir_onset_samples'])
        a=i*4000+delay;b=min(a+4000,len(dry))
        if b>a:arriving[a:b] |= source_mask[i*4000:i*4000+b-a]
    occupancy=arriving.reshape(-1,1600).mean(1)
    wet=np.sqrt(np.mean(wave.reshape(-1,1600,4).astype(np.float64)**2,axis=(1,2)))
    audible=wet>max(float(wet.max())*1e-4,1e-9)
    ignore=((occupancy>0)&(occupancy<.8))|((occupancy==0)&audible)|((occupancy>=.8)&(~audible))
    return (occupancy>=.8)&(~ignore),~ignore,wet<1e-7


class Analysis:
    def __init__(self,root):
        self.root=root
        self.candidate=Candidate(Path('/workspaces/avn_foa_dense15h_finetune_v02/code_snapshot'),
            Path('/workspaces/dsf_localization_runtime_v1'),
            Path('/workspaces/localization_candidates/avn_foa_dense15h_v02_epoch097/best.pt'))
        self.rows=[];self.raw={}

    def predict(self,key,wave,q,active,valid,quiet,truth=None,extra=None):
        # No target class, target direction, source position or mask enters predict().
        result=self.candidate.predict(np.ascontiguousarray(wave,dtype=np.float32),RATE,q)
        raw=np.asarray(result['raw_accdoa'],np.float32)
        scores=np.linalg.norm(raw.reshape(100,3,3,21),axis=2).max(1)
        decoded=np.zeros((100,21),bool);counts=Counter();angles=[]
        for i,events in enumerate(result['events_by_frame']):
            for e in events:
                c=e['class_id'];decoded[i,c]=True;counts[self.candidate.description['class_names'][c]]+=1
            if truth is not None and active[i] and decoded[i,0]:
                v=np.asarray(truth[i]);v=v/max(np.linalg.norm(v),1e-12)
                errors=[np.degrees(np.arccos(np.clip(np.dot(v,e['direction_unit_xyz']),-1,1))) for e in events if e['class_id']==0]
                angles.append(float(min(errors)))
        use=active&valid;negative=(~active)&valid
        local=world_foa_to_listener(wave,q)
        with torch.inference_mode():
            feature=self.candidate.front(torch.from_numpy(local.T.copy())[None].cuda())
            normalized=(feature-feature.mean(dim=2,keepdim=True))/feature.std(dim=2,keepdim=True).clamp(min=1e-8)
            feature_stats=dict(shape=list(feature.shape),per_channel_mean=feature.mean(dim=(0,2,3)).cpu().tolist(),
                per_channel_std=feature.std(dim=(0,2,3)).cpu().tolist(),
                normalized_per_channel_std=normalized.std(dim=(0,2,3)).cpu().tolist(),
                finite=bool(torch.isfinite(feature).all()))
        row=dict(key=key,input=level(wave),local_input=level(local),features=feature_stats,
            active_valid_frames=int(use.sum()),valid_background_frames=int(negative.sum()),
            target_detected_active_frames=int((use&decoded[:,0]).sum()),
            target_frame_recall=float(decoded[use,0].mean()) if use.any() else None,
            any_class_detected_active_frames=int((use&decoded.any(1)).sum()),
            other_class_only_active_frames=int((use&decoded.any(1)&(~decoded[:,0])).sum()),
            background_alarm_frames=int((negative&decoded.any(1)).sum()),
            numerical_quiet_frames=int(quiet.sum()),numerical_quiet_alarm_frames=int((quiet&decoded.any(1)).sum()),
            target_score_active=stats(scores[use,0]),target_score_all=stats(scores[:,0]),
            decoded_class_frame_counts=dict(counts),conditional_target_angle_deg=(dict(n=len(angles),
                median=float(np.median(angles)),q90=float(np.quantile(angles,.9)),maximum=float(max(angles))) if angles else None),
            conditional_target_LE_mean_deg=float(np.mean(angles)) if angles else None,
            target_frame_recall_within_20deg=sum(e<=20 for e in angles)/int(use.sum()) if use.any() else None,
            metric_scope='framewise class/activity and conditional geometry errors; not macro21 DSF benchmark SELD; overlap is not independent',
            **(extra or {}))
        self.rows.append(row);self.raw[key]=raw
        print('INFERRED',key,'chair_recall',row['target_frame_recall'],'max',row['target_score_all']['maximum'],flush=True)
        return raw


def truth_in_anchor(logs,source,q):
    receiver=np.asarray([r['receiver_position'] for r in logs])
    # 10Hz frame center is in quarter chunk floor(center/.25).
    ids=np.minimum(((np.arange(100)+.5)*.1/.25).astype(int),len(logs)-1)
    vectors=(np.asarray(source)-receiver[ids])@np.asarray(listener_basis(q)).T
    return vectors/np.linalg.norm(vectors,axis=1,keepdims=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    out=a.data/'analysis';out.mkdir(exist_ok=False)
    torch.manual_seed(7);np.random.seed(7)
    analysis=Analysis(a.data)
    from avn_foa_sameroom.synthesis import read_source
    from avn_foa_sameroom.dense_synthesis import synthesize_dense
    source=read_source(a.data/'original_chair.wav')
    design=json.loads((a.data/'design.json').read_text())
    cases=[]
    for item in design['cases']:
        name=item['case'];root=a.data/name
        c=json.loads((root/'collection.json').read_text());cases.append(c)
        wave,rate=sf.read(root/'foa_world.wav',dtype='float32',always_2d=True)
        dry,dryrate=sf.read(root/'emitted_source.wav',dtype='float32')
        assert rate==dryrate==RATE and wave.shape==(256000,4)
        logs=[json.loads(l) for l in (root/'actions.jsonl').read_text().splitlines()]
        with np.load(root/'initial_rir.npz') as z:initial_ir=z['native']
        # First bootstrap case predates per-step delay logging, fixed receiver only.
        if 'rir_onset_samples' not in logs[0]:
            assert c['motion']=='static'
            support=np.flatnonzero(np.max(np.abs(initial_ir),axis=1)>max(float(np.max(np.abs(initial_ir)))*1e-5,1e-12))
            for row in logs:row['rir_onset_samples']=int(support[0])
        active,valid,quiet=diagnostic_masks(dry,wave,logs)
        for start in (0,3,6):
            sample=wave[start*RATE:(start+10)*RATE].copy()
            chunklogs=logs[start*4:(start+10)*4];q=chunklogs[-1]['rotation_xyzw']
            mask=slice(start*10,(start+10)*10)
            assert active[mask].sum()>=20, 'No useful audio in 10s window'
            truth=truth_in_anchor(chunklogs,c['evaluation_only']['source_position'],q)
            common=dict(case=name,motion=c['motion'],schedule=c['schedule'],seed=c['seed'],
                temporalCoherence=c['temporalCoherence'],start_s=start,anchor_observed_time_s=start+9.75)
            analysis.predict(f'{name}__{start}__native',sample,q,active[mask],valid[mask],quiet[mask],truth,
                dict(common,view='native',global_gain=1.))
            scaled,gain=to_training_level(sample)
            analysis.predict(f'{name}__{start}__level',scaled,q,active[mask],valid[mask],quiet[mask],truth,
                dict(common,view='uniform_training_level',global_gain=gain))
        # Bridge controls need only one source seed and native temporal setting.
        if name=='static_continuous_seed7':
            q=logs[39]['rotation_xyzw'];truth=truth_in_anchor(logs[:40],c['evaluation_only']['source_position'],q)
            fixed,fs=sf.read(root/'frozen_initial_rir_full.wav',dtype='float32',always_2d=True)
            fixed=fixed[:160000];fixed,gain=to_training_level(fixed)
            analysis.predict('bridge__same_native_stream_frozen_rir',fixed,q,active[:100],valid[:100],quiet[:100],truth,
                dict(view='bridge_fixed_rir_native_stream',global_gain=gain))
            irworld=native_to_world(initial_ir)
            dense,labels,details=synthesize_dense(source,irworld,np.asarray([1.,0,0]),0,21,np.random.default_rng(7))
            dense,gain=to_training_level(dense)
            mask=labels['activity'].astype(bool);v=~labels['ignore_mask'].astype(bool)
            analysis.predict('bridge__same_geometry_dense_crops',dense,q,mask,v,np.zeros(100,bool),truth,
                dict(view='bridge_dense_crops',global_gain=gain,source_active_frames=int(mask.sum())))
            sf.write(str(out/'bridge_same_geometry_dense.wav'),dense,RATE,subtype='FLOAT')
            native_source,sr=sf.read(root/'native_normalized_source.wav',dtype='float32')
            ratio=float(np.dot(source,native_source[:len(source)])/np.dot(source,source))
            error=native_source[:len(source)]-source*ratio
            source_comparison=dict(original_resampler='training scipy resample_poly; native librosa.load default',
                best_scalar_native_over_training=ratio,correlation=float(np.corrcoef(source,native_source[:len(source)])[0,1]),
                scalar_aligned_relative_rmse=float(np.sqrt(np.mean(error**2))/np.sqrt(np.mean(native_source[:len(source)]**2))))
            # Feature extraction must happen after joining 40 chunks, not 40 reflect-padded STFTs.
            local=world_foa_to_listener(wave[:160000],q)
            with torch.inference_mode():
                correct=analysis.candidate.front(torch.from_numpy(local.T.copy())[None].cuda())
                incorrect=torch.cat([analysis.candidate.front(torch.from_numpy(x.T.copy())[None].cuda()) for x in np.split(local,40)],dim=2)
                discrepancy=float((correct-incorrect).abs().max())
                badraw=analysis.candidate.model(incorrect).float().cpu().numpy()[0]
            np.testing.assert_array_equal(np.concatenate(np.split(wave[:160000],40)),wave[:160000])
            try:analysis.candidate.predict(wave[:4000],RATE,q)
            except ValueError:short_rejected=True
            else:raise AssertionError('Quarter-second input should be rejected')
            window_check=dict(sample_chunk_s=.25,train_window_s=10,online_window_s=10,chunks_per_input=40,
                concatenation_waveform_max_error=0,short_input_rejected=short_rejected,
                correct_feature_shape=list(correct.shape),wrong_per_chunk_feature_shape=list(incorrect.shape),
                wrong_per_chunk_STFT_feature_max_abs_difference=discrepancy,
                wrong_per_chunk_STFT_not_used_in_production=True,
                wrong_per_chunk_STFT_target_score_max=float(np.linalg.norm(badraw.reshape(100,3,3,21),axis=2).max((0,1))[0]))
    # Stored offline controls use their already-local WAV directly: identity does not rotate it.
    offline=a.data/'offline_controls';stored=np.load(offline/'stored_predictions.npz',allow_pickle=False)
    lookup=dict(zip(stored['sample_id'].tolist(),stored['accdoa']))
    parity=[]
    for row in (json.loads(s) for s in (offline/'metadata.jsonl').read_text().splitlines()):
        wave,fs=sf.read(offline/row['audio'],dtype='float32',always_2d=True)
        with np.load(offline/row['labels']) as label:
            active=label['activity'].astype(bool);valid=~label['ignore_mask'].astype(bool)
            truth=label['direction_xyz']
        raw=analysis.predict('offline__'+row['sample_id'],wave,[0,0,0,1],active,valid,
            np.sqrt(np.mean(wave.reshape(100,1600,4).astype(np.float64)**2,axis=(1,2)))<1e-7,
            truth,dict(view='stored_offline',split=row['split'],sample_id=row['sample_id'],scene_id=row['scene_id']))
        if row['sample_id'] in lookup:
            ref=lookup[row['sample_id']];scores=lambda x:np.linalg.norm(x.reshape(100,3,3,21),axis=2).max(1)>.5
            parity.append(dict(sample_id=row['sample_id'],max_raw_abs_error=float(np.abs(raw-ref).max()),
                class_activity_agreement=float((scores(raw)==scores(ref)).mean())))
    with (out/'diagnostics.jsonl').open('w') as f:
        for row in analysis.rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    np.savez_compressed(out/'raw_predictions.npz',**analysis.raw)
    # Paired source/pose invariants, independent of predictions.
    paired=[]
    for seed in (7,8,9):
        for schedule in ('continuous','intermittent'):
            left=a.data/f'static_{schedule}_seed{seed}';right=a.data/f'motion_{schedule}_seed{seed}'
            x,_=sf.read(left/'emitted_source.wav',dtype='float32');y,_=sf.read(right/'emitted_source.wav',dtype='float32')
            assert np.array_equal(x,y)
            paired.append(dict(seed=seed,schedule=schedule,emitted_source_identical=True))
    summary=dict(status='completed',candidate_description=analysis.candidate.description,
        conditions=len(cases),inference_views=len(analysis.rows),window_check=window_check,
        source_preprocessing=source_comparison,offline_prediction_parity=parity,paired_source_checks=paired,
        threshold=.5,no_training=True,no_navigation_policy=True,no_online_tool_changes=True,
        protocol_limit='Single episode/source; overlapping windows not independent. gain/dense/smoothing are diagnostics, not selected production changes.',
        collection_summaries=cases)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('ANALYSIS_COMPLETE',len(analysis.rows),flush=True)


if __name__=='__main__':
    main()
