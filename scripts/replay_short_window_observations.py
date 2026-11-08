"""Read-only prior FOA replay. No simulator, Agent, oracle controller or future samples."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
import torch


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.experiment/'code_snapshot'))
    from avn_foa_sameroom.infer_finetuned import load
    from avn_foa_sameroom.short_window import patch_short_context,pool_current_step
    from spatial_audio.foa import world_foa_to_listener,CONTRACT_VERSION
    sys.path.insert(0,'/workspaces/dsf_localization_runtime_v1')
    import dsf_decode
    a.output.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    observations=[json.loads(s) for s in (a.recording/'observations.jsonl').read_text().splitlines()]
    wave,rate=sf.read(a.recording/'foa_ambix.wav',dtype='float32',always_2d=True)
    assert rate==16000 and wave.shape==(len(observations)*4000,4)
    for i,obs in enumerate(observations):
        assert obs['step']==i and abs(obs['timestamp_s']-i*.25)<1e-8
        assert obs['audio_format']['coordinate_frame']=='habitat_world'
        assert obs['audio_format']['normalization']=='SN3D' and obs['audio_format']['channels']==['W','Y','Z','X']
    summaries={}
    for label in ('w025','w100','w150'):
        model,front,config=load(a.experiment/'runs'/label/'run_seed7_full100/best.pt',Path('/workspaces/dsf_localization_runtime_v1'))
        patch_short_context(model);dsf_decode.CLASSES=tuple(config['classes']);dsf_decode.ACTIVITY_THRESHOLD=.5
        n=round(config['window_s']*16000);rows=[]
        for i,obs in enumerate(observations):
            end=(i+1)*4000
            if end<n:continue
            # The full input is already observed and rotated once to this observed anchor.
            local=world_foa_to_listener(wave[end-n:end].copy(),obs['pose']['rotation_xyzw'])
            raw=pool_current_step(model(front(torch.from_numpy(local.T.copy())[None].cuda())),config['window_s'])[0].float().cpu().numpy()
            events=dsf_decode.decode(raw)[0]
            scores=np.linalg.norm(raw.reshape(3,3,21),axis=1).max(0)
            current=wave[i*4000:end]
            rows.append(dict(observation_id=obs['observation_id'],input_start_s=(end-n)/16000,
                audio_available_at_s=end/16000,current_step_interval_s=[i*.25,(i+1)*.25],
                current_rms=float(np.sqrt(np.mean(current.astype(float)**2))),
                direction_frame='latest_observed_listener',events=events,raw_accdoa=raw[0].tolist(),
                chair_score=float(scores[0]),max_class_score=float(scores.max()),
                only_observed_audio_and_pose_used=True,common_coverage=i>=5))
        (a.output/f'{label}_predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
        common=[r for r in rows if r['common_coverage']]
        # Official schedule and category are evaluation-only, after all predictions exist.
        on=[r for r in common if 2<=r['current_step_interval_s'][0]<8]
        quiet=[r for r in common if r['current_rms']<1e-7]
        heard=[r for r in common if r['current_rms']>=1e-7]
        has_chair=lambda r:any(e['class_id']==0 for e in r['events'])
        classes=Counter(e['class_name'] for r in common for e in r['events'])
        summaries[label]=dict(window_s=config['window_s'],predictions=len(rows),common_step_count=len(common),
            earliest_result_at_s=rows[0]['audio_available_at_s'],source_emission_on_steps=len(on),
            chair_detected_emission_on_steps=sum(has_chair(r) for r in on),
            numerically_audible_steps=len(heard),chair_detected_numerically_audible_steps=sum(has_chair(r) for r in heard),
            numerical_quiet_steps=len(quiet),numerical_quiet_steps_with_any_prediction=sum(bool(r['events']) for r in quiet),
            decoded_class_events=dict(classes),maximum_chair_score_during_source_on=max(r['chair_score'] for r in on),
            scope='single previously inspected moving episode replay, not a new held-out online evaluation; numerical RMS is not semantic ground truth')
        print('REPLAYED',label,json.dumps(summaries[label]),flush=True)
        del model,front;torch.cuda.empty_cache()
    report=dict(status='completed',episode_id='1000000008',runs=summaries,conversion_contract=CONTRACT_VERSION,
        source_schedule_evaluation_only=[2,8],source_category_evaluation_only='chair',
        simulator_executed=False,LLM_called=False,actions_changed=False,navigation_control=False,
        original_recording_unchanged=True,model_inputs='only already observed world FOA and current observed quaternion',
        caveat='Do not equate this one replay with navigation success or multi-scene online accuracy; no thresholds/checkpoints selected on replay')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
