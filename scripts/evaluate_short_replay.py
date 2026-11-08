"""Oracle geometry only after replay predictions are written; never used for inference."""
import argparse
import json
import math
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--truth',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.experiment/'code_snapshot'))
    from spatial_audio.foa import listener_basis
    source=json.loads(a.truth.read_text())['evaluation_only']['source_position']
    observations={x['observation_id']:x for x in (json.loads(s) for s in (a.recording/'observations.jsonl').read_text().splitlines())}
    summary={}
    for label in ('w025','w100','w150'):
        rows=[json.loads(s) for s in (a.predictions/f'{label}_predictions.jsonl').read_text().splitlines()]
        on=[r for r in rows if r['common_coverage'] and 2<=r['current_step_interval_s'][0]<8]
        angles=[];time_errors=[]
        for row in on:
            events=[e for e in row['events'] if e['class_id']==0]
            if not events:continue
            obs=observations[row['observation_id']]
            vector=[s-r for s,r in zip(source,obs['audio_receiver_position'])]
            basis=listener_basis(obs['pose']['rotation_xyzw'])
            local=[sum(v*b for v,b in zip(vector,axis)) for axis in basis]
            length=math.sqrt(sum(v*v for v in local));unit=[v/length for v in local]
            err=min(math.degrees(math.acos(max(-1,min(1,sum(a*b for a,b in zip(unit,e['direction_unit_xyz'])))))) for e in events)
            angles.append(err);time_errors.append(dict(observation_id=row['observation_id'],angle_deg=err))
        summary[label]=dict(source_on_steps=len(on),steps_with_chair=len(angles),
            conditional_current_pose_angle_mean_deg=sum(angles)/len(angles) if angles else None,
            conditional_current_pose_angle_max_deg=max(angles) if angles else None,
            within20deg_steps=sum(x<=20 for x in angles),
            correct_class_and_20deg_step_recall=sum(x<=20 for x in angles)/len(on),
            per_detected_step=time_errors)
    result=dict(status='completed',runs=summary,
        scope='single-episode conditional geometry error and step recall; not official DSF block F20/LE',
        truth_used_after_predictions_only=True,model_inputs_unchanged=True,no_missing_detection_as_zero_angle=True)
    (a.predictions/'geometry_evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
