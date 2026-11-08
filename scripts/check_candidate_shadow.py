"""Post-run audit; privileged truth is read only after both native environments close."""
import argparse,hashlib,json,math,statistics,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
import jsonschema


def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]


def private_path(output):
    token=hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
    return Path('research_outputs/agent_private')/(token+'_'+output.name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--actor-audit',type=Path,default=Path('outputs/candidate_calibration_20261108/actor_environment.json'));a=p.parse_args();run=a.run
    assert json.loads((run/'rollout_complete.json').read_text())['status']=='completed'
    nav=rows(run/'navigation/actions.jsonl');mirror=rows(run/'foa_shadow/actions.jsonl')
    assert len(nav)==len(mirror)>0
    for x,y in zip(nav,mirror):
        for key in ('action','observation_before','observation_after','done'):assert x[key]==y[key],key
        for key in ('pose_before','pose_after'):
            assert math.dist(x[key]['position'],y[key]['position'])<1e-5
            assert abs(sum(a*b for a,b in zip(x[key]['rotation_xyzw'],y[key]['rotation_xyzw'])))>1-1e-5
        assert y['decision']['primary_decision_sha256']==hashlib.sha256(json.dumps(x['decision'],sort_keys=True).encode()).hexdigest()
    accepted=[r['proposal'] for r in rows(run/'model_decisions.jsonl') if r['status']=='accepted']
    assert accepted==[x['decision'] for x in nav]
    calls=rows(run/'codex_calls.jsonl');assert all(c['model']=='gpt-6.1-sol' and not c['shadow_visible'] for c in calls)
    successful_calls=[c for c in calls if c['returncode']==0];assert len(successful_calls)>=len(nav)
    for c in calls:assert not set(c['item_types'])&{'command_execution','mcp_tool_call','web_search','file_change','collab_tool_call'}
    observed=rows(run/'foa_shadow/observations.jsonl');byid={r['observation_id']:r for r in observed}
    predictions=rows(run/'predictions.jsonl');schema=json.loads(Path('docs/localization_tool.candidate.schema.json').read_text())
    for r in predictions:jsonschema.validate(r,schema)
    success=[r for r in predictions if r['status']=='ok']
    for r in success:
        ids=r['provenance']['observation_ids'];assert len(ids)==40 and ids[-1]==r['at']
        assert byid[ids[-1]]['step']-byid[ids[0]]['step']==39
        assert r['timing']['audio_end_s']-r['timing']['audio_start_s']==10
        assert r['timing']['result_available_at_audio_time_s']==r['timing']['audio_end_s']
        assert np.asarray(r['raw_accdoa']).shape==(100,189)
        for frame in r['events']:
            for e in frame:
                assert e['orientation_pose_ref']==ids[-1] and e['pose_ref'] in ids
                assert abs(np.linalg.norm(e['direction_unit_habitat_xyz'])-1)<1e-5
    # Actor container is separate from the simulator and sees no scene/data/output mounts.
    actor=json.loads(a.actor_audit.read_text())
    allowed={'/usr/local/bin/codex','/codex_home/auth.json','/codex_home/config.toml','/etc/ssl/certs','/inputs'}
    assert set(actor['mount_destinations'])<=allowed and actor['all_bind_mounts_readonly']
    assert actor['readonly_root'] and actor['user'] not in ('0','0:0','')
    for decision in accepted:assert all(':localization' not in e for e in decision['evidence_ids'])
    truth=json.loads((private_path(run/'foa_shadow')/'evaluation_truth.json').read_text())
    evaluation=json.loads((private_path(run/'navigation')/'evaluation.json').read_text())
    # Offline only. Keep independent from the CLI input directory.
    source=truth['acoustic_source_position'];events_by_class=Counter();angles=[];correct_angles=[];recent=0;numeric_silent=0
    for r in success:
        poses={x['observation_id']:x for x in r['pose_refs']}
        for index,frame in enumerate(r['events']):
            for e in frame:
                events_by_class[e['class_name']]+=1;recent+=int(index>=90)
                if byid[e['pose_ref']]['audio_rms']<=1e-7:numeric_silent+=1;continue
                receiver=poses[e['pose_ref']]['receiver_position_habitat_xyz_m'];v=np.asarray(source)-receiver;v/=np.linalg.norm(v)
                angle=float(np.degrees(np.arccos(np.clip(np.dot(v,e['direction_unit_habitat_xyz']),-1,1))))
                angles.append(angle)
                if e['class_name']=='chair':correct_angles.append(angle)
    def stats(values):return dict(count=len(values),mean_error_deg=statistics.mean(values) if values else None,within20=sum(v<=20 for v in values),over90=sum(v>90 for v in values))
    report=dict(status='passed',episode_id='1000000008',action_count=len(nav),movement_steps=len(observed)-1,
        observation_count=len(observed),policy_model='gpt-6.1-sol',actual_successful_codex_requests=len(successful_calls),failed_codex_requests=len(calls)-len(successful_calls),
        pose_and_action_lockstep_verified=True,shadow_not_sent_to_policy=True,actor_has_no_scene_assets=True,
        candidate_status_counts=dict(Counter(r['status'] for r in predictions)),unique_successful_anchor_ids=len({r['at'] for r in success}),cached_successful_results=sum(r['provenance']['cache_hit'] for r in success),first_result_audio_time_s=success[0]['timing']['audio_end_s'] if success else None,
        first_and_last_numerically_active_quarter_start_s=([r['timestamp_s'] for r in observed if r['audio_rms']>1e-7][0], [r['timestamp_s'] for r in observed if r['audio_rms']>1e-7][-1]) if any(r['audio_rms']>1e-7 for r in observed) else None,
        events_by_class=dict(events_by_class),events_in_latest_one_second_summed_over_calls=recent,
        events_on_numerically_quiet_quarter_second_windows=numeric_silent,
        all_class_conditional_direction=stats(angles),chair_conditional_direction=stats(correct_angles),
        conditional_metric_note='Offline renderer-truth diagnostic on numerical RMS>1e-7; overlapping predictions counted, not independent samples; not SELD/F20 or Agent success',
        navigation=evaluation,checkpoint_sha256=json.loads((run/'model.json').read_text())['checkpoint_sha256'])
    (run/'shadow_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
