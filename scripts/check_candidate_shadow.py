"""Post-run audit; privileged truth is read only after both native environments close."""
import argparse,hashlib,json,math,statistics,subprocess,csv
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
    model=json.loads((run/'model.json').read_text());short=model.get('backend')=='avn_foa_short1s'
    support_count=4 if short else 40;output_frames=1 if short else 100;window_s=1 if short else 10
    predictions=rows(run/'predictions.jsonl');schema=json.loads(Path('docs/localization_tool.short.schema.json' if short else 'docs/localization_tool.candidate.schema.json').read_text())
    for r in predictions:jsonschema.validate(r,schema)
    success=[r for r in predictions if r['status']=='ok']
    for r in success:
        ids=r['provenance']['observation_ids'];assert len(ids)==support_count and ids[-1]==r['at']
        assert byid[ids[-1]]['step']-byid[ids[0]]['step']==support_count-1
        assert r['timing']['audio_end_s']-r['timing']['audio_start_s']==window_s
        assert r['timing']['result_available_at_audio_time_s']==r['timing']['audio_end_s']
        assert np.asarray(r['raw_accdoa']).shape==(output_frames,189)
        if short:
            assert r['timing']['current_step_interval_s']==[byid[r['at']]['timestamp_s'],r['timing']['audio_end_s']]
            assert r['timing']['frame_times_s']==[byid[r['at']]['timestamp_s']]
        for frame in r['events']:
            for e in frame:
                assert e['orientation_pose_ref']==ids[-1] and e['pose_ref'] in ids
                if short:assert e['pose_ref']==ids[-1]
                assert abs(np.linalg.norm(e['direction_unit_habitat_xyz'])-1)<1e-5
    # Actor container is separate from the simulator and sees no scene/data/output mounts.
    actor=json.loads(a.actor_audit.read_text())
    allowed={'/usr/local/bin/codex','/codex_home/auth.json','/codex_home/config.toml','/etc/ssl/certs','/inputs'}
    assert set(actor['mount_destinations'])<=allowed and actor['all_bind_mounts_readonly']
    assert actor['readonly_root'] and actor['user'] not in ('0','0:0','')
    for decision in accepted:assert all(':localization' not in e for e in decision['evidence_ids'])
    for item in rows(run/'model_observations.jsonl'):
        payload=json.dumps(item)
        assert ':localization' not in payload and 'raw_accdoa' not in payload and 'avn_short_context_1s_epoch083' not in payload
    commits=[r for r in rows(run/'shadow.jsonl') if 'decision_already_committed_sha256' in r]
    actions_by_anchor={x['observation_before']:x for x in nav}
    for r in commits:
        expected=hashlib.sha256(json.dumps(actions_by_anchor[r['at']]['decision'],sort_keys=True).encode()).hexdigest()
        assert r['decision_already_committed_sha256']==expected
    truth=json.loads((private_path(run/'foa_shadow')/'evaluation_truth.json').read_text())
    evaluation=json.loads((private_path(run/'navigation')/'evaluation.json').read_text())
    # Offline only. Keep independent from the CLI input directory.
    source=truth['acoustic_source_position'];events_by_class=Counter();angles=[];correct_angles=[];recent=0;numeric_silent=0
    goal=truth['official_episode']['goals'][0];onset,offset=goal['onset'],goal['offset']
    source_on_steps=chair_on_steps=quiet_steps=quiet_alarms=0;source_on_20=0;source_on_angles=[];step_rows=[]
    scored_success=success
    if short:
        unique={}
        for r in success:
            if r['at'] in unique:
                previous=unique[r['at']]
                assert r['provenance']['cache_hit'] and r['raw_accdoa']==previous['raw_accdoa']
                assert r['provenance']['support_sample_interval']==previous['provenance']['support_sample_interval']
            else:unique[r['at']]=r
        scored_success=list(unique.values())
        assert [r['at'] for r in scored_success]==[r['observation_id'] for r in observed[3:]]
        assert all(r['status'] in ('ok','not_ready') for r in predictions)
    for r in scored_success:
        poses={x['observation_id']:x for x in r['pose_refs']}
        if short:
            current=byid[r['at']];on=onset<=current['timestamp_s']<offset;quiet=current['audio_rms']<=1e-7
            source_on_steps+=int(on);quiet_steps+=int(quiet)
            chair_on_steps+=int(on and any(e['class_name']=='chair' for e in r['events'][0]))
            quiet_alarms+=int(quiet and bool(r['events'][0]));current_chair_angles=[]
        for index,frame in enumerate(r['events']):
            for e in frame:
                events_by_class[e['class_name']]+=1;recent+=int(short or index>=90)
                if byid[e['pose_ref']]['audio_rms']<=1e-7:numeric_silent+=1;continue
                receiver=poses[e['pose_ref']]['receiver_position_habitat_xyz_m'];v=np.asarray(source)-receiver;v/=np.linalg.norm(v)
                angle=float(np.degrees(np.arccos(np.clip(np.dot(v,e['direction_unit_habitat_xyz']),-1,1))))
                angles.append(angle)
                if e['class_name']=='chair':
                    correct_angles.append(angle)
                    if short and on:current_chair_angles.append(angle)
        if short and current_chair_angles:
            source_on_angles.append(min(current_chair_angles))
            if min(current_chair_angles)<=20:source_on_20+=1
        if short:
            scores=np.linalg.norm(np.asarray(r['raw_accdoa']).reshape(1,3,3,21),axis=2).max(axis=1)[0]
            step_rows.append(dict(observation_id=r['at'],step=current['step'],
                input_start_s=r['timing']['audio_start_s'],input_end_s=r['timing']['audio_end_s'],
                output_start_s=current['timestamp_s'],output_end_s=r['timing']['audio_end_s'],
                actual_audio_rms=current['audio_rms'],source_on_evaluation_only=on,numerical_quiet=quiet,
                classes=';'.join(e['class_name'] for e in r['events'][0]),
                chair_raw_score=float(scores[0]),max_class_raw_score=float(scores.max()),
                chair_source_on_angle_deg=min(current_chair_angles) if current_chair_angles else None,
                action=actions_by_anchor.get(r['at'],{}).get('action'),
                model_inference_ms=r['timing']['inference']*1000,
                support_ids=';'.join(r['provenance']['observation_ids'])))
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
        navigation=evaluation,checkpoint_sha256=model['checkpoint_sha256'])
    if short:
        first_chair=next((x['output_end_s'] for x in step_rows if x['source_on_evaluation_only'] and 'chair' in x['classes'].split(';')),None)
        report.update(model_window_s=1,output_update_period_s=.25,source_on_steps=source_on_steps,
            chair_detected_source_on_steps=chair_on_steps,correct_class_and20deg_source_on_steps=source_on_20,
            numerical_quiet_steps=quiet_steps,numerical_quiet_steps_with_prediction=quiet_alarms,
            source_schedule_s=[onset,offset],current_step_outputs_verified=True,
            chair_source_on_conditional_direction=stats(source_on_angles),
            unique_current_step_outputs_scored=len(scored_success),
            cached_duplicate_outputs_excluded=len(success)-len(scored_success),
            first_chair_result_audio_time_s=first_chair,
            source_onset_to_first_chair_result_s=first_chair-onset if first_chair is not None else None,
            actions_committed_before_localization_verified=True)
        if step_rows:
            with (run/'per_step_localization.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(step_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(step_rows)
    (run/'shadow_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
