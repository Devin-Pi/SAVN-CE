"""Post-run audit; privileged truth is read only after both native environments close."""
import argparse,hashlib,json,math,statistics,subprocess,csv,sys
from pathlib import Path
from collections import Counter
import numpy as np
import jsonschema

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent_episode.foa_arrival import FOA_STOP_METHODS


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
        assert x.get('terminal_yaw_deg')==y.get('terminal_yaw_deg')
        for key in ('action','observation_before','observation_after','done'):assert x[key]==y[key],key
        for key in ('pose_before','pose_after'):
            assert math.dist(x[key]['position'],y[key]['position'])<1e-5
            assert abs(sum(a*b for a,b in zip(x[key]['rotation_xyzw'],y[key]['rotation_xyzw'])))>1-1e-5
        assert y['decision']['primary_decision_sha256']==hashlib.sha256(json.dumps(x['decision'],sort_keys=True).encode()).hexdigest()
    accepted=[r['proposal'] for r in rows(run/'model_decisions.jsonl') if r['status']=='accepted']
    config=json.loads((run/'run_config.json').read_text());mode=config.get('control_mode')
    foa_method=FOA_STOP_METHODS.get(config.get('stop_rule'))
    foa_stop=foa_method is not None
    stationary=mode=='stationary_listen_v1'
    segmented=mode in ('synchronous_segments_v1','stationary_listen_v1')
    controlled=[row for row in nav if row['action']!='listen']
    listening_windows=[];confirmation_windows=[];retry_windows={}
    if stationary:
        acquisition=rows(run/'listening_cycles.jsonl');assert len(acquisition)%2==0
        for start,end in zip(acquisition[::2],acquisition[1::2]):
            assert start['event']=='listening_started' and end['event']=='listening_finished' and start['cycle']==end['cycle']
            assert start['state']==end['start_state'] and end['unchanged_pose_verified']
            n=end['native_steps'];assert 1<=n<=4 and end['complete']==(n==4)
            assert end['final_state']['step']-start['state']['step']==n
            assert end['advanced_sim_time_s']==n*.25
            block=nav[start['state']['step']:end['final_state']['step']]
            assert len(block)==n and all(x['action']=='listen' for x in block)
            assert [x['observation_after'] for x in block]==end['observation_ids']
            assert all(x['pose_before']==x['pose_after']==start['state']['pose'] for x in block)
            assert end['audio_interval_s'][1]-end['audio_interval_s'][0]==n*.25
            purpose=end.get('purpose')
            if purpose in ('stop_confirmation','stop_confirmation_retry'):
                first=listening_windows[-1]
                expected='decision' if purpose=='stop_confirmation' else 'stop_confirmation'
                assert start.get('purpose')==purpose
                assert first['cycle']==end['cycle'] and first['purpose']==expected
                assert first['final_state']==start['state'] and first['audio_interval_s'][1]==end['audio_interval_s'][0]
                assert not set(first['observation_ids'])&set(end['observation_ids'])
                if purpose=='stop_confirmation':
                    confirmation_windows.append((first,end))
                else:
                    assert end['cycle'] not in retry_windows and config['max_stop_confirmation_retries']==1
                    retry_windows[end['cycle']]=end
            if end['complete']:listening_windows.append(end)
    cycle_summary=None
    if segmented:
        lifecycle=rows(run/'decision_cycles.jsonl');assert len(lifecycle)==3*len(accepted)
        offset=0;durations=[];reasons=Counter()
        for index,decision in enumerate(accepted):
            started,returned,finished=lifecycle[3*index:3*index+3]
            assert [r['event'] for r in (started,returned,finished)]==['decision_started','decision_returned','segment_finished']
            assert all(r['cycle']==index for r in (started,returned,finished))
            assert started['state']==returned['state']==finished['decision_anchor']
            assert returned['frozen_verified'] and returned['decision']==decision
            assert returned['wall_time_s']>=started['wall_time_s']
            count=finished['native_steps'];assert 0<=count<=returned['max_native_steps']<=4
            assert count*.25==finished['advanced_sim_time_s']
            block=controlled[offset:offset+(1 if decision['action']=='stop' else count)]
            assert block and all(r['decision']==decision for r in block)
            assert block[0]['observation_before']==decision['observation_id']==started['state']['observation_id']
            assert block[-1]['observation_after']==finished['final_state']['observation_id']
            assert block[0]['pose_before']==started['state']['pose']
            assert block[-1]['pose_after']==finished['final_state']['pose']
            facing=finished.get('terminal_facing')
            if facing is not None:
                assert config.get('terminal_foa_facing') and decision['action']=='stop' and finished['done']
                assert math.dist(block[0]['pose_before']['position'],block[0]['pose_after']['position'])<=1e-7
                if facing['status']=='aligned':
                    assert count==1 and block[0]['terminal_yaw_deg']==facing['yaw_deg']
                    assert abs(facing['residual_deg'])<=1.
                else:
                    assert count==0 and 'terminal_yaw_deg' not in block[0]
            offset+=len(block);durations.append(count*.25);reasons[finished['reason']]+=1
        assert offset==len(controlled)
        assert len(rows(run/'model_observations.jsonl'))==len(accepted)
        cycle_summary=dict(status='passed',decision_count=len(accepted),frozen_during_all_decisions=True,
            segment_duration_s=durations,segment_end_reasons=dict(reasons),native_step_s=.25,nominal_decision_period_s=1)
        if stationary:
            cycle_summary.update(mode=mode,nominal_decision_period_s=None,listening_window_s=1,
                complete_stationary_windows=len(listening_windows),stop_confirmation_windows=len(confirmation_windows),fixed_pose_for_all_windows=True,
                stop_confirmation_retry_windows=len(retry_windows),
                listening_steps=sum(x['action']=='listen' for x in nav),movement_steps=sum(x['action'] not in ('listen','stop') or 'terminal_yaw_deg' in x for x in nav))
    else:assert accepted==[x['decision'] for x in nav]
    calls=rows(run/'codex_calls.jsonl');assert all(c['model']=='gpt-6.1-sol' and not c['shadow_visible'] for c in calls)
    if foa_stop:assert all(c.get('foa_stop_evidence_visible') for c in calls)
    successful_calls=[c for c in calls if c['returncode']==0 and 'agent_message' in c['item_types']]
    assert len(successful_calls)>=len(accepted)
    for c in calls:assert not set(c['item_types'])&{'command_execution','mcp_tool_call','web_search','file_change','collab_tool_call'}
    observed=rows(run/'foa_shadow/observations.jsonl');byid={r['observation_id']:r for r in observed}
    if stationary:
        full_windows={x['final_state']['observation_id']:x for x in listening_windows}
        for item in rows(run/'model_observations.jsonl'):
            window=full_windows[item['observation_id']]
            assert item['control']['listening_observation_ids']==item['stationary_audio']['observation_ids']==window['observation_ids']
            assert item['control']['latest_pose']==byid[item['observation_id']]['pose']
            assert item['stationary_audio']['same_pose_verified'] and item['stationary_audio']['duration_s']==1
            if foa_stop:
                gate=item['stop_verification'];assert gate['method']==foa_method
                assert 'inferred_source_world_xyz' not in gate
                assert gate['evidence_id']==item['observation_id']+':foa_stop'
        evidence=rows(run/'shadow.jsonl')
        assert all(x['phase']=='stationary_listen' and x['policy_input_unchanged_verified'] for x in evidence)
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
    if foa_stop:
        gates=rows(run/'foa_stop_evidence.jsonl');byanchor={x['at']:x for x in gates}
        registry=rows(run/'derived_evidence_registry.jsonl')
        assert len(registry)==len(gates)==len(byanchor)
        for row in registry:
            current=row['current_audio_observation_ids'];anchor=current[-1]
            assert len(current)==4 and all(x in byid for x in current)
            assert row['evidence_id']==byanchor[anchor]['evidence_id'] and not row['new_physical_observation']
            assert all(x in byanchor and byid[x]['step']<=byid[anchor]['step'] for x in row['supporting_anchor_ids'])
        if config.get('stop_rule')=='foa_geometry':
            model_inputs={r['observation_id']:r for r in rows(run/'model_observations.jsonl')}
            assert all(g['config']['angular_tolerance_deg']==config['foa_tolerance_deg'] for g in gates)
            eligible_ready=set()
            for first,second in confirmation_windows:
                anchor=first['final_state']['observation_id']
                assert byanchor[anchor]['request_stationary_confirmation'] and not byanchor[anchor]['ready']
                assert anchor not in model_inputs
                final=second['final_state']['observation_id']
                retry=retry_windows.get(second['cycle'])
                if retry is not None:
                    missed=byanchor[final]
                    assert missed['request_stationary_retry'] and not missed['ready']
                    assert missed['confirmation_reference_anchor']==anchor and missed['confirmation_count']==1
                    assert final not in model_inputs
                    assert not set(first['observation_ids'])&set(retry['observation_ids'])
                    pred=next(r for r in success if r['at']==final)
                    assert pred['events']==[[]]
                    final=retry['final_state']['observation_id']
                if final in model_inputs:
                    control=model_inputs[final]['control']
                    assert (retry or second)['complete'] and control['first_near_anchor']==anchor
                    if retry is not None:
                        assert control['missed_confirmation_anchor']==second['final_state']['observation_id']
                if (retry or second)['complete']:
                    eligible_ready.add(final)
                    if byanchor[final]['ready'] and config.get('max_stop_confirmation_retries'):
                        assert byanchor[final]['confirmation_reference_anchor']==anchor
            assert {r['at'] for r in gates if r.get('ready')}<=eligible_ready
        for row in rows(run/'model_observations.jsonl'):
            supplied=row['stop_verification'];gate=byanchor[row['observation_id']]
            allowed={'method','evidence_id','ready','reason','checks','estimated_range_interval_m','calibration_status'}
            assert set(supplied)<=allowed and all(gate[k]==v for k,v in supplied.items())
        for row in rows(run/'model_decisions.jsonl'):
            gate=byanchor[row['observation_id']]
            if row['status']=='accepted' and row['proposal']['action']=='stop':assert gate['ready'] and all(gate['checks'].values())
            if row['status']=='stop_vetoed':assert not gate['ready']
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
    runtime_schedule=truth.get('runtime_sound_schedule',{})
    onset=runtime_schedule.get('runtime_onset_s',onset);offset=runtime_schedule.get('runtime_offset_s',offset)
    if runtime_schedule:
        assert json.loads((run/'navigation/sound_schedule.json').read_text())==runtime_schedule
        assert json.loads((run/'foa_shadow/sound_schedule.json').read_text())==runtime_schedule
        assert hashlib.sha256(Path(truth['episode_file']).read_bytes()).hexdigest()==truth['episode_file_sha256']
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
        expected=([x['final_state']['observation_id'] for x in listening_windows] if stationary else [r['observation_id'] for r in observed[3:]])
        assert [r['at'] for r in scored_success]==expected
        if stationary:
            for r in scored_success:
                assert r['provenance']['observation_ids']==full_windows[r['at']]['observation_ids']
                poses=[byid[x]['pose'] for x in r['provenance']['observation_ids']]
                assert all(p==poses[0] for p in poses)
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
    report=dict(status='passed',episode_id=config['episode_id'],action_count=len(nav),movement_steps=evaluation['movement_steps'],
        total_native_steps=len(observed)-1,listening_steps=sum(x['action']=='listen' for x in nav),
        observation_count=len(observed),policy_model='gpt-6.1-sol',actual_successful_codex_requests=len(successful_calls),failed_codex_requests=len(calls)-len(successful_calls),
        pose_and_action_lockstep_verified=True,shadow_not_sent_to_policy=True,actor_has_no_scene_assets=True,
        candidate_status_counts=dict(Counter(r['status'] for r in predictions)),unique_successful_anchor_ids=len({r['at'] for r in success}),cached_successful_results=sum(r['provenance']['cache_hit'] for r in success),first_result_audio_time_s=success[0]['timing']['audio_end_s'] if success else None,
        first_and_last_numerically_active_quarter_start_s=([r['timestamp_s'] for r in observed if r['audio_rms']>1e-7][0], [r['timestamp_s'] for r in observed if r['audio_rms']>1e-7][-1]) if any(r['audio_rms']>1e-7 for r in observed) else None,
        events_by_class=dict(events_by_class),events_in_latest_one_second_summed_over_calls=recent,
        events_on_numerically_quiet_quarter_second_windows=numeric_silent,
        all_class_conditional_direction=stats(angles),chair_conditional_direction=stats(correct_angles),
        conditional_metric_note='Offline renderer-truth diagnostic on numerical RMS>1e-7; overlapping predictions counted, not independent samples; not SELD/F20 or Agent success',
        navigation=evaluation,checkpoint_sha256=model['checkpoint_sha256'])
    if cycle_summary:report['synchronous_control']=cycle_summary
    if foa_stop:
        report.update(shadow_not_sent_to_policy=False,raw_localization_not_sent_to_policy=True,
            localization_used_for_stop_gate=True,localization_used_for_motion_direction=False,
            foa_stop_ready_anchors=[x['at'] for x in gates if x['ready']],
            foa_stop_rule_note='Heuristic multi-view signed-ray Euclidean range, not oracle distance or calibrated confidence; STOP still proposed by Agent')
    if runtime_schedule:report['runtime_sound_schedule']=runtime_schedule
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
        if stationary:
            report.pop('actions_committed_before_localization_verified')
            report.update(output_update_period_s=None,model_output_duration_s=.25,
                localization_invocation_mode='complete stationary windows only, before policy; never exposed',
                localization_does_not_mutate_policy_input_verified=True,
                conditional_metric_note='Fixed-pose 1s inputs, one current250ms output per completed listening phase. Same episode is not independent evidence; not SELD/F20 or Agent success.')
            if foa_stop:
                report.pop('localization_does_not_mutate_policy_input_verified')
                report.update(localization_invocation_mode='complete stationary windows; derived STOP evidence exposed, raw output withheld',
                    raw_inference_does_not_mutate_base_observation_verified=True,
                    derived_stop_evidence_provenance_verified=True)
        if step_rows:
            with (run/'per_step_localization.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(step_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(step_rows)
    (run/'shadow_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
