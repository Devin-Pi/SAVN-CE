"""Two synchronized native environments: unchanged binaural policy and FOA shadow."""
import argparse,json,math,sys,hashlib,time
from pathlib import Path
from collections import deque
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'agent_episode'))
from llm_rollout import ACTIONS,MODALITIES,EPISODES,append,jsonl_process,choose
from codex_backend import CodexBackend
from localization_tool import LocalizationTool,Worker,compact,worker_command
from synchronous_control import run_synchronous
from foa_arrival import DEFAULT_FOA_TOLERANCE_DEG,FOA_STOP_METHODS,policy_evidence


def assert_pose(primary,foa):
    assert primary['observation_id']==foa['observation_id'] and primary['timestamp_s']==foa['timestamp_s']
    assert math.dist(primary['pose']['position'],foa['pose']['position'])<1e-5
    assert abs(sum(a*b for a,b in zip(primary['pose']['rotation_xyzw'],foa['pose']['rotation_xyzw'])))>1-1e-5


def run_segmented(primary,foa,*,sim,shadow,extract,tool,backend,history,observed,out,max_steps,listen_s=0,stop_rule='legacy',foa_tolerance_deg=None):
    latest={};stationary_blocks=deque(maxlen=4)
    arrival=FOA_STOP_METHODS.get(stop_rule)
    def prepare(obs):
        assert_pose(obs,foa)
        if latest.get('id')==obs['observation_id']:return latest['data']
        tool.add_observation(foa)
        data=extract(obs);summary=data['summary'];ident=summary['observation_id']
        if arrival is not None:
            data['stop_gate']=dict(method=arrival,ready=False,reason='Await fresh complete FOA acquisition',checks={'fresh_foa':False})
        observed.update(f'{ident}:{m}' for m in MODALITIES)
        append(out/'sensor_summaries.jsonl',summary)
        latest.update(id=ident,data=data)
        return data
    def decide(data):
        summary=data['summary'];append(out/'model_observations.jsonl',summary)
        center=summary['depth'].get('center')
        allowed=tuple(action for action in ACTIONS if action!='move_forward' or (center is not None and center['p10_m']>=.45))
        return choose(backend,'evidence',data,history,allowed,observed,out/'model_decisions.jsonl')
    def before_action(obs,decision):
        if listen_s:return
        digest=hashlib.sha256(json.dumps(decision,sort_keys=True).encode()).hexdigest()
        result=tool.analyze([obs['observation_id']]);append(out/'predictions.jsonl',dict(at=obs['observation_id'],**result))
        append(out/'shadow.jsonl',dict(at=obs['observation_id'],decision_already_committed_sha256=digest,**compact(result)))
        assert digest==hashlib.sha256(json.dumps(decision,sort_keys=True).encode()).hexdigest()
    def advance(action,decision,*,stop_yaw_deg=None):
        nonlocal foa
        summary=latest['data']['summary']
        terminal={} if stop_yaw_deg is None else {'stop_yaw_deg':stop_yaw_deg}
        reply=sim(dict(command='step',action=action,decision=decision,**terminal))
        digest=hashlib.sha256(json.dumps(decision,sort_keys=True).encode()).hexdigest()
        mirror=shadow(dict(command='step',action=action,decision={'source':'primary_action_mirror','primary_decision_sha256':digest},**terminal))
        foa=mirror['observation'];obs=reply['observation'];assert reply['done']==mirror['done'];assert_pose(obs,foa)
        history.append({k:summary[k] for k in ('observation_id','step','sim_time_s','xz_m','forward_xz','collided','forward_blocked','audio','depth')}|{'action':action})
        print(json.dumps(dict(observation=summary['observation_id'],action=action,step=obs['step'],done=reply['done'])),flush=True)
        return obs,reply['done']
    def read_current():
        obs=sim({'command':'state'})['state']
        other=shadow({'command':'state'})['state'];assert_pose(obs,other)
        return obs
    def listen_advance(cycle):
        nonlocal foa
        summary=latest['data']['summary']
        decision=dict(source='coordinator_stationary_listening',cycle=cycle)
        digest=hashlib.sha256(json.dumps(decision,sort_keys=True).encode()).hexdigest()
        reply=sim(dict(command='listen',decision=decision))
        mirror=shadow(dict(command='listen',decision=dict(source='primary_action_mirror',primary_decision_sha256=digest)))
        foa=mirror['observation'];obs=reply['observation'];assert reply['done']==mirror['done'];assert_pose(obs,foa)
        stationary_blocks.append(obs)
        history.append({k:summary[k] for k in ('observation_id','step','sim_time_s','xz_m','forward_xz','collided','forward_blocked','audio','depth')}|{'action':'listen'})
        print(json.dumps(dict(observation=summary['observation_id'],action='listen',step=obs['step'],done=reply['done'])),flush=True)
        return obs,reply['done']
    def after_listen(obs,window,data):
        ids=[row['observation_id'] for row in window]
        assert ids==[row['observation_id'] for row in stationary_blocks]
        aggregate=extract(dict(command='stationary_summary',observations=list(stationary_blocks)))
        result_data=dict(data,summary=dict(data['summary'],stationary_audio=aggregate['stationary_audio']),audio_png=aggregate['audio_png'])
        digest=hashlib.sha256(json.dumps(result_data,sort_keys=True).encode()).hexdigest()
        result=tool.analyze([obs['observation_id']])
        assert result['status']=='ok' and result['provenance']['observation_ids']==ids
        assert digest==hashlib.sha256(json.dumps(result_data,sort_keys=True).encode()).hexdigest()
        append(out/'predictions.jsonl',dict(at=obs['observation_id'],**result))
        append(out/'shadow.jsonl',dict(at=obs['observation_id'],phase='stationary_listen',policy_input_sha256=digest,policy_input_unchanged_verified=True,**compact(result)))
        if arrival is not None:
            gate=extract(dict(command='foa_stop_gate',result=result,audio_active=aggregate['stationary_audio']['active'],rule=stop_rule,tolerance_deg=foa_tolerance_deg))['gate']
            append(out/'foa_stop_evidence.jsonl',dict(at=obs['observation_id'],**gate))
            append(out/'derived_evidence_registry.jsonl',dict(evidence_id=gate['evidence_id'],
                kind='foa_stop_only',current_audio_observation_ids=ids,
                supporting_anchor_ids=gate.get('support_ids',[]),new_physical_observation=False,
                model_checkpoint_sha256=result['provenance']['checkpoint_sha256'],
                method=gate['method'],record_file='foa_stop_evidence.jsonl'))
            observed.add(gate['evidence_id'])
            result_data['stop_gate']=gate
            result_data['summary']['stop_verification']=policy_evidence(gate)
        return result_data
    primary,cycles=run_synchronous(primary,prepare=prepare,decide=decide,advance=advance,
        read_current=read_current,before_action=before_action,face_on_stop=arrival is not None,
        log_event=lambda row:append(out/'decision_cycles.jsonl',row),max_steps=max_steps,
        listen_advance=listen_advance if listen_s else None,after_listen=after_listen if listen_s else None,
        log_listening=(lambda row:append(out/'listening_cycles.jsonl',row)) if listen_s else None)
    if not listen_s:
        result=tool.analyze([foa['observation_id']]);append(out/'predictions.jsonl',dict(at=foa['observation_id'],**result))
        append(out/'shadow.jsonl',dict(at=foa['observation_id'],terminal=True,**compact(result)))
    return primary,cycles


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--sim-workdir',required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--episode-id',default='1000000008');p.add_argument('--steps',type=int,default=64);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--decision-period-s',type=float,choices=(.25,1.),default=.25,help='1 enables synchronous, guarded segments of up to four native steps')
    p.add_argument('--listen-before-decide-s',type=float,choices=(0.,1.),default=0.,help='Collect four fresh fixed-pose blocks before every decision; counts toward --steps')
    p.add_argument('--sound-schedule',choices=('episode','sustained'),default='episode',help='sustained preserves onset, extends runtime offset beyond the budget; original episode unchanged')
    p.add_argument('--stop-rule',choices=('legacy',*FOA_STOP_METHODS),default='legacy')
    p.add_argument('--source-placement',choices=('episode','object_center'),default='episode',
                   help='Use the episode source or place it 1.5 m above the semantic object center')
    p.add_argument('--binaural-audio-threads',type=int,choices=(1,4),default=4)
    p.add_argument('--foa-audio-threads',type=int,choices=(1,4),default=1,help='Number of FOA acoustic rendering threads')
    p.add_argument('--foa-tolerance-deg',type=float,help='Default is 15 degrees for foa_multiview or 5 for foa_geometry')
    p.add_argument('--profile',choices=('avn_foa_candidate','avn_foa_short1s'),default='avn_foa_candidate');a=p.parse_args()
    if a.steps<=0:p.error('steps must be positive')
    uses_foa=a.stop_rule in FOA_STOP_METHODS
    if a.foa_tolerance_deg is None and uses_foa:a.foa_tolerance_deg=DEFAULT_FOA_TOLERANCE_DEG[a.stop_rule]
    if a.foa_tolerance_deg is not None and (not uses_foa or not 0.<a.foa_tolerance_deg<=45.):p.error('FOA tolerance requires a FOA rule and an angle in (0, 45] degrees')
    if uses_foa and not a.listen_before_decide_s:p.error('FOA stop requires stationary1s acquisition')
    if a.listen_before_decide_s and (a.decision_period_s!=1 or a.profile!='avn_foa_short1s'):p.error('Stationary listening requires 1s movement segments and 1s localization')
    if a.decision_period_s==1 and a.profile!='avn_foa_short1s':p.error('1s decision mode requires the 1s localization profile')
    if a.output.is_absolute():raise ValueError('Use a path relative to the mounted project worktree')
    a.output.mkdir(parents=True,exist_ok=False);out=a.output
    (out/'run_config.json').write_text(json.dumps(dict(episode_id=a.episode_id,episode_file=EPISODES,seed=a.seed,max_steps=a.steps,
        model='gpt-6.1-sol',provider_host='code.ppchat.vip',auth_mode='api_key',policy='existing evidence policy, no yaw probe',
        policy_observations='primary RGB-D/binaural/pose; FOA multiview STOP verification' if uses_foa else 'primary RGB-D/binaural/pose; existing passive STOP gate',
        shadow='synchronized FOA env; raw output withheld; derived STOP evidence supplied' if uses_foa else 'separate FOA env, exact same executed actions, never fed to policy',
        input_rate=16000,model_window_s=1 if a.profile=='avn_foa_short1s' else 10,
        localization_profile=a.profile,localization_update_period_s=None if a.listen_before_decide_s else (.25 if a.profile=='avn_foa_short1s' else 1),resolution=512,
        decision_period_s=a.decision_period_s,listen_before_decide_s=a.listen_before_decide_s,sound_schedule=a.sound_schedule,control_mode='stationary_listen_v1' if a.listen_before_decide_s else ('synchronous_segments_v1' if a.decision_period_s==1 else 'per_native_step'),
        step_budget_unit='native .25s steps',foa_audio_threads=a.foa_audio_threads,binaural_audio_threads=a.binaural_audio_threads,
        source_placement=a.source_placement,stop_rule=a.stop_rule,foa_tolerance_deg=a.foa_tolerance_deg,
        max_stop_confirmation_retries=int(a.stop_rule=='foa_geometry'),localization_used_for_stop=uses_foa,localization_used_for_motion_direction=False,terminal_foa_facing=uses_foa,
        segment_guards='sensor-only; no target truth'),indent=2)+'\n')
    backend=CodexBackend(a.inputs,out/'codex_calls.jsonl');history=deque(maxlen=8);observed=set()
    simcmd=['docker','exec','-i','-e','HABITAT_SIM_LOG=quiet','-e','MAGNUM_LOG=quiet','-w',a.sim_workdir,'savnce_dev','python','-m']
    with ExitStack() as stack:
        extract=stack.enter_context(jsonl_process(simcmd+['agent_episode.llm_observation'],out/'feature_worker.log'))
        worker=Worker(worker_command('avseld_final_test',a.profile),out/'candidate_worker.log');stack.callback(worker.close)
        tool=LocalizationTool(worker.request,profile=a.profile);tool.reset_episode(a.episode_id)
        desc=tool.describe();(out/'model.json').write_text(json.dumps(desc,indent=2)+'\n');assert desc['status']=='ok'
        sim=stack.enter_context(jsonl_process(simcmd+['agent_episode.serve','--inline-observations','--finalize-rollout'],out/'primary_simulator.log'))
        shadow=stack.enter_context(jsonl_process(simcmd+['agent_episode.serve','--inline-observations','--finalize-rollout'],out/'shadow_simulator.log'))
        reset=dict(command='reset',episode_id=a.episode_id,episode_file=EPISODES,resolution=512,seed=a.seed,max_steps=a.steps,sound_schedule=a.sound_schedule,source_placement=a.source_placement)
        primary=sim(dict(reset,output=str(out/'navigation'),audio_mode='binaural',audio_threads=a.binaural_audio_threads))['observation']
        foa=shadow(dict(reset,output=str(out/'foa_shadow'),audio_mode='foa',audio_threads=a.foa_audio_threads))['observation']
        done=False
        cycles=None
        try:
            if a.decision_period_s==1:
                primary,cycles=run_segmented(primary,foa,sim=sim,shadow=shadow,extract=extract,tool=tool,backend=backend,
                    history=history,observed=observed,out=out,max_steps=a.steps,listen_s=a.listen_before_decide_s,stop_rule=a.stop_rule,foa_tolerance_deg=a.foa_tolerance_deg)
                done=True
            while not done:
                assert_pose(primary,foa);tool.add_observation(foa)
                data=extract(primary);summary=data['summary'];ident=summary['observation_id']
                observed.update(f'{ident}:{m}' for m in MODALITIES)
                append(out/'model_observations.jsonl',summary)
                # Decide first, then run shadow. No localization function/result enters choose().
                decision=choose(backend,'evidence',data,history,ACTIONS,observed,out/'model_decisions.jsonl')
                decision_hash=hashlib.sha256(json.dumps(decision,sort_keys=True).encode()).hexdigest()
                decision_ready=time.time()
                if a.profile=='avn_foa_short1s' or primary['step']<40 or (primary['step']-39)%4==0:
                    result=tool.analyze([ident]);append(out/'predictions.jsonl',dict(at=ident,**result))
                    append(out/'shadow.jsonl',dict(at=ident,decision_already_committed_sha256=decision_hash,decision_ready_unix=decision_ready,**compact(result)))
                assert decision_hash==hashlib.sha256(json.dumps(decision,sort_keys=True).encode()).hexdigest()
                reply=sim(dict(command='step',action=decision['action'],decision=decision))
                mirror=shadow(dict(command='step',action=decision['action'],decision={'source':'primary_action_mirror','primary_decision_sha256':decision_hash}))
                primary,foa,done=reply['observation'],mirror['observation'],reply['done']
                assert done==mirror['done'];assert_pose(primary,foa)
                history.append({k:summary[k] for k in ('observation_id','xz_m','forward_xz','collided','forward_blocked','audio','depth')}|{'action':decision['action']})
                print(json.dumps(dict(observation=ident,action=decision['action'],step=primary['step'],done=done)),flush=True)
            if a.decision_period_s==.25:
                tool.add_observation(foa)
                result=tool.analyze([foa['observation_id']]);append(out/'predictions.jsonl',dict(at=foa['observation_id'],**result));append(out/'shadow.jsonl',dict(at=foa['observation_id'],terminal=True,**compact(result)))
        finally:
            sim({'command':'close'});shadow({'command':'close'})
    actions=[json.loads(line) for line in (out/'navigation/actions.jsonl').read_text().splitlines()]
    listen_count=sum(row['action']=='listen' for row in actions)
    (out/'rollout_complete.json').write_text(json.dumps(dict(status='completed',movement_steps=primary['step']-listen_count,listening_steps=listen_count,total_native_steps=primary['step'],shadow_used_for_navigation=a.stop_rule!='legacy',localization_used_for_stop=uses_foa,localization_used_for_motion_direction=False,terminal_foa_facing=uses_foa,pose_lockstep_verified=True,policy_model='gpt-6.1-sol',decision_cycles=cycles,decision_period_s=a.decision_period_s),indent=2)+'\n')

if __name__=='__main__':main()
