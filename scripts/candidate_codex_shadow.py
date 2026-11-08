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


def assert_pose(primary,foa):
    assert primary['observation_id']==foa['observation_id'] and primary['timestamp_s']==foa['timestamp_s']
    assert math.dist(primary['pose']['position'],foa['pose']['position'])<1e-5
    assert abs(sum(a*b for a,b in zip(primary['pose']['rotation_xyzw'],foa['pose']['rotation_xyzw'])))>1-1e-5


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--sim-workdir',required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--episode-id',default='1000000008');p.add_argument('--steps',type=int,default=64);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--profile',choices=('avn_foa_candidate','avn_foa_short1s'),default='avn_foa_candidate');a=p.parse_args()
    if a.output.is_absolute():raise ValueError('Use a path relative to the mounted project worktree')
    a.output.mkdir(parents=True,exist_ok=False);out=a.output
    (out/'run_config.json').write_text(json.dumps(dict(episode_id=a.episode_id,episode_file=EPISODES,seed=a.seed,max_steps=a.steps,
        model='gpt-6.1-sol',provider_host='code.ppchat.vip',auth_mode='api_key',policy='existing evidence policy, no yaw probe',
        policy_observations='primary RGB-D/binaural/pose; existing passive STOP gate',shadow='separate FOA env, exact same executed actions, never fed to policy',
        input_rate=16000,model_window_s=1 if a.profile=='avn_foa_short1s' else 10,
        localization_profile=a.profile,localization_update_period_s=.25 if a.profile=='avn_foa_short1s' else 1,resolution=512),indent=2)+'\n')
    backend=CodexBackend(a.inputs,out/'codex_calls.jsonl');history=deque(maxlen=8);observed=set()
    simcmd=['docker','exec','-i','-e','HABITAT_SIM_LOG=quiet','-e','MAGNUM_LOG=quiet','-w',a.sim_workdir,'savnce_dev','python','-m']
    with ExitStack() as stack:
        extract=stack.enter_context(jsonl_process(simcmd+['agent_episode.llm_observation'],out/'feature_worker.log'))
        worker=Worker(worker_command('avseld_final_test',a.profile),out/'candidate_worker.log');stack.callback(worker.close)
        tool=LocalizationTool(worker.request,profile=a.profile);tool.reset_episode(a.episode_id)
        desc=tool.describe();(out/'model.json').write_text(json.dumps(desc,indent=2)+'\n');assert desc['status']=='ok'
        sim=stack.enter_context(jsonl_process(simcmd+['agent_episode.serve','--inline-observations','--finalize-rollout'],out/'primary_simulator.log'))
        shadow=stack.enter_context(jsonl_process(simcmd+['agent_episode.serve','--inline-observations','--finalize-rollout'],out/'shadow_simulator.log'))
        reset=dict(command='reset',episode_id=a.episode_id,episode_file=EPISODES,resolution=512,seed=a.seed,max_steps=a.steps)
        primary=sim(dict(reset,output=str(out/'navigation'),audio_mode='binaural'))['observation']
        foa=shadow(dict(reset,output=str(out/'foa_shadow'),audio_mode='foa'))['observation']
        done=False
        try:
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
            tool.add_observation(foa)
            result=tool.analyze([foa['observation_id']]);append(out/'predictions.jsonl',dict(at=foa['observation_id'],**result));append(out/'shadow.jsonl',dict(at=foa['observation_id'],terminal=True,**compact(result)))
        finally:
            sim({'command':'close'});shadow({'command':'close'})
    (out/'rollout_complete.json').write_text(json.dumps(dict(status='completed',movement_steps=primary['step'],shadow_used_for_navigation=False,pose_lockstep_verified=True,policy_model='gpt-6.1-sol'),indent=2)+'\n')

if __name__=='__main__':main()
