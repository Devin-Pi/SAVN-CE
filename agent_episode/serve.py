"""JSON lines stdin/stdout protocol for an external agent. Raw arrays are in saved NPZ/WAV/PNG."""
import sys,json,contextlib,traceback,os,base64
from pathlib import Path
INLINE_OBSERVATIONS='--inline-observations' in sys.argv
FINALIZE_ROLLOUT='--finalize-rollout' in sys.argv
PROTOCOL_OUT=os.fdopen(os.dup(1),'w',buffering=1)
os.dup2(2,1)
sys.stdout=sys.stderr
from .environment import OfficialEpisodeEnv,DEFAULT_EPISODES

def public(o):
    audio_key = 'foa_audio' if 'foa_audio' in o else 'binaural_audio'
    sensor_keys = ['rgb', 'depth_m', audio_key]
    d={k:v for k,v in o.items() if k not in sensor_keys}
    d['shapes']={k:list(o[k].shape) for k in sensor_keys}
    if INLINE_OBSERVATIONS:
        d.pop('files',None)
        d['sensor_payload']={'encoding':'npz_base64','data':base64.b64encode(Path(o['files']['npz']).read_bytes()).decode(),'allowed_array_keys':sensor_keys + ['position','rotation_xyzw']}
    return d

def main():
    env=None
    distances=[]
    for line in sys.stdin:
        try:
            req=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                cmd=req['command']
                if cmd=='reset':
                    if env:env.close()
                    env=OfficialEpisodeEnv(req['output'],req.get('episode_id','410'),req.get('resolution',256),req.get('max_steps',64),req.get('seed',7),episodes_path=req.get('episode_file',DEFAULT_EPISODES),audio_mode=req.get('audio_mode','binaural'));response={'observation':public(env.reset()),'actions':['move_forward','turn_left','turn_right','stop']}
                    if FINALIZE_ROLLOUT:distances=[env.initial_distance]
                elif cmd=='observe':response={'observation':public(env.observe())}
                elif cmd=='state':response={'state':env.current_state()}
                elif cmd in ('step','stop'):
                    o,done=env.step('stop' if cmd=='stop' else req['action'],req.get('decision'));response={'observation':public(o),'done':done}
                    if FINALIZE_ROLLOUT and env.index==len(distances):distances.append(env._distance())
                elif cmd=='close':
                    try:
                        if env.audio_mode == 'foa':
                            env.export_foa_wav()
                        if FINALIZE_ROLLOUT and distances:
                            from .rollout_export import export_rollout
                            export_rollout(env, distances)
                    finally:
                        env.close()
                    response={'closed':True}
                else:raise ValueError('Unknown command')
            print(json.dumps({'ok':True,**response}),file=PROTOCOL_OUT,flush=True)
            if cmd=='close':break
        except Exception as e:print(json.dumps({'ok':False,'error':str(e)}),file=PROTOCOL_OUT,flush=True)
    if env:env.close()
if __name__=='__main__':main()
