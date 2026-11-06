from pathlib import Path
import argparse,json,hashlib,subprocess
import numpy as np
import soundfile as sf
import imageio.v2 as imageio
import imageio_ffmpeg
from .environment import OfficialEpisodeEnv,DEFAULT_EPISODES
from .policies import AudioDepthAgent

def run(output,policy='evidence',episode_id='410',steps=64,resolution=256,seed=7,episode_file=DEFAULT_EPISODES, *, stop_probe=True):
    env=OfficialEpisodeEnv(output,episode_id=episode_id,resolution=resolution,max_steps=steps,seed=seed,episodes_path=episode_file)
    agent = AudioDepthAgent(verify=policy == 'evidence', stop_probe=stop_probe)
    p=Path(output);rgb=[];audio=[]
    try:
        o=env.reset()
        rgb.append(o['rgb'])
        audio.append(o['binaural_audio'])
        while not env.done:
            previous_observation_id = o['observation_id']
            d=agent.decide(o)
            with (p/'estimates.jsonl').open('a') as f:f.write(json.dumps({'observation_id':o['observation_id'],**d['estimate']})+'\n')
            o,done=env.step(d['action'],d)
            print(o['step'],d['action'],d['estimate']['accepted_for_action'],flush=True)
            # STOP returns the current observation without advancing simulation time.
            if o['observation_id'] != previous_observation_id:
                rgb.append(o['rgb'])
                audio.append(o['binaural_audio'])
        result=env.evaluate();result.update({'policy':policy,'stop_probe_enabled':policy == 'evidence' and stop_probe,'official_association':True,'episode_file':episode_file,'is_author_official_benchmark':episode_file==DEFAULT_EPISODES,'controller':'transparent audio/depth rule baseline; not an LLM','raw_observations':len(rgb),'sample_rate':env.sr,'dt_s':env.dt})
        (env.private_output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
        sf.write(p/'binaural.wav',np.concatenate(audio),env.sr,subtype='FLOAT');imageio.mimwrite(p/'rgb.mp4',rgb,fps=1/env.dt,quality=7,macro_block_size=1)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',str(p/'rgb.mp4'),'-i',str(p/'binaural.wav'),'-c:v','copy','-c:a','aac','-shortest',str(p/'avn_demo.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        (p/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(p))+'\n' for f in sorted(p.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
        print(json.dumps(result,indent=2));return result
    finally:env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--policy', choices=['baseline', 'evidence'], default='evidence')
    parser.add_argument('--episode-id', default='410')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--episode-file', default=DEFAULT_EPISODES)
    parser.add_argument('--no-stop-probe', dest='stop_probe', action='store_false',
                        help='Use passive range and depth for STOP without an active yaw sweep')
    run(**vars(parser.parse_args()))
if __name__=='__main__':main()
