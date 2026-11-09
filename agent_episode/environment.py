"""Step-level official SAVN-CE environment; privileged truth is not in observations."""
from pathlib import Path
import copy,gzip,json,runpy,hashlib
import numpy as np
import soundfile as sf
import imageio.v2 as imageio
from habitat_sim.utils.common import quat_to_coeffs
from savnce.datasets.savnce_dataset import SAVNCE_Dataset
from savnce.tasks.savnce_task import merge_sim_episode_config
from savnce.sims.savnce_simulator import SAVNCE_Simulator
from .foa import audio_format, to_ambix
from .sound_schedule import resolve_sound_schedule

ACTIONS={0:'stop',1:'move_forward',2:'turn_left',3:'turn_right'}
DEFAULT_EPISODES='audit/official_samples/savnce_dataset/mp3d/v1/train/content/17DRP5sb8fy.json.gz'

class OfficialEpisodeEnv:
    def __init__(self,output,episode_id='410',resolution=256,max_steps=64,seed=7,episodes_path=DEFAULT_EPISODES,private_output=None,*,audio_mode='binaural',sound_schedule='episode',audio_threads=None):
        if audio_mode not in ('binaural', 'foa'):
            raise ValueError("audio_mode must be 'binaural' or 'foa'")
        self.audio_mode = audio_mode
        if audio_threads is None:
            audio_threads = 1 if audio_mode == 'foa' else 4
        if type(audio_threads) is not int or audio_threads < 1:
            raise ValueError('audio_threads must be a positive integer')
        self.audio_threads = audio_threads
        if sound_schedule not in ('episode','sustained'):raise ValueError('Unknown sound schedule')
        self.sound_schedule=sound_schedule
        self.audio_key = 'foa_audio' if audio_mode == 'foa' else 'binaural_audio'
        self.output=Path(output)
        if self.output.exists() and any(self.output.iterdir()):raise FileExistsError('Use an empty output directory')
        self.output.mkdir(parents=True,exist_ok=True);(self.output/'observations').mkdir()
        private_root=Path('research_outputs/agent_private');private_root.mkdir(parents=True,exist_ok=True);private_root.chmod(0o700)
        token=hashlib.sha256(str(self.output.resolve()).encode()).hexdigest()[:12]
        self.private_output=Path(private_output) if private_output else private_root/(token+'_'+self.output.name)
        self.private_output.mkdir(parents=True,exist_ok=True);self.private_output.chmod(0o700)
        self.episode_id=str(episode_id);self.resolution=resolution;self.max_steps=max_steps;self.seed=seed
        self.sim=None;self.index=0;self.done=False;self._cached=None;self._last_collision=False;self._last_blocked=False
        self._episode_path=Path(episodes_path)

    def reset(self):
        if self.sim is not None:self.sim._sim.close()
        if self.index:raise RuntimeError('Create a fresh output/environment for another rollout')
        np.random.seed(self.seed)
        cfg=runpy.run_path('savnce_baselines/magnet/config/default.py')['get_task_config']();cfg.defrost()
        cfg.DATASET.DATA_PATH=str(self._episode_path.resolve());cfg.DATASET.SCENES_DIR='/datasets/mp3d';cfg.DATASET.CONTENT_SCENES=[]
        dataset=SAVNCE_Dataset(cfg.DATASET)
        self._episode=next(e for e in dataset.episodes if str(e.episode_id)==self.episode_id)
        c=cfg.SIMULATOR;c.TYPE='SAVNCE_Simulator';c.AGENT_0.SENSORS=['RGB_SENSOR','DEPTH_SENSOR']
        c.RGB_SENSOR.WIDTH=c.RGB_SENSOR.HEIGHT=self.resolution;c.DEPTH_SENSOR.WIDTH=c.DEPTH_SENSOR.HEIGHT=self.resolution
        c.AUDIO.MIC_TYPE='ambisonics' if self.audio_mode == 'foa' else 'binaural';c.AUDIO.THREADS=self.audio_threads;c.AUDIO.EVERLASTING=False;c.AUDIO.HAS_DISTRACTOR_SOUND=False;c.AUDIO.HAS_NOISE_SOUND=False
        c.AUDIO.SOURCE_SOUND_DIR=str(Path('data/sounds/semantic_splits').resolve())
        assert (Path(c.AUDIO.SOURCE_SOUND_DIR)/self._episode.goals[0].sound_id).is_file()
        merge_sim_episode_config(c,self._episode)
        self.runtime_sound_schedule=resolve_sound_schedule(self.sound_schedule,c.AGENT_0.ONSET,c.AGENT_0.OFFSET,c.STEP_TIME,self.max_steps)
        c.defrost()  # Official merge freezes the simulator and agent config.
        c.AGENT_0.ONSET=self.runtime_sound_schedule['runtime_onset_s'];c.AGENT_0.OFFSET=self.runtime_sound_schedule['runtime_offset_s']
        (self.output/'sound_schedule.json').write_text(json.dumps(self.runtime_sound_schedule,indent=2)+'\n')
        cfg.freeze();self.config=c
        self.sim=SAVNCE_Simulator(c);self.sim.seed(self.seed);self.sim.reconfigure(c);self._native=self.sim.reset()
        self.dt=c.STEP_TIME;self.sr=c.AUDIO.RIR_SAMPLING_RATE
        if self.audio_mode == 'foa':
            self.audio_format = dict(audio_format(self.sr, self.dt), rendering_threads=self.audio_threads)
            (self.output/'audio_format.json').write_text(json.dumps(self.audio_format, indent=2)+'\n')
        self.initial_distance=self._distance()
        raw=json.loads(gzip.open(self._episode_path,'rt').read())
        original=next(e for e in raw['episodes'] if str(e['episode_id'])==self.episode_id)
        source=Path(c.AUDIO.SOURCE_SOUND_DIR)/self._episode.goals[0].sound_id
        truth={'official_episode':original,'episode_file':str(self._episode_path),'episode_file_sha256':hashlib.sha256(self._episode_path.read_bytes()).hexdigest(),'source_wav_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'runtime_goal_position':self._episode.goals[0].position,'acoustic_source_position':(np.array(self._episode.goals[0].position)+[0,1.5,0]).tolist(),'initial_geodesic_distance':self.initial_distance,'agent_receives_ground_truth':False,'is_author_official_benchmark':self._episode_path==Path(DEFAULT_EPISODES),'association':'official object/sound association; task may be newly generated'}
        truth['runtime_sound_schedule']=self.runtime_sound_schedule
        (self.private_output/'evaluation_truth.json').write_text(json.dumps(truth,indent=2)+'\n')
        (self.private_output/'resolved_simulator.yaml').write_text(str(c))
        return self.observe()

    def _distance(self):return float(self.sim.geodesic_distance(self.sim.get_agent_state().position,[self._episode.goals[0].position]))

    def current_state(self):
        """Read live robot pose without advancing time or returning target truth."""
        state=self.sim.get_agent_state()
        return dict(observation_id=f'obs_{self.index:04d}',step=self.index,timestamp_s=self.index*self.dt,
                    pose=dict(position=state.position.tolist(),rotation_xyzw=quat_to_coeffs(state.rotation).tolist()))

    def observe(self):
        if self._cached is not None:return self._cached
        state=self.sim.get_agent_state();audio=self.sim.get_current_audiogoal_observation().copy()
        if self.audio_mode == 'foa':
            audio = to_ambix(audio)
        rgb=self._native['rgb'][...,:3].copy();depth=self._native['depth'].copy()
        if self.config.DEPTH_SENSOR.NORMALIZE_DEPTH:
            depth=depth*(self.config.DEPTH_SENSOR.MAX_DEPTH-self.config.DEPTH_SENSOR.MIN_DEPTH)+self.config.DEPTH_SENSOR.MIN_DEPTH
        pose={'position':state.position.tolist(),'rotation_xyzw':quat_to_coeffs(state.rotation).tolist()}
        ident=f'obs_{self.index:04d}';base=self.output/'observations'/ident
        np.savez_compressed(str(base)+'.npz',rgb=rgb,depth_m=depth,position=state.position,rotation_xyzw=quat_to_coeffs(state.rotation),**{self.audio_key: audio})
        sf.write(str(base)+'.wav',audio,self.sr,subtype='FLOAT');imageio.imwrite(str(base)+'.png',rgb)
        refs={'npz':str(base)+'.npz','rgb':str(base)+'.png','audio':str(base)+'.wav'}
        public={'observation_id':ident,'step':self.index,'timestamp_s':self.index*self.dt,'rgb':rgb,'depth_m':depth,self.audio_key:audio,'pose':pose,'sample_rate':self.sr,'dt_s':self.dt,'collided_previous_step':self._last_collision,'blocked_previous_step':self._last_blocked,'files':refs}
        if self.audio_mode == 'foa':
            public['audio_format'] = self.audio_format
            sensor = self.sim._sim.get_agent(0)._sensors[self.sim._audio_goal_sensor_uuid]
            public['audio_receiver_position'] = np.asarray(sensor.node.absolute_translation).tolist()
        row={k:v for k,v in public.items() if k not in ['rgb','depth_m',self.audio_key]}
        row['shapes']={'rgb':list(rgb.shape),'depth_m':list(depth.shape),self.audio_key:list(audio.shape)}
        row['audio_rms']=float(np.sqrt(np.mean(audio**2)))
        registry=[{'evidence_id':ident+':audio','modality':self.audio_key,'file':refs['audio'],'sample_rate':self.sr,'time_interval_s':[self.index*self.dt,(self.index+1)*self.dt]},
                  {'evidence_id':ident+':depth','modality':'metric_depth','file':refs['npz'],'array_key':'depth_m','units':'metres'},
                  {'evidence_id':ident+':rgb','modality':'RGB','file':refs['rgb']},
                  {'evidence_id':ident+':pose','modality':'robot_pose','file':refs['npz'],'array_keys':['position','rotation_xyzw']}]
        with (self.output/'evidence_registry.jsonl').open('a') as f:
            for item in registry:f.write(json.dumps(item)+'\n')
        with (self.output/'observations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self._cached=public;return public

    def step(self,action,decision=None):
        if self.done:raise RuntimeError('Episode finished; create/reset another environment')
        if isinstance(action,str):action={v:k for k,v in ACTIONS.items()}[action]
        action=int(action)
        if action not in ACTIONS:raise ValueError('Invalid action')
        old=self.observe();before=np.array(old['pose']['position'])
        if action==0:
            self.done=True;new=old
        else:
            self._native=self.sim.step(action);self.index+=1;self._last_collision=bool(self.sim._prev_sim_obs.get('collided',False));self._last_blocked=bool(action==1 and np.linalg.norm(self.sim.get_agent_state().position-before)<.02);self._cached=None;new=self.observe();self.done=self.index>=self.max_steps
        row={'transition_id':f'transition_{len(self._actions):04d}' if hasattr(self,'_actions') else 'transition_0000','observation_before':old['observation_id'],'action_id':action,'action':ACTIONS[action],'observation_after':new['observation_id'],'pose_before':old['pose'],'pose_after':new['pose'],'translation_m':float(np.linalg.norm(np.array(new['pose']['position'])-before)),'collision':self._last_collision,'forward_blocked':bool(self._last_blocked),'done':self.done,'decision':decision}
        if not hasattr(self,'_actions'):self._actions=[]
        self._actions.append(row)
        with (self.output/'actions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return new,self.done

    def evaluate(self):
        distance=self._distance();stopped=bool(getattr(self,'_actions',[]) and self._actions[-1]['action_id']==0)
        actions=getattr(self,'_actions',[]);listening=sum(x['action']=='listen' for x in actions)
        return {'official_episode_id':self.episode_id,'initial_distance_m':self.initial_distance,'final_distance_m':distance,'stopped':stopped,'success_stop_within_1m':stopped and distance<1.0,'movement_steps':self.index-listening,'listening_steps':listening,'total_steps':self.index,'movement_time_s':(self.index-listening)*self.dt,'listening_time_s':listening*self.dt,'num_actions':len(actions),'termination':'stop' if stopped else 'step_budget','note':'Evaluator-only ground truth; not sent to policy.'}

    def listen_step(self,decision=None):
        """Advance native audio/time with a zero-displacement engine action.

        This coordinator-only action does not replace STOP or reset acoustics.
        The official episode, primitive policy action space and simulator step
        implementation remain unchanged.
        """
        if self.done:raise RuntimeError('Episode finished')
        import habitat_sim
        key='savnce_stationary_listen'
        space=self.sim._sim.get_agent(0).agent_config.action_space
        if key not in space:
            space[key]=habitat_sim.ActionSpec('move_forward',habitat_sim.ActuationSpec(amount=0.0))
        old=self.observe();before=self.current_state()
        self._native=self.sim.step(key);self.index+=1
        self._last_collision=False;self._last_blocked=False;self._cached=None
        new=self.observe();after=self.current_state()
        if not np.allclose(before['pose']['position'],after['pose']['position'],rtol=0,atol=1e-7) or not np.allclose(before['pose']['rotation_xyzw'],after['pose']['rotation_xyzw'],rtol=0,atol=1e-7):
            raise RuntimeError('Stationary listening changed robot pose')
        self.done=self.index>=self.max_steps
        if not hasattr(self,'_actions'):self._actions=[]
        row=dict(transition_id=f'transition_{len(self._actions):04d}',observation_before=old['observation_id'],action_id=key,action='listen',observation_after=new['observation_id'],pose_before=old['pose'],pose_after=new['pose'],translation_m=0.0,collision=False,forward_blocked=False,done=self.done,decision=decision)
        self._actions.append(row)
        with (self.output/'actions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return new,self.done
    def export_foa_wav(self):
        windows = []
        for path in sorted((self.output/'observations').glob('obs_*.wav')):
            audio, _ = sf.read(path, dtype='float32', always_2d=True)
            windows.append(audio)
        if windows:
            sf.write(self.output/'foa_ambix.wav', np.concatenate(windows), self.sr, subtype='FLOAT')

    def close(self):
        if self.sim:self.sim._sim.close();self.sim=None
