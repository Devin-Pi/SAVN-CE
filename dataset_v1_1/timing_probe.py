"""Privileged online timing probe; stores scalar diagnostics only, never sensor sequences."""
from pathlib import Path
import argparse,json,math,random,runpy
import numpy as np
import habitat,savnce
import savnce.tasks.nav
from savnce.datasets.savnce_dataset import SAVNCE_Dataset

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='research_outputs/agent_dataset_v1_1');args=p.parse_args();root=Path(args.output)
 dest=root/'private/audio_timing_probe.json';assert not dest.exists(),'Use a fresh probe output'
 cfg=runpy.run_path('savnce_baselines/magnet/config/default.py')['get_task_config']('configs/savnce/magnet/mp3d/savnce_clean.yaml');cfg.defrost();cfg.DATASET.DATA_PATH=str(Path('data/datasets/savnce_generated/mp3d/v1/train/train.json.gz').resolve());cfg.DATASET.SCENES_DIR='/datasets/mp3d';cfg.DATASET.CONTENT_SCENES=['*'];cfg.SIMULATOR.AUDIO.SOURCE_SOUND_DIR=str(Path('data/sounds/semantic_splits').resolve());cfg.SIMULATOR.AUDIO.THREADS=4;cfg.SIMULATOR.AGENT_0.SENSORS=['RGB_SENSOR','DEPTH_SENSOR'];cfg.SIMULATOR.RGB_SENSOR.WIDTH=cfg.SIMULATOR.RGB_SENSOR.HEIGHT=128;cfg.SIMULATOR.DEPTH_SENSOR.WIDTH=cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT=128;cfg.freeze()
 ds=SAVNCE_Dataset(cfg.DATASET);env=habitat.Env(config=cfg,dataset=ds);random.seed(20261105);np.random.seed(20261105);env.seed(20261105);rows=[]
 try:
  env.episode_iterator=iter(ds.episodes)
  for ep in ds.episodes:
   env.reset();g=ep.goals[0];windows=[];n=math.ceil((g.offset+1.25)/.25)
   for i in range(n):
    audio=env.sim.get_current_audiogoal_observation();assert audio.shape==(4000,2) and np.isfinite(audio).all()
    t=env.sim._episode_step_count*.25;expected=g.onset<=t<g.offset;assert env.sim.is_silent== (not expected)
    windows.append({'step':i,'start_s':t,'end_s':t+.25,'rms':float(np.sqrt(np.mean(audio**2))),'source_emission_enabled':not env.sim.is_silent})
    if i<n-1:env.step(2 if i%2==0 else 3) # Legal turn/wait; does not use goal-directed actions.
   pre=[w for w in windows if w['end_s']<=g.onset];active=[w for w in windows if w['source_emission_enabled']];post=[w for w in windows if w['start_s']>=g.offset]
   assert pre and active and post and max(w['rms'] for w in pre)<1e-8 and max(w['rms'] for w in active)>1e-5
   row={'episode_id':str(ep.episode_id),'control':'privileged_alternating_legal_turns_not_agent','scheduled_interval_s':[g.onset,g.offset],'collected_interval_s':[0,windows[-1]['end_s']],'pre_onset_silence_confirmed':True,'active_audio_above_1e_5':True,'post_offset_window_sampled':True,'post_offset_max_rms':max(w['rms'] for w in post),'last_window_rms':windows[-1]['rms'],'native_emission_gate_matches_schedule':True,'tail_interpretation':'Nonzero post-offset audio can be previously accumulated convolution tail, not evidence of new emission. Continuous moving-source physics is not tested.','windows':windows};rows.append(row);print('TIMING_PASS',ep.episode_id,'post_max',row['post_offset_max_rms'],flush=True)
 finally:env.close()
 dest.write_text(json.dumps({'seed':20261105,'rows':rows},indent=2,allow_nan=False)+'\n');print('TIMING_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
