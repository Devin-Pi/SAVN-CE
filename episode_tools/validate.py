"""Validate official schema, native Dataset loading, and full Habitat task replay."""
from pathlib import Path
import argparse,gzip,hashlib,json,runpy,copy
import numpy as np
import habitat
import savnce
import savnce.tasks.nav
from savnce.datasets.savnce_dataset import SAVNCE_Dataset

def main():
 a=argparse.ArgumentParser();a.add_argument('--dataset-root',required=True);a.add_argument('--split',default='train');a.add_argument('--render-count',type=int,default=20);args=a.parse_args();root=Path(args.dataset_root)
 manifest=json.loads((root/'manifest.json').read_text());source=json.loads(gzip.open(manifest['source_episode_file'],'rt').read())['episodes'];source_map={str(e['episode_id']):e for e in source}
 index=root/args.split/(args.split+'.json.gz');content=next((root/args.split/'content').glob('*.json.gz'));raw=json.loads(gzip.open(content,'rt').read())['episodes'];assert len(raw)==manifest['count']
 provenance={e['episode_id']:e for e in manifest['episodes']}
 for e in raw:
  original=source_map[provenance[e['episode_id']]['official_template_episode_id']];g=e['goals'][0];og=original['goals'][0]
  assert set(e)==set(original) and set(g)==set(og)
  assert all(g[k]==og[k] for k in ['object_id','object_category','sound_id','position'])
  assert g['offset']-g['onset']==g['duration'] and e['oracle_actions'][-1]==0 and len(e['oracle_actions'])==e['info']['num_action']
  assert all(x in [0,1,2,3] for x in e['oracle_actions']) and abs(np.linalg.norm(e['start_rotation'])-1)<1e-6
  assert manifest['arguments']['min_distance']<=e['info']['geodesic_distance']<=manifest['arguments']['max_distance']
  assert not any(e['start_position']==s['start_position'] for s in source)
 cfg=runpy.run_path('savnce_baselines/magnet/config/default.py')['get_task_config']('configs/savnce/magnet/mp3d/savnce_clean.yaml');cfg.defrost()
 cfg.DATASET.DATA_PATH=str(index.resolve());cfg.DATASET.SCENES_DIR=manifest['arguments']['scenes_root'];cfg.DATASET.CONTENT_SCENES=['*'];cfg.DATASET.SPLIT=args.split
 cfg.SIMULATOR.AUDIO.SOURCE_SOUND_DIR=str(Path(manifest['arguments']['sounds_root']).resolve());cfg.SIMULATOR.AUDIO.THREADS=4
 cfg.SIMULATOR.AGENT_0.SENSORS=['RGB_SENSOR','DEPTH_SENSOR'];cfg.SIMULATOR.RGB_SENSOR.WIDTH=cfg.SIMULATOR.RGB_SENSOR.HEIGHT=128;cfg.SIMULATOR.DEPTH_SENSOR.WIDTH=cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT=128;cfg.freeze()
 ds=SAVNCE_Dataset(cfg.DATASET);assert len(ds.episodes)==len(raw);assert {str(e.episode_id) for e in ds.episodes}=={e['episode_id'] for e in raw}
 # Full original serializer/deserializer round trip on a copy, because to_json mutates its input.
 clone=copy.deepcopy(ds);encoded=clone.to_json();roundtrip=SAVNCE_Dataset();roundtrip._config=cfg.DATASET.clone();roundtrip._config.defrost();roundtrip._config.CONTINUOUS=False;roundtrip._config.freeze();roundtrip.from_json(encoded,scenes_dir=None)
 assert len(roundtrip.episodes)==len(ds.episodes)
 original_map={e['episode_id']:e for e in raw}
 results=[];env=habitat.Env(config=cfg,dataset=ds)
 try:
  env.seed(manifest['seed']);selected=ds.episodes[:min(args.render_count,len(ds.episodes))];env.episode_iterator=iter(selected)
  for e in selected:
   obs=env.reset();assert str(env.current_episode.episode_id)==str(e.episode_id);max_rms=0.;collisions=0;steps=0
   for action in original_map[str(e.episode_id)]['oracle_actions']:
    audio=env.sim.get_current_audiogoal_observation();assert audio.shape==(4000,2) and np.isfinite(audio).all();max_rms=max(max_rms,float(np.sqrt(np.mean(audio**2))))
    assert obs['rgb'].shape==(128,128,3) and obs['depth'].shape==(128,128,1) and np.isfinite(obs['depth']).all()
    assert obs['binaural_extractor'].shape==(4,256,26)
    obs=env.step(action);steps+=1
    if action!=0:collisions+=int(env.sim._prev_sim_obs.get('collided',False))
   metrics=env.get_metrics();success=bool(metrics['success']);assert env.episode_over and success and max_rms>1e-7,(e.episode_id,metrics,max_rms)
   result={'episode_id':str(e.episode_id),'success':success,'final_distance_m':float(metrics['distance_to_goal']),'spl':float(metrics['spl']),'actions_including_stop':steps,'max_audio_rms':max_rms,'native_collisions':collisions,'rgb_depth_and_audio_finite':True}
   results.append(result);print(json.dumps(result),flush=True)
 finally:env.close()
 report={'schema_keys_identical_to_official':True,'official_object_sound_goal_associations_preserved':True,'new_starts_verified':True,'native_dataset_loaded':len(ds.episodes),'native_serializer_roundtrip':True,'full_habitat_env_rendered_and_replayed':len(results),'all_replayed_successful_and_audible':all(r['success'] and r['max_audio_rms']>0 for r in results),'reference_controller':'privileged oracle actions for dataset validation, not Agent performance','results':results}
 (root/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 (root/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(root))+'\n' for f in sorted(root.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
 print('VALIDATION PASS',json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
if __name__=='__main__':main()
