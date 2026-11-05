"""Additional reset-only metric/engine checks, without re-exporting trajectories."""
from pathlib import Path
import argparse,collections,hashlib,json,runpy,os
import habitat,savnce
import savnce.tasks.nav
from savnce.datasets.savnce_dataset import SAVNCE_Dataset

ROOT=Path('research_outputs/agent_dataset_v1_1')
def sha(p):
 p=str(p).replace('/media/4tbssd/00_PROJECT_CODE/SAVN-CE','/workspace/SAVN-CE').replace('/media/4tbssd/DATA/mp3d/v1/tasks/mp3d','/datasets/mp3d')
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 global ROOT
 parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT));args=parser.parse_args();ROOT=Path(args.output)
 rows=[json.loads(x) for x in (ROOT/'episode_audit.jsonl').read_text().splitlines()];byid={x['episode_id']:x for x in rows};engine=json.loads((ROOT/'environment.json').read_text())
 cfg=runpy.run_path('savnce_baselines/magnet/config/default.py')['get_task_config']('configs/savnce/magnet/mp3d/savnce_clean.yaml');cfg.defrost();cfg.DATASET.DATA_PATH=str(Path('data/datasets/savnce_generated/mp3d/v1/train/train.json.gz').resolve());cfg.DATASET.SCENES_DIR='/datasets/mp3d';cfg.DATASET.CONTENT_SCENES=['*'];cfg.SIMULATOR.AUDIO.SOURCE_SOUND_DIR=str(Path('data/sounds/semantic_splits').resolve());cfg.SIMULATOR.AUDIO.THREADS=4;cfg.SIMULATOR.AGENT_0.SENSORS=['RGB_SENSOR','DEPTH_SENSOR'];cfg.SIMULATOR.RGB_SENSOR.WIDTH=cfg.SIMULATOR.RGB_SENSOR.HEIGHT=128;cfg.SIMULATOR.DEPTH_SENSOR.WIDTH=cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT=128;cfg.freeze();ds=SAVNCE_Dataset(cfg.DATASET);env=habitat.Env(config=cfg,dataset=ds)
 try:
  env.episode_iterator=iter(ds.episodes)
  for e in ds.episodes:
   env.reset();row=byid[str(e.episode_id)];distance=float(env.get_metrics()['distance_to_goal']);delta=abs(distance-row['native_replay']['initial_distance_m']);assert delta<.05
   row['native_replay'].update(start_navigable=bool(env.sim.pathfinder.is_navigable(env.sim.get_agent_state().position)),actual_initial_distance_m=distance,declared_distance_error_m=delta)
   assert row['native_replay']['start_navigable']
   if not engine.get('acoustic_export',{}).get('export_succeeded'):
    f=ROOT/'private/base_acoustic_geometry.obj';sensor=env.sim._sim.get_agent(0)._sensors['audio_goal_sensor'];ok=sensor.writeSceneMeshOBJ(str(f));counts=collections.Counter();lo=[float('inf')]*3;hi=[-float('inf')]*3
    if f.exists():
     with f.open(errors='replace') as stream:
      for line in stream:
       vals=line.split();counts[vals[0] if vals else '']+=1
       if vals and vals[0]=='v':
        for k in range(3):lo[k]=min(lo[k],float(vals[k+1]));hi[k]=max(hi[k],float(vals[k+1]))
     engine['acoustic_export']={'export_succeeded':bool(ok),'vertices':counts['v'],'faces':counts['f'],'bounds':[lo,hi],'sha256':sha(f),'status':'base RLRA mesh exported; exact visual/collision/semantic surface equivalence unverified','file_removed_after_statistics':True};f.unlink()
 finally:env.close()
 (ROOT/'episode_audit.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in rows));(ROOT/'environment.json').write_text(json.dumps(engine,indent=2,allow_nan=False)+'\n')
 metadata=[json.loads(x) for x in (ROOT/'scenario_metadata.jsonl').read_text().splitlines()]
 selection={g:[x['episode_id'] for x in metadata if x['scenario_group']==g] for g in ['G0','G1','unassigned']};selection.update({'annotation_status':'provisional_instrumented; manual RGB similarity review pending','new_development_tasks_added':0,'reason':'Existing tasks include at least one G0 and G1 candidate; do not expand unnecessarily.'});(ROOT/'study_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 before=json.loads((ROOT/'input_hashes_before.json').read_text());after={f:sha(f) for f in before};assert before==after;(ROOT/'input_hashes_after.json').write_text(json.dumps(after,indent=2)+'\n')
 print('FINALIZE_PASS',selection,flush=True)
if __name__=='__main__':main()
