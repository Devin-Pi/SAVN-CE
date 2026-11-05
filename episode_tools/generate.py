"""New official-format tasks using audited official object/sound/goal templates."""
from pathlib import Path
import argparse,copy,gzip,hashlib,json,collections,subprocess
import numpy as np
import soundfile as sf
import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs

SOURCE='audit/official_samples/savnce_dataset/mp3d/v1/train/content/17DRP5sb8fy.json.gz'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def dump_gz(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('wb') as raw:
  with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:f.write(json.dumps(data,allow_nan=False).encode())
def dist(sim,a,b):
 path=habitat_sim.ShortestPath();path.requested_start=np.array(a,dtype=np.float32);path.requested_end=np.array(b,dtype=np.float32)
 return float(path.geodesic_distance) if sim.pathfinder.find_path(path) else float('inf')

def generate(args):
 out=Path(args.output)
 if out.exists() and any(out.iterdir()):raise FileExistsError('Refusing to overwrite a nonempty dataset')
 if not 0<args.min_distance<args.max_distance:raise ValueError('Invalid distance interval')
 source=Path(args.source);raw=json.loads(gzip.open(source,'rt').read());rng=np.random.default_rng(args.seed)
 scene_ids={e['scene_id'] for e in raw['episodes']}
 if len(scene_ids)!=1:raise ValueError('This small generator requires a single-scene content file')
 scene_id=next(iter(scene_ids));scene_path=Path(args.scenes_root)/scene_id
 sc=habitat_sim.SimulatorConfiguration();sc.scene_id=str(scene_path);sc.enable_physics=False;sc.create_renderer=False
 agent=habitat_sim.AgentConfiguration();agent.height=1.5;agent.radius=.1
 agent.action_space={1:habitat_sim.ActionSpec('move_forward',habitat_sim.ActuationSpec(amount=.25)),2:habitat_sim.ActionSpec('turn_left',habitat_sim.ActuationSpec(amount=15)),3:habitat_sim.ActionSpec('turn_right',habitat_sim.ActuationSpec(amount=15))}
 sim=habitat_sim.Simulator(habitat_sim.Configuration(sc,[agent]));sim.seed(args.seed);sim.pathfinder.seed(args.seed)
 try:
  objects={int(o.id.split('_')[-1]):o for o in sim.semantic_scene.objects if o is not None}
  templates={};rejected=collections.Counter()
  for e in raw['episodes']:
   g=e['goals'][0];oid=int(g['object_id']);sound=Path(args.sounds_root)/g['sound_id']
   if oid not in objects or objects[oid].category.name()!=g['object_category']:rejected['template_semantic_mismatch']+=1;continue
   if not sound.is_file():rejected['template_missing_sound']+=1;continue
   if sound.stem!=g['object_category']:rejected['template_sound_category_mismatch']+=1;continue
   if sf.info(sound).frames==0:continue
   key=(oid,g['sound_id'],tuple(g['position']))
   templates.setdefault(key,e)
  templates=list(templates.values());rng.shuffle(templates)
  if not templates:raise ValueError('No verified object/sound templates')
  generated=[];provenance=[];seen=set();source_starts={tuple(e['start_position']) for e in raw['episodes']}
  for attempt in range(args.count*500):
   if len(generated)==args.count:break
   template=templates[attempt%len(templates)];goal=copy.deepcopy(template['goals'][0]);target=np.array(goal['position'],dtype=np.float32)+np.array([0,.1,0],dtype=np.float32)
   snap=sim.pathfinder.snap_point(target)
   if not np.isfinite(snap).all() or np.linalg.norm((snap-target)[[0,2]])>.25:rejected['goal_not_near_navmesh']+=1;continue
   start=sim.pathfinder.get_random_navigable_point()
   if not np.isfinite(start).all() or abs(float(start[1]-snap[1]))>.3:rejected['different_floor']+=1;continue
   distance=dist(sim,start,target)
   if not args.min_distance<=distance<=args.max_distance:rejected['distance_or_unreachable']+=1;continue
   key=(int(goal['object_id']),*np.round(start,3))
   if key in seen or tuple(start.tolist()) in source_starts:rejected['duplicate_start']+=1;continue
   yaw=rng.uniform(-np.pi,np.pi);rotation=[0.,float(np.sin(yaw/2)),0.,float(np.cos(yaw/2))]
   state=sim.get_agent(0).get_state();state.position=start;state.rotation=quat_from_coeffs(np.array(rotation));state.sensor_states={};sim.get_agent(0).set_state(state)
   follower=sim.make_greedy_follower(goal_radius=.9,stop_key=0,forward_key=1,left_key=2,right_key=3)
   actions=[];collisions=0
   try:
    for step in range(args.max_actions):
     a=int(follower.next_action_along(target));actions.append(a)
     if a==0:break
     native=sim.step(a);collisions+=int(native['collided'])
   except habitat_sim.errors.GreedyFollowerError:rejected['follower_error']+=1;continue
   final_distance=dist(sim,sim.get_agent(0).get_state().position,target)
   if not actions or actions[-1]!=0 or final_distance>=1:rejected['oracle_did_not_reach']+=1;continue
   travel_time=(len(actions)-1)*.25
   onset_max=min(3,int(np.floor(travel_time-1.0)))
   if onset_max<1:rejected['too_short_to_hear_sound']+=1;continue
   onset=int(rng.integers(1,onset_max+1));duration=int(rng.integers(5,13));goal.update(onset=onset,offset=onset+duration,duration=duration)
   e={'episode_id':str(args.id_start+len(generated)),'scene_id':scene_id,'start_position':start.tolist(),'start_rotation':rotation,'info':{'geodesic_distance':distance,'num_action':len(actions)},'start_room':None,'shortest_paths':None,'goals':[goal],'distractor':None,'oracle_actions':actions}
   generated.append(e);seen.add(key)
   provenance.append({'episode_id':e['episode_id'],'official_template_episode_id':str(template['episode_id']),'object_id':goal['object_id'],'category':goal['object_category'],'scene_object_id':objects[int(goal['object_id'])].id,'sound_id':goal['sound_id'],'sound_sha256':sha(Path(args.sounds_root)/goal['sound_id']),'goal_position_preserved':goal['position']==template['goals'][0]['position'],'planner_final_distance_m':final_distance,'planner_collisions':collisions,'oracle_travel_time_s':travel_time,'at_least_one_second_of_sound_before_reference_stop':travel_time-onset>=1})
   print('accepted',e['episode_id'],goal['object_category'],round(distance,2),len(actions),flush=True)
  if len(generated)!=args.count:raise RuntimeError(f'Only {len(generated)}/{args.count} tasks after sampling budget; rejected={dict(rejected)}')
  split=out/args.split;content=split/'content'/Path(scene_id).with_suffix('.json.gz').name
  dump_gz(content,{'episodes':generated});dump_gz(split/(args.split+'.json.gz'),{'episodes':[],'content_scenes_path':'{data_path}/content/{scene}.json.gz'})
  manifest={'generator':'episode_tools.generate','format':'unmodified SAVNCE_Episode/SAVNCE_AudioGoal schema','is_author_official_benchmark':False,'association_source':'official object_id/category/sound_id/goal.position retained; new starts/rotations/timing','scene_id':scene_id,'count':len(generated),'seed':args.seed,'arguments':vars(args),'source_episode_file':str(source),'source_episode_sha256':sha(source),'source_repo_commit':subprocess.check_output(['git','-c','safe.directory='+str(Path.cwd()),'rev-parse','HEAD'],text=True).strip(),'navmesh_sha256':sha(scene_path.with_suffix('.navmesh')),'house_sha256':sha(scene_path.with_suffix('.house')),'verified_template_count':len(templates),'rejected':dict(rejected),'episodes':provenance,'reference_actions_note':'Greedy geodesic reference, not a proof of minimum action count. Privileged oracle actions are for validation only.','coordinates_note':'Serialized raw goal position is preserved; original loader applies y+0.1, renderer adds y+1.5. No new meshes or obstacles are generated.'}
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
  (out/'SHA256SUMS').write_text(''.join(sha(f)+'  '+str(f.relative_to(out))+'\n' for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
  print(json.dumps({'dataset_root':str(out),'count':len(generated),'content_file':str(content),'categories':dict(collections.Counter(e['goals'][0]['object_category'] for e in generated))},indent=2),flush=True)
 finally:sim.close()

def main():
 a=argparse.ArgumentParser();a.add_argument('--source',default=SOURCE);a.add_argument('--output',required=True);a.add_argument('--count',type=int,default=20);a.add_argument('--split',default='train');a.add_argument('--seed',type=int,default=20261105);a.add_argument('--id-start',type=int,default=1000000000);a.add_argument('--min-distance',type=float,default=2.);a.add_argument('--max-distance',type=float,default=6.);a.add_argument('--max-actions',type=int,default=160);a.add_argument('--scenes-root',default='/datasets/mp3d');a.add_argument('--sounds-root',default='data/sounds/semantic_splits');args=a.parse_args();generate(args)
if __name__=='__main__':main()
