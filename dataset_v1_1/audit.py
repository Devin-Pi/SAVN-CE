"""Offline privileged audit. Never supplies semantic IDs, future views or truth to an Agent."""
from pathlib import Path
import argparse,collections,copy,gzip,hashlib,itertools,json,os,runpy,sys
import numpy as np
import imageio.v2 as imageio
import habitat_sim,habitat,savnce
import savnce.tasks.nav
from habitat_sim.utils.common import quat_from_coeffs,quat_to_coeffs,quat_rotate_vector
from savnce.datasets.savnce_dataset import SAVNCE_Dataset
from .variants import SceneVariantSpec,ExistingSceneBackend

ROOT=Path('research_outputs/agent_dataset_v1_1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write_json(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def fov_relation(obj,sensor_state,hfov=90):
    corners=[np.array(obj.aabb.center)+np.array(sign)*np.array(obj.aabb.sizes)/2 for sign in itertools.product([-1,1],repeat=3)]
    cs=np.array([quat_rotate_vector(sensor_state.rotation.inverse(),v-sensor_state.position) for v in corners]);z=-cs[:,2];tan=np.tan(np.radians(hfov/2))
    outside=any(np.all(x) for x in [z<.01,cs[:,0]>z*tan,cs[:,0]<-z*tan,cs[:,1]>z*tan,cs[:,1]<-z*tan])
    return 'outside_camera_frustum_bbox' if outside else 'intersects_frustum_bbox_proxy'
def build_visual_sim(scene):
    c=habitat_sim.SimulatorConfiguration();c.scene_id=scene;c.enable_physics=False;c.load_semantic_mesh=True
    specs=[]
    for name,typ in [('audit_rgb',habitat_sim.SensorType.COLOR),('audit_depth',habitat_sim.SensorType.DEPTH),('audit_semantic',habitat_sim.SensorType.SEMANTIC)]:
        s=habitat_sim.CameraSensorSpec();s.uuid=name;s.sensor_type=typ;s.resolution=[128,128];s.position=[0,1.25,0];specs.append(s)
    a=habitat_sim.AgentConfiguration();a.height=1.5;a.radius=.1;a.sensor_specifications=specs
    a.action_space={1:habitat_sim.ActionSpec('move_forward',habitat_sim.ActuationSpec(amount=.25)),2:habitat_sim.ActionSpec('turn_left',habitat_sim.ActuationSpec(amount=15)),3:habitat_sim.ActionSpec('turn_right',habitat_sim.ActuationSpec(amount=15))}
    return habitat_sim.Simulator(habitat_sim.Configuration(c,[a]))
def sample_views(sim,episode,objects,folder):
    goal=episode['goals'][0];target=objects[int(goal['object_id'])];sid=int(target.semantic_id);cat=goal['object_category']
    plans=[('initial',[]),('left45',[2]*3),('right45',[3]*3)]
    prefix=[a for a in episode['oracle_actions'][:8] if a!=0]
    plans += [('move_prefix',prefix),('move_left45',prefix+[2]*3),('move_right45',prefix+[3]*3)]
    views=[]
    for name,actions in plans:
        state=sim.get_agent(0).get_state();state.position=np.array(episode['start_position']);state.rotation=quat_from_coeffs(np.array(episode['start_rotation']));state.sensor_states={};sim.get_agent(0).set_state(state)
        collisions=0
        for a in actions:collisions+=int(sim.step(a)['collided'])
        obs=sim.get_sensor_observations();state=sim.get_agent(0).get_state();sem=obs['audit_semantic'];values,counts=np.unique(sem,return_counts=True);hist=dict(zip(map(int,values),map(int,counts)))
        candidates=[{'object_id':oid,'semantic_id':int(o.semantic_id),'category':cat,'pixels':hist.get(int(o.semantic_id),0),'evidence':'same-class semantic instance visible at >=20 pixels; RGB appearance similarity unverified','emitter_evidence':'unverified_from_agent_observation'} for oid,o in objects.items() if o.category.name()==cat and hist.get(int(o.semantic_id),0)>=20]
        pixels=hist.get(sid,0);relation=fov_relation(target,state.sensor_states['audit_rgb'])
        item={'view_id':episode['episode_id']+':'+name,'name':name,'action_sequence_from_start':actions,'position':state.position.tolist(),'rotation_xyzw':quat_to_coeffs(state.rotation).tolist(),'legal_navmesh_pose':bool(sim.pathfinder.is_navigable(state.position)),'collisions_on_legal_action_sequence':collisions,'target_semantic_pixels':pixels,'target_visibility':'visible_in_semantic_render' if pixels>0 else 'not_visible_in_semantic_render','target_rgb_recognizability':'unverified','target_fov_relation':relation,'target_occlusion':'unknown' if pixels==0 and relation!='outside_camera_frustum_bbox' else ('not_fully_occluded_in_semantic_render' if pixels>0 else 'unverified_outside_fov'),'same_class_candidates':candidates}
        # A few privileged audit snapshots, not fixed-trajectory training sequences.
        folder.mkdir(parents=True,exist_ok=True);imageio.imwrite(folder/(name+'.png'),obs['audit_rgb'][...,:3]);np.savez_compressed(folder/(name+'_semantic.npz'),instance_id=sem)
        item['audit_rgb_file']=str(folder/(name+'.png'));item['audit_instance_file']=str(folder/(name+'_semantic.npz'));views.append(item)
    return views

def audit(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);root.chmod(0o700);private=root/'private';private.mkdir(exist_ok=True);private.chmod(0o700)
    if (root/'episode_audit.jsonl').exists():raise FileExistsError('Use a fresh output or archive previous audit')
    dataset=Path(args.dataset_root);manifest=json.loads((dataset/'manifest.json').read_text());raw=json.loads(gzip.open(next((dataset/'train/content').glob('*.json.gz')),'rt').read())['episodes'];source=json.loads(gzip.open(manifest['source_episode_file'],'rt').read())['episodes'];source_map={str(e['episode_id']):e for e in source};provenance={e['episode_id']:e for e in manifest['episodes']}
    scene_id=raw[0]['scene_id'];scene_path=str(Path('/datasets/mp3d')/scene_id);base_id=scene_id.split('/')[0];spec=SceneVariantSpec('base:'+base_id,base_id);ExistingSceneBackend().validate(spec)
    sim=build_visual_sim(scene_path);sim.seed(20261105);objects={int(o.id.split('_')[-1]):o for o in sim.semantic_scene.objects if o is not None}
    mapping={str(k):{'semantic_id':int(v.semantic_id),'category':v.category.name(),'scene_object_id':v.id} for k,v in objects.items()};write_json(private/'semantic_instance_mapping.json',mapping)
    stage=sim.get_stage_initialization_template();engine={'python':sys.version,'habitat_sim_version':habitat_sim.__version__,'habitat_sim_module':habitat_sim.__file__,'built_with_bullet':bool(habitat_sim.built_with_bullet),'physics_enabled_in_task':False,'rigid_object_count':sim.get_rigid_object_manager().get_num_objects(),'visual_asset':stage.render_asset_handle,'collision_asset':stage.collision_asset_handle,'semantic_asset':stage.semantic_asset_handle,'navmesh_asset':stage.navmesh_asset_handle,'recompute_navmesh_doc':habitat_sim.Simulator.recompute_navmesh.__doc__,'audio_bindings':[x for x in dir(habitat_sim.bindings.AudioSensor) if not x.startswith('_')],'semantic_pixels_meaning':'Runtime SemanticObject.semantic_id; joined semantic mesh; image recognizability is not inferred','source_build_commit_correspondence':'unverified; local C++ source has matching API but installed binary exact commit unknown'}
    write_json(root/'environment.json',engine);write_json(root/'scene_variant_base.json',spec.to_dict())
    audits=[]
    try:
        for e in raw:
            official=source_map[provenance[e['episode_id']]['official_template_episode_id']];g=e['goals'][0];o=objects.get(int(g['object_id']))
            association={'object_exists':o is not None,'scene_category_matches':o is not None and o.category.name()==g['object_category'],'official_fields_match':all(g[k]==official['goals'][0][k] for k in ['object_id','object_category','sound_id','position']),'sound_file_exists':(Path('data/sounds/semantic_splits')/g['sound_id']).is_file()}
            views=sample_views(sim,e,objects,private/'views'/e['episode_id']);initial=views[0];initial_ids={x['object_id'] for x in initial['same_class_candidates']}
            changed=[v for v in views[1:] if v['legal_navmesh_pose'] and np.linalg.norm(np.array(v['position'])-np.array(initial['position']))>.25 and {x['object_id'] for x in v['same_class_candidates']}!=initial_ids]
            opportunity={'visual_candidate_sets_change_after_legal_movement':bool(changed),'supporting_view_ids':[v['view_id'] for v in changed],'identifies_actual_sound_emitter_from_observations':'unverified','note':'Offline reachable view tests; not future observations supplied to an Agent. Class labels come from privileged semantics.'}
            if initial['target_semantic_pixels']>=20 and len(initial_ids)==1:group='G0'
            elif len(initial_ids)>=2:group='G1'
            else:group='unassigned'
            row={'episode_id':e['episode_id'],'dataset_version':'1.1','scenario_group':group,'scene_variant_id':spec.scene_variant_id,'association':association,'private_ground_truth':{'target_object_id':g['object_id'],'category':g['object_category'],'raw_goal_position':g['position'],'sound_id':g['sound_id']},'initial_view':initial,'legal_audit_views':views[1:],'verification_opportunity':opportunity,'visual_similarity':'unverified; same semantic category is not an appearance judgment','emitter_visual_evidence':'unverified; visual presence is not evidence of sound production','native_replay':None,'audio_activity':None}
            audits.append(row);print('visibility',e['episode_id'],group,'target_px',initial['target_semantic_pixels'],'candidates',len(initial_ids),flush=True)
    finally:sim.close()
    # Native Dataset and Env replay: do not write raw fixed-trajectory image/audio sequences.
    cfg=runpy.run_path('savnce_baselines/magnet/config/default.py')['get_task_config']('configs/savnce/magnet/mp3d/savnce_clean.yaml');cfg.defrost();cfg.DATASET.DATA_PATH=str((dataset/'train/train.json.gz').resolve());cfg.DATASET.SCENES_DIR='/datasets/mp3d';cfg.DATASET.CONTENT_SCENES=['*'];cfg.SIMULATOR.AUDIO.SOURCE_SOUND_DIR=str(Path('data/sounds/semantic_splits').resolve());cfg.SIMULATOR.AUDIO.THREADS=4;cfg.SIMULATOR.AGENT_0.SENSORS=['RGB_SENSOR','DEPTH_SENSOR'];cfg.SIMULATOR.RGB_SENSOR.WIDTH=cfg.SIMULATOR.RGB_SENSOR.HEIGHT=128;cfg.SIMULATOR.DEPTH_SENSOR.WIDTH=cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT=128;cfg.freeze()
    ds=SAVNCE_Dataset(cfg.DATASET);assert len(ds.episodes)==20
    clone=copy.deepcopy(ds);text=clone.to_json();roundtrip=SAVNCE_Dataset();roundtrip._config=cfg.DATASET.clone();roundtrip._config.defrost();roundtrip._config.CONTINUOUS=False;roundtrip._config.freeze();roundtrip.from_json(text);assert len(roundtrip.episodes)==20
    env=habitat.Env(config=cfg,dataset=ds);byid={a['episode_id']:a for a in audits};rawmap={e['episode_id']:e for e in raw}
    try:
        env.seed(20261105);env.episode_iterator=iter(ds.episodes)
        for e in ds.episodes:
            obs=env.reset();row=byid[str(e.episode_id)];actual_start_distance=float(env.get_metrics()['distance_to_goal']);g=rawmap[str(e.episode_id)]['goals'][0];windows=[];collisions=0
            for action in rawmap[str(e.episode_id)]['oracle_actions']:
                audio=env.sim.get_current_audiogoal_observation();assert audio.shape==(4000,2) and np.isfinite(audio).all();assert obs['rgb'].shape==(128,128,3) and np.isfinite(obs['depth']).all()
                t=env.sim._episode_step_count*.25;rms=float(np.sqrt(np.mean(audio**2)));windows.append({'step':env.sim._episode_step_count,'start_s':t,'end_s':t+.25,'rms':rms,'above_1e_5':rms>1e-5,'emission_window_overlap':t<g['offset'] and t+.25>g['onset']})
                obs=env.step(action)
                if action!=0:collisions+=int(env.sim._prev_sim_obs.get('collided',False))
            metrics=env.get_metrics();success=bool(metrics['success']);assert success and env.episode_over and max(w['rms'] for w in windows)>1e-7
            row['native_replay']={'control':'privileged_reference_not_agent','success':success,'start_navigable':bool(row['initial_view']['legal_navmesh_pose']),'initial_distance_m':actual_start_distance,'distance_vs_declared_error_m':abs(actual_start_distance-rawmap[str(e.episode_id)]['info']['geodesic_distance']),'final_distance_m':float(metrics['distance_to_goal']),'actions_including_stop':len(windows),'collisions':collisions,'rgb_depth_audio_finite':True}
            offset_seen=windows[-1]['end_s']>g['offset']+.25
            row['audio_activity']={'scheduled_onset_s':g['onset'],'scheduled_offset_s':g['offset'],'scheduled_duration_s':g['duration'],'actual_collection_window_s':[0,windows[-1]['end_s']],'nonzero_energy_windows':sum(w['above_1e_5'] for w in windows),'effective_audio_observed':any(w['above_1e_5'] for w in windows),'offset_behavior':'sampled_after_offset; any remaining energy may be reverb' if offset_seen else 'unverified_collection_ends_before_post_offset_window','audio_semantic_recognition':'unverified','audio_to_visible_instance_attribution':'unverified_from_agent_sensors','windows':windows}
            print('native_replay',e.episode_id,'success',success,'audio_windows',row['audio_activity']['nonzero_energy_windows'],flush=True)
            if 'acoustic_export' not in engine:
                try:
                    path=private/'base_acoustic_geometry.obj';sensor=env.sim._sim.get_agent(0)._sensors['audio_goal_sensor'];ok=sensor.writeSceneMeshOBJ(str(path));stats={'export_succeeded':bool(ok),'sha256':sha(path) if path.exists() else None}
                    if path.exists():
                        counts=collections.Counter()
                        with path.open(errors='replace') as f:
                            for line in f:counts[line.split(' ',1)[0]]+=1
                        stats['vertices']=counts['v'];stats['faces']=counts['f'];stats['file_bytes']=path.stat().st_size;path.unlink()
                    engine['acoustic_export']=stats
                except Exception as ex:engine['acoustic_export']={'status':'unverified','error':str(ex)}
                write_json(root/'environment.json',engine)
    finally:env.close()
    with (root/'episode_audit.jsonl').open('w') as f:
        for row in audits:f.write(json.dumps(row,allow_nan=False)+'\n')
    with (root/'scenario_metadata.jsonl').open('w') as f:
        for row in audits:
            initial=row['initial_view'];f.write(json.dumps({'episode_id':row['episode_id'],'scenario_group':row['scenario_group'],'scene_variant_id':row['scene_variant_id'],'target_visibility':{'initial':initial['target_visibility'],'pixels':initial['target_semantic_pixels'],'rgb_recognizability':'unverified','occlusion':initial['target_occlusion'],'fov_relation':initial['target_fov_relation']},'visible_same_class_candidates':initial['same_class_candidates'],'audio_activity_status':{k:v for k,v in row['audio_activity'].items() if k!='windows'},'visual_ambiguity':{'same_class_visible_count':len(initial['same_class_candidates']),'semantic_candidate_ambiguity':len(initial['same_class_candidates'])>=2,'appearance_similarity':'unverified','actual_emitter_evidence':'unverified'},'verification_opportunity':row['verification_opportunity'],'annotation_method':'offline aligned semantic-instance render + legal viewpoint actions + native audio/trajectory replay','annotation_status':'instrumented_provisional_requires_human_review','dataset_version':'1.1','access_class':'researcher_only_privileged_not_agent_input'},allow_nan=False)+'\n')
    counts=collections.Counter(a['scenario_group'] for a in audits);write_json(root/'audit_summary.json',{'episodes':len(audits),'groups':dict(counts),'all_native_successful':all(a['native_replay']['success'] for a in audits),'all_audio_effective':all(a['audio_activity']['effective_audio_observed'] for a in audits),'future_fixed_sequences_exported':False,'note':'Six selected offline audit views per task, not all-frame RGBD/audio trajectories. All audit artifacts are privileged.'});print('AUDIT_COMPLETE',dict(counts),flush=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('--dataset-root',default='data/datasets/savnce_generated/mp3d/v1');a.add_argument('--output',default=str(ROOT));args=a.parse_args();audit(args)
if __name__=='__main__':main()
