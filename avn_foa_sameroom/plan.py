"""Geometry qualification in complete, unchanged MP3D buildings."""
import argparse,collections,json,time
from pathlib import Path
import numpy as np
import habitat_sim as hs
from .common import write_json,sha,config_hash
from .geometry import Rooms,clear_segment,free_connected_point


def make_sim(mesh,seed,renderer=False):
    cfg=hs.SimulatorConfiguration();cfg.scene_id=str(mesh);cfg.create_renderer=renderer
    cfg.enable_physics=True;cfg.load_semantic_mesh=True
    agent=hs.agent.AgentConfiguration();agent.height=1.5;agent.radius=.1
    if renderer:
        spec=hs.CameraSensorSpec();spec.uuid='audit_rgb';spec.sensor_type=hs.SensorType.COLOR
        spec.resolution=[192,192];spec.position=[0,1.5,0];agent.sensor_specifications=[spec]
    sim=hs.Simulator(hs.Configuration(cfg,[agent]));sim.seed(seed);sim.pathfinder.seed(seed)
    if not sim.pathfinder.is_loaded:raise RuntimeError('Missing original NavMesh')
    stage=sim.get_stage_initialization_template()
    assert stage.render_asset_handle==stage.collision_asset_handle, 'Separate collision proxy requires audit'
    return sim


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--scenes',type=Path,default=Path('/datasets/mp3d'))
    p.add_argument('--seed',type=int,default=20261107);a=p.parse_args()
    rows=[json.loads(x) for x in (a.root/'official_goals.jsonl').read_text().splitlines()]
    groups=collections.defaultdict(list)
    for row in rows:groups[row['scene_id']].append(row)
    rng=np.random.default_rng(a.seed);accepted=[];reject=collections.Counter();per_scene={}
    out=a.root/'geometry_candidates.jsonl'
    if out.exists():raise FileExistsError(out)
    with out.open('w') as stream,(a.root/'geometry_rejections.jsonl').open('w') as reject_stream:
        for scene,goals in sorted(groups.items()):
            mesh=a.scenes/scene/(scene+'.glb');rooms=Rooms(mesh.with_suffix('.house'))
            sim=make_sim(mesh,a.seed)
            objects={int(o.id.split('_')[-1]):o for o in sim.semantic_scene.objects if o is not None}
            scene_reject=collections.Counter();count=0
            geometry_hashes={'glb':sha(mesh),'house':sha(mesh.with_suffix('.house')),'navmesh':sha(mesh.with_suffix('.navmesh'))}
            try:
                rng.shuffle(goals)
                for goal in goals:
                    oid=goal['source_object_id'];obj=objects.get(oid);room_obj=rooms.objects.get(oid)
                    source=np.asarray(goal['source_position'])
                    reason=None
                    if obj is None or room_obj is None:reason='missing_object_instance'
                    elif obj.category.name()!=goal['official_category']:reason='semantic_category_mismatch'
                    source_room=rooms.locate(source)
                    if not reason and (source_room is None or source_room!=room_obj['room_id']):reason='source_room_unknown_or_mismatch'
                    if not reason and free_connected_point(sim,source) is None:reason='source_free_space_unverified'
                    if reason:
                        reject[reason]+=1;scene_reject[reason]+=1
                        reject_stream.write(json.dumps(dict(scene_id=scene,source_object_id=oid,source_episode_id=goal['source_episode_id'],reason=reason))+'\n');continue
                    made=0
                    for lo,hi in ((.5,1),(1,2),(2,4),(4,8)):
                        for attempt in range(24):
                            radius=float(rng.uniform(lo,hi));ang=float(rng.uniform(-np.pi,np.pi))
                            desired=source+np.array([radius*np.cos(ang),-1.5,radius*np.sin(ang)])
                            base=np.asarray(sim.pathfinder.snap_point(desired),dtype=np.float64)
                            if not np.isfinite(base).all() or np.linalg.norm((base-desired)[[0,2]])>.3:scene_reject['receiver_snap_failed']+=1;continue
                            receiver=base+[0,1.5,0];distance=float(np.linalg.norm(source-receiver))
                            if not lo<=distance<hi:scene_reject['distance_bin']+=1;continue
                            if not sim.pathfinder.is_navigable(base):scene_reject['receiver_not_navigable']+=1;continue
                            if rooms.locate(receiver)!=source_room or rooms.locate(base+[0,.15,0])!=source_room:scene_reject['receiver_not_same_room']+=1;continue
                            # Verify the entire direct segment also stays in this annotated region.
                            n=max(3,int(distance/.15)+1)
                            if any(rooms.locate(v)!=source_room for v in np.linspace(source,receiver,n)):scene_reject['segment_leaves_room_or_boundary_unknown']+=1;continue
                            if not clear_segment(sim,source,receiver):scene_reject['geometric_occlusion']+=1;continue
                            if free_connected_point(sim,receiver) is None:scene_reject['receiver_free_space_unverified']+=1;continue
                            row=dict(goal,room_id=source_room,base_position=base.tolist(),receiver_position=receiver.tolist(),
                                distance_m=distance,distance_bin=[lo,hi],same_room=True,geometric_los=True,
                                source_in_valid_space=True,receiver_in_valid_space=True,
                                region_method='house floor-surface polygons + unique height-aware containment of source/receiver/base and direct segment; object region agrees',
                                valid_space_method='clear bidirectional segment to NavMesh-connected air + six-axis 4cm surface clearance; conservative subset',
                                collision_method='Bullet bidirectional static triangle-mesh segment rays, same GLB asset as render/acoustics; no mesh simplification',
                                geometry_sha256=geometry_hashes)
                            # Positions are quantized only for duplicate detection, never for rendering.
                            row['geometry_id']=config_hash({k:row[k] for k in ('scene_id','source_position','receiver_position')})[:24]
                            stream.write(json.dumps(row)+'\n');stream.flush();accepted.append(row);count+=1;made+=1;break
                    if made==0:
                        reject['no_legal_receiver']+=1
                        reject_stream.write(json.dumps(dict(scene_id=scene,source_object_id=oid,source_episode_id=goal['source_episode_id'],reason='no_legal_receiver'))+'\n')
            finally:sim.close()
            per_scene[scene]=dict(accepted_configurations=count,rejections=dict(scene_reject))
            print(scene,'accepted',count,'classes',sorted({g['official_category'] for g in accepted if g['scene_id']==scene}),flush=True)
    write_json(a.root/'geometry_audit.json',{'accepted_configurations':len(accepted),'goal_rejections':dict(reject),'per_scene':per_scene,'seed':a.seed})
    print('TOTAL',len(accepted),flush=True)
if __name__=='__main__':main()
