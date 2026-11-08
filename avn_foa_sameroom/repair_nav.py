"""Replace isolated NavMesh-island receivers; quarantine all rejected draft records."""
import argparse,collections,gzip,json,shutil,time
from pathlib import Path
import habitat_sim as hs
import numpy as np
from .common import sha,write_json,config_hash,acoustic_config
from .generate import audio_spec,save_record
from .plan import make_sim
from .acoustic_mesh import AcousticMesh
from .synthesis import read_source
from .geometry import Rooms,clear_segment,free_connected_point


def resample_receiver(templates,used,starts,navs,rng,seed):
    """Scarce classes may have no unused inventory row: sample a new legal receiver."""
    order=rng.permutation(len(templates))
    for index in order:
        t=templates[int(index)];scene=t['scene_id'];nav=navs.get(scene)
        if scene not in starts:
            doc=json.loads(gzip.open(t['source_episode_file'],'rt').read());starts[scene]={str(e['episode_id']):e['start_position'] for e in doc['episodes']}
            nav=hs.PathFinder();assert nav.load_nav_mesh('/datasets/mp3d/'+scene+'/'+scene+'.navmesh');navs[scene]=nav
        origin=np.array(t['base_position']);mesh=Path('/datasets/mp3d')/scene/(scene+'.glb')
        sim=make_sim(mesh,seed);rooms=Rooms(mesh.with_suffix('.house'))
        try:
            for attempt in range(80):
                desired=origin+np.array([rng.uniform(-.8,.8),0,rng.uniform(-.8,.8)])
                base=np.asarray(nav.snap_point(desired),dtype=np.float64)
                if not np.isfinite(base).all() or np.linalg.norm(base-origin)<.06 or np.linalg.norm((base-desired)[[0,2]])>.2:continue
                rec=base+[0,1.5,0];source=np.array(t['source_position']);distance=float(np.linalg.norm(source-rec))
                if not .5<=distance<8 or rooms.locate(rec)!=t['room_id'] or rooms.locate(base+[0,.15,0])!=t['room_id']:continue
                path=hs.ShortestPath();path.requested_start=np.array(starts[scene][t['source_episode_id']],dtype=np.float32);path.requested_end=base.astype(np.float32)
                if not nav.find_path(path):continue
                if not clear_segment(sim,source,rec) or free_connected_point(sim,rec) is None:continue
                if any(rooms.locate(v)!=t['room_id'] for v in np.linspace(source,rec,max(3,int(distance/.15)+1))):continue
                row=dict(t,base_position=base.tolist(),receiver_position=rec.tolist(),distance_m=distance,
                    distance_bin=next([lo,hi] for lo,hi in ((.5,1),(1,2),(2,4),(4,8)) if lo<=distance<hi),
                    extra_receiver_resampled=True,extra_receiver_method='seeded jitter within0.8m of qualified receiver; all same-room/free-space/LOS/global-navigation checks repeated')
                row['geometry_id']=config_hash({k:row[k] for k in ('scene_id','source_position','receiver_position')})[:24]
                if row['geometry_id'] not in used:return row
        finally:sim.close()
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--assets',type=Path,required=True);p.add_argument('--quarantine',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20268107);a=p.parse_args()
    check=json.loads((a.root/'navmesh_reachability_check.json').read_text());bad={r['geometry_id'] for r in check['failed']}
    if not bad:write_json(a.root/'navmesh_repair.json',dict(status='not_needed',replaced_configurations=0));return
    a.quarantine.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(l) for l in (a.root/'metadata.jsonl').read_text().splitlines()]
    rirrows=[json.loads(l) for l in (a.root/'rir_manifest.jsonl').read_text().splitlines()]
    groups=collections.defaultdict(list)
    for row in rows:
        if row['geometry_id'] in bad:groups[row['geometry_id']].append(row)
    used={r['geometry_id'] for r in rows};pool=collections.defaultdict(list);all_pool=collections.defaultdict(list)
    for l in (a.assets/'geometry_candidates.jsonl').open():
        r=json.loads(l);all_pool[r['split'],r['class_id']].append(r)
        if r['geometry_id'] not in used:pool[r['split'],r['class_id']].append(r)
    rng=np.random.default_rng(a.seed)
    for p in pool.values():rng.shuffle(p)
    nclasses=len(json.loads((a.root/'class_map.json').read_text())['official_category_to_class_id'])
    shutil.copyfile(a.root/'metadata.jsonl',a.quarantine/'metadata_before_repair.jsonl')
    shutil.copyfile(a.root/'rir_manifest.jsonl',a.quarantine/'rir_manifest_before_repair.jsonl')
    starts={};navs={};meshes={};replacements={};newrirs=[];notes=[];spec=audio_spec();cfg_hash=config_hash({'acoustic':acoustic_config(spec),'enableMaterials':False})
    for oldid,oldrows in sorted(groups.items()):
        original=oldrows[0];key=original['split'],original['class_id'];candidate=None
        while pool[key]:
            c=pool[key].pop();scene=c['scene_id']
            if scene not in starts:
                doc=json.loads(gzip.open(c['source_episode_file'],'rt').read());starts[scene]={str(e['episode_id']):e['start_position'] for e in doc['episodes']}
                nav=hs.PathFinder();assert nav.load_nav_mesh('/datasets/mp3d/'+scene+'/'+scene+'.navmesh');navs[scene]=nav
            path=hs.ShortestPath();path.requested_start=np.array(starts[scene][c['source_episode_id']],dtype=np.float32);path.requested_end=np.array(c['base_position'],dtype=np.float32)
            if navs[scene].find_path(path):
                meshpath=a.root/'acoustic_geometry'/(scene+'.obj')
                if not meshpath.exists():continue
                if scene not in meshes:meshes.clear();meshes[scene]=AcousticMesh(None,meshpath)
                if not meshes[scene].clear(c['source_position'],c['receiver_position']):continue
                candidate=c;break
        if candidate is None:candidate=resample_receiver(all_pool[key],used,starts,navs,rng,a.seed)
        if candidate is None:raise RuntimeError('No reachable replacement after extra sampling for '+str(key))
        used.add(candidate['geometry_id'])
        row=dict(candidate,generation_seed=a.seed,nav_reachable_from_official_start=True)
        scene=row['scene_id'];sim=make_sim(Path('/datasets/mp3d')/scene/(scene+'.glb'),a.seed)
        try:
            sim.add_sensor(spec);sensor=sim.get_agent(0)._sensors[spec.uuid];sensor.reset()
            state=hs.AgentState();state.position=np.array(row['base_position']);sim.get_agent(0).set_state(state)
            sensor.setAudioSourceTransform(np.array(row['source_position']))
            ir=np.asarray(sim.get_sensor_observations()[spec.uuid],dtype=np.float32).T.copy()
            meshpath=a.quarantine/(scene+'_replacement_acoustic.obj');am=AcousticMesh(sensor,meshpath)
            assert am.clear(row['source_position'],row['receiver_position']) and np.isfinite(ir).all() and np.max(np.abs(ir))>1e-10
            rirrel='rir/'+row['geometry_id']+'.npz'
            from .common import native_to_local
            assert not (a.root/rirrel).exists()
            np.savez_compressed(a.root/rirrel,native=ir,ambix_world=native_to_local(ir,0))
            rr=dict(geometry_id=row['geometry_id'],scene_id=scene,room_id=row['room_id'],split=row['split'],path=rirrel,sha256=sha(a.root/rirrel),
                source_position=row['source_position'],receiver_position=row['receiver_position'],native_channels=['W','+HabitatY','+HabitatZ','+HabitatX'],
                native_normalization='N3D',acoustic_mesh_export_sha256=am.hash,acoustic_mesh_triangles=am.triangles,sourceIsVisible_diagnostic_only=bool(sensor.sourceIsVisible()),
                sample_rate_hz=16000,samples=len(ir),acoustic_config_hash=cfg_hash,source_visible_in_acoustic_geometry=True,
                ray_efficiency=float(sensor.getRayEfficiency()),history_reset=True,geometry_sha256=row['geometry_sha256'])
            newrirs.append(rr)
            dry=read_source(row['path'])
            for old in oldrows:
                for k in ('audio','labels','dcase_cartesian_csv'):
                    path=a.root/old[k];dest=a.quarantine/old[k];dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(path,dest)
                idx=int(old['sample_id'].split('_')[-1])
                new=save_record(a.root,row,ir,rirrel,cfg_hash,idx,old['variant_index'],nclasses,dry,rng,old['is_no_event'])
                new.update(replaced_geometry_id=oldid,replacement_reason='receiver was on NavMesh island unreachable from cited official start',repair_seed=a.seed)
                assert new['sample_id']==old['sample_id'];replacements[old['sample_id']]=new
        finally:sim.close()
        oldrir=original['rir_reference'];dest=a.quarantine/oldrir;dest.parent.mkdir(exist_ok=True);shutil.move(a.root/oldrir,dest)
        notes.append(dict(old_geometry_id=oldid,new_geometry_id=row['geometry_id'],records=len(oldrows),class_id=original['class_id'],split=original['split']))
        print('replaced',oldid,'with',row['geometry_id'],len(oldrows),flush=True)
    fixed=[replacements.get(row['sample_id'],row) for row in rows]
    rirs=[r for r in rirrows if r['geometry_id'] not in bad]+newrirs
    temp=a.root/'metadata.repaired.jsonl';temp.write_text(''.join(json.dumps(r)+'\n' for r in fixed));temp.replace(a.root/'metadata.jsonl')
    (a.root/'rir_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rirs));write_json(a.root/'rir_manifest.json',rirs)
    write_json(a.root/'navmesh_repair.json',dict(status='completed',replaced_configurations=len(bad),replaced_records=len(replacements),
        repair_seed=a.seed,quarantine=str(a.quarantine),changes=notes,original_scene_or_episode_modified=False,repair_code_sha256=sha(__file__)))
if __name__=='__main__':main()
