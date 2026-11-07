"""Render native static FOA RIRs, then synthesize traceable single-source records."""
import argparse,collections,csv,json,math,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
import habitat_sim as hs
from habitat_sim.utils.common import quat_from_angle_axis
from .common import RATE,DURATION,VERSION,sha,write_json,config_hash,acoustic_config,native_to_local,local_unit
from .plan import make_sim
from .acoustic_mesh import AcousticMesh
from .synthesis import read_source,synthesize


def audio_spec():
    s=hs.AudioSensorSpec();s.uuid='dataset_foa';s.position=[0,1.5,0];s.enableMaterials=False
    s.channelLayout.type=hs.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics;s.channelLayout.channelCount=4
    s.acousticsConfig.sampleRate=RATE;s.acousticsConfig.threadCount=4
    s.acousticsConfig.indirectRayCount=500;s.acousticsConfig.temporalCoherence=False
    s.acousticsConfig.transmission=True
    return s


def select(rows,number,rng):
    """Approximately class-balanced, cycling different buildings and distance bins."""
    pools=collections.defaultdict(list)
    for r in rows:pools[r['class_id']].append(r)
    for v in pools.values():rng.shuffle(v)
    picked=[];seen=set();class_use=collections.Counter();room_use=collections.Counter();scene_use=collections.Counter();bin_use=collections.Counter()
    while len(picked)<number:
        choices=[c for c,v in pools.items() if v]
        if not choices:raise RuntimeError(f'Insufficient unique geometry {len(picked)}/{number}')
        c=min(choices,key=lambda c:(class_use[c],float(rng.random())))
        # Prefer a new room/instance over oversampling a popular building.
        v=pools[c]
        pos=min(range(len(v)),key=lambda i:(scene_use[v[i]['scene_id']],room_use[(v[i]['scene_id'],v[i]['room_id'])],bin_use[tuple(v[i]['distance_bin'])],float(rng.random())))
        r=v.pop(pos)
        if r['geometry_id'] in seen:continue
        picked.append(r);seen.add(r['geometry_id']);class_use[c]+=1;scene_use[r['scene_id']]+=1;room_use[r['scene_id'],r['room_id']]+=1;bin_use[tuple(r['distance_bin'])]+=1
    return picked


def save_record(root,row,ir,native_rir_reference,spec_hash,index,variant,nclasses,dry,rng,negative):
    row=dict(row);row.pop('original_recordings',None);row['original_wav_files']=1;row['original_recording_count']='unknown'
    yaw=float(rng.uniform(-np.pi,np.pi))
    local_ir=native_to_local(ir,yaw)
    unit=local_unit(row['source_position'],np.asarray(row['receiver_position']),yaw)
    for attempt in range(10):
        wave,labels,details=synthesize(dry,local_ir,unit,row['class_id'],nclasses,rng,negative)
        if negative or details['supervised_active_frames']>=3:break
    else:raise RuntimeError('No usable activity after crop retries')
    sample_id=f"{row['split']}_{index:06d}"
    wavrel=f"audio/{row['split']}/{sample_id}.wav";labrel=f"labels/{row['split']}/{sample_id}.npz";csvrel=labrel.replace('.npz','.csv')
    sf.write(root/wavrel,wave,RATE,subtype='FLOAT');np.savez_compressed(root/labrel,**labels)
    with (root/csvrel).open('w') as f:
        for frame in np.flatnonzero(labels['activity']):
            f.write(f"{frame},{row['class_id']},0,{unit[0]:.9f},{unit[1]:.9f},{unit[2]:.9f}\n")
    rotation=[0.,math.sin(yaw/2),0.,math.cos(yaw/2)]
    # Orientation of local audio XYZ expressed in Habitat world coordinates (columns).
    c,s=math.cos(yaw),math.sin(yaw)
    matrix=[[-s,-c,0],[0,0,1],[-c,s,0]]
    record=dict(row,sample_id=sample_id,version=VERSION,audio=wavrel,labels=labrel,dcase_cartesian_csv=csvrel,
        audio_sha256=sha(root/wavrel),labels_sha256=sha(root/labrel),
        rir_reference=native_rir_reference,acoustic_config_hash=spec_hash,
        receiver_yaw_rad=yaw,receiver_rotation_xyzw=rotation,R_world_from_audio=matrix,
        receiver_relative_to_base=[0,1.5,0],camera_relative_to_base=[0,1.5,0],
        dataset_camera='audit only; no RGB-D training sequence',audio_direction_xyz=unit.tolist(),
        sample_rate_hz=RATE,render_sample_rate_hz=RATE,channels=['W','Y','Z','X'],normalization='SN3D',
        wave_subtype='FLOAT',duration_s=DURATION,resampling='dry source resample_poly to 16k if needed; RIR is native16k',
        resampling_bandwidth_hz=8000,random_seed=int(row['generation_seed']),rng_state_after_sample=rng.bit_generator.state,
        waveform_family_shared_between_internal_splits=True,variant_index=variant,**details)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenes',type=Path,default=Path('/datasets/mp3d'));p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--smoke',action='store_true');p.add_argument('--acceptance',type=Path)
    a=p.parse_args()
    if not a.smoke:
        if a.acceptance is None:raise ValueError('Batch requires passed pilot acceptance JSON')
        accept=json.loads(a.acceptance.read_text());assert accept['status']=='passed' and accept['records']>=30 and accept['foa_calibration']=='passed' and accept['dsf_forward']=='passed' and accept['native_association']=='passed'
        if not (a.assets/'geometry_audit.json').exists():raise ValueError('Full geometry audit must finish before batch selection')
    rows=[json.loads(l) for l in (a.assets/'geometry_candidates.jsonl').read_text().splitlines()]
    cmap=json.loads((a.assets/'class_map.json').read_text());nclasses=len(cmap['official_category_to_class_id'])
    split_manifest=json.loads((a.assets/'split_manifest.json').read_text())
    calibration=json.loads((a.assets.parent/'foa_calibration.json').read_text());assert calibration['status']=='passed'
    rng=np.random.default_rng(a.seed+(1 if a.smoke else 1000))
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ('class_map.json','split_manifest.json','class_coverage.csv','asset_audit.json','source_assets.json','geometry_audit.json'):
        if (a.assets/name).exists():shutil.copyfile(a.assets/name,a.output/name)
    write_json(a.output/'foa_calibration.json',calibration)
    config=dict(version=VERSION,seed=a.seed,sampling_seed_offset=1 if a.smoke else 1000,sample_rate_hz=RATE,duration_s=DURATION,label_rate_hz=10,foa='local ACN/SN3D W/Y/Z/X',
        acoustic=acoustic_config(audio_spec()),enableMaterials=False,material_model='default engine material; semantic material database disabled',
        temporal_state='AudioSensor.reset() destroys RLR context before each independent RIR; temporalCoherence=False',
        native_engine_seed='No exposed RLR seed API found; Habitat/NavMesh/Python sampling seeded; actual RIR and wave hashes retained',
        source_count_max=1,source_motion=False,receiver_motion=False,background=None,
        repeated_rir_variants=3,planned_records={'train':24,'validation':9,'test':9} if a.smoke else split_manifest['planned_records'],
        negative_fraction=.1,split_protocol=split_manifest['protocol'],stage_assets_unchanged=True)
    import yaml
    (a.output/'generation_config.yaml').write_text(yaml.safe_dump(config,sort_keys=True))
    write_json(a.output/'environment.json',dict(python=sys.version,habitat_sim=hs.__version__,habitat_sim_binary=str(hs.__file__),
        built_with_bullet=bool(hs.built_with_bullet),git_commit=subprocess.check_output(['git','-c','safe.directory='+str(Path.cwd()),'rev-parse','HEAD'],text=True).strip(),
        generator_source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    (a.output/'rir').mkdir();plans=[];counter=collections.Counter()
    for split,total in config['planned_records'].items():
        (a.output/'audio'/split).mkdir(parents=True);(a.output/'labels'/split).mkdir(parents=True)
        negative=int(round(total*.1));positive=total-negative
        ngeom=int(math.ceil(positive/3))
        pool=[r for r in rows if r['split']==split]
        picked=select(pool,ngeom,rng)
        for geom_idx,row in enumerate(picked):
            npositive=min(3,positive-geom_idx*3)
            plans.append((row,npositive,int(geom_idx<negative)))
    # Group by building: one visual/collision scene, fresh acoustic context per RIR.
    plans.sort(key=lambda x:(x[0]['scene_id'],x[0]['geometry_id']))
    used_geometry={p[0]['geometry_id'] for p in plans}
    fallback=collections.defaultdict(list)
    for r in rows:
        if r['geometry_id'] not in used_geometry:fallback[(r['split'],r['class_id'])].append(r)
    render_rejections=[]
    spec=audio_spec();ac_hash=config_hash({'acoustic':acoustic_config(spec),'enableMaterials':False})
    source_cache={};sim=None;scene=None;acmesh=None;rir_rows=[];records=[];started=time.monotonic()
    (a.output/'acoustic_geometry').mkdir()
    meta=a.output/'metadata.jsonl';rirfile=a.output/'rir_manifest.jsonl'
    try:
        with meta.open('w') as output,rirfile.open('w') as rirout:
            for row,npositive,nnegative in plans:
                row=dict(row,generation_seed=a.seed)
                if scene!=row['scene_id']:
                    if sim is not None:sim.close()
                    scene=row['scene_id'];acmesh=None;sim=make_sim(a.scenes/scene/(scene+'.glb'),a.seed)
                    sim.add_sensor(spec)
                sensor=sim.get_agent(0)._sensors[spec.uuid]
                sensor.reset() # new RLR context, not merely a new transform
                state=hs.AgentState();state.position=np.asarray(row['base_position']);state.rotation=quat_from_angle_axis(0,np.array([0,1,0]))
                sim.get_agent(0).set_state(state)
                receiver=np.asarray(sensor.node.absolute_translation)
                np.testing.assert_allclose(receiver,row['receiver_position'],atol=1e-5)
                sensor.setAudioSourceTransform(np.asarray(row['source_position']))
                ir=np.asarray(sim.get_sensor_observations()[spec.uuid],dtype=np.float32).T.copy()
                if acmesh is None:acmesh=AcousticMesh(sensor,a.output/'acoustic_geometry'/(scene+'.obj'))
                reason=None
                if not acmesh.clear(row['source_position'],row['receiver_position']):reason='acoustic_mesh_los_failed'
                if ir.ndim!=2 or ir.shape[1]!=4 or not np.isfinite(ir).all() or np.max(np.abs(ir))<1e-10:reason='invalid_native_foa_rir'
                if reason:
                    render_rejections.append(dict(geometry_id=row['geometry_id'],reason=reason,scene_id=scene))
                    write_json(a.output/'render_rejections.json',render_rejections)
                    candidates=fallback[row['split'],row['class_id']]
                    if not candidates or len(render_rejections)>20:raise RuntimeError('Cannot replace invalid render; stop for audit')
                    replacement=candidates.pop();used_geometry.add(replacement['geometry_id'])
                    plans.append((replacement,npositive,nnegative));continue
                rirrel=f"rir/{row['geometry_id']}.npz"
                np.savez_compressed(a.output/rirrel,native=ir,ambix_world=native_to_local(ir,0))
                rr=dict(geometry_id=row['geometry_id'],scene_id=scene,room_id=row['room_id'],split=row['split'],
                    path=rirrel,sha256=sha(a.output/rirrel),source_position=row['source_position'],receiver_position=row['receiver_position'],
                    native_channels=['W','+HabitatY','+HabitatZ','+HabitatX'],native_normalization='N3D',acoustic_mesh_export_sha256=acmesh.hash,acoustic_mesh_triangles=acmesh.triangles,sourceIsVisible_diagnostic_only=bool(sensor.sourceIsVisible()),
                    sample_rate_hz=RATE,samples=len(ir),acoustic_config_hash=ac_hash,source_visible_in_acoustic_geometry=True,
                    ray_efficiency=float(sensor.getRayEfficiency()),history_reset=True,geometry_sha256=row['geometry_sha256'])
                rirout.write(json.dumps(rr)+'\n');rirout.flush();rir_rows.append(rr)
                if row['path'] not in source_cache:source_cache[row['path']]=read_source(row['path'])
                for variant in range(npositive+nnegative):
                    negative=variant>=npositive
                    counter[row['split']]+=1
                    record=save_record(a.output,row,ir,rirrel,ac_hash,counter[row['split']],variant,nclasses,source_cache[row['path']],rng,negative)
                    output.write(json.dumps(record)+'\n');output.flush();records.append(record)
                print(f"{len(records)} records, {len(rir_rows)} RIRs; {scene}; elapsed {time.monotonic()-started:.1f}s",flush=True)
    finally:
        if sim is not None:sim.close()
    write_json(a.output/'rir_manifest.json',rir_rows)
    write_json(a.output/'render_rejections.json',render_rejections)
    write_json(a.output/'generation_summary.json',dict(records=len(records),hours=sum(r['duration_s'] for r in records)/3600,
        split_counts=dict(counter),unique_native_rirs=len(rir_rows),elapsed_s=time.monotonic()-started))
if __name__=='__main__':main()
