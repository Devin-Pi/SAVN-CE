"""Generate a new high-duty corpus using certified, byte-identical static FOA RIRs."""
import argparse
import collections
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf

from .common import RATE,DURATION,sha,write_json,native_to_local,local_unit
from .dense_synthesis import VERSION,MIN_ACTIVE_FRAMES,synthesize_dense,active_regions
from .synthesis import read_source

_CTX={}


def init_worker(source,output,nclasses,seed):
    os.environ['OMP_NUM_THREADS']='1'
    _CTX.update(source=Path(source),output=Path(output),nclasses=nclasses,seed=seed,dry={},rir={})


def generate_one(task):
    row,split,index,negative=task
    split_number=('train','validation','test').index(split)
    attempt_errors=[]
    if row['path'] not in _CTX['dry']:
        _CTX['dry'][row['path']]=read_source(row['path'])
    dry=_CTX['dry'][row['path']]
    ref=row['rir_reference']
    if ref not in _CTX['rir']:
        with np.load(_CTX['source']/ref,allow_pickle=False) as z:_CTX['rir'][ref]=z['native'].copy()
    native=_CTX['rir'][ref]
    for attempt in range(30):
        rng=np.random.default_rng(np.random.SeedSequence([_CTX['seed'],split_number,index,attempt]))
        yaw=float(rng.uniform(-np.pi,np.pi))
        ir=native_to_local(native,yaw)
        unit=local_unit(row['source_position'],np.asarray(row['receiver_position']),yaw)
        try:
            wave,labels,details=synthesize_dense(dry,ir,unit,row['class_id'],_CTX['nclasses'],rng,negative)
            break
        except ValueError as exc:attempt_errors.append(str(exc))
    else:raise RuntimeError(f"Cannot meet activity contract for {row['official_category']}: {attempt_errors[-1]}")
    sid=f'dense15h_{split}_{index:06d}'
    wav=f'audio/{split}/{sid}.wav';lab=f'labels/{split}/{sid}.npz';csv=lab.replace('.npz','.csv')
    root=_CTX['output']
    sf.write(root/wav,wave,RATE,subtype='FLOAT');np.savez_compressed(root/lab,**labels)
    with (root/csv).open('w') as stream:
        for frame in np.flatnonzero(labels['activity']):
            stream.write(f"{frame},{row['class_id']},0,{unit[0]:.9f},{unit[1]:.9f},{unit[2]:.9f}\n")
    # Remove every temporal/acoustic-derived annotation from the parent record.
    provenance_keys=('scene_id','source_episode_id','source_episode_file','source_episode_file_sha256',
        'source_object_id','raw_official_goal_position','source_position','height_offsets','official_category','class_id',
        'source_sound_id','path','source_audio_hash','source_family_id','original_split','original_sample_rate_hz',
        'family_resolution','room_id','base_position','receiver_position','distance_m','distance_bin','same_room',
        'geometric_los','source_in_valid_space','receiver_in_valid_space','region_method','valid_space_method',
        'collision_method','geometry_sha256','geometry_id','acoustic_config_hash')
    record={k:row[k] for k in provenance_keys}
    c,s=math.cos(yaw),math.sin(yaw)
    record.update(sample_id=sid,version=VERSION,split=split,audio=wav,labels=lab,dcase_cartesian_csv=csv,
        audio_sha256=sha(root/wav),labels_sha256=sha(root/lab),rir_reference=ref,
        receiver_yaw_rad=yaw,receiver_rotation_xyzw=[0.,math.sin(yaw/2),0.,math.cos(yaw/2)],
        R_world_from_audio=[[-s,-c,0],[0,0,1],[-c,s,0]],audio_direction_xyz=unit.tolist(),
        receiver_relative_to_base=[0,1.5,0],camera_relative_to_base=[0,1.5,0],
        sample_rate_hz=RATE,render_sample_rate_hz=RATE,channels=['W','Y','Z','X'],normalization='SN3D',wave_subtype='FLOAT',
        duration_s=DURATION,resampling='dry WAV resample_poly to16k; certified RIR already native16k',
        original_wav_files=1,original_recording_count='unknown',waveform_family_shared_between_internal_splits=True,
        source_dataset='AVN-FOA-SameRoom-v0.1',source_sample_id_for_geometry=row['sample_id'],
        rir_reused_without_rerender=True,generation_seed=_CTX['seed'],
        per_record_seed_components=[_CTX['seed'],split_number,index,attempt],generation_attempts=attempt+1,
        rejected_temporal_attempt_reasons=attempt_errors,**details)
    return record


def make_tasks(base,counts,seed,pilot):
    pools=collections.defaultdict(list)
    for row in base.values():pools[row['split'],row['class_id']].append(row)
    nclasses=len({c for _,c in pools})
    rng=np.random.default_rng(seed)
    for key in sorted(pools):
        pools[key].sort(key=lambda row:row['geometry_id']);rng.shuffle(pools[key])
    tasks=[];counters=collections.Counter()
    if pilot:
        for c in range(nclasses):
            for offset in (0,1):
                split=('train','validation','test')[(c+offset)%3]
                counters[split]+=1;tasks.append((pools[split,c][0],split,counters[split],False))
        for k,split in enumerate(('train','validation','test')):
            counters[split]+=1;tasks.append((pools[split,k][0],split,counters[split],True))
    else:
        for split,total in counts.items():
            negative=total//10;positive=total-negative
            for isnegative,number in ((False,positive),(True,negative)):
                classes=np.arange(number)%nclasses;rng.shuffle(classes)
                used=collections.Counter()
                for c in classes:
                    c=int(c);pool=pools[split,c]
                    row=pool[used[c]%len(pool)];used[c]+=1
                    counters[split]+=1;tasks.append((row,split,counters[split],isnegative))
    return tasks,dict(counters)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261107);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--acceptance',type=Path)
    args=parser.parse_args()
    if not args.pilot:
        accepted=json.loads(args.acceptance.read_text()) if args.acceptance else {}
        assert accepted.get('status')=='passed' and accepted.get('records')==45
        assert accepted.get('active_grid_fraction',0)>=.8 and accepted.get('dsf_forward')=='passed'
    parent=json.loads((args.source/'validation_report.json').read_text())
    assert parent['status']=='passed' and parent['covered_classes']==21 and parent['navmesh_reachability']=='passed'
    old=[json.loads(line) for line in (args.source/'metadata.jsonl').read_text().splitlines()]
    base={}
    for row in old:
        if not row['is_no_event']:base.setdefault(row['geometry_id'],row)
    cmap=json.loads((args.source/'class_map.json').read_text());nclasses=len(cmap['official_category_to_class_id'])
    counts={'train':4320,'validation':540,'test':540}
    tasks,actual_counts=make_tasks(base,counts,args.seed,args.pilot)
    assert len(tasks)==(45 if args.pilot else 5400)
    if args.output.exists():
        if any(args.output.iterdir()):raise FileExistsError('Output directory is not empty')
    else:args.output.mkdir(parents=True,exist_ok=False)
    for split in counts:
        (args.output/'audio'/split).mkdir(parents=True);(args.output/'labels'/split).mkdir(parents=True)
    for name in ('class_map.json','source_assets.json','asset_audit.json','foa_calibration.json','geometry_audit.json'):
        shutil.copy2(args.source/name,args.output/name)
    manifest=json.loads((args.source/'split_manifest.json').read_text())
    manifest.update(dataset_version=VERSION,planned_records=actual_counts,
        source_dataset='AVN-FOA-SameRoom-v0.1',parent_seal_sha256=sha(args.source/'SHA256SUMS'),
        independent_source_recordings_added=0,independent_acoustic_configurations_added=0,
        development_protocol='same certified geometry and known-source families; changed temporal density; not a new pristine official test')
    write_json(args.output/'split_manifest.json',manifest)
    used={row['geometry_id'] for row,_,_,_ in tasks}
    references=[r for r in json.loads((args.source/'rir_manifest.json').read_text()) if r['geometry_id'] in used]
    assert len(references)==len(used)
    (args.output/'rir').mkdir()
    for ref in references:
        source=args.source/ref['path'];assert sha(source)==ref['sha256']
        shutil.copy2(source,args.output/ref['path'])
    write_json(args.output/'rir_manifest.json',references)
    (args.output/'rir_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in references))
    certificate={name:dict(path=str(args.source/name),sha256=sha(args.source/name)) for name in
        ('validation_report.json','navmesh_reachability_check.json','native_association_check.json','native_audio_pipeline_check.json','SHA256SUMS')}
    write_json(args.output/'parent_certificates.json',certificate)
    config=dict(version=VERSION,seed=args.seed,workers=args.workers,source_dataset=str(args.source),
        source_dataset_seal_sha256=sha(args.source/'SHA256SUMS'),planned_records=actual_counts,
        total_hours=len(tasks)*10/3600,sample_rate_hz=16000,channels=['W','Y','Z','X'],normalization='SN3D',
        target_active_grid_fraction=.8,minimum_positive_active_frames=MIN_ACTIVE_FRAMES,
        negative_fraction=3/45 if args.pilot else .1,source_selection_relative_db=-40,activity_label_relative_db=-50,
        background=None,source_count_max=1,source_motion=False,receiver_motion=False,
        cached_rir_count=len(references),new_rendered_rirs=0,
        acoustic_configuration='unchanged; inherited from certified native16k SoundSpaces RIRs',
        dense_playback='actual non-silent source crops, logged random retriggering and5ms crossfade; single full-history convolution',
        training_started=False,official_wav_families=21,verified_independent_original_recordings='unknown',
        code_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('dense_synthesis.py'))})
    write_json(args.output/'generation_config.json',config)
    profiles=[]
    for cid in range(nclasses):
        row=next(r for r in base.values() if r['class_id']==cid)
        dry=read_source(row['path']);_,profile=active_regions(dry)
        profiles.append(dict(class_id=cid,category=row['official_category'],source_audio_hash=row['source_audio_hash'],**profile))
    write_json(args.output/'source_activity_audit.json',profiles)
    start=time.monotonic();records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
            initargs=(str(args.source),str(args.output),nclasses,args.seed)) as pool,(args.output/'metadata.jsonl').open('w') as stream:
        for row in pool.map(generate_one,tasks,chunksize=4):
            stream.write(json.dumps(row)+'\n');stream.flush();records.append(row)
            if len(records)%50==0 or len(records)==len(tasks):
                print(json.dumps(dict(completed=len(records),planned=len(tasks),elapsed_seconds=time.monotonic()-start)),flush=True)
    write_json(args.output/'generation_summary.json',dict(status='generated_pending_independent_validation',records=len(records),
        hours=len(records)/360,records_by_split=actual_counts,active_grid_fraction=sum(r['supervised_active_frames'] for r in records)/(len(records)*100),
        no_event_records=sum(r['is_no_event'] for r in records),elapsed_seconds=time.monotonic()-start,cached_native_rirs=len(references),new_rirs_rendered=0))


if __name__=='__main__':main()
