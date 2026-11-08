"""Whole-corpus checks; declared known-sound sharing is explicit, never called unheard."""
import argparse,collections,csv,json,math,shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from .common import RATE,DURATION,VERSION,sha,write_json,native_to_local,local_unit,angles
from .geometry import Rooms
from .synthesis import read_source
from scipy.signal import fftconvolve


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--forward-report',type=Path)
    p.add_argument('--scenes',type=Path,default=Path('/datasets/mp3d'));p.add_argument('--pilot',action='store_true');a=p.parse_args()
    root=a.root;rows=[json.loads(l) for l in (root/'metadata.jsonl').read_text().splitlines()]
    manifest=json.loads((root/'split_manifest.json').read_text());cmap=json.loads((root/'class_map.json').read_text())['official_category_to_class_id'];nclasses=len(cmap)
    calibration=json.loads((root/'foa_calibration.json').read_text());assert calibration['status']=='passed'
    if a.forward_report:
        if a.forward_report.resolve()!=(root/'dsf_forward_smoke.json').resolve():shutil.copyfile(a.forward_report,root/'dsf_forward_smoke.json')
    forward=json.loads((root/'dsf_forward_smoke.json').read_text());assert forward['status']=='passed' and not forward['training_started']
    native=json.loads((root/'native_association_check.json').read_text());assert native['status']=='passed'
    if not a.pilot:
        nav=json.loads((root/'navmesh_reachability_check.json').read_text());assert nav['status']=='passed' and nav['checked_configurations']==len({r['geometry_id'] for r in rows})
        chain=json.loads((root/'native_audio_pipeline_check.json').read_text());assert chain['status']=='passed' and chain['native_simulator_executed']
    else:nav={'status':'not_required_for_pilot'}
    assert forward['original_adpit_label_converter_checks']>=1
    rir_rows=[json.loads(l) for l in (root/'rir_manifest.jsonl').read_text().splitlines()]
    rirs={r['path']:r for r in rir_rows};assert len(rirs)==len(rir_rows)
    scene_sets=collections.defaultdict(set);geometry_sets=collections.defaultdict(set);family_sets=collections.defaultdict(set)
    split_counts=collections.Counter();counts=collections.defaultdict(collections.Counter);rooms=collections.defaultdict(set)
    instances=set();orientations=set();distance=[];azimuth=[];elevation=[];activity=0;ignore=0;seconds=0;negative=0;source_cache={};reconstruction=0
    per_class_sources=collections.defaultdict(set);catalog={};native_cache={};room_models={};sample_ids=set();energy_train=0;values_train=0;max_delay_error=0
    source_checks={};episode_checks={};geometry_checks={};reconstruction_by_split=collections.Counter()
    for index,row in enumerate(rows):
        sid=row['sample_id'];assert sid not in sample_ids;sample_ids.add(sid)
        split=row['split'];assert row['scene_id'] in manifest['scene_ids'][split]
        source_checks[row['path']]=row['source_audio_hash'];episode_checks[row['source_episode_file']]=row['source_episode_file_sha256'];geometry_checks[row['scene_id']]=row['geometry_sha256']
        assert row['source_sound_id'].startswith('train/') and row['original_split']=='train'
        assert row['same_room'] is True and row['geometric_los'] is True and row['source_in_valid_space'] and row['receiver_in_valid_space']
        assert row['class_id']==cmap[row['official_category']]
        np.testing.assert_allclose(np.asarray(row['raw_official_goal_position'])+[0,1.6,0],row['source_position'],atol=1e-8)
        np.testing.assert_allclose(np.asarray(row['base_position'])+[0,1.5,0],row['receiver_position'],atol=1e-5)
        matrix=np.asarray(row['R_world_from_audio']);np.testing.assert_allclose(matrix.T@matrix,np.eye(3),atol=1e-7);assert np.linalg.det(matrix)>.9999
        delta=np.subtract(row['source_position'],row['receiver_position']);u=matrix.T@delta;u/=np.linalg.norm(u)
        np.testing.assert_allclose(u,local_unit(row['source_position'],np.array(row['receiver_position']),row['receiver_yaw_rad']),atol=1e-7)
        np.testing.assert_allclose(u,row['audio_direction_xyz'],atol=1e-7)
        assert abs(np.linalg.norm(delta)-row['distance_m'])<1e-7
        if row['scene_id'] not in room_models:room_models[row['scene_id']]=Rooms(a.scenes/row['scene_id']/(row['scene_id']+'.house'))
        model=room_models[row['scene_id']]
        assert model.locate(row['source_position'])==model.locate(row['receiver_position'])==row['room_id']
        wave,rate=sf.read(root/row['audio'],dtype='float32',always_2d=True);info=sf.info(root/row['audio'])
        assert rate==RATE and wave.shape==(RATE*DURATION,4) and info.subtype=='FLOAT'
        assert np.isfinite(wave).all() and np.max(np.abs(wave))<=.951
        assert sha(root/row['audio'])==row['audio_sha256'] and sha(root/row['labels'])==row['labels_sha256']
        with np.load(root/row['labels'],allow_pickle=False) as z:
            active=z['activity'].astype(bool);mask=z['ignore_mask'].astype(bool);adpit=z['adpit']
            assert adpit.shape==(100,6,4,nclasses) and np.isfinite(adpit).all()
            assert not np.any(active&mask) and not adpit[~active].any() and not adpit[:,1:].any()
            assert np.all(adpit[active,0,0,row['class_id']]==1)
            for k in range(3):np.testing.assert_allclose(adpit[active,0,k+1,row['class_id']],np.full(active.sum(),u[k]),atol=1e-6)
            assert np.allclose(np.linalg.norm(z['direction_xyz'][active],axis=1),1)
            assert not z['direction_xyz'][~active].any()
            csvrows=list(csv.reader((root/row['dcase_cartesian_csv']).open()))
            assert {int(x[0]) for x in csvrows}==set(np.flatnonzero(active))
            activity+=int(active.sum());ignore+=int(mask.sum())
        if row['is_no_event']:
            assert not wave.any() and not active.any() and not mask.any() and not row['events'];negative+=1
        else:
            assert wave.any() and active.any() and row['events']
        ref=rirs[row['rir_reference']];assert ref['split']==split and ref['history_reset'] is True
        assert ref['source_visible_in_acoustic_geometry'] is True
        expected=row['distance_m']/343
        max_delay_error=max(max_delay_error,abs(row['rir_onset_s']-expected))
        # Only a small, deterministic subset requires full waveform reconstruction.
        if reconstruction_by_split[split]<4 and not row['is_no_event']:
            if row['path'] not in source_cache:source_cache[row['path']]=read_source(row['path'])
            dry=source_cache[row['path']];signal=np.zeros(RATE*DURATION,dtype=np.float32)
            for e in row['events']:
                start,end=[round(x*RATE) for x in e['source_crop_interval_s']];crop=dry[start:end].copy()
                fade=round(e['fade_duration_s']*RATE);ramp=np.linspace(0,1,fade,dtype=np.float32);crop[:fade]*=ramp;crop[-fade:]*=ramp[::-1]
                onset=round(e['emission_interval_s'][0]*RATE);signal[onset:onset+len(crop)]+=crop
            with np.load(root/ref['path'],allow_pickle=False) as z:
                native=z['native'];np.testing.assert_allclose(native_to_local(native,0),z['ambix_world'],atol=1e-8)
            recovered=fftconvolve(signal[:,None],native_to_local(native,row['receiver_yaw_rad']),axes=0)[:len(signal)].astype(np.float32)*row['global_gain']
            np.testing.assert_allclose(recovered,wave,rtol=1e-5,atol=1e-7);reconstruction+=1;reconstruction_by_split[split]+=1
        if split=='train':energy_train+=float(np.sum(wave.astype(np.float64)**2));values_train+=wave.size
        split_counts[split]+=1;counts[split][row['class_id']]+=int(not row['is_no_event'])
        scene_sets[split].add(row['scene_id']);geometry_sets[split].add(row['geometry_id']);family_sets[split].add(row['source_family_id'])
        rooms[split].add((row['scene_id'],row['room_id']));instances.add((row['scene_id'],row['source_object_id']));orientations.add((row['geometry_id'],row['receiver_yaw_rad']))
        per_class_sources[row['class_id']].add(row['source_family_id']);distance.append(row['distance_m']);azimuth.append(row['azimuth_deg']);elevation.append(row['elevation_deg']);seconds+=row['activity_duration_s']
    for ref in rir_rows:assert sha(root/ref['path'])==ref['sha256']
    for path,digest in {**source_checks,**episode_checks}.items():assert sha(path)==digest, path
    for scene,hashes in geometry_checks.items():
        for name,ext in (('glb','.glb'),('house','.house'),('navmesh','.navmesh')):assert sha(a.scenes/scene/(scene+ext))==hashes[name]
    intersections={}
    for left,right in (('train','validation'),('train','test'),('validation','test')):
        assert not scene_sets[left]&scene_sets[right] and not geometry_sets[left]&geometry_sets[right]
        intersections[left+'__'+right]=dict(scenes=0,rir_configurations=0,declared_known_sound_families=len(family_sets[left]&family_sets[right]))
    assert manifest['protocol']=='known-sound / unseen-scene' and not manifest['sound_families_disjoint']
    expected={'train':24,'validation':9,'test':9} if a.pilot else manifest['planned_records']
    assert dict(split_counts)==expected,(dict(split_counts),expected)
    if a.pilot:assert 30<=len(rows)<=50
    else:assert len(rows)==3600
    actual_hours={s:n*DURATION/3600 for s,n in split_counts.items()}
    byclass=[]
    for cat,cid in sorted(cmap.items(),key=lambda x:x[1]):
        byclass.append(dict(class_id=cid,official_category=cat,available_official_train_wav_files=1,verified_independent_recordings='unknown',used_source_families=len(per_class_sources[cid]),
            **{s+'_positive_records':counts[s][cid] for s in expected},total_positive_records=sum(counts[s][cid] for s in expected),
            status='covered' if sum(counts[s][cid] for s in expected) else 'no_selected_legal_configuration'))
    with (root/'class_coverage.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(byclass[0]));writer.writeheader();writer.writerows(byclass)
    result=dict(status='passed',version=VERSION,records=len(rows),hours=sum(actual_hours.values()),hours_by_split=actual_hours,
        records_by_split=dict(split_counts),official_class_count=nclasses,covered_classes=sum(x['total_positive_records']>0 for x in byclass),
        buildings={s:len(v) for s,v in scene_sets.items()},rooms={s:len(v) for s,v in rooms.items()},target_instances=len(instances),
        native_rirs=len(rir_rows),record_spatial_configurations=len(orientations),same_native_rir_has_up_to_3_oriented_positive_crops=True,
        no_event_records=negative,no_event_fraction=negative/len(rows),supervised_active_frame_fraction=activity/(len(rows)*100),ignore_frame_fraction=ignore/(len(rows)*100),
        direct_activity_duration_s=seconds,audio_validated_files=len(rows),full_history_reconstruction_checks=reconstruction,full_history_reconstruction_by_split=dict(reconstruction_by_split),
        azimuth_histogram_12_bins=np.histogram(azimuth,bins=np.linspace(-180,180,13))[0].tolist(),
        elevation_histogram_18_bins=np.histogram(elevation,bins=np.linspace(-90,90,19))[0].tolist(),elevation_range_deg=[min(elevation),max(elevation)],
        distance_histogram=np.histogram(distance,bins=[0,.5,1,2,4,8,100])[0].tolist(),distance_range_m=[min(distance),max(distance)],
        maximum_rir_onset_vs_euclidean_delay_error_s=max_delay_error,
        split_intersections=intersections,unheard_sound_generalization=False,official_test_resources_used=False,original_assets_unchanged=True,verified_original_sound_files=len(source_checks),verified_original_episode_files=len(episode_checks),
        train_only_scalar_rms=math.sqrt(energy_train/values_train),normalization='raw float wave gain shared by all4channels; original model per-window temporal feature normalization; no corpus-wide/test statistics applied',
        foa_calibration='passed',dsf_forward='passed',native_association='passed',navmesh_reachability=nav['status'],training_started=False,geometry_los='Collision mesh and actual exported acoustic triangle mesh agree for every selected config',
        limitations=['Known-sound sources are deliberately shared across internal splits','Source-family origin beyond official WAV unknown; no unseen-recording claim','Room polygons and original scanned meshes are imperfect; conservative boundary exclusion is used','Within-room LOS only; not a cross-room/NLOS dataset','No distance head or navigation agent training','SoundSpaces RLR internal RNG not exposed; bitwise regeneration not guaranteed'])
    write_json(root/'validation_report.json',result)
    write_json(root/'train_normalization_statistics.json',dict(train_only_scalar_rms=result['train_only_scalar_rms'],used_for_normalization=False))
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
