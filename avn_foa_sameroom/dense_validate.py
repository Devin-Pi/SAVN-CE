"""Independently verify every dense waveform, original sources, labels and split."""
import argparse
import collections
import concurrent.futures
import csv
import json
from pathlib import Path
import math
import time

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from .common import RATE,DURATION,sha,write_json,native_to_local
from .synthesis import read_source

_CTX={}


def init_worker(root):
    _CTX.update(root=Path(root),dry={},rir={})


def check_one(task):
    row,convolution=task;root=_CTX['root']
    wave,rate=sf.read(root/row['audio'],dtype='float32',always_2d=True)
    assert rate==16000 and wave.shape==(160000,4) and sf.info(root/row['audio']).subtype=='FLOAT'
    assert np.isfinite(wave).all() and float(np.abs(wave).max())<=.951
    assert sha(root/row['audio'])==row['audio_sha256'] and sha(root/row['labels'])==row['labels_sha256']
    if row['path'] not in _CTX['dry']:_CTX['dry'][row['path']]=read_source(row['path'])
    dry=_CTX['dry'][row['path']];signal=np.zeros(160000,np.float32)
    for event in row['events']:
        a,b=event['source_start_sample_16k'],event['source_end_sample_16k'];onset=event['emission_start_sample_16k'];fade=event['fade_samples_16k']
        assert 0<=a<b<=len(dry) and 0<=onset and onset+b-a<=160000
        assert not event['looped'] and 1<=fade<=len(dry)//4
        np.testing.assert_allclose([a/RATE,b/RATE],event['source_crop_interval_s'],atol=1e-12)
        np.testing.assert_allclose([onset/RATE,(onset+b-a)/RATE],event['emission_interval_s'],atol=1e-12)
        crop=dry[a:b].copy();ramp=np.linspace(0,1,fade,dtype=np.float32)
        crop[:fade]*=ramp;crop[-fade:]*=ramp[::-1];signal[onset:onset+len(crop)]+=crop
    ref=row['rir_reference']
    if ref not in _CTX['rir']:
        with np.load(root/ref,allow_pickle=False) as z:
            native=z['native'].copy();np.testing.assert_allclose(native_to_local(native,0),z['ambix_world'],atol=1e-8)
            _CTX['rir'][ref]=native
    ir=native_to_local(_CTX['rir'][ref],row['receiver_yaw_rad'])
    if convolution and not row['is_no_event']:
        reconstructed=fftconvolve(signal[:,None],ir,axes=0)[:160000].astype(np.float32)*row['global_gain']
        np.testing.assert_allclose(reconstructed,wave,atol=1e-7,rtol=1e-5)
    signal*=row['global_gain']
    energy=np.sqrt(np.mean(signal.reshape(1000,160)**2,axis=1))
    threshold=max(float(energy.max())*10**(-50/20),1e-8)
    support=np.max(np.abs(ir),axis=1);indices=np.flatnonzero(support>max(float(support.max())*1e-5,1e-12))
    assert len(indices)
    delay=int(indices[0]);arriving=np.zeros(160000,bool)
    for index in np.flatnonzero(energy>threshold):
        first=int(index*160+delay);last=min(160000,first+160)
        arriving[first:last]=True
    occupancy=arriving.reshape(100,1600).sum(1)/1600
    wet=np.sqrt(np.mean(wave.reshape(100,1600,4)**2,axis=(1,2)))
    audible=wet>max(float(wet.max())*1e-4,1e-9)
    expected_ignore=((occupancy>0)&(occupancy<.8))|((occupancy==0)&audible)|((occupancy>=.8)&(~audible))
    expected=(occupancy>=.8)&(~expected_ignore)
    unit=np.asarray(row['R_world_from_audio']).T@np.subtract(row['source_position'],row['receiver_position']);unit/=np.linalg.norm(unit)
    np.testing.assert_allclose(unit,row['audio_direction_xyz'],atol=1e-7)
    with np.load(root/row['labels'],allow_pickle=False) as z:
        active=z['activity'].astype(bool);ignore=z['ignore_mask'].astype(bool)
        np.testing.assert_array_equal(active,expected);np.testing.assert_array_equal(ignore,expected_ignore)
        assert z['adpit'].shape==(100,6,4,21) and np.isfinite(z['adpit']).all()
        expected_adpit=np.zeros_like(z['adpit']);expected_adpit[active,0,0,row['class_id']]=1
        for axis in range(3):expected_adpit[active,0,axis+1,row['class_id']]=unit[axis]
        np.testing.assert_allclose(z['adpit'],expected_adpit,atol=1e-6)
        np.testing.assert_allclose(z['direction_xyz'][active],np.tile(unit,(int(active.sum()),1)),atol=1e-6)
        assert not z['direction_xyz'][~active].any()
    assert int(active.sum())==row['supervised_active_frames']
    assert int(ignore.sum())==row['ignore_frames']
    assert abs(row['rir_onset_s']-delay/RATE)<1e-12
    assert abs(row['activity_duration_s']-float(arriving.sum())/RATE)<1e-8
    with (root/row['dcase_cartesian_csv']).open() as stream:
        entries=list(csv.reader(stream))
    assert [int(entry[0]) for entry in entries]==list(np.flatnonzero(active))
    for entry in entries:
        assert int(entry[1])==row['class_id'] and int(entry[2])==0
        np.testing.assert_allclose(np.array(entry[3:6],float),unit,atol=1e-7)
    if row['is_no_event']:
        assert not wave.any() and not active.any() and not ignore.any() and not row['events']
    else:
        assert int(active.sum())>=92 and wave.any() and np.all(audible[active])
    return dict(active=int(active.sum()),ignore=int(ignore.sum()),audible=int(audible.sum()),
        convolution_checked=bool(convolution and not row['is_no_event']),
        energy=float(np.sum(wave.astype(np.float64)**2)) if row['split']=='train' else 0,
        values=int(wave.size) if row['split']=='train' else 0,event_count=len(row['events']))


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--forward-report',type=Path);a=p.parse_args();start=time.monotonic()
    config=json.loads((a.root/'generation_config.json').read_text());manifest=json.loads((a.root/'split_manifest.json').read_text())
    rows=[json.loads(line) for line in (a.root/'metadata.jsonl').read_text().splitlines()]
    assert len({r['sample_id'] for r in rows})==len(rows)
    counts=collections.Counter(r['split'] for r in rows);assert dict(counts)==manifest['planned_records']
    assert len(rows) in (45,5400)
    source=Path(config['source_dataset']);old={}
    for line in (source/'metadata.jsonl').read_text().splitlines():
        row=json.loads(line);old.setdefault(row['geometry_id'],row)
    assert sha(source/'SHA256SUMS')==manifest['parent_seal_sha256']
    parent=json.loads((source/'validation_report.json').read_text());assert parent['status']=='passed'
    for name,cert in json.loads((a.root/'parent_certificates.json').read_text()).items():assert sha(Path(cert['path']))==cert['sha256']
    refs=json.loads((a.root/'rir_manifest.json').read_text());oldrefs={r['geometry_id']:r for r in json.loads((source/'rir_manifest.json').read_text())}
    assert len(refs)==len({r['geometry_id'] for r in rows})
    for ref in refs:
        assert ref==oldrefs[ref['geometry_id']]
        assert sha(a.root/ref['path'])==ref['sha256']==sha(source/ref['path'])
    cmap=json.loads((a.root/'class_map.json').read_text())['official_category_to_class_id']
    scenes=collections.defaultdict(set);geometries=collections.defaultdict(set);rooms=collections.defaultdict(set)
    sourcefiles={};episodefiles={};first=set();tasks=[]
    for row in rows:
        original=old[row['geometry_id']]
        for key in ('split','scene_id','room_id','class_id','source_object_id','source_position','receiver_position','base_position','geometry_sha256','source_audio_hash','source_episode_file_sha256'):
            assert row[key]==original[key],key
        assert row['same_room'] and row['geometric_los'] and row['source_in_valid_space'] and row['receiver_in_valid_space']
        assert row['class_id']==cmap[row['official_category']]
        assert row['source_sound_id'].startswith('train/') and row['original_split']=='train'
        assert row['scene_id'] in manifest['scene_ids'][row['split']]
        sourcefiles[row['path']]=row['source_audio_hash'];episodefiles[row['source_episode_file']]=row['source_episode_file_sha256']
        scenes[row['split']].add(row['scene_id']);geometries[row['split']].add(row['geometry_id']);rooms[row['split']].add((row['scene_id'],row['room_id']))
        check=(len(rows)==45 or (not row['is_no_event'] and row['geometry_id'] not in first))
        if not row['is_no_event']:first.add(row['geometry_id'])
        tasks.append((row,check))
    for path,digest in {**sourcefiles,**episodefiles}.items():assert sha(Path(path))==digest,path
    for left,right in (('train','validation'),('train','test'),('validation','test')):
        assert not scenes[left]&scenes[right] and not geometries[left]&geometries[right]
    positive_hashes=[r['audio_sha256'] for r in rows if not r['is_no_event']]
    assert len(set(positive_hashes))==len(positive_hashes)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=init_worker,initargs=(str(a.root),)) as pool:
        checks=list(pool.map(check_one,tasks,chunksize=8))
    split_report={};class_report=[]
    for split in ('train','validation','test'):
        selected=[(r,c) for r,c in zip(rows,checks) if r['split']==split]
        active=sum(c['active'] for r,c in selected);ignore=sum(c['ignore'] for r,c in selected)
        split_report[split]=dict(records=len(selected),hours=len(selected)/360,
            positive_records=sum(not r['is_no_event'] for r,c in selected),negative_records=sum(r['is_no_event'] for r,c in selected),
            active_grid_fraction=active/(100*len(selected)),active_nonignored_fraction=active/(100*len(selected)-ignore),
            ignore_grid_fraction=ignore/(100*len(selected)),actual_wet_audible_grid_fraction=sum(c['audible'] for r,c in selected)/(100*len(selected)),
            buildings=len(scenes[split]),rooms=len(rooms[split]),rir_configurations=len(geometries[split]))
        assert split_report[split]['active_grid_fraction']>=.8
    for cat,cid in sorted(cmap.items(),key=lambda item:item[1]):
        result=dict(class_id=cid,category=cat,source_wav_families=1,verified_independent_original_recordings='unknown')
        for split in ('train','validation','test'):
            selected=[(r,c) for r,c in zip(rows,checks) if r['split']==split and r['class_id']==cid]
            positive=[(r,c) for r,c in selected if not r['is_no_event']]
            result[split+'_positive_records']=len(positive)
            result[split+'_negative_records']=len(selected)-len(positive)
            result[split+'_all_record_active_fraction']=sum(c['active'] for r,c in selected)/(100*len(selected)) if selected else None
            result[split+'_positive_active_fraction']=sum(c['active'] for r,c in positive)/(100*len(positive)) if positive else None
            if len(rows)==5400:
                assert positive and result[split+'_positive_active_fraction']>=.92
                assert sum(c['active'] for r,c in selected)/(100*len(selected))>=.8
        class_report.append(result)
    with (a.root/'class_coverage.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(class_report[0]));w.writeheader();w.writerows(class_report)
    forward=None
    if a.forward_report:forward=json.loads(a.forward_report.read_text());assert forward['status']=='passed' and not forward['training_started']
    report=dict(status='passed' if forward else 'pending_forward_compatibility',version=config['version'],records=len(rows),hours=len(rows)/360,
        split=split_report,active_grid_fraction=sum(c['active'] for c in checks)/(100*len(rows)),
        ignore_grid_fraction=sum(c['ignore'] for c in checks)/(100*len(rows)),
        active_nonignored_fraction=sum(c['active'] for c in checks)/(100*len(rows)-sum(c['ignore'] for c in checks)),
        positive_minimum_active_frames=min(c['active'] for r,c in zip(rows,checks) if not r['is_no_event']),
        no_event_records=sum(r['is_no_event'] for r in rows),
        full_waveform_reconstruction_checks=sum(c['convolution_checked'] for c in checks),
        every_record_source_activity_and_label_recomputed=True,every_waveform_hash_and_format_checked=True,
        cached_native_rirs=len(refs),new_rirs_rendered=0,official_classes=len(cmap),
        independent_new_source_recordings=0,official_wav_families=len(sourcefiles),
        train_only_scalar_rms=math.sqrt(sum(c['energy'] for c in checks)/sum(c['values'] for c in checks)),
        event_count_range_positive=[min(c['event_count'] for r,c in zip(rows,checks) if not r['is_no_event']),max(c['event_count'] for r,c in zip(rows,checks) if not r['is_no_event'])],
        dsf_forward='passed' if forward else 'pending',training_started=False,elapsed_seconds=time.monotonic()-start,
        original_assets_unchanged=True,source_certificates_reused_explicitly=True,unheard_sound_generalization=False,
        known_sound_families_shared_between_splits=True,scene_and_rir_splits_disjoint=True,
        baseline_supervised_active_grid_fraction=parent['supervised_active_frame_fraction'],baseline_hours=parent['hours'])
    assert report['active_grid_fraction']>=.8
    if len(rows)==5400:
        assert report['hours']==15 and report['no_event_records']==540
    write_json(a.root/'validation_report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
