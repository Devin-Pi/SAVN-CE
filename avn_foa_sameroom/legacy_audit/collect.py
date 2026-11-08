"""Collect only acoustics evidence. Never copy logger credentials/config.yaml."""
import argparse,collections,hashlib,json
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--archives',type=Path,default=Path('/media/8tbhdd/rdpi/workspaces'))
    p.add_argument('--workspace',type=Path,default=Path('/media/4tbssd/av_seld/workspaces'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    raw=a.archives/'audio_bb_mamba_15_MB_RAW';sc=a.archives/'SpatialScaper_';runtime=a.workspace/'dsf_localization_runtime_v1'
    checksum=sha(raw/'checkpoints/epoch17_val_F0.8022_val_seld_scr0.1549.ckpt')
    assert checksum==sha(a.workspace/'av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt')=='85e380a6a8d2f1e545aa093e3f086f0a4d7830a15c41f7f79d047d7d9c1f9f90'
    assert sha(raw/'models/audio_encoder.py')==sha(runtime/'models/audio_encoder.py')
    lists={}
    for folder in ('list_dataset','list_dataset_raw'):
        for f in sorted((sc/'output'/folder).glob('*.txt')):
            if 'testedition' in f.name:continue
            items=f.read_text().splitlines();count=collections.Counter(Path(s).name.split('_mix')[0] for s in items)
            lists[str(f.relative_to(sc/'output'))]=dict(count=len(items),rooms=dict(sorted(count.items())),sha256=sha(f))
    log=raw/'logs/file_2025-04-16_13-56-56_653527.log'
    epoch17=[l for l in log.read_text().splitlines() if 'Validation results: EPOCH: 17,' in l]
    observed={}
    for key,name in [('wav_train','legacy_foa_train_measurement.json'),('wav_val','legacy_foa_val_measurement.json'),('tau','legacy_tau_rir_measurement.json'),('acs','legacy_foa_acs_check.json'),('scale','legacy_foa_scale_sensitivity.json')]:
        obj=json.loads((a.workspace/name).read_text());observed[key]=obj
        (a.output/name).write_text(json.dumps(obj,indent=2)+'\n')
    assert observed['tau']['official_foa_norm']=='sn3d' and observed['tau']['official_foa_channel_order']=='wyzx'
    assert observed['acs']['status']=='passed'
    sources=[raw/'config/tools_cfg.py',raw/'models/audio_encoder.py',raw/'dataloaders/dataloader.py',raw/'utils/tools.py',raw/'utils/cls_feature_class.py',raw/'utils/extract_IV.py',raw/'utils/stft.py',sc/'example_generation.py',sc/'scripts/prepare_rirs.py',sc/'spatialscaper/core.py',sc/'spatialscaper/sofa_utils.py',sc/'spatialscaper/spatialize.py',sc/'spatialscaper/utils.py']
    missing_sources=[str(s) for s in sources if not s.exists()]
    source_hashes={str(s):sha(s) for s in sources if s.exists()}
    other=a.archives/'SpatialScaper/spatialscaper/sofa_utils.py'
    source_hashes[str(other)]=sha(other)
    train=lists['list_dataset/dcase2023t3_foa_devtrain_audiovisual.txt']['rooms']
    assert sum(train.values())==960
    result=dict(status='partially_verified',checkpoint_sha256=checksum,archived_checkpoint_matches_current=True,
        checkpoint_hparams_data_path='/workspaces/SpatialScaper/output/',checkpoint_epoch=17,checkpoint_global_step=517860,
        training_log=dict(path=str(log),sha256=sha(log),epoch17_lines=epoch17),
        loader_resolved_archived_lists=lists,
        loader_resolved_train_count=960,tau_train_count=train['fold1_room1']+train['fold1_room3'],metu_train_count=train['fold1_room2']+train['fold2_room7'],
        historical_list_limit='These are archived loader-resolved lists, not a dataset hash embedded in the checkpoint; exact historical per-clip consumption is not cryptographically proved.',
        room_mapping_source='Archived example_generation.py filename comments: fold1_room1/sc203, fold1_room2/metu, fold1_room3/se203, fold2_room4/metu, fold2_room5/sc203, fold2_room6/se203, fold2_room7/metu; comments support provenance but are not per-wave RIR manifests.',
        tau_format=dict(status='confirmed_for_official_TAU_RIR_and_preserving_conversion',normalization='SN3D',channel_order=['W','Y','Z','X'],metadata_file=str(a.workspace/'avn14_spatialscaper_v1/tau_raw/rirdata.mat'),metadata_sha256=observed['tau']['metadata_sha256'],provider='https://zenodo.org/records/6408611',numeric_rirs=sum(v['count'] for v in observed['tau']['summary'].values()),
            original_sofa_byte_identity='unverified: historical prepared SOFA missing; provider metadata and converter code inspected'),
        metu_format=dict(status='unverified',reason='Historical metu_foa.sofa missing. Archived prepare_metu builds only mic; FOA was an external pre-encoded download. Official old link returns HTTP404. Retained labeled METU clip has only6 labeled100ms frames; 4 eligible interior single-source frames cannot uniquely establish normalization or channel permutation.',official_foa_url='https://drive.google.com/file/d/1zamCd6OR6Tr5M40RdDhswYbT1wbGo2ZO/view',observed_http_status=404),
        channel_coordinate_contract=dict(status='confirmed_for_archived_loader_and_TAU_retained_training_audio',channels=['W','Y','Z','X'],xyz_axes=['front','left','up'],handedness='right',azimuth='0 front; +90 left',elevation='positive up',acs_checks=observed['acs']['cases'],acs_max_error=observed['acs']['max_error'],actual_train_audio_best_mapping=observed['wav_train']['ranked_channel_mappings'][0],new_avn_format_matches_contract=True),
        full_training_pool_single_normalization='unverified: cannot certify METU portion or every deleted waveform from TAU metadata alone',
        comparison_to_new_data=dict(new_format='ACN/SN3D, listener-local +X front/+Y left/+Z up',tau_norm_order_axes_match=True,all_legacy_sources_certified=False,new_data_conversion_changed=False,extra_sqrt3_conversion_recommended=False,
            sample_rate_note='Legacy24k/nativeAVN16k frontend frequency/time differences remain as previously documented'),
        source_sha256=source_hashes,missing_archive_source_files=missing_sources,sofa_helper_inspected_in_other_archive=str(other),audio_and_weights_modified=False,training_started=False,
        evidence_boundaries=['Declared metadata determines TAU normalization; reverberant W-to-directional RMS/coherent gains alone do not determine SN3D/N3D.','Four labeled METU interior frames do not support a unique global format conclusion.','Prediction insensitivity to gain is not historical-format evidence.','No new data channel or sign adjustment was selected using per-sample truth.'])
    (a.output/'legacy_foa_format_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','loader_resolved_train_count','tau_train_count','metu_train_count','full_training_pool_single_normalization')},ensure_ascii=False))
if __name__=='__main__':main()
