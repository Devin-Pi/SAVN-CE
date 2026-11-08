"""Read-only asset audit, stable SAVN IDs, train-pool-only building split."""
import argparse,ast,collections,csv,gzip,json,math
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
from .common import sha,write_json,VERSION


def mappings(repo):
    tree=ast.parse((repo/'savnce/mp3d_utils.py').read_text())
    values={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('CATEGORY_INDEX_MAPPING','SCENE_SPLITS'):
                values[node.targets[0].id]=ast.literal_eval(node.value)
    return values['CATEGORY_INDEX_MAPPING'],values['SCENE_SPLITS']


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--episodes',type=Path,required=True)
    p.add_argument('--sounds',type=Path,required=True);p.add_argument('--scenes',type=Path,default=Path('/datasets/mp3d'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20261107);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cmap,official_splits=mappings(a.repo)
    write_json(a.output/'class_map.json',{'version':VERSION,'official_category_to_class_id':cmap,'source_file_sha256':sha(a.repo/'savnce/mp3d_utils.py')})
    sources={};wave_groups=collections.defaultdict(list);missing=[]
    for cat,cid in sorted(cmap.items(),key=lambda x:x[1]):
        sound_id='train/'+cat+'.wav';path=a.sounds/sound_id
        if not path.is_file():missing.append(cat);continue
        wave,sr=sf.read(path,dtype='float32',always_2d=True)
        wave=wave.mean(axis=1)
        if sr!=16000:
            d=math.gcd(sr,16000);wave=resample_poly(wave,16000//d,sr//d)
        import hashlib
        family=hashlib.sha256(np.asarray(wave,dtype='<f4').tobytes()).hexdigest()
        row=dict(official_category=cat,class_id=cid,source_sound_id=sound_id,path=str(path),
                 source_audio_hash=sha(path),source_family_id=family,original_split='train',
                 original_sample_rate_hz=int(sr),duration_s=len(wave)/16000,original_wav_files=1,original_recording_count='unknown',
                 family_resolution='canonical decoded mono 16k PCM hash; provenance beyond WAV unknown')
        sources[cat]=row;wave_groups[family].append(cat)
    write_json(a.output/'source_assets.json',list(sources.values()))
    episode_docs=[];goals={};rejected=collections.Counter();scene_counts={};episode_hashes={}
    for path in sorted(a.episodes.glob('*.json.gz')):
        sid=path.name.split('.')[0]
        if sid not in official_splits['train']:rejected['not_official_train_scene']+=1;continue
        mesh=a.scenes/sid/(sid+'.glb')
        if not all(mesh.with_name(sid+ext).is_file() for ext in ('.glb','.navmesh','.house')):rejected['missing_scene_asset']+=1;continue
        doc=json.loads(gzip.open(path,'rt').read());episode_hashes[sid]=sha(path);scene_counts[sid]=len(doc['episodes'])
        for ep in doc['episodes']:
            for g in ep.get('goals',[])[:1]:
                cat=g.get('object_category');sound=g.get('sound_id')
                if cat not in sources or sound!=sources[cat]['source_sound_id']:rejected['missing_or_non_train_sound']+=1;continue
                if ep['scene_id'].split('/')[-2]!=sid:raise ValueError('Scene id mismatch')
                key=(sid,int(g['object_id']),sound,tuple(g['position']))
                if key in goals:continue
                goals[key]=dict(scene_id=sid,source_episode_id=str(ep['episode_id']),source_episode_file=str(path),
                    source_episode_file_sha256=episode_hashes[sid],source_object_id=int(g['object_id']),
                    raw_official_goal_position=g['position'],source_position=(np.asarray(g['position'])+[0,1.6,0]).tolist(),
                    height_offsets={'continuous_dataset_y':.1,'audio_renderer_y':1.5},**sources[cat])
    populated=sorted({g['scene_id'] for g in goals.values()})
    if len(populated)<3:raise RuntimeError('At least three official TRAIN buildings required')
    # Stable seeded assignment, before geometry/RIR rendering. Choose from train pool only.
    rng=np.random.default_rng(a.seed);rng.shuffle(populated)
    n=len(populated);nv=max(2,int(round(n*.15)));nt=max(2,int(round(n*.15)))
    split_scenes={'train':sorted(populated[nv+nt:]),'validation':sorted(populated[:nv]),'test':sorted(populated[nv:nv+nt])}
    lookup={s:k for k,ss in split_scenes.items() for s in ss}
    rows=[dict(g,split=lookup[g['scene_id']]) for g in goals.values()]
    (a.output/'official_goals.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in rows))
    shared=[{'family_id':h,'categories':cats} for h,cats in wave_groups.items() if len(cats)>1]
    split=dict(protocol='known-sound / unseen-scene',official_resource_pool='train only',seed=a.seed,
        scene_ids=split_scenes,planned_records={'train':2880,'validation':360,'test':360},
        shared_sound_families=sorted(wave_groups),sound_families_disjoint=False,
        explanation='One official train WAV per category. The same train source family is explicitly shared across internal splits. No unheard-sound claim. No official val/test resources are consumed.',
        scene_hashes=episode_hashes,split_created_before_rendering=True)
    write_json(a.output/'split_manifest.json',split)
    counts=collections.Counter(g['official_category'] for g in rows)
    with (a.output/'class_coverage.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['class_id','official_category','original_recordings','source_families','official_goal_configurations','buildings','status']);w.writeheader()
        for cat,cid in sorted(cmap.items(),key=lambda x:x[1]):
            w.writerow(dict(class_id=cid,official_category=cat,original_recordings=int(cat in sources),source_families=int(cat in sources),official_goal_configurations=counts[cat],buildings=len({g['scene_id'] for g in rows if g['official_category']==cat}),status='pending_geometry' if counts[cat] else 'missing_sound' if cat not in sources else 'no_official_train_target'))
    write_json(a.output/'asset_audit.json',dict(version=VERSION,official_class_count=len(cmap),sound_assets=len(sources),
        missing_sounds=missing,unique_waveform_families=len(wave_groups),cross_category_identical_waveforms=shared,
        unique_goal_configurations=len(rows),official_train_scene_episode_counts=scene_counts,exclusions=dict(rejected),
        official_val_test_used=False,original_assets_modified=False,source_height_rule='raw official goal + 0.1m dataset offset + 1.5m renderer offset, once each',
        source_family_limitation='Canonical PCM hashes cannot prove independent recordings or detect every same-origin excerpt. Each official category WAV is conservatively one source family; same-family generalization is disclosed.'))
    print(json.dumps({'scenes':{k:len(v) for k,v in split_scenes.items()},'classes':len(cmap),'goals':len(rows),'shared_waveform_classes':shared}),flush=True)
if __name__=='__main__':main()
