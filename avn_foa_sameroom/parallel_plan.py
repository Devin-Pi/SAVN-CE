"""Independent per-building geometry audit; deterministic across worker counts."""
import argparse,concurrent.futures,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from .common import write_json


def worker(task):
    scene,root,seed=task
    with (root/'worker.log').open('w') as log:
        subprocess.run([sys.executable,'-u','-m','avn_foa_sameroom.plan',str(root),'--seed',str(seed)],check=True,stdout=log,stderr=subprocess.STDOUT)
    return scene,json.loads((root/'geometry_audit.json').read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20261107);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ('class_map.json','split_manifest.json','class_coverage.csv','asset_audit.json','source_assets.json','official_goals.jsonl'):
        shutil.copyfile(a.assets/name,a.output/name)
    groups={}
    for l in (a.assets/'official_goals.jsonl').read_text().splitlines():
        row=json.loads(l);groups.setdefault(row['scene_id'],[]).append(l)
    tasks=[]
    for scene,rows in sorted(groups.items()):
        root=a.output/'geometry_shards'/scene;root.mkdir(parents=True)
        (root/'official_goals.jsonl').write_text('\n'.join(rows)+'\n')
        seed=(a.seed+int(hashlib.sha256(scene.encode()).hexdigest()[:8],16))%(2**31-1)
        tasks.append((scene,root,seed))
    stats={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for scene,result in pool.map(worker,tasks):
            stats[scene]=result;print(scene,result['accepted_configurations'],flush=True)
    counts={};total=0;per_scene={}
    with (a.output/'geometry_candidates.jsonl').open('w') as c,(a.output/'geometry_rejections.jsonl').open('w') as r:
        for scene,root,seed in tasks:
            c.write((root/'geometry_candidates.jsonl').read_text());r.write((root/'geometry_rejections.jsonl').read_text())
            result=stats[scene];total+=result['accepted_configurations'];per_scene.update(result['per_scene'])
            for key,n in result['goal_rejections'].items():counts[key]=counts.get(key,0)+n
    write_json(a.output/'geometry_audit.json',dict(accepted_configurations=total,goal_rejections=counts,per_scene=per_scene,seed=a.seed,
        scene_seed_rule='(global_seed + int(SHA256(scene_id)[:8],16)) % (2**31-1)',workers=a.workers,
        spatial_sampling_reproducible_across_worker_counts=True))
    print('TOTAL',total,flush=True)
if __name__=='__main__':main()
