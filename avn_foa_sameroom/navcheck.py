"""Independently verify receiver reachability from the cited official episode start."""
import argparse,collections,gzip,json
from pathlib import Path
import numpy as np
import habitat_sim as hs
from .common import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    groups=collections.defaultdict(dict)
    for l in (a.root/'metadata.jsonl').open():
        r=json.loads(l);groups[r['scene_id']][r['geometry_id']]=r
    passed=[];failed=[]
    for scene,rows in groups.items():
        first=next(iter(rows.values()))
        doc=json.loads(gzip.open(first['source_episode_file'],'rt').read())
        starts={str(e['episode_id']):e['start_position'] for e in doc['episodes']}
        nav=hs.PathFinder();assert nav.load_nav_mesh('/datasets/mp3d/'+scene+'/'+scene+'.navmesh')
        for gid,row in rows.items():
            path=hs.ShortestPath();path.requested_start=np.asarray(starts[row['source_episode_id']],dtype=np.float32)
            path.requested_end=np.asarray(row['base_position'],dtype=np.float32)
            ok=nav.find_path(path)
            result=dict(geometry_id=gid,scene_id=scene,reachable_from_official_start=bool(ok),geodesic_distance_m=float(path.geodesic_distance) if ok else None)
            (passed if ok else failed).append(result)
    result=dict(status='passed' if not failed else 'failed',checked_configurations=len(passed)+len(failed),failed=failed,
        maximum_geodesic_from_official_start_m=max((r['geodesic_distance_m'] for r in passed),default=0),
        note='Original official episode start to generated receiver base using unchanged original NavMesh. This is an offline geometry check, not Agent success.')
    write_json(a.root/'navmesh_reachability_check.json',result);print(result,flush=True)
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
