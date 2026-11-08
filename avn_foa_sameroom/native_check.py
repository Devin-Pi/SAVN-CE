"""Read-only official Dataset association and native overlap-add consistency checks."""
import argparse,gzip,json,runpy
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
import habitat_sim as hs
from habitat_sim.utils.common import quat_from_angle_axis
import imageio.v2 as imageio
from savnce.datasets.savnce_dataset import SAVNCE_Dataset
from savnce.tasks.savnce_task import merge_sim_episode_config
from .common import write_json
from .plan import make_sim
from .synthesis import read_source


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--scenes',type=Path,default=Path('/datasets/mp3d'));a=p.parse_args()
    rows=[json.loads(l) for l in (a.root/'metadata.jsonl').read_text().splitlines()]
    selected=[];seen=set()
    for row in rows:
        if row['scene_id'] not in seen and not row['is_no_event']:
            selected.append(row);seen.add(row['scene_id'])
        if len(selected)==4:break
    report=[];audit=a.root/'audit_views';audit.mkdir(exist_ok=True)
    for i,row in enumerate(selected):
        cfg=runpy.run_path('savnce_baselines/magnet/config/default.py')['get_task_config']();cfg.defrost()
        cfg.DATASET.DATA_PATH=row['source_episode_file'];cfg.DATASET.CONTENT_SCENES=[];cfg.DATASET.SCENES_DIR=str(a.scenes)
        dataset=SAVNCE_Dataset(cfg.DATASET)
        episode=next(e for e in dataset.episodes if str(e.episode_id)==row['source_episode_id'])
        goal=episode.goals[0]
        np.testing.assert_allclose(np.array(goal.position)+[0,1.5,0],row['source_position'],atol=1e-8)
        assert int(goal.object_id)==row['source_object_id'] and goal.object_category==row['official_category'] and goal.sound_id==row['source_sound_id']
        # Follow the official merger; never use the serialized position twice.
        merge_sim_episode_config(cfg.SIMULATOR,episode)
        np.testing.assert_allclose(np.array(cfg.SIMULATOR.AGENT_0.GOAL_POSITION)+[0,1.5,0],row['source_position'],atol=1e-8)
        with np.load(a.root/row['rir_reference']) as z:ir=z['native']
        dry=read_source(row['path'])[:16000*2]
        # Compare the official 250ms source-chunk overlap-add algorithm against full-history convolution.
        full=fftconvolve(dry[:,None],ir,axes=0);ola=np.zeros_like(full)
        for start in range(0,len(dry),4000):
            wet=fftconvolve(dry[start:start+4000,None],ir,axes=0);ola[start:start+len(wet)]+=wet
        maxerr=float(np.max(np.abs(full-ola)));assert maxerr<2e-5
        sim=make_sim(a.scenes/row['scene_id']/(row['scene_id']+'.glb'),7,renderer=True)
        try:
            delta=np.subtract(row['source_position'],row['receiver_position']);yaw=float(np.arctan2(-delta[0],-delta[2]))
            state=hs.AgentState();state.position=np.array(row['base_position']);state.rotation=quat_from_angle_axis(yaw,np.array([0,1,0]))
            sim.get_agent(0).set_state(state);rgb=sim.get_sensor_observations()['audit_rgb'][...,:3]
            imageio.imwrite(audit/(row['sample_id']+'.png'),rgb)
        finally:sim.close()
        report.append(dict(sample_id=row['sample_id'],native_dataset_association='passed',source_height='passed',
            official_chunk_overlap_add_vs_full_max_error=maxerr,
            audit_view=str((audit/(row['sample_id']+'.png')).relative_to(a.root)),audit_camera_yaw_rad=yaw,audit_camera_position=row['receiver_position'],audit_camera_hfov_deg=90,audit_view_uses_record_yaw=False,
            note='Native Dataset and official config merger executed. Convolution math matches simulator static-source overlap-add; no Agent success claim. View points at source for offline inspection only.'))
    write_json(a.root/'native_association_check.json',dict(status='passed',samples=report,original_episodes_modified=False))
    print('native checks passed',len(report),flush=True)
if __name__=='__main__':main()
