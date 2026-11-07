"""Execute the actual SAVN audio pipeline, including cached overlap-add and tail."""
import argparse,json,runpy
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
from savnce.datasets.savnce_dataset import SAVNCE_Dataset
from savnce.tasks.savnce_task import merge_sim_episode_config
from savnce.sims.savnce_simulator import SAVNCE_Simulator
from .common import write_json,native_to_local


def main():
    p=argparse.ArgumentParser();p.add_argument('pilot',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    row=next(json.loads(l) for l in (a.pilot/'metadata.jsonl').open() if not json.loads(l)['is_no_event'])
    cfg=runpy.run_path('savnce_baselines/magnet/config/default.py')['get_task_config']();cfg.defrost()
    cfg.DATASET.DATA_PATH=row['source_episode_file'];cfg.DATASET.CONTENT_SCENES=[];cfg.DATASET.SCENES_DIR='/datasets/mp3d'
    ds=SAVNCE_Dataset(cfg.DATASET);ep=next(e for e in ds.episodes if str(e.episode_id)==row['source_episode_id'])
    # In-memory diagnostic schedule. Source/instance/WAV/position remain original.
    ep.goals[0].onset=0;ep.goals[0].offset=1;ep.goals[0].duration=1
    ep.start_position=row['base_position'];ep.start_rotation=[0,0,0,1]
    c=cfg.SIMULATOR;c.AGENT_0.SENSORS=[];c.AUDIO.MIC_TYPE='ambisonics';c.AUDIO.THREADS=4
    c.AUDIO.EVERLASTING=False;c.AUDIO.HAS_DISTRACTOR_SOUND=False;c.AUDIO.HAS_NOISE_SOUND=False
    c.AUDIO.SOURCE_SOUND_DIR=str(Path('data/sounds/semantic_splits').resolve())
    merge_sim_episode_config(c,ep);cfg.freeze()
    sim=SAVNCE_Simulator(c);sim.seed(7);sim.reconfigure(c);sim.reset()
    errors=[];signals=[];tail_rms=[];active=[]
    reference=np.zeros((16000*8,4),dtype=np.float64)
    try:
        sensor=sim._sim.get_agent(0)._sensors[sim._audio_goal_sensor_uuid]
        receiver=np.asarray(sensor.node.absolute_translation)
        np.testing.assert_allclose(receiver,row['receiver_position'],atol=1e-5)
        for step in range(12):
            n=sim._num_samples_per_step;offset=sim._audio_index
            ir=np.asarray(sim._prev_sim_obs[sim._audio_goal_sensor_uuid]).T
            isactive=sim._sound_onset_step<=sim._episode_step_count<sim._sound_offset_step
            if isactive:
                start=sim._current_sound_sample_index
                ids=(start+np.arange(n))%len(sim.current_source_sound)
                chunk=sim.current_source_sound[ids]
                wet=fftconvolve(chunk[:,None],ir,axes=0)
                reference[offset:offset+len(wet)]+=wet
            observed=sim.get_current_audiogoal_observation().copy()
            expected=reference[offset:offset+n].astype(np.float32)
            np.testing.assert_allclose(observed,expected,atol=2e-6,rtol=1e-4)
            errors.append(float(np.max(np.abs(observed-expected))));signals.append(float(np.sqrt(np.mean(observed**2))));active.append(bool(isactive))
            if not isactive:tail_rms.append(signals[-1])
            # Rotations keep the base fixed; RIR is taken from the actual native step.
            sim.step(2)
        assert any(active) and any(not b for b in active) and any(v>1e-9 for v in tail_rms)
    finally:sim._sim.close()
    result=dict(status='passed',native_simulator_executed=True,steps=12,dt_s=float(c.STEP_TIME),
        source_episode_id=row['source_episode_id'],sound_id=row['source_sound_id'],source_position=row['source_position'],receiver_position=receiver.tolist(),
        max_native_audio_vs_reference_error=max(errors),rms_by_step=signals,source_active_by_step=active,
        post_offset_tail_present=True,original_episodes_modified=False,
        scope='Actual SAVNCE_Simulator audio observation compared with independently accumulated convolutions of its current source chunk and actual step RIR; verifies history/tail and source offset. Dataset uses fixed RIR and explicit non-looped crops/global gain, not native source-normalization/periodic playback.')
    write_json(a.output,result);print(result,flush=True)
if __name__=='__main__':main()
