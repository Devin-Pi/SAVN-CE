"""Diagnostic-only native FOA factorial; never edits Episodes or invokes an Agent."""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_audio.foa import native_to_world, CONTRACT_VERSION
from avn_foa_sameroom.common import acoustic_config
from savnce.datasets.savnce_dataset import SAVNCE_Dataset
from savnce.tasks.savnce_task import merge_sim_episode_config
from savnce.sims.savnce_simulator import SAVNCE_Simulator
from habitat_sim.agent import ActionSpec, ActuationSpec
from habitat_sim.utils.common import quat_to_coeffs

EPISODES = 'data/datasets/savnce_generated/mp3d/v1/train/content/17DRP5sb8fy.json.gz'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--actions', type=Path, required=True)
    p.add_argument('--motion', choices=['static', 'motion'], required=True)
    p.add_argument('--schedule', choices=['continuous', 'intermittent'], required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--temporal', choices=['true', 'false'], default='true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    episode_sha = digest(EPISODES)
    cfg = runpy.run_path('savnce_baselines/magnet/config/default.py')['get_task_config']()
    cfg.defrost()
    cfg.DATASET.DATA_PATH = EPISODES
    cfg.DATASET.CONTENT_SCENES = []
    cfg.DATASET.SCENES_DIR = '/datasets/mp3d'
    ep = next(e for e in SAVNCE_Dataset(cfg.DATASET).episodes if str(e.episode_id) == '1000000008')
    official_schedule = dict(onset=ep.goals[0].onset, offset=ep.goals[0].offset)
    # Counterfactual schedule lives only in this process, never in the dataset.
    ep.goals[0].onset = 0
    ep.goals[0].offset = 16
    ep.goals[0].duration = 16
    c = cfg.SIMULATOR
    c.AGENT_0.SENSORS = []
    c.AUDIO.MIC_TYPE = 'ambisonics'
    c.AUDIO.THREADS = 4
    c.AUDIO.EVERLASTING = False
    c.AUDIO.HAS_DISTRACTOR_SOUND = False
    c.AUDIO.HAS_NOISE_SOUND = False
    c.AUDIO.SOURCE_SOUND_DIR = str(Path('data/sounds/semantic_splits').resolve())
    merge_sim_episode_config(c, ep)
    cfg.freeze()
    temporal = a.temporal == 'true'

    class DiagnosticSim(SAVNCE_Simulator):
        def add_acoustic_config(self, mic_type, sensor_uuid):
            # Exact native settings, with this single explicitly registered toggle.
            import habitat_sim as hs
            spec = hs.AudioSensorSpec()
            spec.uuid = sensor_uuid
            spec.channelLayout.type = hs.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics
            spec.channelLayout.channelCount = 4
            self._audio_channel_count = 4
            spec.enableMaterials = False
            spec.acousticsConfig.sampleRate = self._sampling_rate
            spec.acousticsConfig.threadCount = self.config.AUDIO.THREADS
            spec.acousticsConfig.indirectRayCount = 500
            spec.acousticsConfig.temporalCoherence = temporal
            spec.acousticsConfig.transmission = True
            self.diagnostic_acoustics = acoustic_config(spec)
            self._sim.add_sensor(spec)

    replay = [json.loads(l)['action_id'] for l in a.actions.read_text().splitlines()]
    assert len(replay) == 64 and set(replay) <= {1, 2, 3}
    np.random.seed(a.seed)
    sim = DiagnosticSim(c)
    sim.seed(a.seed)
    np.random.seed(a.seed)
    sim.reconfigure(c)
    sim.reset()
    sensor = sim._sim.get_agent(0)._sensors[sim._audio_goal_sensor_uuid]
    sim._sim.get_agent(0).agent_config.action_space[99] = ActionSpec('move_forward', ActuationSpec(amount=0.0))
    waves, drys, poses, rotations, emission, logs, native_irs = [], [], [], [], [], [], []
    initial_source_offset = int(sim._current_sound_sample_index)
    source_path = Path(sim.source_sound_dir) / sim._current_sound
    source_sha = digest(source_path)
    sf.write(str(a.output/'native_normalized_source.wav'), sim.current_source_sound, 16000, subtype='FLOAT')
    try:
        for step in range(64):
            state = sim.get_agent_state()
            q = quat_to_coeffs(state.rotation).tolist()
            enabled = a.schedule == 'continuous' or step % 16 < 8
            # Native gate only; source sample clock continues through silent periods.
            sim._sound_onset_step = 0 if enabled else 500
            sim._sound_offset_step = 500 if enabled else 501
            n = sim._num_samples_per_step
            assert n == 4000 and sim._audio_index == step*n
            ids = (sim._current_sound_sample_index + np.arange(n)) % len(sim.current_source_sound)
            dry = sim.current_source_sound[ids].copy() if enabled else np.zeros(n, np.float32)
            ir = np.asarray(sim._prev_sim_obs[sim._audio_goal_sensor_uuid], dtype=np.float32).T.copy()
            support = np.flatnonzero(np.max(np.abs(ir),axis=1) > max(float(np.max(np.abs(ir)))*1e-5,1e-12))
            assert len(support)
            wave = sim.get_current_audiogoal_observation().copy()
            # Re-observe must not append a second convolution or advance source time.
            np.testing.assert_array_equal(wave, sim.get_current_audiogoal_observation())
            waves.append(native_to_world(wave))
            drys.append(dry)
            poses.append(state.position.tolist())
            rotations.append(q)
            emission.append(enabled)
            native_irs.append(ir)
            action = replay[step] if a.motion == 'motion' else 99
            logs.append(dict(step=step, time_s=step*.25, action_id=action,
                replay_action_id=replay[step], source_enabled=enabled,
                source_sample_index=int(sim._current_sound_sample_index),
                position=poses[-1], rotation_xyzw=q,
                receiver_position=np.asarray(sensor.node.absolute_translation).tolist(),
                rir_onset_samples=int(support[0]), rir_samples=len(ir),
                rms=float(np.sqrt(np.mean(wave.astype(np.float64)**2)))))
            if step < 63:
                sim.step(action)
        world = np.concatenate(waves)
        dry = np.concatenate(drys)
        sf.write(str(a.output/'foa_world.wav'), world, 16000, subtype='FLOAT')
        sf.write(str(a.output/'emitted_source.wav'), dry, 16000, subtype='FLOAT')
        # Actual per-step IR reference isolates chunk convolution arithmetic.
        maxir = max(map(len, native_irs))
        reference = np.zeros((len(dry)+maxir-1, 4), np.float64)
        for i, ir in enumerate(native_irs):
            wet = fftconvolve(drys[i][:, None], ir, axes=0)
            reference[i*4000:i*4000+len(wet)] += wet
        reference_world = native_to_world(reference[:len(dry)].astype(np.float32))
        consistency = float(np.max(np.abs(world-reference_world)))
        assert consistency < 2e-6, consistency
        # Same first RIR, same emission: full vs quarter-second OLA, no state change.
        full = fftconvolve(dry[:, None], native_irs[0], axes=0).astype(np.float32)
        chunked = np.zeros_like(full)
        for i, chunk in enumerate(drys):
            wet = fftconvolve(chunk[:, None], native_irs[0], axes=0)
            chunked[i*4000:i*4000+len(wet)] += wet
        frozen_error = float(np.max(np.abs(full-chunked)))
        assert frozen_error < 2e-6, frozen_error
        sf.write(str(a.output/'frozen_initial_rir_full.wav'), native_to_world(full[:len(dry)]), 16000, subtype='FLOAT')
        np.savez_compressed(a.output/'initial_rir.npz', native=native_irs[0])
        np.savez_compressed(a.output/'poses.npz', positions=np.asarray(poses), rotations=np.asarray(rotations), emission=np.asarray(emission))
        (a.output/'actions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in logs))
        translation = float(np.linalg.norm(np.diff(np.asarray(poses), axis=0), axis=1).sum())
        if a.motion == 'static':
            assert translation == 0
            np.testing.assert_array_equal(rotations, np.repeat(np.asarray(rotations[:1]),64,axis=0))
        summary = dict(status='collected', episode_id='1000000008', seed=a.seed,
            motion=a.motion, schedule=a.schedule, temporalCoherence=temporal,
            duration_s=16, sample_rate_hz=16000, samples=len(world), chunk_s=.25,
            model_window_s=10, analysis_window_starts_s=[0,3,6],
            schedule_definition='continuous [0,16); intermittent [0,2),[4,6),[8,10),[12,14)',
            model_receives_only='FOA waveform and observed anchor quaternion; no labels/positions/source direction',
            navigation_policy='none: prerecorded action replay; hold is zero-distance native move_forward',
            initial_source_offset_samples=initial_source_offset, source_path=str(source_path), source_sha256=source_sha,
            source_id=sim._current_sound, official_schedule_unchanged=official_schedule,
            original_episode_file_sha256=episode_sha, episode_file_unchanged=digest(EPISODES)==episode_sha,
            conversion_contract_version=CONTRACT_VERSION, acoustic_config=sim.diagnostic_acoustics,
            actual_translation_m=translation, native_ola_reference_max_abs_error=consistency,
            frozen_rir_full_vs_chunked_max_abs_error=frozen_error,
            evaluation_only=dict(source_position=(np.asarray(c.AGENT_0.GOAL_POSITION)+[0,1.5,0]).tolist(),
                class_id=0, class_name='chair'),
            truth_used_for_audio_conversion=False, weights_modified=False)
        (a.output/'collection.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary), flush=True)
    finally:
        sim._sim.close()


if __name__ == '__main__':
    main()
