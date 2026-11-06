import argparse
import json
import tempfile
from pathlib import Path

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
import numpy as np
import trimesh

from agent_episode.foa import to_ambix


def measure(mesh):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(mesh)
    sim_cfg.create_renderer = False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [habitat_sim.agent.AgentConfiguration()]))
    try:
        spec = habitat_sim.AudioSensorSpec()
        spec.uuid = 'foa_calibration'
        spec.enableMaterials = False
        spec.position = [0.0, 0.0, 0.0]
        spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics
        spec.channelLayout.channelCount = 4
        ac = spec.acousticsConfig
        ac.sampleRate = 16000
        ac.threadCount = 1
        ac.direct = True
        ac.indirect = ac.diffraction = ac.transmission = ac.temporalCoherence = False
        sim.add_sensor(spec)
        agent = sim.get_agent(0)
        sensor = agent._sensors[spec.uuid]
        axes = np.eye(3)
        directions = np.concatenate((axes, -axes, np.array([[1, 2, 3], [-2, 3, -1]]) / np.sqrt(14)))
        cases = []
        for yaw in (0, 90, 180):
            angle = np.deg2rad(yaw)
            state = habitat_sim.AgentState()
            state.position = np.zeros(3)
            state.rotation = quat_from_angle_axis(angle, axes[1])
            agent.set_state(state)
            for direction in directions:
                sensor.setAudioSourceTransform(2 * direction)
                rir = np.asarray(sim.get_sensor_observations()[spec.uuid], dtype=float)
                assert rir.ndim == 2 and rir.shape[0] == 4 and np.isfinite(rir).all()
                peak = np.linalg.norm(rir, axis=0).argmax()
                assert 60 < peak < 140, (yaw, direction, peak)
                x, y, z = direction
                native = rir[:, peak]
                assert native[0] > 0, native
                expected = np.array([1, -x, y, -z])
                converted = to_ambix(native[None, :])[0] / native[0]
                np.testing.assert_allclose(converted, expected, atol=1e-5, rtol=0)
                cases.append(dict(
                    yaw_deg=yaw, source_world_direction=direction.tolist(), peak_sample=int(peak),
                    native_relative_to_w=(native/native[0]).tolist(), ambix_relative_to_w=converted.tolist(),
                    expected_ambix=expected.tolist(), max_error=float(np.max(np.abs(converted-expected))),
                ))
        return cases
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        mesh = Path(directory) / 'room.glb'
        room = trimesh.creation.box(extents=[20, 20, 20])
        room.invert()
        room.export(mesh)
        cases = measure(mesh)
    result = json.dumps(dict(
        sample_rate_hz=16000, distance_m=2, direct_only=True, case_count=len(cases),
        max_abs_error=max(row['max_error'] for row in cases), cases=cases,
    ), indent=2) + '\n'
    if args.output:
        args.output.write_text(result)
    else:
        print(result)


if __name__ == '__main__':
    main()
