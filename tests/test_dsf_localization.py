"""Synthetic unit/protocol fixtures. Model quality is evaluated separately on retained WAVs."""
import base64
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent_episode'))
from dsf_decode import decode, vector_event
from dsf_worker import DSF
from llm_rollout import ResponsesBackend
from localization_tool import LocalizationTool, compact, MODEL_FORMAT


def observation(index, *, rate=16000, foa=True):
    data = io.BytesIO()
    np.savez_compressed(data, foa_audio=np.zeros((round(rate/4), 4), dtype=np.float32))
    return {'observation_id': f'obs_{index:04d}', 'step': index, 'sample_rate': rate, 'dt_s': .25,
            'timestamp_s': index/4, 'audio_format': MODEL_FORMAT if foa else {},
            'sensor_payload': {'encoding': 'npz_base64', 'data': base64.b64encode(data.getvalue()).decode()},
            'shapes': {'foa_audio': [round(rate/4), 4]}, 'pose': {'position': [index,0,0], 'rotation_xyzw': [0,0,0,1]},
            'audio_receiver_position': [index,1.5,0]}


def worker(request):
    if request['command'] == 'describe':
        return {'status': 'ok', 'model_variant': 'synthetic unit test', 'checkpoint_sha256': 'test', 'source_sha256': {}}
    assert len(request['windows_npy_base64']) == 40
    return {'status': 'ok', 'events_by_frame': [[{'direction_unit_xyz': [0.,1.,0.], 'class_name': 'test', 'azimuth_deg': 90,
                 'elevation_deg': 0, 'activity_score': 1}] for _ in range(100)],
            'raw_accdoa': [], 'timing_s': {}, 'peak_cuda_allocated_mib': 0}


class TestLocalization(unittest.TestCase):
    def setUp(self):
        self.backend = Mock(side_effect=worker)
        self.tool = LocalizationTool(self.backend)
        self.tool.reset_episode('episode-1')

    def test_format_and_references(self):
        self.assertEqual(self.tool.analyze([])['status'], 'unsupported_input')
        self.tool.add_observation(observation(0, rate=44100))
        self.assertEqual(self.tool.analyze(['obs_0000'])['status'], 'unsupported_input')
        self.assertEqual(self.tool.analyze(['obs_9999'])['status'], 'unsupported_input')
        self.tool.reset_episode('episode-2')
        self.assertEqual(self.tool.analyze(['obs_0000'])['status'], 'unsupported_input')
        self.tool.add_observation(observation(0, foa=False))
        self.assertEqual(self.tool.analyze(['obs_0000'])['status'], 'unsupported_input')

    def test_contiguous_time_overlap_cache_and_pose(self):
        for index in range(40):
            self.assertTrue(self.tool.add_observation(observation(index)))
            self.assertFalse(self.tool.add_observation(observation(index)))
        self.assertEqual(self.tool.analyze(['obs_0038'])['status'], 'not_ready')
        first = self.tool.analyze(['obs_0039'])
        self.assertEqual(first['provenance']['support_sample_interval'], [0, 160000])
        self.assertEqual(first['events'][3][0]['pose_ref'], 'obs_0001')
        self.assertEqual(first['events'][0][0]['direction_unit_habitat_xyz'], [-1,0,0])
        self.assertEqual(first['timing']['result_available_at_audio_time_s'], 10)
        self.assertEqual(first['pose_refs'][1]['receiver_position_habitat_xyz_m'], [1,1.5,0])
        hit = self.tool.analyze(first['provenance']['observation_ids'])
        self.assertTrue(hit['provenance']['cache_hit'])
        self.assertFalse(hit['provenance']['new_physical_observation'])
        self.assertEqual(hit['timing']['result_available_unix_s'], first['timing']['result_available_unix_s'])
        self.assertEqual(self.backend.call_count, 2)
        self.tool.add_observation(observation(40))
        overlap = self.tool.analyze(['obs_0040'])
        self.assertEqual(overlap['provenance']['support_sample_interval'], [4000, 164000])
        self.assertFalse(overlap['provenance']['cache_hit'])
        self.assertEqual(self.tool.analyze(['obs_0002','obs_0040'])['status'], 'unsupported_input')
        self.assertEqual(compact(overlap)['evidence_id'], 'obs_0040:localization')
        self.tool.reset_episode('episode-2')
        self.assertEqual(self.tool.analyze(['obs_0039'])['status'], 'unsupported_input')

    def test_unavailable_worker(self):
        self.backend.side_effect = FileNotFoundError('test fixture missing checkpoint')
        for index in range(40):
            self.tool.add_observation(observation(index))
        self.assertEqual(self.tool.analyze(['obs_0039'])['status'], 'unavailable')

    def test_decoder_axis_activity_nan_and_duplicate_rules(self):
        raw = np.zeros((1,3,3,13), dtype=float)
        self.assertEqual(decode(raw.reshape(1,117)), [[]])
        raw[0,0,:,0] = [.5,0,0]
        self.assertEqual(decode(raw.reshape(1,117)), [[]])
        self.assertIsNone(vector_event(0, np.zeros(3), [0], False)['direction_unit_xyz'])
        for vector, angle in [([1,0,0],0), ([0,1,0],90), ([-1,0,0],180), ([0,-1,0],-90)]:
            raw[0,0,:,0] = vector
            self.assertEqual(decode(raw.reshape(1,117))[0][0]['azimuth_deg'], angle)
        raw[0,0,:,0] = [-1,.01,0]
        raw[0,1,:,0] = [-1,-.01,0]
        self.assertEqual(decode(raw.reshape(1,117)), [[]])  # archived evaluator drops lone pair
        raw[0,2,:,0] = [1,0,0]
        self.assertEqual(decode(raw.reshape(1,117))[0][1]['merged_track_slots'], [0,1])
        raw[0,2,:,0] = [-1,0,0]
        self.assertEqual(decode(raw.reshape(1,117))[0][0]['merged_track_slots'], [0,1,2])
        raw[0,2,:,0] = [np.nan,0,0]
        with self.assertRaises(ValueError):
            decode(raw.reshape(1,117))

    def test_worker_rejects_bad_samples_before_inference(self):
        model = object.__new__(DSF)
        for samples, rate in [(np.empty((0,4)),16000), (np.zeros((160000,2)),16000),
                              (np.full((160000,4), np.nan),16000), (np.zeros((160000,4)),22050)]:
            with self.assertRaises(ValueError):
                model.predict(samples, rate)

    def test_responses_roundtrip_registration_and_tool_result(self):
        backend = object.__new__(ResponsesBackend)
        backend.model, backend.effort = 'test', 'low'
        ident = 'obs_0040'
        evidence = {ident+':audio'}
        call = {'type': 'function_call', 'call_id': 'call_test', 'name': 'localization_tool',
                'arguments': json.dumps({'observation_ids': [ident]})}
        action = {'observation_id': ident, 'action': 'turn_left', 'reason': 'test',
                  'evidence_ids': [ident+':localization']}
        seen = []

        def response(payload):
            seen.append(json.loads(json.dumps(payload)))
            if len(seen) == 1:
                return {'output': [call]}
            return {'output': [{'type': 'message', 'content': [
                {'type': 'output_text', 'text': json.dumps(action)}]}]}

        backend.request = response
        hook = Mock(return_value={'status': 'ok', 'evidence_id': ident+':localization'})
        decision, _ = backend.decide({'observation_id': ident}, [], [], ['turn_left'], evidence, None, hook)
        self.assertEqual(decision, action)
        self.assertEqual(seen[0]['tools'][0]['name'], 'localization_tool')
        self.assertEqual(seen[1]['input'][-1]['call_id'], 'call_test')
        self.assertIn(ident+':localization', seen[1]['text']['format']['schema']['properties']['evidence_ids']['items']['enum'])
        hook.assert_called_once_with('call_test', [ident])


if __name__ == '__main__':
    unittest.main()
