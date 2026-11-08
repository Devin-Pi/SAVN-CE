"""One-second acquisition, no future support, current-step pose and cache guards."""
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'agent_episode'))
from localization_tool import LocalizationTool
from short_candidate_worker import ShortCandidate
from test_dsf_localization import observation


def short_worker(request):
    if request['command']=='describe':
        return dict(status='ok',model_variant='synthetic1s',checkpoint_sha256='test',source_sha256={})
    assert len(request['windows_npy_base64'])==4
    return dict(status='ok',raw_accdoa=[[0.]*189],events_by_frame=[[dict(
        direction_unit_xyz=[1.,0.,0.],class_name='test',azimuth_deg=0,elevation_deg=0,activity_score=1)]],
        timing_s={},peak_cuda_allocated_mib=0)


class ShortTool(unittest.TestCase):
    def test_warmup_current_step_cache_and_no_future(self):
        request=Mock(side_effect=short_worker);tool=LocalizationTool(request,profile='avn_foa_short1s');tool.reset_episode('e')
        for i in range(3):
            tool.add_observation(observation(i))
            self.assertEqual(tool.analyze([f'obs_{i:04d}'])['status'],'not_ready')
        self.assertEqual(request.call_count,0)
        tool.add_observation(observation(3));r=tool.analyze(['obs_0003'])
        self.assertEqual(r['provenance']['support_sample_interval'],[0,16000])
        self.assertEqual(r['timing']['current_step_interval_s'],[.75,1.])
        self.assertEqual(r['timing']['frame_times_s'],[.75])
        self.assertEqual(r['events'][0][0]['pose_ref'],'obs_0003')
        self.assertEqual(r['events'][0][0]['orientation_pose_ref'],'obs_0003')
        self.assertEqual(r['input_format']['model_sample_rate_hz'],16000)
        self.assertIsNone(r['input_format']['resampling'])
        self.assertTrue(tool.analyze(['obs_0003'])['provenance']['cache_hit'])
        self.assertEqual(tool.analyze(['obs_0004'])['status'],'unsupported_input')
        self.assertFalse(tool.add_observation(observation(3)))
        tool.add_observation(observation(4));next_result=tool.analyze(['obs_0004'])
        self.assertEqual(next_result['provenance']['observation_ids'],['obs_0001','obs_0002','obs_0003','obs_0004'])
        self.assertEqual(next_result['timing']['current_step_interval_s'],[1.,1.25])
        self.assertEqual(next_result['provenance']['support_sample_interval'],[4000,20000])
        # Old anchors cannot borrow the newly observed future block.
        self.assertTrue(tool.analyze(['obs_0003'])['provenance']['cache_hit'])
        self.assertEqual(tool.analyze(['obs_0003','obs_0004'])['status'],'unsupported_input')
    def test_wrong_rate_and_shape_rejected(self):
        candidate=object.__new__(ShortCandidate)
        for wave,rate in ((np.zeros((4000,4),np.float32),16000),
                          (np.zeros((16000,4),np.float32),24000),
                          (np.zeros((16000,2),np.float32),16000),
                          (np.full((16000,4),np.nan,np.float32),16000)):
            with self.assertRaises(ValueError):candidate.predict(wave,rate,[0,0,0,1])
        tool=LocalizationTool(short_worker,profile='avn_foa_short1s');tool.reset_episode('e')
        tool.add_observation(observation(0,rate=24000))
        self.assertEqual(tool.analyze(['obs_0000'])['status'],'unsupported_input')
    def test_episode_reset_removes_history(self):
        tool=LocalizationTool(short_worker,profile='avn_foa_short1s');tool.reset_episode('one')
        for i in range(4):tool.add_observation(observation(i))
        self.assertEqual(tool.analyze(['obs_0003'])['status'],'ok')
        tool.reset_episode('two')
        self.assertEqual(tool.analyze(['obs_0003'])['status'],'unsupported_input')
        tool.add_observation(observation(0))
        self.assertEqual(tool.analyze(['obs_0000'])['status'],'not_ready')


if __name__=='__main__':unittest.main()
