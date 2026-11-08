"""Direction/frame and acquisition guards; real checkpoint parity is checked separately."""
import base64,io,math,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'agent_episode'))
from spatial_frames import world_foa_to_listener,listener_to_habitat
from localization_tool import LocalizationTool
from test_dsf_localization import observation,worker
from unittest.mock import Mock


class CandidateFrames(unittest.TestCase):
    def test_identity_all_axes(self):
        x=np.eye(4,dtype=np.float32)
        np.testing.assert_array_equal(world_foa_to_listener(x,[0,0,0,1]),x)
        for v,h in [([1,0,0],[0,0,-1]),([0,1,0],[-1,0,0]),([0,0,1],[0,1,0])]:
            np.testing.assert_allclose(listener_to_habitat(v,[0,0,0,1]),h,atol=1e-7)
    def test_yaw_sign_energy_and_roundtrip(self):
        rng=np.random.default_rng(7);wave=rng.normal(size=(100,4)).astype(np.float32)
        for angle in (math.pi/2,math.pi,.73,-math.pi/2):
            q=[0,math.sin(angle/2),0,math.cos(angle/2)]
            local=world_foa_to_listener(wave,q);c,s=math.cos(angle),math.sin(angle)
            np.testing.assert_allclose(local[:,3],c*wave[:,3]+s*wave[:,1],atol=3e-7)
            np.testing.assert_allclose(local[:,1],-s*wave[:,3]+c*wave[:,1],atol=3e-7)
            np.testing.assert_array_equal(local[:,0],wave[:,0])
            np.testing.assert_allclose(np.sum(local[:,1:]**2,axis=1),np.sum(wave[:,1:]**2,axis=1),rtol=1e-6,atol=2e-6)
            np.testing.assert_allclose(world_foa_to_listener(local,[0,-q[1],0,q[3]]),wave,atol=4e-7)
        front=np.array([[1,0,0,1]],np.float32)
        local=world_foa_to_listener(front,[0,math.sqrt(.5),0,math.sqrt(.5)])
        np.testing.assert_allclose(local,[[1,-1,0,0]],atol=1e-7)
        np.testing.assert_allclose(listener_to_habitat([0,-1,0],[0,math.sqrt(.5),0,math.sqrt(.5)]),[0,0,-1],atol=1e-7)
    def test_invalid_pose_rejected(self):
        for q in ([0,0,0,0],[0,float('nan'),0,1]):
            with self.assertRaises(ValueError):world_foa_to_listener(np.ones((10,4)),q)
    def test_acquisition_anchor_and_no_future(self):
        request=Mock(side_effect=worker);tool=LocalizationTool(request,profile='avn_foa_candidate');tool.reset_episode('e')
        for i in range(39):tool.add_observation(observation(i))
        self.assertEqual(tool.analyze(['obs_0038'])['status'],'not_ready');self.assertEqual(request.call_count,0)
        q=[0,math.sqrt(.5),0,math.sqrt(.5)];o=observation(39);o['pose']['rotation_xyzw']=q;tool.add_observation(o)
        r=tool.analyze(['obs_0039']);self.assertEqual(request.call_args.args[0]['anchor_rotation_xyzw'],q)
        self.assertEqual(r['events'][0][0]['orientation_pose_ref'],'obs_0039')
        np.testing.assert_allclose(r['events'][0][0]['direction_unit_habitat_xyz'],[0,0,1],atol=1e-7)
        self.assertEqual(r['input_format']['model_sample_rate_hz'],16000);self.assertIsNone(r['input_format']['resampling'])
        self.assertEqual(r['timing']['audio_end_s'],10)
        self.assertTrue(tool.analyze(['obs_0039'])['provenance']['cache_hit'])
        self.assertEqual(tool.analyze(['obs_0040'])['status'],'unsupported_input')
        tool.add_observation(observation(40))
        self.assertEqual(tool.analyze(['obs_0039','obs_0040'])['status'],'unsupported_input')
    def test_native16k_only(self):
        tool=LocalizationTool(worker,profile='avn_foa_candidate');tool.reset_episode('e')
        tool.add_observation(observation(0,rate=24000));self.assertEqual(tool.analyze(['obs_0000'])['status'],'unsupported_input')

if __name__=='__main__':unittest.main()
