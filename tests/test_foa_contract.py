"""Frozen 28-case fixture and shared generation/deployment conversion invariants."""
import json,math,unittest
from pathlib import Path
import numpy as np
from spatial_audio.foa import native_to_listener,native_to_world,world_foa_to_listener,CONTRACT_VERSION
from avn_foa_sameroom.common import native_to_local


class SharedFOA(unittest.TestCase):
    def test_frozen28(self):
        d=json.loads((Path(__file__).parent/'fixtures/foa_28_calibrated_build.json').read_text())
        self.assertEqual(len(d['cases']),28)
        for c in d['cases']:
            yaw=c['yaw_rad'];q=[0,math.sin(yaw/2),0,math.cos(yaw/2)]
            native=np.asarray([c['native_w_normalized']],np.float32)
            actual=native_to_listener(native,q)[0]
            np.testing.assert_allclose(actual,c['expected'],atol=3e-7,rtol=0)
            np.testing.assert_array_equal(native_to_local(native,yaw)[0],actual)
            np.testing.assert_array_equal(world_foa_to_listener(native_to_world(native),q)[0],actual)
    def test_declared_repeated_rotation_rejected(self):
        with self.assertRaises(ValueError):world_foa_to_listener(np.ones((10,4)),[0,0,0,1],coordinate_frame='listener_anchor')
        with self.assertRaises(ValueError):world_foa_to_listener(np.ones((10,4)),[0,0,0,1],normalization='N3D')
        with self.assertRaises(ValueError):native_to_world(np.ones((10,4)),native_normalization='SN3D')
    def test_wave_gain_and_normalization(self):
        native=np.array([[2,0,0,2*np.sqrt(3)]],dtype=np.float32)
        np.testing.assert_allclose(native_to_world(native),[[2,-2,0,0]],atol=2e-7)
        self.assertEqual(CONTRACT_VERSION,'savnce-rlr-foa-1.0')

if __name__=='__main__':unittest.main()
