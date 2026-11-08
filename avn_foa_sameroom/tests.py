"""Focused tests for source silence, boundary masks, convolution history and coordinates."""
import unittest
import numpy as np
from scipy.signal import fftconvolve
from .common import native_to_local,local_unit
from .geometry import polygon_relation
from .synthesis import synthesize


class Tests(unittest.TestCase):
    def test_concave_room_and_boundary(self):
        polygon=[[0,0],[3,0],[3,1],[1,1],[1,3],[0,3]]
        self.assertTrue(polygon_relation([.5,2],polygon))
        self.assertFalse(polygon_relation([2,2],polygon)) # bbox would incorrectly pass
        self.assertIsNone(polygon_relation([1,2],polygon))

    def test_world_audio_rotation(self):
        # Source world front becomes listener-right after left 90deg turn.
        native=np.array([[1.,0.,-np.sqrt(3),0]],dtype=np.float32)
        result=native_to_local(native,np.pi/2)[0]
        np.testing.assert_allclose(result,[1,-1,0,0],atol=1e-6)
        np.testing.assert_allclose(local_unit([0,0,-2],np.zeros(3),np.pi/2),[0,-1,0],atol=1e-6)

    def test_complete_tail_and_silence_labels(self):
        dry=np.zeros(16000*12,dtype=np.float32);dry[::16000]=1
        dry=np.convolve(dry,np.ones(1600,dtype=np.float32),mode='same').astype(np.float32)
        ir=np.zeros((8000,4),dtype=np.float32);ir[48]=[1,.5,0,.5];ir[6500]=[.5,-.2,0,.1]
        wave,labels,row=synthesize(dry,ir,np.array([1.,0,0]),0,2,np.random.default_rng(7))
        self.assertGreater(row['ignore_frames'],0)
        self.assertGreater(labels['activity'].sum(),0)
        self.assertLess(labels['activity'].sum(),100)
        self.assertFalse(np.any(labels['activity']&labels['ignore_mask']))
        # Chunked source overlap-add must equal one uninterrupted convolution.
        source=np.random.default_rng(5).normal(size=32000).astype(np.float32)
        full=fftconvolve(source[:,None],ir,axes=0);ola=np.zeros_like(full)
        for start in range(0,len(source),4000):
            wet=fftconvolve(source[start:start+4000,None],ir,axes=0)
            ola[start:start+len(wet)]+=wet
        np.testing.assert_allclose(full,ola,atol=1e-5,rtol=1e-4)

    def test_same_seed_source_schedule_is_reproducible(self):
        dry=np.random.default_rng(2).normal(0,.1,16000*12).astype(np.float32)
        ir=np.zeros((1000,4),np.float32);ir[50]=[1,.2,.1,.5]
        a=synthesize(dry,ir,np.array([1.,0,0]),0,2,np.random.default_rng(7))
        b=synthesize(dry,ir,np.array([1.,0,0]),0,2,np.random.default_rng(7))
        np.testing.assert_array_equal(a[0],b[0])
        self.assertEqual(a[2],b[2])
        for key in a[1]:np.testing.assert_array_equal(a[1][key],b[1][key])

    def test_noevent_is_true_silence(self):
        ir=np.zeros((100,4),np.float32);ir[1]=1
        wave,label,row=synthesize(np.ones(20000),ir,np.array([1,0,0]),0,2,np.random.default_rng(1),True)
        self.assertFalse(wave.any());self.assertFalse(label['adpit'].any());self.assertFalse(label['ignore_mask'].any())

if __name__=='__main__':unittest.main()
