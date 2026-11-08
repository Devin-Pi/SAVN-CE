"""Checks for real dense activity, full convolution, source provenance and silence."""
import unittest
import numpy as np
from scipy.signal import fftconvolve
from .dense_synthesis import synthesize_dense,reconstruct_timeline,active_regions


class DenseTests(unittest.TestCase):
    def setUp(self):
        # Source itself is only 12% active, with short real pulses.
        self.dry=np.zeros(16000*3,np.float32)
        t=np.arange(1920,dtype=np.float32)/16000
        for start in (1600,17600,33600):self.dry[start:start+len(t)]=.2*np.sin(2*np.pi*420*t)
        self.ir=np.zeros((8000,4),np.float32);self.ir[48]=[1,.4,.1,.7];self.ir[5000]=[.25,-.1,0,.1]
    def test_sparse_source_becomes_measured_dense_not_padded_labels(self):
        wave,label,row=synthesize_dense(self.dry,self.ir,np.array([1.,0,0]),2,21,np.random.default_rng(7))
        self.assertGreaterEqual(int(label['activity'].sum()),92)
        self.assertGreater(len(row['events']),10)
        self.assertFalse(np.any(label['activity']&label['ignore_mask']))
        dry=reconstruct_timeline(self.dry,row['events'])
        recovered=fftconvolve(dry[:,None],self.ir,axes=0)[:160000].astype(np.float32)*row['global_gain']
        np.testing.assert_allclose(recovered,wave,atol=1e-7,rtol=1e-5)
        rms=np.sqrt(np.mean(wave.reshape(100,1600,4)**2,axis=(1,2)))
        self.assertTrue(np.all(rms[label['activity'].astype(bool)]>row['wet_audibility_rms_floor']))
        self.assertEqual(np.count_nonzero(label['adpit'][:,:,0,:]),int(label['activity'].sum()))
    def test_seed_and_trace_reproducibility(self):
        a=synthesize_dense(self.dry,self.ir,np.array([1.,0,0]),2,21,np.random.default_rng(7))
        b=synthesize_dense(self.dry,self.ir,np.array([1.,0,0]),2,21,np.random.default_rng(7))
        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[2],b[2])
        for key in a[1]:np.testing.assert_array_equal(a[1][key],b[1][key])
        for event in a[2]['events']:
            self.assertLessEqual(event['source_end_sample_16k'],len(self.dry))
            self.assertGreaterEqual(event['source_start_sample_16k'],0)
    def test_true_negative_has_no_activity_or_ignored_frames(self):
        wave,label,row=synthesize_dense(self.dry,self.ir,np.array([1.,0,0]),2,21,np.random.default_rng(7),negative=True)
        self.assertFalse(wave.any());self.assertFalse(label['adpit'].any());self.assertFalse(label['ignore_mask'].any());self.assertFalse(row['events'])
    def test_silent_source_cannot_be_fabricated_into_positive(self):
        with self.assertRaises(ValueError):active_regions(np.zeros(32000,np.float32))
        with self.assertRaises(ValueError):synthesize_dense(np.zeros(32000,np.float32),self.ir,np.array([1.,0,0]),2,21,np.random.default_rng(7))


if __name__=='__main__':unittest.main()
