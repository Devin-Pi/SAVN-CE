"""Run in savnce_dev: real feature extractor on synthetic fixed-pose blocks."""
import base64,copy,io,unittest
import numpy as np
from agent_episode.llm_observation import stationary_summary


def blocks():
    result=[]
    for i in range(4):
        wave=np.zeros((4000,2),np.float32);wave[:,0]=.01*np.sin(np.arange(4000)*.1+i);wave[:,1]=wave[:,0]
        b=io.BytesIO();np.savez(b,binaural_audio=wave,position=np.array([1.,0.,2.]),rotation_xyzw=np.array([0.,0.,0.,1.]))
        result.append(dict(observation_id=f'obs_{i+1:04d}',step=i+1,timestamp_s=(i+1)*.25,sample_rate=16000,dt_s=.25,
            pose=dict(position=[1.,0.,2.],rotation_xyzw=[0.,0.,0.,1.]),sensor_payload=dict(data=base64.b64encode(b.getvalue()).decode())))
    return result


class StationaryAudioTests(unittest.TestCase):
    def test_full_second_features_and_panel_use_all_four_blocks(self):
        obs=blocks();r=stationary_summary(obs)
        self.assertEqual(r['stationary_audio']['duration_s'],1.)
        self.assertEqual(r['stationary_audio']['audio_interval_s'],[.25,1.25])
        self.assertEqual(r['stationary_audio']['observation_ids'],[x['observation_id'] for x in obs])
        self.assertTrue(r['stationary_audio']['active'])
        self.assertTrue(base64.b64decode(r['audio_png']).startswith(b'\x89PNG'))
    def test_moving_duplicate_and_noncontiguous_blocks_rejected(self):
        for kind in ('pose','id','time','short'):
            obs=blocks()
            if kind=='pose':obs[1]['pose']['position'][0]+=.1
            if kind=='id':obs[1]['observation_id']=obs[0]['observation_id']
            if kind=='time':obs[1]['timestamp_s']+=.25
            if kind=='short':obs.pop()
            with self.assertRaises(ValueError):stationary_summary(obs)


if __name__=='__main__':unittest.main()
