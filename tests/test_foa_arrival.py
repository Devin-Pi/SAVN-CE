import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'agent_episode'))
from foa_arrival import FoaArrival,policy_evidence


def result(index,origin,source=(0,1.5,0),label='synthetic'):
    ids=[f'obs_{index*8+i:04d}' for i in range(1,5)]
    p=np.asarray(origin,float);ray=np.asarray(source,float)-p;ray/=np.linalg.norm(ray)
    poses=[dict(observation_id=i,receiver_position_habitat_xyz_m=list(origin),robot_pose=dict(rotation_xyzw=[0,0,0,1])) for i in ids]
    return dict(status='ok',provenance=dict(observation_ids=ids,episode_ref='synthetic_episode'),pose_refs=poses,
        timing=dict(audio_start_s=index*2+.25,audio_end_s=index*2+1.25),
        events=[[dict(class_name=label,pose_ref=ids[-1],orientation_pose_ref=ids[-1],direction_unit_habitat_xyz=ray.tolist())]])


POSITIONS=[(-2,1.5,0),(1,1.5,1),(1,1.5,-1),(0,1.5,-.4),(0,1.5,-.25)]


class FoaArrivalTests(unittest.TestCase):
    def test_behind_camera_can_stop_without_any_depth_or_back_hypothesis(self):
        gate=FoaArrival();reports=[gate.update(result(i,p),audio_active=True) for i,p in enumerate(POSITIONS)]
        self.assertFalse(reports[-2]['ready'])
        self.assertTrue(reports[-1]['ready'],reports[-1])
        self.assertLessEqual(reports[-1]['estimated_range_interval_m'][1],1.+1e-6)
        self.assertNotIn('front',reports[-1]);self.assertNotIn('back',reports[-1])
        self.assertNotIn('inferred_source_world_xyz',policy_evidence(reports[-1]))
    def test_same_position_turns_do_not_create_range_evidence(self):
        gate=FoaArrival()
        for i in range(5):r=gate.update(result(i,(0,1.5,-.4)),audio_active=True)
        self.assertFalse(r['ready']);self.assertFalse(r['checks']['distinct_positions'])
    def test_parallel_approach_cannot_infer_range(self):
        gate=FoaArrival()
        for i,z in enumerate((-3,-2,-1,-.7,-.4)):r=gate.update(result(i,(0,1.5,z)),audio_active=True)
        self.assertFalse(r['ready']);self.assertFalse(r['checks'].get('parallax',False))
    def test_far_source_is_not_accepted(self):
        gate=FoaArrival()
        for i,p in enumerate(POSITIONS):r=gate.update(result(i,p,source=(4,1.5,0)),audio_active=True)
        self.assertFalse(r['ready'])
    def test_wrong_signed_ray_is_not_mirrored_to_fit(self):
        gate=FoaArrival()
        for i,p in enumerate(POSITIONS[:-1]):gate.update(result(i,p),audio_active=True)
        bad=result(4,POSITIONS[-1]);event=bad['events'][0][0];event['direction_unit_habitat_xyz']=(-np.array(event['direction_unit_habitat_xyz'])).tolist()
        self.assertFalse(gate.update(bad,audio_active=True)['ready'])
    def test_cached_window_cannot_add_confirmation(self):
        gate=FoaArrival()
        for i,p in enumerate(POSITIONS[:4]):r=gate.update(result(i,p),audio_active=True)
        cached=gate.update(result(3,POSITIONS[3]),audio_active=True)
        self.assertFalse(cached['ready']);self.assertEqual(cached['confirmation_count'],r['confirmation_count'])
    def test_missing_multiple_events_and_class_switch_fail_closed(self):
        for kind in ('silence','multiple','class'):
            gate=FoaArrival()
            for i,p in enumerate(POSITIONS[:-1]):gate.update(result(i,p),audio_active=True)
            r=result(4,POSITIONS[-1])
            if kind=='multiple':r['events'][0].append(copy.deepcopy(r['events'][0][0]))
            if kind=='class':r['events'][0][0]['class_name']='other'
            self.assertFalse(gate.update(r,audio_active=kind!='silence')['ready'])
    def test_moving_window_and_old_pose_are_rejected(self):
        for kind in ('moving','pose'):
            r=result(0,POSITIONS[0])
            if kind=='moving':r['pose_refs'][0]['receiver_position_habitat_xyz_m'][0]+=.1
            else:r['events'][0][0]['pose_ref']='old'
            self.assertFalse(FoaArrival().update(r,audio_active=True)['ready'])
    def test_new_episode_cannot_reuse_previous_arrival(self):
        gate=FoaArrival()
        for i,p in enumerate(POSITIONS):r=gate.update(result(i,p),audio_active=True)
        self.assertTrue(r['ready'])
        fresh=result(4,POSITIONS[-1]);fresh['provenance']['episode_ref']='another_episode'
        self.assertFalse(gate.update(fresh,audio_active=True)['ready'])
    def test_partially_overlapping_windows_cannot_build_support(self):
        gate=FoaArrival();first=result(0,POSITIONS[0]);gate.update(first,audio_active=True)
        next_window=result(1,POSITIONS[1]);next_window['provenance']['observation_ids'][0]=first['provenance']['observation_ids'][-1]
        r=gate.update(next_window,audio_active=True)
        self.assertFalse(r['ready']);self.assertIn('Overlapping',r['reason'])


class GeometryArrivalTests(unittest.TestCase):
    def first_near(self,tolerance=None):
        gate=FoaArrival(rule='foa_geometry',tolerance_deg=tolerance)
        for i,pose in enumerate(POSITIONS[:4]):
            report=gate.update(result(i,pose,label=f'class_{i}'),audio_active=True)
        self.assertEqual(report['config']['angular_tolerance_deg'],2. if tolerance is None else tolerance)
        self.assertTrue(report['request_stationary_confirmation'],report)
        self.assertFalse(report['ready'])
        self.assertEqual(len(set(report['supporting_classes'])),4)
        self.assertNotIn('feasible_xz_samples',policy_evidence(report))
        return gate

    def test_geometry_confirmation_allows_class_change(self):
        gate=self.first_near()
        again=result(4,POSITIONS[3],label='different_again')
        again['timing'].update(audio_start_s=7.25,audio_end_s=8.25)
        report=gate.update(again,audio_active=True)
        self.assertTrue(report['ready'],report)
        self.assertEqual(report['confirmation_count'],2)

    def test_confirmation_needs_same_pose_and_consecutive_window(self):
        for origin,delta in ((POSITIONS[4],0.),(POSITIONS[3],.25)):
            gate=self.first_near()
            again=result(4,origin)
            again['timing'].update(audio_start_s=7.25+delta,audio_end_s=8.25+delta)
            self.assertFalse(gate.update(again,audio_active=True)['ready'])

    def test_signed_outlier_is_excluded(self):
        gate=FoaArrival(rule='foa_geometry')
        for i,pose in enumerate(POSITIONS[:4]):
            packet=result(i,pose)
            if i==0:
                event=packet['events'][0][0]
                event['direction_unit_habitat_xyz']=(-np.array(event['direction_unit_habitat_xyz'])).tolist()
            report=gate.update(packet,audio_active=True)
        self.assertTrue(report['request_stationary_confirmation'],report)
        self.assertEqual(len(report['support_ids']),3)
        self.assertNotIn('obs_0004',report['support_ids'])

    def test_one_active_miss_can_be_followed_by_fresh_confirmation(self):
        gate=self.first_near(5.)
        missed=result(4,POSITIONS[3])
        missed.update(events=[[]],timing=dict(audio_start_s=7.25,audio_end_s=8.25))
        retry=gate.update(missed,audio_active=True)
        self.assertTrue(retry['request_stationary_retry'])
        self.assertFalse(retry['ready'])
        self.assertEqual(retry['confirmation_count'],1)
        recovered=result(5,POSITIONS[3],label='new_label')
        recovered['timing'].update(audio_start_s=8.25,audio_end_s=9.25)
        ready=gate.update(recovered,audio_active=True)
        self.assertTrue(ready['ready'],ready)
        self.assertEqual(ready['confirmation_reference_anchor'],'obs_0028')
        self.assertNotIn('obs_0036',ready['support_ids'])

    def test_silence_motion_or_second_miss_cannot_keep_confirmation(self):
        for mode in ('silence','motion','miss_twice'):
            with self.subTest(mode=mode):
                gate=self.first_near()
                origin=POSITIONS[4] if mode=='motion' else POSITIONS[3]
                missed=result(4,origin)
                missed.update(events=[[]],timing=dict(audio_start_s=7.25,audio_end_s=8.25))
                first=gate.update(missed,audio_active=mode!='silence')
                self.assertEqual(bool(first.get('request_stationary_retry')),mode=='miss_twice')
                third=result(5,origin)
                third['timing'].update(audio_start_s=8.25,audio_end_s=9.25)
                if mode=='miss_twice':third['events']=[[]]
                last=gate.update(third,audio_active=True)
                self.assertFalse(last['ready'])
                self.assertFalse(last.get('request_stationary_retry',False))


if __name__=='__main__':unittest.main()
