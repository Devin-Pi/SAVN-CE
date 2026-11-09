"""Stationary phase must contain four fresh audio ticks before policy access."""
import copy
import unittest
from test_synchronous_control import Harness,observation


class StationaryHarness(Harness):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.acquisitions=[];self.windows=[];self.listens=[]
    def listen(self,cycle):
        pose=copy.deepcopy(self.obs['pose']);self.obs=observation(self.obs['step']+1);self.obs['pose']=pose
        self.listens.append(self.obs['step'])
        return self.obs,self.obs['step']>=self.budget
    def aggregate(self,obs,window,data):
        self.windows.append(copy.deepcopy(window));return data
    def run(self,**kwargs):
        return super().run(listen_advance=self.listen,after_listen=self.aggregate,
                           log_listening=self.acquisitions.append,**kwargs)


class StationaryTests(unittest.TestCase):
    def test_listen_then_decide_then_move_and_repeat(self):
        h=StationaryHarness(budget=16);obs,cycles=h.run()
        self.assertEqual((obs['step'],cycles),(16,2))
        self.assertEqual(h.listens,[1,2,3,4,9,10,11,12])
        self.assertEqual([x['summary']['observation_id'] for x in h.inputs],['obs_0004','obs_0012'])
        self.assertEqual(h.reads,[4,4,12,12])
        self.assertEqual([[x['step'] for x in w] for w in h.windows],[[1,2,3,4],[9,10,11,12]])
        for window in h.windows:
            self.assertTrue(all(x['pose']==window[0]['pose'] for x in window))
        self.assertEqual(len(h.steps),8)
        self.assertEqual(h.inputs[0]['summary']['control']['mode'],'stationary_listen_v1')
    def test_partial_movement_budget_is_separate_from_listening(self):
        h=StationaryHarness(budget=6);_,cycles=h.run()
        self.assertEqual((len(h.listens),len(h.steps),cycles),(4,2,1))
        self.assertEqual(h.inputs[0]['summary']['control']['max_native_steps'],2)
    def test_partial_listening_never_reaches_model_or_localizer(self):
        h=StationaryHarness(budget=3);_,cycles=h.run()
        self.assertEqual((cycles,len(h.inputs),len(h.windows),len(h.steps)),(0,0,0,0))
        self.assertFalse(h.acquisitions[-1]['complete'])
        self.assertEqual(len(h.listens),3)
    def test_terminal_full_window_can_be_logged_without_policy_action(self):
        h=StationaryHarness(budget=4);_,cycles=h.run()
        self.assertEqual((cycles,len(h.windows),len(h.inputs)),(0,1,0))
    def test_stop_after_listening_does_not_move_or_listen_again(self):
        h=StationaryHarness(budget=16,action='stop');obs,cycles=h.run()
        self.assertEqual((obs['step'],cycles,len(h.listens),len(h.steps)),(4,1,4,1))
        self.assertEqual(h.events[-1]['advanced_sim_time_s'],0)
    def confirmation_run(self,budget):
        h=StationaryHarness(budget=budget,action='stop')
        def aggregate(obs,window,data):
            h.aggregate(obs,window,data)
            return dict(data,stop_gate={'request_stationary_confirmation':obs['step']==4,'ready':obs['step']==8})
        obs,cycles=Harness.run(h,listen_advance=h.listen,after_listen=aggregate,log_listening=h.acquisitions.append)
        return h,obs,cycles
    def test_near_window_defers_agent_until_new_stationary_window(self):
        h,obs,cycles=self.confirmation_run(16)
        self.assertEqual((obs['step'],cycles,len(h.steps)),(8,1,1))
        self.assertEqual([a['purpose'] for a in h.acquisitions[1::2]],['decision','stop_confirmation'])
        control=h.inputs[0]['summary']['control']
        self.assertEqual(control['first_near_anchor'],'obs_0004')
        self.assertEqual(control['listening_observation_ids'],[f'obs_{i:04d}' for i in range(5,9)])
    def test_confirmation_cannot_exceed_native_budget(self):
        h,obs,cycles=self.confirmation_run(6)
        self.assertEqual((obs['step'],cycles,len(h.inputs),len(h.windows)),(6,0,0,1))
        self.assertFalse(h.acquisitions[-1]['complete'])
    def test_pose_drift_in_listening_fails_before_policy(self):
        h=StationaryHarness()
        def move(cycle):
            obs,done=h.listen(cycle);obs['pose']['position'][0]+=.01;return obs,done
        # Call the base harness to override the acquisition callback.
        with self.assertRaisesRegex(RuntimeError,'pose changed'):
            Harness.run(h,listen_advance=move,after_listen=h.aggregate,log_listening=h.acquisitions.append)
        self.assertEqual(h.inputs,[])


if __name__=='__main__':unittest.main()
