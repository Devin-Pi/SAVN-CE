"""Executable scheduling semantics, without Habitat or external API access."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'agent_episode'))
from synchronous_control import run_synchronous
from codex_backend import CodexBackend


def observation(step):
    return dict(observation_id=f'obs_{step:04d}',step=step,timestamp_s=step*.25,
                pose=dict(position=[step*.25,0,0],rotation_xyzw=[0,0,0,1]),
                collided_previous_step=False,blocked_previous_step=False)


class Harness:
    def __init__(self,budget=8,action='turn_left'):
        self.obs=observation(0);self.budget=budget;self.action=action
        self.steps=[];self.events=[];self.inputs=[];self.reads=[];self.summaries=[]
        self.close_at=None;self.collide_at=None;self.arrive_at=None
    def prepare(self,obs):
        self.summaries.append(obs['step'])
        return dict(summary=dict(observation_id=obs['observation_id'],depth=dict(center=dict(p10_m=.3 if obs['step']==self.close_at else 2.))),stop_gate=dict(ready=obs['step']==self.arrive_at))
    def decide(self,data):
        self.inputs.append(copy.deepcopy(data))
        return dict(observation_id=self.obs['observation_id'],action=self.action,reason='test',evidence_ids=[self.obs['observation_id']+':pose'])
    def advance(self,action,decision):
        self.steps.append((self.obs['step'],action,decision))
        if action=='stop':return self.obs,True
        previous=self.obs['pose']['position'][0]
        self.obs=observation(self.obs['step']+1)
        self.obs['pose']['position'][0]=previous+.25
        self.obs['collided_previous_step']=self.obs['step']==self.collide_at
        return self.obs,self.obs['step']>=self.budget
    def read(self):
        self.reads.append(self.obs['step']);return copy.deepcopy(self.obs)
    def run(self,**overrides):
        kwargs=dict(prepare=self.prepare,decide=self.decide,advance=self.advance,read_current=self.read,
                    before_action=lambda o,d:None,log_event=self.events.append,max_steps=self.budget)
        kwargs.update(overrides)
        return run_synchronous(copy.deepcopy(self.obs),**kwargs)


class SynchronousTests(unittest.TestCase):
    def test_full_period_latest_pose_and_frozen_wait(self):
        h=Harness();final,cycles=h.run()
        self.assertEqual((final['step'],cycles),(8,2))
        self.assertEqual([x['summary']['control']['latest_pose']['position'][0] for x in h.inputs],[0,1])
        self.assertEqual(h.reads,[0,0,4,4])
        self.assertEqual(h.summaries,list(range(9)))
        self.assertEqual([x['advanced_sim_time_s'] for x in h.events if x['event']=='segment_finished'],[1,1])
        self.assertEqual(h.steps[4][2]['observation_id'],'obs_0004')
    def test_partial_final_budget(self):
        h=Harness(budget=6);h.run()
        self.assertEqual([x['summary']['control']['max_native_steps'] for x in h.inputs],[4,2])
        self.assertEqual(len(h.steps),6)
    def test_forward_guard_interrupts_before_next_primitive(self):
        h=Harness(budget=4,action='move_forward');h.close_at=1
        def decide(data):
            h.action='turn_left' if data['summary']['depth']['center']['p10_m']<.45 else 'move_forward'
            return h.decide(data)
        h.run(decide=decide)
        finished=[x for x in h.events if x['event']=='segment_finished']
        self.assertEqual(finished[0]['native_steps'],1)
        self.assertEqual(finished[0]['reason'],'forward_depth_guard')
        self.assertEqual(h.steps[1][1],'turn_left')
    def test_collision_and_arrival_interrupt(self):
        for setting,reason in [('collide_at','collision_or_blockage'),('arrive_at','existing_arrival_evidence')]:
            h=Harness();setattr(h,setting,1);h.run()
            self.assertEqual(h.events[2]['reason'],reason)
            self.assertEqual(h.events[2]['native_steps'],1)
    def test_stop_is_immediate_without_time_or_pose_advance(self):
        h=Harness(action='stop');obs,cycles=h.run()
        self.assertEqual((obs['step'],cycles,len(h.steps)),(0,1,1))
        self.assertEqual(h.events[-1]['reason'],'stop')
        self.assertEqual(h.events[-1]['native_steps'],0)
    def test_motion_during_model_request_fails_closed(self):
        h=Harness()
        def decide(data):
            d=h.decide(data);h.obs=observation(1);return d
        with self.assertRaisesRegex(RuntimeError,'advanced'):h.run(decide=decide)
        self.assertEqual(h.steps,[])
    def test_old_observation_and_api_failure_execute_nothing(self):
        h=Harness()
        with self.assertRaises(RuntimeError):h.run(read_current=lambda:observation(1))
        self.assertEqual(h.inputs,[])
        def fail(data):raise RuntimeError('API unavailable')
        h=Harness()
        with self.assertRaises(RuntimeError):h.run(decide=fail)
        self.assertEqual(h.steps,[])
        self.assertEqual(h.events[-1]['event'],'decision_failed')
    def test_initial_unsafe_forward_is_rejected(self):
        h=Harness(action='move_forward');h.close_at=0
        with self.assertRaisesRegex(ValueError,'Unsafe forward'):h.run()
        self.assertEqual(h.steps,[])
    def test_codex_prompt_explains_segment_but_old_default_is_unchanged(self):
        for segmented in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                backend=CodexBackend(folder,Path(folder)/'calls.jsonl')
                summary=dict(observation_id='obs_0000')
                if segmented:summary['control']=dict(mode='synchronous_segments_v1',max_native_steps=4)
                decision=dict(observation_id='obs_0000',action='turn_left',reason='test',evidence_ids=['obs_0000:pose'])
                response=SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(dict(type='item.completed',item=dict(type='agent_message',text=json.dumps(decision)))))
                with patch('codex_backend.subprocess.run',return_value=response) as run:
                    backend.decide(summary,[],[],['turn_left'],{'obs_0000:pose'},None)
                prompt=run.call_args.kwargs['input']
                self.assertEqual('A full four-step segment can move 1m or rotate 60 degrees' in prompt,segmented)
                self.assertEqual('Choose exactly one primitive action' in prompt,not segmented)


if __name__=='__main__':unittest.main()
