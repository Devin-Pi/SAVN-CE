"""STOP evidence can affect stop acceptance without exposing source coordinates."""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'agent_episode'))
from codex_backend import CodexBackend
from llm_rollout import choose


class FakeBackend:
    def __init__(self,actions):self.actions=iter(actions)
    def decide(self,summary,history,images,actions,evidence,feedback):
        return dict(observation_id='obs_0004',action=next(self.actions),reason='observed evidence',evidence_ids=['obs_0004:foa_stop']),{}


def data(ready):
    gate=dict(method='foa-signed-rays-v1',ready=ready,reason='test range evidence',checks=dict(range_supported=ready),evidence_id='obs_0004:foa_stop')
    return dict(summary=dict(observation_id='obs_0004',passive_acoustic_fits={},stop_verification=gate),stop_gate=gate,rgb_png=None,audio_png=None)


class FoaStopProtocol(unittest.TestCase):
    def select(self,ready,actions):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'decisions.jsonl'
            decision=choose(FakeBackend(actions),'evidence',data(ready),[],('move_forward','turn_left','turn_right','stop'),{'obs_0004:foa_stop'},path)
            return decision,[json.loads(x) for x in path.read_text().splitlines()]
    def test_supported_stop_does_not_require_legacy_arrival_or_depth(self):
        d,rows=self.select(True,['stop']);self.assertEqual(d['action'],'stop');self.assertEqual(rows[0]['status'],'accepted')
    def test_unknown_range_rejects_stop_and_requests_a_navigation_action(self):
        d,rows=self.select(False,['stop','turn_left']);self.assertEqual(d['action'],'turn_left');self.assertEqual([x['status'] for x in rows],['stop_vetoed','accepted'])
    def test_ready_does_not_force_agent_to_stop(self):
        d,_=self.select(True,['turn_right']);self.assertEqual(d['action'],'turn_right')
    def test_codex_receives_stop_evidence_and_audits_its_use(self):
        with tempfile.TemporaryDirectory() as folder:
            audit=Path(folder)/'calls.jsonl';backend=CodexBackend(folder,audit)
            answer=dict(observation_id='obs_0004',action='stop',reason='ready evidence',evidence_ids=['obs_0004:foa_stop'])
            response=SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(dict(type='item.completed',item=dict(type='agent_message',text=json.dumps(answer)))))
            with patch('codex_backend.subprocess.run',return_value=response) as run:
                backend.decide(data(True)['summary'],[],[],['stop'],{'obs_0004:foa_stop'},None)
            prompt=run.call_args.kwargs['input']
            self.assertIn('For STOP only',prompt);self.assertNotIn('inferred_source_world_xyz',prompt)
            self.assertTrue(json.loads(audit.read_text())['foa_stop_evidence_visible'])


if __name__=='__main__':unittest.main()
