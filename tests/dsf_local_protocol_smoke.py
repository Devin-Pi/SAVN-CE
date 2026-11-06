"""Local Responses fixture + real DSF + real simulator. No external model call."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sim-workdir', required=True)
    args = parser.parse_args()
    actions = [json.loads(line)['action'] for line in args.actions.read_text().splitlines()]
    assert len(actions) >= 40 and 'stop' not in actions[:40]
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path == '/v1/responses'
            assert self.headers['Authorization'] == 'Bearer local-protocol-only'
            assert payload['tools'][0]['name'] == 'localization_tool'
            data = json.loads(payload['input'][0]['content'][0]['text'])
            step, ident = (data['current'][key] for key in ('step', 'observation_id'))
            assert data['current']['audio']['format'] == 'FOA ACN/SN3D'
            evidence_id = ident+':rgb'
            if step == 39 and len(payload['input']) == 1:
                response = {'type': 'function_call', 'name': 'localization_tool', 'call_id': 'local_fixture_call',
                            'arguments': json.dumps({'observation_ids': [ident]})}
            else:
                if step == 39:
                    tool = payload['input'][-1]
                    assert tool['type'] == 'function_call_output' and tool['call_id'] == 'local_fixture_call'
                    result = json.loads(tool['output'])
                    assert result['status'] == 'ok' and result['audio_interval_s'] == [0.0,10.0]
                    evidence_id = result['evidence_id']
                decision = dict(observation_id=ident, action=actions[step], evidence_ids=[evidence_id],
                                reason='Scripted local transport test; rule trace unchanged.')
                response = {'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(decision)}]}
            requests.append((step,response['type']))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'completed', 'model': payload['model'], 'output': [response]}).encode())

        def log_message(self, *_):
            pass

    with HTTPServer(('127.0.0.1', 0), Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        env = dict(os.environ, SAVNCE_API_KEY='local-protocol-only',
                   SAVNCE_API_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1', no_proxy='127.0.0.1')
        subprocess.run([sys.executable, 'agent_episode/llm_rollout.py', '--policy', 'ordinary',
                        '--episode-id', '1000000008', '--output', str(args.output), '--steps', '40',
                        '--sim-workdir', args.sim_workdir, '--acoustic-backend', 'dsf_audio_only'], env=env, check=True)
    assert len(requests) == 41 and requests[-2:] == [(39,'function_call'),(39,'message')]
    call = json.loads((args.output/'localization_calls.jsonl').read_text())
    assert call['result']['status'] == 'ok' and len(call['result']['raw_accdoa']) == 100
    decisions = (args.output/'model_decisions.jsonl').read_text().splitlines()
    executed = (args.output/'rollout/actions.jsonl').read_text().splitlines()
    assert len(decisions) == len(executed) == 40
    final = json.loads(executed[-1])
    assert final['decision']['evidence_ids'] == ['obs_0039:localization']
    assert final['action'] == actions[39] and final['done']
    (args.output/'protocol_result.json').write_text(json.dumps({
        'gateway': 'local scripted protocol fixture', 'external_llm_calls': 0,
        'actual_dsf_calls': 1, 'simulator_actions': 40, 'tool_result': 'ok', 'execution_verified': True,
    },indent=2)+'\n')
    print('PASS: real DSF and simulator, local scripted Responses fixture', flush=True)


if __name__ == '__main__':
    main()
