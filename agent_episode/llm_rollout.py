"""Run an external visual LLM policy over SAVN-CE JSONL, from the Docker host."""

import argparse
import base64
from collections import deque
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

if __package__:
    from .localization_tool import FUNCTION, MODEL_NAME, LocalizationTool, Worker, compact, worker_command
else:
    from localization_tool import FUNCTION, MODEL_NAME, LocalizationTool, Worker, compact, worker_command


ACTIONS = ('move_forward', 'turn_left', 'turn_right', 'stop')
MODALITIES = ('rgb', 'depth', 'audio', 'pose')
EPISODES = 'data/datasets/savnce_generated/mp3d/v1/train/content/17DRP5sb8fy.json.gz'
PROMPT_VERSION = 'savnce-llm-v1'
INSTRUCTIONS = """Navigate to the audible emitter in a simulated room. Choose exactly one primitive action.
move_forward advances 0.25 m, turn_left and turn_right rotate 15 degrees, stop ends the episode.
Only use the RGB, acoustic panel, sensor summaries and executed action history given to you.
The first image is current RGB; the second is a stereo evidence panel computed from the current 0.25 s window.
Prefer the numerical acoustic features. Positive ITD means left leads; positive bearing is toward the left.
ITD admits both front/back bearings, RMS only describes activity, and depth measures surfaces rather than source distance.
Passive multi-view fits can be invalid or ambiguous. Never infer a sound from appearance alone.
If uncertain, choose a safe useful movement that changes your viewpoint. Avoid oscillating or repeatedly moving into a blockage.
Do not carry out a dedicated yaw sweep. Prefer translation when safely possible. Turns for normal navigation are allowed.
Stop when observed evidence indicates you are likely within 1 m of the emitter. Don't stop simply because a wall or a plausible object is close.
Use a short reason and cite the evidence IDs you actually used. The observation_id must match the current frame."""


def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


@contextmanager
def jsonl_process(command, log_path):
    with log_path.open('w') as log:
        process = subprocess.Popen(command, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        try:
            def request(row):
                try:
                    process.stdin.write(json.dumps(row, allow_nan=False) + '\n')
                    process.stdin.flush()
                    line = process.stdout.readline()
                    if not line:
                        raise RuntimeError('JSONL process exited without a reply')
                    response = json.loads(line)
                    if not response.get('ok'):
                        raise RuntimeError('JSONL request failed: ' + response.get('error', 'unknown error'))
                    return response
                except (OSError, ValueError) as exc:
                    raise RuntimeError('JSONL transport failed') from exc
            yield request
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=30)
            process.stdout.close()


def decision_schema(observation_id, actions, evidence):
    return {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'observation_id': {'type': 'string', 'enum': [observation_id]},
            'action': {'type': 'string', 'enum': list(actions)},
            'reason': {'type': 'string'},
            'evidence_ids': {'type': 'array', 'items': {'type': 'string', 'enum': sorted(evidence)}},
        },
        'required': ['observation_id', 'action', 'reason', 'evidence_ids'],
    }


def validate(decision, ident, actions, evidence):
    if not isinstance(decision, dict) or set(decision) != {'observation_id', 'action', 'reason', 'evidence_ids'}:
        raise ValueError('Decision fields do not match the schema')
    if decision['observation_id'] != ident or decision['action'] not in actions:
        raise ValueError('Unexpected observation ID or invalid action')
    reason, refs = decision['reason'], decision['evidence_ids']
    if not isinstance(reason, str) or not 0 < len(reason.strip()) <= 400:
        raise ValueError('Expected a short action reason')
    if (not isinstance(refs, list) or not 0 < len(refs) <= 12
            or any(not isinstance(ref, str) or ref not in evidence for ref in refs)
            or len(refs) != len(set(refs))):
        raise ValueError('Expected unique evidence IDs from the model input')


class GatewayAuthError(RuntimeError):
    pass


class ResponsesBackend:
    def __init__(self, model):
        self.model = model
        self.key = os.environ['SAVNCE_API_KEY']
        self.url = os.environ['SAVNCE_API_BASE_URL'].rstrip('/') + '/responses'
        self.effort = 'low'
        self.opener = urllib.request.build_opener()

    def request(self, payload):
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key},
        )
        try:
            with self.opener.open(req, timeout=90) as res:
                result = json.load(res)
        except urllib.error.HTTPError as exc:
            message = f'Model endpoint returned HTTP {exc.code}'
            if exc.code in (401, 403):
                raise GatewayAuthError(message) from None
            raise RuntimeError(message) from None
        if result.get('status') != 'completed' or result.get('model') != self.model:
            raise ValueError('Model response is incomplete or reports a different model')
        return result

    def health(self):
        response = self.request({'model': self.model, 'input': 'Reply with exactly OK.',
                                 'store': False, 'reasoning': {'effort': self.effort}, 'max_output_tokens': 128})
        text = ''.join(part['text'] for item in response.get('output', []) if item.get('type') == 'message'
                       for part in item.get('content', []) if part.get('type') == 'output_text')
        if not text.strip():
            raise ValueError('Responses health check returned no text')
        return response['model']

    def decide(self, summary, history, images, actions, evidence, feedback, call_tool=None):
        ident = summary['observation_id']
        text = json.dumps({'current': summary, 'past_steps_oldest_first': list(history),
                           'available_evidence_ids': sorted(evidence), 'available_actions': actions,
                           'feedback_on_previous_proposal': feedback}, allow_nan=False, separators=(',', ':'))
        payload = {
            'model': self.model, 'store': False, 'tools': [], 'reasoning': {'effort': self.effort},
            'max_output_tokens': 700, 'instructions': INSTRUCTIONS,
            'input': [{'role': 'user', 'content': [
                {'type': 'input_text', 'text': text},
                *[{'type': 'input_image', 'detail': 'low', 'image_url': 'data:image/png;base64,' + image}
                  for image in images],
            ]}],
            'text': {'format': {'type': 'json_schema', 'name': 'savnce_decision', 'strict': True,
                                'schema': decision_schema(ident, actions, evidence)}},
        }
        if call_tool:
            payload.update(tools=[FUNCTION], parallel_tool_calls=False,
                           include=['reasoning.encrypted_content'])
            first = INSTRUCTIONS.index('Only use the RGB')
            last = INSTRUCTIONS.index('Passive multi-view')
            payload['instructions'] = INSTRUCTIONS[:first] + (
                'Use only the observed RGB-D, audio summaries, tool results and executed action history. '
                'The image is RGB. Audio is world-aligned FOA. Use localization_tool when needed; '
                'give it an already observed ID, normally the current one. A full 10-second history is required. '
                'Tool azimuth is relative to the fixed world axes. Treat overlapping windows as shared evidence. '
                'Do not infer range, identity, or STOP success from activity scores. '
            ) + INSTRUCTIONS[last:]
        result = self.request(payload)
        usage = {key: (result.get('usage') or {}).get(key, 0) or 0
                 for key in ('input_tokens', 'output_tokens')}
        for call_index in range(2):
            calls = [item for item in result.get('output', []) if item.get('type') == 'function_call']
            if not calls:
                break
            if call_tool is None or len(calls) != 1 or calls[0].get('name') != MODEL_NAME:
                raise ValueError('Unexpected tool request')
            call = calls[0]
            arguments = json.loads(call['arguments'])
            if not isinstance(arguments, dict) or set(arguments) != {'observation_ids'}:
                raise ValueError('Invalid localization_tool arguments')
            report = call_tool(call['call_id'], arguments['observation_ids'])
            if report.get('evidence_id'):
                evidence.add(report['evidence_id'])
            payload['input'] += result['output'] + [{
                'type': 'function_call_output', 'call_id': call['call_id'],
                'output': json.dumps(report, allow_nan=False),
            }]
            payload['text']['format']['schema'] = decision_schema(ident, actions, evidence)
            # Two model-initiated calls per decision. The last response must be an action.
            if call_index == 1:
                payload['tool_choice'] = 'none'
            result = self.request(payload)
            for key in usage:
                usage[key] += (result.get('usage') or {}).get(key, 0) or 0
        parts = [content['text'] for item in result.get('output', []) if item.get('type') == 'message'
                 for content in item.get('content', []) if content.get('type') == 'output_text']
        return json.loads(''.join(parts)), usage


def recent_evidence(summary, history, observed):
    current = summary['observation_id']
    refs = {f'{current}:{modality}' for modality in MODALITIES}
    for row in history:
        ident = row['observation_id']
        # Prior RGB images are not in the current context.
        refs.update(f'{ident}:{modality}' for modality in ('audio', 'depth', 'pose'))
    for fit in summary['passive_acoustic_fits'].values():
        refs.update(fit.get('supporting_observation_ids', []))
    return refs & observed


def gate_feedback(gate):
    arrival = gate['arrival']
    return {
        'stop_vetoed': True,
        'reason': 'Insufficient passive audio/depth support for arrival; select a navigation action.',
        'latest_fit_selection': gate['latest_fit_selection'],
        'failed_arrival_checks': [key for key, value in arrival['checks'].items() if not value] if arrival else [],
    }


def choose(backend, variant, data, history, actions, observed, log_path, call_tool=None):
    summary, gate = data['summary'], data['stop_gate']
    images = [image for image in (data['rgb_png'], data['audio_png']) if image is not None]
    ident = summary['observation_id']
    evidence = recent_evidence(summary, history, observed)
    allowed = actions
    feedback = None
    for attempt in range(1, 4):
        started = time.monotonic()
        row = {'observation_id': ident, 'attempt': attempt}
        try:
            if call_tool:
                decision, usage = backend.decide(summary, history, images, allowed, evidence, feedback, call_tool)
            else:
                decision, usage = backend.decide(summary, history, images, allowed, evidence, feedback)
            row.update(proposal=decision, usage=usage)
            validate(decision, ident, allowed, evidence)
        except (RuntimeError, ValueError, OSError) as exc:
            row.update(status='request_or_validation_error', error=str(exc), latency_s=round(time.monotonic()-started, 3))
            append(log_path, row)
            if isinstance(exc, GatewayAuthError):
                raise
            if attempt == 1 or (attempt == 2 and feedback and feedback.get('stop_vetoed')):
                time.sleep(1)
                continue
            raise RuntimeError('Model decision failed after retry') from exc
        row['latency_s'] = round(time.monotonic()-started, 3)
        if variant == 'evidence' and decision['action'] == 'stop':
            row['verification'] = gate
            if not gate['ready']:
                feedback = gate_feedback(gate)
                allowed = tuple(a for a in actions if a != 'stop')
                row['status'] = 'stop_vetoed'
                append(log_path, row)
                continue
        row['status'] = 'accepted'
        append(log_path, row)
        return decision
    raise RuntimeError('Model did not produce an executable decision')


def run(args):
    backend = ResponsesBackend(args.model)
    use_dsf = args.acoustic_backend == 'dsf_audio_only'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'audio_evidence').mkdir()
    (output/'run_config.json').write_text(json.dumps({
        'model': args.model, 'reasoning_effort': backend.effort, 'seed': args.seed, 'max_steps': args.steps,
        'policy': args.policy, 'episode_id': args.episode_id, 'episode_file': args.episode_file,
        'stop_probe': False, 'prompt_version': 'savnce-llm-foa-v1' if use_dsf else PROMPT_VERSION, 'history_steps': 8,
        'acoustic_backend': args.acoustic_backend, 'observation_profile': 'foa_ambix_world' if use_dsf else 'binaural',
        'foa_stop_gate': 'unavailable; evidence policy vetoes unverified STOP' if use_dsf else None,
    }, indent=2) + '\n')
    docker = ['docker', 'exec', '-i', '-w', args.sim_workdir, args.container, 'python', '-m']
    history = deque(maxlen=8)
    observed = set()
    with ExitStack() as resources:
        extract = resources.enter_context(jsonl_process(docker + ['agent_episode.llm_observation'],
                                                       output/'feature_worker.log'))
        tool = None
        if use_dsf:
            worker = Worker(worker_command(args.dsf_container), output/'dsf_worker.log')
            resources.callback(worker.close)
            tool = LocalizationTool(worker.request)
            tool.reset_episode(args.episode_id)
            description = tool.describe()
            (output/'dsf_backend.json').write_text(json.dumps(description, indent=2)+'\n')
            if description['status'] != 'ok':
                raise RuntimeError('DSF model unavailable; see dsf_backend.json')

        def call_tool(call_id, ids):
            result = tool.analyze(ids)
            append(output/'localization_calls.jsonl', {'call_id': call_id, 'requested_ids': ids, 'result': result})
            return compact(result)
        with jsonl_process(docker + ['agent_episode.serve', '--inline-observations', '--finalize-rollout'],
                           output/'simulator.log') as sim:
            # Server requires an empty directory; model-side logs are siblings of this subdirectory.
            reset = sim(dict(command='reset', output=str(output/'rollout'), episode_id=args.episode_id,
                             episode_file=args.episode_file, resolution=256, seed=args.seed, max_steps=args.steps,
                             audio_mode='foa' if use_dsf else 'binaural'))
            actions = tuple(reset['actions'])
            if actions != ACTIONS:
                raise RuntimeError('Unexpected simulator action space')
            obs, done = reset['observation'], False
            try:
                while not done:
                    if tool:
                        tool.add_observation(obs)
                    data = extract(obs)
                    summary = data['summary']
                    ident = summary['observation_id']
                    observed.update(f'{ident}:{modality}' for modality in MODALITIES)
                    if data['audio_png']:
                        (output/'audio_evidence'/f'{ident}.png').write_bytes(base64.b64decode(data['audio_png']))
                    append(output/'model_observations.jsonl', summary)
                    decision = choose(backend, args.policy, data, history, actions, observed,
                                      output/'model_decisions.jsonl', call_tool if tool else None)
                    reply = sim(dict(command='step', action=decision['action'], decision=decision))
                    obs, done = reply['observation'], reply['done']
                    if decision['action'] == 'stop':
                        if not done or obs['observation_id'] != ident:
                            raise RuntimeError('Unexpected STOP transition')
                    elif obs['step'] != summary['step'] + 1 or obs['observation_id'] == ident:
                        raise RuntimeError('Unexpected movement transition')
                    history.append({k: summary[k] for k in ('observation_id', 'xz_m', 'forward_xz',
                                                           'collided', 'forward_blocked', 'audio', 'depth')}
                                   | {'action': decision['action']})
                    print(f"{args.policy} {args.episode_id} {ident} {decision['action']} done={done}", flush=True)
            finally:
                sim({'command': 'close'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-api', action='store_true', help='Send one minimal authenticated Responses request')
    parser.add_argument('--policy', choices=('ordinary', 'evidence'))
    parser.add_argument('--episode-id')
    parser.add_argument('--output')
    parser.add_argument('--model', default='gpt-5.6-sol')
    parser.add_argument('--episode-file', default=EPISODES)
    parser.add_argument('--container', default='savnce_dev')
    parser.add_argument('--sim-workdir', default='/workspace/SAVN-CE')
    parser.add_argument('--acoustic-backend', choices=('binaural', 'dsf_audio_only'), default='binaural')
    parser.add_argument('--dsf-container', default='avseld_final_test')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--steps', type=int, default=64)
    args = parser.parse_args()
    if not all(os.getenv(key) for key in ('SAVNCE_API_KEY', 'SAVNCE_API_BASE_URL')):
        parser.exit(2, 'Responses credentials unavailable: set SAVNCE_API_KEY and SAVNCE_API_BASE_URL.\n')
    if args.check_api:
        print(json.dumps({'status': 'ok', 'model': ResponsesBackend(args.model).health()}))
        return
    if not all((args.policy, args.episode_id, args.output)):
        parser.error('Rollout requires --policy, --episode-id and --output')
    if args.steps <= 0 or Path(args.output).is_absolute():
        parser.error('Use a positive step budget and a path relative to the mounted repo')
    run(args)


if __name__ == '__main__':
    main()
