"""Run an external visual LLM policy over SAVN-CE JSONL, from the Docker host."""

import argparse
import base64
from collections import deque
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


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


class ResponsesBackend:
    def __init__(self, model):
        self.model = model
        self.key = os.environ['SAVNCE_API_KEY']
        self.url = os.environ['SAVNCE_API_BASE_URL'].rstrip('/') + '/responses'
        self.effort = 'low'
        self.opener = urllib.request.build_opener()

    def decide(self, summary, history, images, actions, evidence, feedback):
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
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key},
        )
        try:
            with self.opener.open(req, timeout=90) as res:
                result = json.load(res)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'Model endpoint returned HTTP {exc.code}') from None
        if result.get('status') != 'completed' or result.get('model') != self.model:
            raise ValueError('Model response is incomplete or reports a different model')
        parts = [content['text'] for item in result.get('output', []) if item.get('type') == 'message'
                 for content in item.get('content', []) if content.get('type') == 'output_text']
        usage = result.get('usage') or {}
        return json.loads(''.join(parts)), {key: usage.get(key) for key in ('input_tokens', 'output_tokens')}


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


def choose(backend, variant, data, history, actions, observed, log_path):
    summary, gate = data['summary'], data['stop_gate']
    images = [data['rgb_png'], data['audio_png']]
    ident = summary['observation_id']
    evidence = recent_evidence(summary, history, observed)
    allowed = actions
    feedback = None
    for attempt in range(1, 4):
        started = time.monotonic()
        row = {'observation_id': ident, 'attempt': attempt}
        try:
            decision, usage = backend.decide(summary, history, images, allowed, evidence, feedback)
            row.update(proposal=decision, usage=usage)
            validate(decision, ident, allowed, evidence)
        except (RuntimeError, ValueError, OSError) as exc:
            row.update(status='request_or_validation_error', error=str(exc), latency_s=round(time.monotonic()-started, 3))
            append(log_path, row)
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
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'audio_evidence').mkdir()
    (output/'run_config.json').write_text(json.dumps({
        'model': args.model, 'reasoning_effort': backend.effort, 'seed': args.seed, 'max_steps': args.steps,
        'policy': args.policy, 'episode_id': args.episode_id, 'episode_file': args.episode_file,
        'stop_probe': False, 'prompt_version': PROMPT_VERSION, 'history_steps': 8,
    }, indent=2) + '\n')
    docker = ['docker', 'exec', '-i', args.container, 'python', '-m']
    history = deque(maxlen=8)
    observed = set()
    with jsonl_process(docker + ['agent_episode.llm_observation'], output/'feature_worker.log') as extract:
        with jsonl_process(docker + ['agent_episode.serve', '--inline-observations', '--finalize-rollout'],
                           output/'simulator.log') as sim:
            # Server requires an empty directory; model-side logs are siblings of this subdirectory.
            reset = sim(dict(command='reset', output=str(output/'rollout'), episode_id=args.episode_id,
                             episode_file=args.episode_file, resolution=256, seed=args.seed, max_steps=args.steps))
            actions = tuple(reset['actions'])
            if actions != ACTIONS:
                raise RuntimeError('Unexpected simulator action space')
            obs, done = reset['observation'], False
            try:
                while not done:
                    data = extract(obs)
                    summary = data['summary']
                    ident = summary['observation_id']
                    observed.update(f'{ident}:{modality}' for modality in MODALITIES)
                    (output/'audio_evidence'/f'{ident}.png').write_bytes(base64.b64decode(data['audio_png']))
                    append(output/'model_observations.jsonl', summary)
                    decision = choose(backend, args.policy, data, history, actions, observed,
                                      output/'model_decisions.jsonl')
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
    parser.add_argument('--policy', choices=('ordinary', 'evidence'), required=True)
    parser.add_argument('--episode-id', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='gpt-5.6-sol')
    parser.add_argument('--episode-file', default=EPISODES)
    parser.add_argument('--container', default='savnce_dev')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--steps', type=int, default=64)
    args = parser.parse_args()
    if args.steps <= 0 or Path(args.output).is_absolute():
        parser.error('Use a positive step budget and a path relative to the mounted repo')
    run(args)


if __name__ == '__main__':
    main()
