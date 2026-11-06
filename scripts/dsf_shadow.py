"""Replay recorded actions through native FOA and run DSF without controlling actions."""
import argparse
from pathlib import Path
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent_episode'))
from llm_rollout import append, jsonl_process
from localization_tool import LocalizationTool, Worker, compact, worker_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode-id', required=True)
    parser.add_argument('--episode-file', required=True)
    parser.add_argument('--sim-workdir', required=True)
    args = parser.parse_args()
    actions = [json.loads(row) for row in args.actions.read_text().splitlines()]
    if not actions:
        parser.error('Expected nonempty recorded actions')
    args.output.mkdir(parents=True, exist_ok=False)
    model = Worker(worker_command('avseld_final_test'), args.output/'dsf_worker.stderr.log')
    try:
        tool = LocalizationTool(model.request)
        tool.reset_episode(args.episode_id)
        description = tool.describe()
        (args.output/'model.json').write_text(json.dumps(description, indent=2)+'\n')
        if description['status'] != 'ok':
            raise RuntimeError('Model unavailable; see model.json')
        docker = ['docker', 'exec', '-i', '-e', 'HABITAT_SIM_LOG=quiet', '-e', 'MAGNUM_LOG=quiet',
                  '-w', args.sim_workdir, 'savnce_dev', 'python', '-m', 'agent_episode.serve',
                  '--inline-observations', '--finalize-rollout']
        with jsonl_process(docker, args.output/'simulator.stderr.log') as sim:
            obs = sim(dict(command='reset', output=str(args.output/'rollout'), episode_id=args.episode_id,
                           episode_file=args.episode_file, seed=7, max_steps=len(actions), audio_mode='foa'))['observation']
            try:
                for index in range(len(actions)+1):
                    tool.add_observation(obs)
                    if index < 40 or (index-39) % 4 == 0 or index == len(actions):
                        result = tool.analyze([obs['observation_id']])
                        append(args.output/'shadow.jsonl', {'at': obs['observation_id'], **compact(result)})
                        append(args.output/'predictions.jsonl', {'at': obs['observation_id'], **result})
                        if result['status'] == 'ok':
                            print(obs['observation_id'], result['timing']['inference'], flush=True)
                    if index == len(actions):
                        break
                    action = actions[index]
                    reply = sim(dict(command='step', action=action['action']))
                    obs = reply['observation']
                    assert obs['observation_id'] == action['observation_after']
                    assert math.dist(obs['pose']['position'], action['pose_after']['position']) < 1e-5
                    dot = sum(a*b for a,b in zip(obs['pose']['rotation_xyzw'], action['pose_after']['rotation_xyzw']))
                    assert abs(dot) > 0.99999 and reply['done'] == action['done']
                    if reply['done'] and action['action'] == 'stop':
                        break
            finally:
                sim({'command': 'close'})
    finally:
        model.close()


if __name__ == '__main__':
    main()
