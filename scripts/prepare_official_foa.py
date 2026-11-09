"""Index the official clean test split and capture a small FOA sensor pilot."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import runpy
import shlex
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_episode.environment import OfficialEpisodeEnv
from savnce.datasets.savnce_dataset import SAVNCE_Dataset


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def official_index(directory):
    files = sorted(directory.glob('test-split_*.json.gz'))
    if len(files) != 10:
        raise ValueError('Expected the ten original clean test shards')
    scenes_root = Path('/datasets/mp3d')
    config = runpy.run_path('savnce_baselines/magnet/config/default.py')['get_task_config']().DATASET.clone()
    config.defrost()
    config.SCENES_DIR = str(scenes_root)
    config.CONTENT_SCENES = []
    sound_root = Path('data/sounds/semantic_splits')
    scenes, sounds = set(), set()
    records = []
    for path in files:
        with gzip.open(path, 'rt') as stream:
            episodes = json.load(stream)['episodes']
        config.DATA_PATH = str(path.resolve())
        native = SAVNCE_Dataset(config).episodes
        raw_keys = [(str(e['episode_id']), e['scene_id']) for e in episodes]
        native_keys = [(str(e.episode_id), str(Path(e.scene_id).relative_to(scenes_root))) for e in native]
        if raw_keys != native_keys or len({key[0] for key in raw_keys}) != len(raw_keys):
            raise ValueError(f'Shard identity is not preserved by the native loader: {path}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for e in episodes:
            if e.get('distractor') is not None or len(e['goals']) != 1:
                raise ValueError('Expected a single clean target')
            goal = e['goals'][0]
            episode_id = str(e['episode_id'])
            scene = Path(e['scene_id'])
            sound = goal['sound_id']
            scenes.add(scene)
            sounds.add(sound)
            entry = dict(key=f"{path.name.removesuffix('.json.gz')}-{episode_id}", episode_id=episode_id,
                         episode_file=str(path), episode_file_sha256=digest, scene_id=str(scene))
            records.append((entry, goal))
    for scene in scenes:
        stem = scenes_root / scene.with_suffix('')
        for suffix in ('.glb', '.navmesh', '.house', '_semantic.ply'):
            if not Path(str(stem) + suffix).is_file():
                raise FileNotFoundError(str(stem) + suffix)
    for sound in sounds:
        if sf.info(sound_root / sound).frames <= 0:
            raise ValueError(f'Empty sound: {sound}')
    if len(records) != 1000:
        raise ValueError('Expected 1,000 official clean test records')
    return records, scenes, sounds


def select_pilot(records, seed):
    anchor = next(row for row in records if row[0]['episode_id'] == '3990'
                  and row[0]['scene_id'] == 'pa4otMbVnkk/pa4otMbVnkk.glb')
    candidates = sorted(records, key=lambda row: row[0]['key'])
    random.Random(seed).shuffle(candidates)
    selected = [anchor]
    for entry, goal in candidates:
        if (goal['onset'] < 2 or goal['offset'] - goal['onset'] < 2
                or any(entry['scene_id'] == e['scene_id'] or goal['object_category'] == g['object_category']
                       for e, g in selected)):
            continue
        selected.append((entry, goal))
        if len(selected) == 5:
            return selected
    raise ValueError('Expected five eligible cases in distinct scenes and categories')


def capture(output, entry, goal, seed):
    key = entry['key']
    dest = output / 'captures' / key
    private = output / 'evaluation_only' / key
    onset = goal['onset']
    capture_steps = (onset + 2) * 4 - 1
    env = OfficialEpisodeEnv(dest, episode_id=entry['episode_id'], episodes_path=entry['episode_file'],
                             private_output=private, resolution=128, max_steps=capture_steps,
                             seed=seed, audio_mode='foa', audio_threads=1, sound_schedule='episode')
    try:
        start = env.reset()
        for _ in range(capture_steps):
            final, _ = env.listen_step()
        if start['pose'] != final['pose']:
            raise ValueError(f'Pilot pose drift: {key}')
        env.export_foa_wav()
        wave, rate = sf.read(dest / 'foa_ambix.wav', dtype='float32', always_2d=True)
        if rate != 16000 or wave.shape != ((capture_steps + 1) * 4000, 4) or not np.isfinite(wave).all():
            raise ValueError(f'Invalid FOA capture: {key}')
        fmt = start['audio_format']
        if (fmt['channels'], fmt['normalization'], fmt['coordinate_frame']) != (
                ['W', 'Y', 'Z', 'X'], 'SN3D', 'habitat_world'):
            raise ValueError(f'Unexpected FOA format: {key}')
        truth = json.loads((private / 'evaluation_truth.json').read_text())
        semantic = [o for o in env.sim._sim.semantic_scene.objects if o
                    and o.id.rsplit('_', 1)[-1] == str(goal['object_id'])]
        if (len(semantic) != 1 or semantic[0].category.name() != goal['object_category']
                or not np.isfinite(truth['initial_geodesic_distance'])
                or truth['episode_file_sha256'] != entry['episode_file_sha256']
                or not np.allclose(truth['runtime_goal_position'], np.asarray(goal['position']) + [0, .1, 0])
                or not np.allclose(truth['acoustic_source_position'], np.asarray(goal['position']) + [0, 1.6, 0])
                or truth['runtime_sound_schedule']['runtime_offset_s'] != goal['offset']):
            raise ValueError(f'Invalid official target association: {key}')
        windows = {}
        for name, seconds in (('before_onset', .25), ('source_on', onset)):
            begin = int(seconds * rate)
            sample = wave[begin:begin + rate]
            sf.write(dest / f'{name}_1s.wav', sample, rate, subtype='FLOAT')
            windows[name] = dict(start_s=seconds, end_s=seconds + 1, anchor_step=int(seconds * 4) + 3,
                                 rms=float(np.sqrt(np.mean(sample**2))))
        if windows['before_onset']['rms'] > 1e-8 or windows['source_on']['rms'] <= 1e-7:
            raise ValueError(f'Pilot audio activity check failed: {key}, {windows}')
        return dict(**entry, object_category=goal['object_category'], object_id=semantic[0].id,
                    official_onset_s=onset, official_offset_s=goal['offset'],
                    initial_geodesic_distance_m=truth['initial_geodesic_distance'],
                    capture_duration_s=len(wave) / rate, windows=windows)
    finally:
        env.close()


def write_agent_launcher(output, selected, seed):
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail',
             ': "${SIM_WORKDIR:?Set the project worktree path inside savnce_dev}"',
             ': "${ACTOR_INPUTS:?Set the host path to the isolated Codex Actor inputs}"',
             ': "${RUN_ROOT:?Set a fresh relative output root}"', 'steps=${STEPS:-128}']
    for entry, _ in selected:
        command = shlex.join(['python3', '-u', 'scripts/candidate_codex_shadow.py',
            '--episode-id', entry['episode_id'], '--episode-file', entry['episode_file'],
            '--profile', 'avn_foa_short1s', '--decision-period-s', '1', '--listen-before-decide-s', '1',
            '--sound-schedule', 'episode', '--source-placement', 'episode', '--stop-rule', 'foa_geometry',
            '--foa-tolerance-deg', '2', '--foa-audio-threads', '1', '--binaural-audio-threads', '1',
            '--seed', str(seed)])
        lines.append(command + ' --steps "$steps" --sim-workdir "$SIM_WORKDIR" --inputs "$ACTOR_INPUTS"'
                     + ' --output "$RUN_ROOT"/' + shlex.quote(entry['key']))
    (output / 'run_agent_pilot.sh').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    records, scenes, sounds = official_index(args.episode_dir)
    selected = select_pilot(records, args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'evaluation_only').mkdir(mode=0o700)
    with (output / 'official_index.jsonl').open('w') as stream:
        for entry, _ in records:
            stream.write(json.dumps(entry) + '\n')
    write_json(output / 'pilot_plan.json', [entry for entry, _ in selected])
    write_agent_launcher(output, selected, args.seed)
    pilot = []
    for entry, goal in selected:
        result = capture(output, entry, goal, args.seed)
        pilot.append(result)
        write_json(output / 'capture_progress.json', pilot)
        print(json.dumps(result), flush=True)
    write_json(output / 'validation.json', dict(status='passed', native_loaded_records=len(records),
        distinct_episode_ids=len({entry['episode_id'] for entry, _ in records}),
        scenes=len(scenes), wav_assets=len(sounds), seed=args.seed, pilot=pilot,
        agent_navigation_run=False))


if __name__ == '__main__':
    main()
