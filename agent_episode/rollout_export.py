"""Trusted export after a JSONL rollout. Evaluation never enters the protocol."""

import json
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import soundfile as sf


def export_rollout(env, distances):
    result = env.evaluate()
    if not env.done:
        result['termination'] = 'interrupted'
    result['stop_distance_m'] = result['final_distance_m'] if result['stopped'] else None
    finite = [distance for distance in distances if np.isfinite(distance)]
    result['min_geodesic_distance_m'] = min(finite, default=None)
    first_entry = next((index for index, distance in enumerate(distances) if distance < 1), None)
    result['first_within_1m_step'] = first_entry
    result['movement_steps_after_first_entry'] = env.index - first_entry if first_entry is not None else None
    for key in ('initial_distance_m', 'final_distance_m', 'stop_distance_m'):
        if result[key] is not None and not np.isfinite(result[key]):
            result[key] = None
    (env.private_output/'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    files = sorted((env.output/'observations').glob('obs_*.npz'))
    audio = []
    with imageio.get_writer(env.output/'rgb.mp4', fps=1/env.dt, quality=7, macro_block_size=1,
                            output_params=['-threads', '2']) as video:
        for path in files:
            with np.load(path, allow_pickle=False) as obs:
                video.append_data(obs['rgb'])
                if env.audio_mode == 'binaural':
                    audio.append(obs[env.audio_key])
    if env.audio_mode == 'foa':
        return  # Keep the FOA WAV separate from the RGB preview.
    wav = env.output/'binaural.wav'
    sf.write(wav, np.concatenate(audio), env.sr, subtype='FLOAT')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i',
                    str(env.output/'rgb.mp4'), '-i', str(wav), '-c:v', 'copy',
                    '-c:a', 'aac', '-shortest', str(env.output/'avn_demo.mp4')], check=True)
