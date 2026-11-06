"""Trusted evaluation of the logged acoustic predictions, separate from policy inputs."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import statistics


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--truth', required=True, type=Path)
    parser.add_argument('--episodes', required=True, type=Path)
    args = parser.parse_args()
    run = args.run
    predictions = rows(run/'predictions.jsonl')
    observations = rows(run/'rollout/observations.jsonl')
    by_id = {row['observation_id']: row for row in observations}
    active = [row for row in observations if row['audio_rms'] > 1e-7]
    truth = json.loads(args.truth.read_text())
    source = truth['acoustic_source_position']
    counts, events_at, angles, distances = Counter(), {}, [], []
    stats = []
    for record in predictions:
        if record['status'] != 'ok':
            continue
        end = record['timing']['audio_end_s']
        stats.append(record['timing'])
        positions = {row['observation_id']: row['receiver_position_habitat_xyz_m'] for row in record['pose_refs']}
        start = record['timing']['audio_start_s']
        for index, (ts, events) in enumerate(zip(record['timing']['frame_times_s'], record['events'])):
            if abs(start*10-round(start*10)) > 1e-7:
                continue  # avoid mixing the two sample grids of overlapping windows
            pose_ref = record['pose_refs'][min(39, index*2//5)]['observation_id']
            events_at.setdefault(round(ts*10), (events, by_id[pose_ref]['audio_rms'] > 1e-7, positions, end))
    counts['unique_frames'] = len(events_at)
    for events, energetic, positions, _ in events_at.values():
        counts['frames_without_output'] += not events
        counts['frames_with_energy'] += energetic
        counts['frames_with_energy_and_output'] += energetic and bool(events)
        counts['frames_without_energy_and_output'] += not energetic and bool(events)
        for event in events:
            counts['predicted_events'] += 1
            counts[event['class_name']] += 1
            if energetic:
                ray = event['direction_unit_habitat_xyz']
                receiver = positions[event['pose_ref']]
                v = [s-r for s,r in zip(source,receiver)]
                norm = math.sqrt(sum(x*x for x in v))
                dot = sum(a*b for a,b in zip(ray,v))/norm
                angles.append(math.degrees(math.acos(max(-1,min(1,dot)))))
                distances.append(norm)
    successful = [row for row in predictions if row['status'] == 'ok']
    latest = [event for record in successful for events in record['events'][-10:] for event in events]
    actions = rows(run/'rollout/actions.jsonl')
    offline_eval = args.truth.with_name('evaluation.json')
    result = {
        'trajectory_mode': 'fixed recorded rule-controller actions, shadow only',
        'observation_count': len(observations), 'action_count': len(actions),
        'translation_m': sum(action['translation_m'] for action in actions),
        'collisions': sum(action['collision'] for action in actions),
        'first_and_last_energetic_audio_s': ([active[0]['timestamp_s'], active[-1]['timestamp_s']+.25] if active else None),
        'window_status_counts': dict(Counter(row['status'] for row in predictions)),
        'first_result_available_audio_time_s': successful[0]['timing']['audio_end_s'] if successful else None,
        'predicted_events_in_most_recent_second_summed_over_calls': len(latest),
        'earliest_window_prediction_per_nominal_frame': dict(counts),
        'conditional_doa_on_energetic_frames': {
            'count': len(angles), 'mean_error_deg': statistics.mean(angles) if angles else None,
            'errors_over_90_deg': sum(angle > 90 for angle in angles),
            'within_20_deg': sum(angle <= 20 for angle in angles),
            'receiver_source_3d_m_bounds': [min(distances),max(distances)] if distances else None,
        },
        'note': 'This uses the actual renderer source offline. Class coverage of the simulator audio is unverified; '
                'conditional angles are not class-correct recall. RMS cutoff is numerical, not annotation of silence.',
        'warm_median_s': {key: statistics.median(t[key] for t in stats[1:])
                          for key in ('preprocess','inference','postprocess')} if len(stats)>1 else None,
        'peak_cuda_allocated_mib': max((row['peak_cuda_allocated_mib'] for row in successful), default=None),
        'navigation': json.loads(offline_eval.read_text()) if offline_eval.exists() else None,
    }
    (run/'shadow_evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    with gzip.open(args.episodes) as stream:
        episodes = json.load(stream)['episodes']
    coverage = [
        {'episode_id': str(ep['episode_id']), 'visual_object_class': ep['goals'][0]['object_category'],
         'sound_asset': ep['goals'][0]['sound_id'], 'audio_event_class': None,
         'association': 'official simulator object/source association; closed-set acoustic coverage unverified'}
        for ep in episodes
    ]
    (run/'offline_category_coverage.json').write_text(json.dumps(coverage, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
