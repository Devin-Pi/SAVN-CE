"""Compare the adapter to the archived raw-audio validation path on retained samples."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import soundfile as sf
import torch

from dsf_worker import DSF, sha256


def direction(azimuth, elevation):
    a, e = np.radians([azimuth, elevation])
    return [np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)]


def score(events, labels):
    errors, total_pred, total_ref, matched = [], 0, 0, 0
    quiet, quiet_positive = 0, 0
    for index, predictions in enumerate(events):
        actual = labels.get(index, [])
        total_pred += len(predictions)
        total_ref += len(actual)
        if not actual:
            quiet += 1
            quiet_positive += bool(predictions)
        for class_id in {p['class_id'] for p in predictions} & {row[0] for row in actual}:
            p = [event['direction_unit_xyz'] for event in predictions if event['class_id'] == class_id]
            r = [row[1] for row in actual if row[0] == class_id]
            degrees = np.degrees(np.arccos(np.clip(np.asarray(p) @ np.asarray(r).T, -1, 1)))
            i, j = linear_sum_assignment(degrees)
            errors.extend(degrees[i, j].tolist())
            matched += np.sum(degrees[i, j] <= 20).item()
    return {'reference_events': total_ref, 'predicted_events': total_pred, 'same_class_pairs': len(errors),
            'same_class_mean_angle_deg': sum(errors)/len(errors) if errors else None,
            'same_class_front_back_errors': sum(angle > 90 for angle in errors),
            'same_class_within_20deg': matched, 'reference_recall_20deg': matched/total_ref if total_ref else None,
            'unannotated_frames': quiet, 'unannotated_frames_with_predictions': quiet_positive,
            'frames_without_output': sum(not frame for frame in events)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    backend = DSF(args.source, args.checkpoint)
    from utils import tools
    labels = {}
    with args.labels.open() as stream:
        for frame, class_id, _, az, el, *_ in csv.reader(stream):
            labels.setdefault(int(frame), []).append((int(class_id), direction(float(az), float(el))))
    predictions, parity, samples = [], [], []
    info = sf.info(args.wav)
    assert info.samplerate == 24000 and info.channels == 4
    for start in range(0, (info.frames//240000)*10, 10):
        audio, rate = sf.read(args.wav, dtype='float32', start=start*24000, frames=240000, always_2d=True)
        t = time.monotonic()
        actual = backend.predict(audio, rate)
        total = time.monotonic()-t
        native, _ = tools.load_audio_file(str(args.wav), start)
        native = tools.swap_audio_channels(native, 3)
        with torch.inference_mode():
            features = backend.extract(torch.from_numpy(native.T.copy()).float().unsqueeze(0))
            raw_reference = backend.model(features.cuda()).cpu().numpy()[0]
        np.testing.assert_allclose(actual['raw_accdoa'], raw_reference, atol=1e-6, rtol=1e-5)
        parity.append(float(np.max(np.abs(raw_reference-np.asarray(actual['raw_accdoa'])))))
        samples.append(dict(start_s=start, total_s=total, timing_s=actual['timing_s'],
                            peak_cuda_allocated_mib=actual['peak_cuda_allocated_mib']))
        predictions.extend(actual['events_by_frame'])
        np.savez_compressed(args.output/f'window_{start:03d}.npz', raw_accdoa=actual['raw_accdoa'],
                            native_accdoa=raw_reference)
    (args.output/'events.json').write_text(json.dumps(predictions, indent=2)+'\n')
    (args.output/'labels.json').write_text(json.dumps(labels, indent=2)+'\n')
    result = {'input': str(args.wav), 'input_sha256': sha256(args.wav), 'labels': str(args.labels),
              'labels_sha256': sha256(args.labels), 'model': backend.description, 'samples': samples,
              'max_native_vs_adapter_error': max(parity), 'metrics': score(predictions, labels)}
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['metrics']), flush=True)


if __name__ == '__main__':
    main()
