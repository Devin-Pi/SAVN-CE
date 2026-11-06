"""Decode the archived RAW model's track-major, axis-major ACCDOA."""
import math

import numpy as np

CLASSES = (
    'female speech', 'male speech', 'clapping', 'telephone', 'laughter',
    'domestic sounds', 'footsteps', 'door', 'music', 'musical instrument',
    'water tap', 'bell', 'knock',
)
ACTIVITY_THRESHOLD = 0.5
MERGE_ANGLE_DEG = 15


def vector_event(class_id, vector, slots, active):
    score = float(np.linalg.norm(vector))
    unit = (vector / score).tolist() if active and score > 0 else None
    return {
        'class_id': class_id, 'class_name': CLASSES[class_id],
        'track_slot': slots[0] if len(slots) == 1 else None, 'merged_track_slots': slots,
        'accdoa': vector.tolist(), 'activity_score': score, 'activity_threshold': ACTIVITY_THRESHOLD,
        'active': active, 'direction_unit_xyz': unit,
        'azimuth_deg': math.degrees(math.atan2(unit[1], unit[0])) if unit else None,
        'elevation_deg': math.degrees(math.atan2(unit[2], math.hypot(unit[0], unit[1]))) if unit else None,
        'range_m': None, 'uncertainty': None, 'calibration_status': 'uncalibrated',
    }


def decode(raw):
    if raw.ndim != 2 or raw.shape[1] != 9 * len(CLASSES) or not np.isfinite(raw).all():
        raise ValueError('Invalid ACCDOA tensor')
    vectors = raw.reshape(-1, 3, 3, len(CLASSES)).transpose(0, 3, 1, 2)
    frames = []
    for frame in vectors:
        events = []
        for class_id, tracks in enumerate(frame):
            scores = np.linalg.norm(tracks, axis=1)
            active = [i for i, score in enumerate(scores) if score > ACTIVITY_THRESHOLD]
            pairs = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                if a in active and b in active:
                    cosine = np.dot(tracks[a], tracks[b]) / (scores[a] * scores[b])
                    if math.degrees(math.acos(float(np.clip(cosine, -1, 1)))) < MERGE_ANGLE_DEG:
                        pairs.append([a, b])
            if not pairs:
                groups = [[i] for i in active]
            elif len(pairs) >= 2:
                groups = [[0, 1, 2]]
            else:
                third = next(i for i in range(3) if i not in pairs[0])
                # The archived validation/evaluation emits a lone pair only if the third slot is active.
                groups = [[third], pairs[0]] if third in active else []
            events.extend(vector_event(class_id, tracks[group].mean(axis=0), group, True) for group in groups)
        frames.append(events)
    return frames
