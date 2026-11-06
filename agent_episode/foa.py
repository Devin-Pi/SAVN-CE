"""FOA conversion for the calibrated Habitat-Sim/RLR audio sensor."""

import numpy as np


def to_ambix(native):
    """Convert (..., 4) world-aligned RLR coefficients to ACN/SN3D."""
    if native.ndim != 2 or native.shape[1] != 4:
        raise ValueError('Expected native FOA samples with shape (samples, 4)')
    # Native order is [W, +Habitat Y, +Habitat Z, +Habitat X].
    # AmbiX +X is front (-Habitat Z), +Y left (-Habitat X), +Z up (+Habitat Y).
    output = native[:, [0, 3, 1, 2]].copy()
    output[:, 1:] *= np.asarray([-1, 1, -1], dtype=output.dtype) / np.sqrt(3)
    return output


def audio_format(sample_rate, dt):
    return {
        'encoding': 'AmbiX',
        'ambisonic_order': 1,
        'channel_order': 'ACN',
        'channels': ['W', 'Y', 'Z', 'X'],
        'normalization': 'SN3D',
        'coordinate_frame': 'habitat_world',
        'ambix_positive_axes_in_habitat': {'X': '-Z', 'Y': '-X', 'Z': '+Y'},
        'sample_rate_hz': sample_rate,
        'window_duration_s': dt,
        'wav_subtype': 'FLOAT',
    }
