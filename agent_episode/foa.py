"""FOA conversion for the calibrated Habitat-Sim/RLR audio sensor."""

from spatial_audio.foa import native_to_world,CONTRACT_VERSION
from spatial_audio import foa as contract
import hashlib
from pathlib import Path


def to_ambix(native):
    """Convert (..., 4) world-aligned RLR coefficients to ACN/SN3D."""
    return native_to_world(native)


def audio_format(sample_rate, dt):
    return {
        'encoding': 'AmbiX',
        'conversion_contract_version': CONTRACT_VERSION,
        'conversion_module_sha256': hashlib.sha256(Path(contract.__file__).read_bytes()).hexdigest(),
        'native_layout_scope': 'Locally calibrated Habitat-Sim/RLR build only; not a universal Ambisonics world-frame assertion',
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
