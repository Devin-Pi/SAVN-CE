"""One persistent process in the archived PyTorch/CUDA environment."""
import argparse
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import resample_poly
import torch

from dsf_decode import CLASSES, decode

CHECKPOINT_SHA256 = '85e380a6a8d2f1e545aa093e3f086f0a4d7830a15c41f7f79d047d7d9c1f9f90'
SOURCE_FILES = ('models/audio_encoder.py', 'models/mb_block.py', 'models/mamba/bimamba.py',
                'models/mamba/selective_scan_interface.py', 'utils/extract_IV.py',
                'utils/stft.py', 'config/tools_cfg.py')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


class DSF:
    def __init__(self, source, checkpoint):
        if sha256(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError('Unexpected DSF RAW checkpoint hash')
        sys.path.insert(0, str(source))
        import models.audio_encoder as architecture
        import config.tools_cfg as config
        from utils.extract_IV import LogmelIntensity_Extractor

        expected = {'fs': 24000, 'input_len_sec': 10, 'label_resolution': 10, 'audio_format': 'foa', 'num_classes': 13}
        expected_stft = {'n_fft': 512, 'hoplen': 150, 'numcep': 128, 'fmin': 0, 'fmax': 12000}
        if (any(config.input.get(k) != v for k, v in expected.items())
                or any(config.spectrog_param.get(k) != v for k, v in expected_stft.items())):
            raise ValueError('Archived feature settings do not match this checkpoint profile')
        state = torch.load(str(checkpoint), map_location='cpu', mmap=True)
        weights = state['state_dict']
        if not all(key.startswith('model.') for key in weights):
            raise ValueError('Unexpected checkpoint prefix')
        # All retained ResNet layers are replaced by the checkpoint. Skip ImageNet download at construction.
        resnet18 = architecture.resnet18
        architecture.resnet18 = lambda *, weights: resnet18(weights=None)
        try:
            self.model = architecture.AudioBackBone(batch_size=1, num_video_frames=100)
        finally:
            architecture.resnet18 = resnet18
        keys = self.model.load_state_dict({key.removeprefix('model.'): value for key, value in weights.items()}, strict=True)
        self.extract = LogmelIntensity_Extractor(config).eval()
        self.model = self.model.eval().cuda()
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        versions = {name: sha256(source / name) for name in SOURCE_FILES}
        versions['adapter'] = sha256(Path(__file__))
        versions['postprocess'] = sha256(Path(__file__).with_name('dsf_decode.py'))
        self.description = {
            'backend': 'dsf_audio_only', 'model_variant': 'audio_bb_mamba_15_MB_RAW_epoch17',
            'checkpoint_sha256': CHECKPOINT_SHA256, 'epoch': state['epoch'], 'source_sha256': versions,
            'class_names': CLASSES, 'tracks': 3, 'model_sample_rate_hz': 24000, 'window_s': 10,
            'frame_rate_hz': 10, 'normalization': 'per-window temporal mean/std inside original model',
            'feature_shape': [1, 7, 1600, 128], 'output_shape': [1, 100, 117],
            'strict_load': {'missing': keys.missing_keys, 'unexpected': keys.unexpected_keys},
            'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        }

    @torch.inference_mode()
    def predict(self, wave, rate):
        if (rate not in (16000, 24000) or wave.shape != (rate*10, 4)
                or not np.isfinite(wave).all() or wave.dtype.kind != 'f'):
            raise ValueError('Expected a finite 10-second FOA window at 16 or 24 kHz')
        start = time.monotonic()
        if rate == 16000:
            wave = resample_poly(wave, 3, 2, axis=0)
        features = self.extract(torch.from_numpy(np.ascontiguousarray(wave.T, dtype=np.float32))[None])
        if tuple(features.shape) != (1, 7, 1600, 128) or not torch.isfinite(features).all():
            raise ValueError('Invalid model features')
        before_infer = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        raw = self.model(features.cuda()).cpu().numpy()
        after_infer = time.monotonic()
        if raw.shape != (1, 100, 117):
            raise ValueError('Unexpected model output shape')
        frames = decode(raw[0])
        return {
            'raw_accdoa': raw[0].tolist(), 'events_by_frame': frames,
            'timing_s': {'preprocess': before_infer-start, 'inference': after_infer-before_infer,
                         'postprocess': time.monotonic()-after_infer},
            'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        try:
            model = DSF(args.source, args.checkpoint)
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            model, error = None, str(exc)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if model is None:
                    response = {'status': 'unavailable', 'reason': error}
                elif request['command'] == 'describe':
                    response = {'status': 'ok', **model.description}
                elif request['command'] == 'predict':
                    arrays = [np.load(io.BytesIO(base64.b64decode(value, validate=True)), allow_pickle=False)
                              for value in request['windows_npy_base64']]
                    rate = request['sample_rate']
                    if (rate not in (16000, 24000) or len(arrays) != 40
                            or any(a.shape != (rate//4, 4) or a.dtype != np.float32 for a in arrays)):
                        response = {'status': 'unsupported_input', 'reason': 'Expected 40 native float32 FOA windows'}
                    else:
                        response = {'status': 'ok', **model.predict(np.concatenate(arrays), rate)}
                else:
                    raise ValueError('Unknown model worker command')
            except (ValueError, KeyError, OSError, RuntimeError, TypeError, EOFError) as exc:
                response = {'status': 'error', 'reason': str(exc)}
            print(json.dumps(response, allow_nan=False, separators=(',', ':')), file=protocol, flush=True)


if __name__ == '__main__':
    main()
