"""Frozen epoch83 one-second model, current250ms output, sensor-only inputs."""
import argparse
import base64
import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial_audio.foa import world_foa_to_listener,CONTRACT_VERSION

EXPECTED='e8c1539fb72aca7cdf40c40c9d90368dd19bffef93ca7824f00dc1bf52e38a6d'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


class ShortCandidate:
    def __init__(self,training_source,runtime,checkpoint):
        if sha(checkpoint)!=EXPECTED:raise ValueError('One-second checkpoint SHA mismatch')
        sys.path.insert(0,str(training_source))
        from avn_foa_sameroom.infer_finetuned import load
        from avn_foa_sameroom.short_window import patch_short_context,pool_current_step
        self.model,self.front,self.config=load(checkpoint,runtime)
        patch_short_context(self.model);self.pool=pool_current_step
        assert self.config['window_s']==1 and self.config['step_s']==.25 and len(self.config['classes'])==21
        import dsf_decode
        self.decoder=dsf_decode;dsf_decode.CLASSES=tuple(self.config['classes']);dsf_decode.ACTIVITY_THRESHOLD=.5
        import spatial_audio.foa as contract
        paths=[Path(__file__),Path(contract.__file__)]+[training_source/'avn_foa_sameroom'/name for name in ('loader.py','infer_finetuned.py','short_window.py','train_short.py')]
        paths += [runtime/'models/audio_encoder.py',runtime/'utils/extract_IV.py',runtime/'utils/stft.py',runtime/'dsf_decode.py']
        self.description=dict(status='ok',backend='avn_foa_short1s',model_variant='avn_short_context_1s_epoch083',
            epoch=83,checkpoint_sha256=EXPECTED,source_sha256={str(p):sha(p) for p in paths},
            conversion_contract_version=CONTRACT_VERSION,class_names=self.config['classes'],tracks=3,
            model_sample_rate_hz=16000,window_s=1,output_update_period_s=.25,output_scope='current last250ms only',
            feature_shape=[1,7,160,128],raw_sequence_shape=[1,10,189],output_shape=[1,1,189],
            frontend=dict(sample_rate_hz=16000,n_fft=512,win_length=342,hop_length=100,n_mels=128,fmax=8000),
            strict_load=dict(missing=[],unexpected=[]),
            normalization='ACN/SN3D waveform gain preserved; original feature time mean/std; no scaler/resampling',
            coordinate_profile='entire observed1s world FOA rotated once into latest observed listener frame',
            torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0))

    @torch.inference_mode()
    def predict(self,wave,rate,rotation):
        if rate!=16000 or wave.shape!=(16000,4) or wave.dtype!=np.float32 or not np.isfinite(wave).all():
            raise ValueError('Expected exactly1s native16k float32 world FOA')
        start=time.monotonic();local=world_foa_to_listener(wave,rotation)
        features=self.front(torch.from_numpy(local.T.copy())[None].cuda())
        assert features.shape==(1,7,160,128) and torch.isfinite(features).all()
        torch.cuda.synchronize();before=time.monotonic();torch.cuda.reset_peak_memory_stats()
        sequence=self.model(features);raw=self.pool(sequence,1.)[0].float().cpu().numpy()
        assert sequence.shape==(1,10,189) and raw.shape==(1,189) and np.isfinite(raw).all()
        after=time.monotonic();events=self.decoder.decode(raw);end=time.monotonic()
        return dict(status='ok',raw_accdoa=raw.tolist(),events_by_frame=events,
            timing_s=dict(preprocess=before-start,inference=after-before,postprocess=end-after),
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            native_window_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
            listener_window_sha256=hashlib.sha256(local.tobytes()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--training-source',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
    protocol=sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        model=ShortCandidate(a.training_source,a.runtime,a.checkpoint)
        for line in sys.stdin:
            try:
                req=json.loads(line)
                if req['command']=='describe':result=model.description
                elif req['command']=='predict':
                    blocks=[np.load(io.BytesIO(base64.b64decode(x,validate=True)),allow_pickle=False) for x in req['windows_npy_base64']]
                    if len(blocks)!=4 or any(x.shape!=(4000,4) or x.dtype!=np.float32 for x in blocks):
                        raise ValueError('Exactly four observed250ms FOA blocks required')
                    result=model.predict(np.concatenate(blocks),req['sample_rate'],req['anchor_rotation_xyzw'])
                else:raise ValueError('Unsupported command')
            except (KeyError,ValueError,TypeError,RuntimeError) as exc:result=dict(status='error',reason=str(exc))
            print(json.dumps(result,allow_nan=False,separators=(',',':')),file=protocol,flush=True)


if __name__=='__main__':main()
