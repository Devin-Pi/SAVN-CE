"""Native16k candidate worker. Whole history is expressed in one observed anchor frame."""
import argparse,base64,contextlib,io,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from spatial_audio.foa import world_foa_to_listener,CONTRACT_VERSION

EXPECTED='8bf8cd0e521084f8ca2a685a8e9a12b769ae476ba37880be41bf53f0613c0fc1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


class Candidate:
    def __init__(self,training_source,runtime,checkpoint):
        if sha(checkpoint)!=EXPECTED:raise ValueError('Candidate checkpoint SHA mismatch')
        sys.path.insert(0,str(training_source))
        from avn_foa_sameroom.infer_finetuned import load
        self.model,self.front,self.config=load(checkpoint,runtime)
        import dsf_decode
        self.decoder=dsf_decode;self.decoder.CLASSES=tuple(self.config['classes']);self.decoder.ACTIVITY_THRESHOLD=.5
        assert len(self.config['classes'])==21
        import spatial_audio.foa as contract
        paths=[Path(contract.__file__),Path(__file__),Path(__file__).with_name('spatial_frames.py'),training_source/'avn_foa_sameroom/loader.py',training_source/'avn_foa_sameroom/infer_finetuned.py',runtime/'models/audio_encoder.py',runtime/'utils/extract_IV.py',runtime/'utils/stft.py',runtime/'dsf_decode.py']
        self.description=dict(conversion_contract_version=CONTRACT_VERSION,status='ok',backend='avn_foa_candidate',model_variant='avn_foa_dense15h_v02_epoch097',epoch=97,
            checkpoint_sha256=EXPECTED,source_sha256={str(p):sha(p) for p in paths},class_names=self.config['classes'],tracks=3,
            model_sample_rate_hz=16000,window_s=10,frame_rate_hz=10,feature_shape=[1,7,1600,128],output_shape=[1,100,189],
            strict_load=dict(missing=[],unexpected=[]),normalization='ACN/SN3D waveform gain preserved; no external scaler; model time-axis mean/std; no resampling',
            frontend=dict(sample_rate_hz=16000,n_fft=512,win_length=342,hop_length=100,n_mels=128,fmax=8000),
            coordinate_profile='whole 10s history rotated from fixed world AmbiX into latest observed listener frame; output orientation uses that one anchor',
            torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0))

    @torch.inference_mode()
    def predict(self,wave,rate,rotation):
        if rate!=16000 or wave.shape!=(160000,4) or wave.dtype!=np.float32 or not np.isfinite(wave).all():raise ValueError('Expected native16k 10s float32 FOA')
        t=time.monotonic();local=world_foa_to_listener(wave,rotation)
        features=self.front(torch.from_numpy(np.ascontiguousarray(local.T))[None].cuda())
        assert tuple(features.shape)==(1,7,1600,128) and torch.isfinite(features).all()
        before=time.monotonic();torch.cuda.reset_peak_memory_stats();raw=self.model(features).float().cpu().numpy();after=time.monotonic()
        assert raw.shape==(1,100,189) and np.isfinite(raw).all()
        return dict(status='ok',raw_accdoa=raw[0].tolist(),events_by_frame=self.decoder.decode(raw[0]),
            timing_s=dict(preprocess=before-t,inference=after-before,postprocess=time.monotonic()-after),
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            native_window_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),listener_window_sha256=hashlib.sha256(local.tobytes()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--training-source',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
    protocol=sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        model=Candidate(a.training_source,a.runtime,a.checkpoint)
        for line in sys.stdin:
            try:
                req=json.loads(line)
                if req['command']=='describe':result=model.description
                elif req['command']=='predict':
                    arrays=[np.load(io.BytesIO(base64.b64decode(b,validate=True)),allow_pickle=False) for b in req['windows_npy_base64']]
                    if len(arrays)!=40 or any(x.shape!=(4000,4) or x.dtype!=np.float32 for x in arrays):raise ValueError('Exactly 40 observed quarter-second windows required')
                    result=model.predict(np.concatenate(arrays),req['sample_rate'],req['anchor_rotation_xyzw'])
                else:raise ValueError('Unsupported command')
            except (ValueError,KeyError,TypeError,RuntimeError) as e:result=dict(status='error',reason=str(e))
            print(json.dumps(result,allow_nan=False,separators=(',',':')),file=protocol,flush=True)

if __name__=='__main__':main()
