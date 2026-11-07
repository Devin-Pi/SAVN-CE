"""Read a finetuned checkpoint and16k AmbiX WAV, without target metadata."""
import argparse,json,sys
from pathlib import Path
import numpy as np,soundfile as sf,torch
from .loader import extractor
from .common import write_json


def load(checkpoint,runtime):
    sys.path.insert(0,str(runtime));import models.audio_encoder as architecture
    saved=torch.load(str(checkpoint),map_location='cpu',mmap=True);config=saved['config'];classes=config['classes']
    fn=architecture.resnet18;architecture.resnet18=lambda *,weights:fn(weights=None)
    try:model=architecture.AudioBackBone(batch_size=1,num_video_frames=100)
    finally:architecture.resnet18=fn
    old=model.AVfnn_list[-1];model.AVfnn_list[-1]=torch.nn.Linear(old.in_features,9*len(classes))
    model.load_state_dict(saved['model_state_dict'],strict=True)
    return model.cuda().eval(),extractor(runtime).cuda().eval(),config


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--wav',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    model,front,config=load(a.checkpoint,a.runtime)
    wave,rate=sf.read(a.wav,dtype='float32',always_2d=True)
    if rate!=16000 or wave.shape!=(160000,4):raise ValueError('Expected10s16k4ch listener-local ACN/SN3D WAV')
    with torch.inference_mode():
        feature=front(torch.from_numpy(wave.T.copy())[None].cuda())
        raw=model(feature)
    import dsf_decode as decoder;decoder.CLASSES=tuple(config['classes']);decoder.ACTIVITY_THRESHOLD=.5
    result=dict(sample_rate_hz=rate,window_s=10,classes=config['classes'],frame_rate_hz=10,
                raw_accdoa=raw[0].float().cpu().tolist(),events_by_frame=decoder.decode(raw[0].float().cpu().numpy()),
                input='waveform only; no target category/position/RIR/oracle information',activity_threshold=.5,range_prediction=False)
    write_json(a.output,result)
    print('inference passed',list(raw.shape),flush=True)
if __name__=='__main__':main()
