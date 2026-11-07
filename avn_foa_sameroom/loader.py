"""Native16k bridge into DSF preprocessing and ADPIT target layout."""
import json,sys,types
from pathlib import Path
import numpy as np
import soundfile as sf
import torch


def extractor(runtime):
    sys.path.insert(0,str(runtime))
    from utils.extract_IV import LogmelIntensity_Extractor
    from utils.stft import STFT
    conf=types.SimpleNamespace(input={'fs':16000},spectrog_param=dict(n_fft=512,hoplen=100,numcep=128,fmin=0,fmax=8000))
    model=LogmelIntensity_Extractor(conf)
    # Preserve the old ~21.33ms window while using a real 16k frontend. Zero-pad to FFT512.
    model.stft_extractor=STFT(n_fft=512,hop_length=100,win_length=342,window='hann',center=True,pad_mode='reflect',freeze_parameters=True)
    return model.eval()


class Dataset(torch.utils.data.Dataset):
    def __init__(self,root,split='train',window_s=10,hop_s=.5):
        self.root=Path(root);self.rows=[json.loads(l) for l in (self.root/'metadata.jsonl').read_text().splitlines() if json.loads(l)['split']==split]
        self.window_s=window_s;self.hop_s=hop_s
        if window_s*10%1 or hop_s*10%1 or not 0<window_s<=10:raise ValueError('10Hz-aligned windows/hops required')
        self.starts=np.arange(0,10-window_s+1e-6,hop_s)

    def __len__(self):return len(self.rows)*len(self.starts)

    def __getitem__(self,index):
        row=self.rows[index//len(self.starts)];start=float(self.starts[index%len(self.starts)])
        data,rate=sf.read(self.root/row['audio'],start=round(start*16000),frames=round(self.window_s*16000),dtype='float32',always_2d=True)
        if rate!=16000 or data.shape!=(round(self.window_s*16000),4):raise ValueError('Invalid native16k FOA payload')
        with np.load(self.root/row['labels'],allow_pickle=False) as label:
            a=round(start*10);b=a+round(self.window_s*10)
            target=label['adpit'][a:b].copy();mask=~label['ignore_mask'][a:b].astype(bool)
        return {'waveform':torch.from_numpy(data.T.copy()),'adpit':torch.from_numpy(target),
                'valid_frames':torch.from_numpy(mask),'sample_id':row['sample_id'],'start_s':start}


def masked_adpit_loss(prediction,target,valid_frames):
    """Single-source specialization of original ADPIT auxiliary track duplication.

    Original single instance a0 is copied to all three prediction tracks; zeros
    for inactive classes. Ignore frames are removed BEFORE reduction. Refuse
    multi-instance targets, which require the full original permutation loss.
    """
    if torch.any(target[:,:,1:]!=0):raise ValueError('This dataset bridge supports single source only')
    b,t,_,_,c=target.shape
    pred=prediction.reshape(b,t,3,3,c)
    truth=target[:,:,0,1:,:]*target[:,:,0,0:1,:]
    error=(pred-truth[:,:,None]).square().mean(dim=(2,3,4))
    return error[valid_frames].mean() if valid_frames.any() else error.sum()*0
