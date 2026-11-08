"""Short-context encoder contract and current-250ms supervision; no navigation hooks."""
import json
import math
import types
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from .labels50 import activity
from .finetune import yaw

PROTOCOL='short-context-current250ms-v1'


def ceiling_pool_forward(self,x,pool_type='avg',pool_size=(2,2)):
    residual=self.bn_res(self.downsample_res1x1(x))
    z=F.relu(self.bn1(self.conv1(x)),inplace=True)
    z=F.relu(self.bn2(self.conv2(z)),inplace=True)
    if pool_type!='avg':raise ValueError('Short-window contract requires average pooling')
    # 40 STFT frames -> 5 pre-pool frames -> 3 outputs. Last cell contains
    # one real pre-pool feature; no waveform or future observation is padded.
    z=F.avg_pool2d(z,kernel_size=pool_size,ceil_mode=True,count_include_pad=False)
    assert z.shape==residual.shape
    return z+residual


def patch_short_context(model):
    model.conv_block.forward=types.MethodType(ceiling_pool_forward,model.conv_block)
    return model


def output_edges(window_s):
    n=math.ceil(window_s*10-1e-8)
    return np.minimum(np.arange(n+1,dtype=np.float64)/10,window_s)


def current_step_weights(window_s):
    edges=output_edges(window_s)
    widths=np.maximum(0,np.minimum(edges[1:],window_s)-np.maximum(edges[:-1],window_s-.25))
    assert abs(widths.sum()-.25)<1e-8
    return torch.tensor(widths/.25,dtype=torch.float32)


def pool_current_step(raw,window_s):
    weights=current_step_weights(window_s).to(raw.device,dtype=raw.dtype)
    assert raw.shape[1]==len(weights),(raw.shape,window_s)
    return (raw*weights[None,:,None]).sum(dim=1,keepdim=True)


class ShortData(torch.utils.data.Dataset):
    def __init__(self,data,labels,split,window_s,seed=7,train=False):
        self.root=Path(data);self.labels=Path(labels);self.window_s=window_s
        assert window_s in (.25,1.,1.5)
        self.rows=[json.loads(s) for s in (self.root/'metadata.jsonl').read_text().splitlines() if json.loads(s)['split']==split]
        self.seed=seed;self.epoch=0;self.train=train
        self.ends=np.arange(1.5,10.0001,.25) # common causal warmup across all three contexts
        self.label_cache={}

    def __len__(self):return len(self.rows) if self.train else len(self.rows)*len(self.ends)

    def label(self,index):
        if index not in self.label_cache:
            with np.load(self.labels/(self.rows[index]['sample_id']+'.npz'),allow_pickle=False) as z:
                self.label_cache[index]={k:z[k].copy() for k in ('occupancy','power','audibility_floor','unit','class_id')}
        return self.label_cache[index]

    def step_target(self,parent,end):
        z=self.label(parent);begin=round((end-.25)*20);stop=round(end*20)
        assert stop-begin==5
        a,v=activity(np.array([z['occupancy'][begin:stop].mean()]),np.array([z['power'][begin:stop].mean()]),float(z['audibility_floor']))
        target=np.zeros((1,6,4,21),np.float32)
        if a[0]:
            c=int(z['class_id']);target[0,0,0,c]=1;target[0,0,1:,c]=z['unit']
        return torch.from_numpy(target),torch.tensor(v,dtype=torch.bool)

    def __getitem__(self,index):
        parent=index if self.train else index//len(self.ends)
        if self.train:
            # Shared quarter-second endpoint/target/yaw across all contexts.
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index,0]))
            end=float(self.ends[int(rng.integers(len(self.ends)))])
        else:end=float(self.ends[index%len(self.ends)])
        start=end-self.window_s;row=self.rows[parent]
        wave,rate=sf.read(self.root/row['audio'],start=round(start*16000),frames=round(self.window_s*16000),dtype='float32',always_2d=True)
        if rate!=16000 or wave.shape!=(round(self.window_s*16000),4):raise ValueError('Invalid exact short window')
        target,valid=self.step_target(parent,end)
        wave=torch.from_numpy(wave.T.copy())
        if self.train:
            angle=float(np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index,1])).uniform(-np.pi,np.pi))
            wave,target=yaw(wave,target,angle)
        return dict(waveform=wave,adpit=target,valid_frames=valid,parent_index=parent,
            step_index=round(end*4)-1,end_s=end,start_s=start,sample_id=row['sample_id'])


class StepMetrics:
    def __init__(self,runtime,classes,rate):
        import sys
        from types import SimpleNamespace
        sys.path.insert(0,str(runtime));import dsf_decode as decoder
        from utils.SELD_evaluation_metrics import SELDMetrics
        from utils.cls_feature_class import FeatureClass
        decoder.CLASSES=tuple(classes);decoder.ACTIVITY_THRESHOLD=.5
        self.decoder=decoder;self.scorer=SELDMetrics(doa_threshold=20,nb_classes=21,average='macro')
        self.segment=lambda d,n:FeatureClass.segment_labels(SimpleNamespace(_label_resolution=rate),d,n)
    def add(self,raw,target,valid):
        prediction={i:[[e['class_id'],0,*e['direction_unit_xyz']] for e in events]
            for i,events in enumerate(self.decoder.decode(raw)) if valid[i] and events}
        truth={}
        for i in np.flatnonzero(valid):
            active=np.flatnonzero(target[i,0,0]>.5)
            if len(active):truth[int(i)]=[[int(c),0,*target[i,0,1:,c].tolist()] for c in active]
        self.scorer.update_seld_scores(self.segment(prediction,len(valid)),self.segment(truth,len(valid)))
    def result(self):return dict(zip(('ER','F20','LE','LR','SELD'),map(float,self.scorer.compute_seld_scores()[:5])))
