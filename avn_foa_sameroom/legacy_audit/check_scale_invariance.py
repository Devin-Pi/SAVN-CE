"""Controlled gain test; never use accuracy to guess undocumented historical format."""
import argparse,json,sys
from pathlib import Path
import numpy as np,soundfile as sf,torch
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--wav',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.runtime));from dsf_worker import DSF
dsf=DSF(a.runtime,a.checkpoint);x,sr=sf.read(a.wav,frames=240000,dtype='float32',always_2d=True);assert sr==24000
x2=x.copy();x2[:,1:]*=np.sqrt(3)
with torch.inference_mode():
 f=dsf.extract(torch.from_numpy(x.T.copy())[None]);g=dsf.extract(torch.from_numpy(x2.T.copy())[None])
 def norm(y):return (y-y.mean(dim=2,keepdim=True))/y.std(dim=2,keepdim=True).clamp(min=1e-8)
 nf,ng=norm(f),norm(g)
 y=dsf.model(f.cuda()).cpu();z=dsf.model(g.cuda()).cpu()
report=dict(status='completed',input=str(a.wav),experiment='Multiply every first-order waveform channel by sqrt(3), preserve W and channel order; use identical weights and original24k frontend. No training or data rewrite.',
 normalized_feature_mean_abs_delta=float((nf-ng).abs().mean()),normalized_feature_max_abs_delta=float((nf-ng).abs().max()),
 prediction_mean_abs_delta=float((y-z).abs().mean()),prediction_max_abs_delta=float((y-z).abs().max()),
 note='Model temporal feature normalization removes much constant directional-channel gain; numerical differences/floors remain. Output similarity is NOT evidence of the historical normalization.',training_started=False)
a.output.write_text(json.dumps(report,indent=2));print(report)
