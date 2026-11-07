"""Read official TAU metadata and direct-window FOA coefficients; no synthetic gain correction."""
import argparse,hashlib,itertools,json
from pathlib import Path
import h5py,numpy as np
from scipy.io import loadmat
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
x=loadmat(a.root/'rirdata.mat')['rirdata'][0,0];meta=loadmat(a.root/'measinfo.mat')['measinfo'];fs=int(x['fs'][0,0]);rows=[]
for name,index in [('sa203',4),('sc203',5),('se203',6),('tb103',7)]:
 path=next(a.root.glob('*'+name+'.mat'));paths=x['room'][index,0][2]
 with h5py.File(path) as f:
  references=f['rirs/foa']
  for height in (0,references.shape[0]//2,references.shape[0]-1):
   for traj in range(references.shape[1]):
    cell=f[references[height,traj]][:]
    directions=paths[traj,height][0]
    assert len(cell)==len(directions)
    for j in (0,len(cell)//2,len(cell)-1):
     ir=cell[j].T
     # A short window near the first prominent pressure peak minimizes late reverberation.
     w=ir[:round(.07*fs),0];bound=np.max(np.abs(w));on=np.flatnonzero(np.abs(w)>bound*.2)[0]
     right=min(len(w),on+round(.002*fs));peak=on+np.abs(w[on:right]).argmax()
     left=max(0,peak-round(.001*fs));right=min(len(ir),peak+round(.003*fs));short=ir[left:right]*np.hanning(right-left)[:,None]
     z=np.fft.rfft(short,n=512,axis=0);freq=np.fft.rfftfreq(512,1/fs);mask=(freq>=500)&(freq<=4000)
     q=(np.conj(z[mask,0,None])*z[mask,1:]).real.sum(axis=0)/(np.abs(z[mask,0])**2).sum()
     d=np.asarray(directions[j],dtype=float);d/=np.linalg.norm(d)
     v=q[[2,0,1]];err=float(np.degrees(np.arccos(np.clip(np.dot(v,d)/np.linalg.norm(v),-1,1))))
     rows.append(dict(room=name,height=height,traj=traj,index=j,source_unit_xyz=d.tolist(),coherent_xyz_over_w=v.tolist(),gain=float(np.dot(v,d)),angle_error_deg=err,window_samples=[int(left),int(right)]))
summary={}
for room in ('sa203','sc203','se203','tb103'):
 r=[t for t in rows if t['room']==room];g=[t['gain'] for t in r];e=[t['angle_error_deg'] for t in r]
 summary[room]=dict(count=len(r),gain_quantiles=np.quantile(g,[.1,.5,.9]).tolist(),median_angle_error_deg=float(np.median(e)))
result=dict(source='https://zenodo.org/records/6408611',metadata_sha256=hashlib.sha256((a.root/'rirdata.mat').read_bytes()).hexdigest(),official_foa_norm=str(x['foa_norm'][0]),official_foa_channel_order=str(x['foa_channel_order'][0]),sample_rate_hz=fs,summary=summary,cases=rows,
 caveat='Direct-window coherent gain is affected by real encoding filters, near field and early reflections; metadata, not these gains, determines the declared SH normalization.')
a.output.write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='cases'})
