import argparse,collections,hashlib,itertools,json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft

p=argparse.ArgumentParser();p.add_argument('--wav',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
x,sr=sf.read(a.wav,dtype='float64',always_2d=True);assert x.shape[1]==4
labels=collections.defaultdict(list)
for l in a.labels.read_text().splitlines():
 f,c,s,az,el,*_=map(float,l.split(','));az,el=np.radians([az,el]);labels[int(f)].append([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
frames=[]
for f,dirs in sorted(labels.items()):
 if len(dirs)!=1:continue
 # All labels in the adjacent 100ms bins must refer to a single source as well.
 if any(len(labels.get(g,[]))!=1 for g in (f-1,f+1)):continue
 clip=x[round(f*.1*sr):round((f+1)*.1*sr)]
 if len(clip)<512:continue
 freqs,t,z=stft(clip.T,fs=sr,nperseg=512,noverlap=384,boundary=None,padded=False)
 w=z[0];powers=np.abs(w)**2
 if powers.mean()<1e-10:continue
 bins=(freqs>=500)&(freqs<=4000)
 cross=(np.conj(w)[None]*z[1:]).real
 # Estimate coherent first-order coefficients relative to W, not normalized IV.
 q=cross[:,bins].sum(axis=(1,2))/powers[bins].sum()
 frames.append(dict(frame=f,u=dirs[0],q=q.tolist(),power=float(powers[bins].sum())))
q=np.array([r['q'] for r in frames]);u=np.array([r['u'] for r in frames]);powers=np.array([r['power'] for r in frames])
assert len(q)>=3
rank=[]
for perm in itertools.permutations(range(3)):
 for sign in itertools.product((-1,1),repeat=3):
  v=q[:,perm]*sign
  norms=np.linalg.norm(v,axis=1)
  err=np.degrees(np.arccos(np.clip(np.sum(v*u,axis=1)/np.maximum(norms,1e-20),-1,1)))
  rank.append(dict(perm=list(perm),sign=list(sign),median_error_deg=float(np.median(err)),p25_error_deg=float(np.quantile(err,.25))))
rank.sort(key=lambda r:r['median_error_deg']);best=rank[0]
v=q[:,best['perm']]*best['sign'];cos=np.sum(v*u,axis=1)/np.linalg.norm(v,axis=1);angle=np.degrees(np.arccos(np.clip(cos,-1,1)))
gain=np.sum(v*u,axis=1)
selected=(angle<=10)&(powers>=np.median(powers))
result=dict(wav=str(a.wav),wav_sha256=hashlib.sha256(a.wav.read_bytes()).hexdigest(),label_sha256=hashlib.sha256(a.labels.read_bytes()).hexdigest(),sample_rate_hz=sr,duration_s=len(x)/sr,
 single_source_frames=len(frames),ranked_channel_mappings=rank[:8],selected_frames=int(selected.sum()),selection='angle<=10deg, W bandpower>=median among single-source frames; 500-4000Hz',
 coherent_gain_quantiles=np.quantile(gain[selected],[.1,.25,.5,.75,.9]).tolist() if selected.any() else [],
 all_gain_quantiles=np.quantile(gain,[.1,.25,.5,.75,.9]).tolist(),
 selected_abs_error_to_sn3d=float(np.median(np.abs(gain[selected]-1))) if selected.any() else None,
 selected_abs_error_to_n3d=float(np.median(np.abs(gain[selected]-np.sqrt(3)))) if selected.any() else None,
 best_mapping_median_error_deg=best['median_error_deg'],frames=frames)
a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('frames','ranked_channel_mappings')}));print(best)
