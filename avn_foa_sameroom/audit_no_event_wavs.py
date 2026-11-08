import json,pathlib,numpy as np,soundfile as sf
out={}
for name in ('main','reference'):
 root=pathlib.Path('/datasets')/name;count=frames=0
 for line in (root/'metadata.jsonl').read_text().splitlines():
  r=json.loads(line)
  if r['split']!='test':continue
  with np.load(root/r['labels'],allow_pickle=False) as z:
   active=z['adpit'][:,0,0]>.5;valid=~z['ignore_mask'].astype(bool)
   if active[valid].any():continue
   assert valid.all() and len(valid)==100
   wav,rate=sf.read(root/r['audio'],dtype='float32',always_2d=True)
   assert rate==16000 and wav.shape==(160000,4) and not np.any(wav)
   count+=1;frames+=100
 out[name]=dict(all_no_event_clips=count,all_zero_wav_confirmed=True,valid_frames=frames,no_ignored_frames=True)
pathlib.Path('/experiment/analysis_full100_20261108/no_event_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
