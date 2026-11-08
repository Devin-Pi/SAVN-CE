import argparse,ast,hashlib,json,tempfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--tools',type=Path,required=True);p.add_argument('--labels-code',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def get_function(path,name):
 tree=ast.parse(path.read_text())
 return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
ns={'np':np}
for path,name in [(a.tools,'swap_audio_channels'),(a.labels_code,'load_output_format_file'),(a.labels_code,'convert_output_format_polar_to_cartesian')]:
 node=get_function(path,name);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
angles=[(a,e) for a in range(-180,180,15) for e in (-75,-30,0,25,70)]+[(0,90),(0,-90)]
worst=0;count=0
with tempfile.TemporaryDirectory() as d:
 f=Path(d)/'label.csv'
 for az,el in angles:
  xyz=np.array([np.cos(np.radians(az))*np.cos(np.radians(el)),np.sin(np.radians(az))*np.cos(np.radians(el)),np.sin(np.radians(el))])
  f.write_text(f'0,0,0,{az},{el},1\n')
  for case in range(1,9):
   labels=ns['load_output_format_file'](None,f,case)
   expected=np.array(ns['convert_output_format_polar_to_cartesian'](None,labels)[0][0][2:5])
   for scale in (1,np.sqrt(3)):
    audio=np.array([[1,xyz[1]*scale,xyz[2]*scale,xyz[0]*scale]])
    out=ns['swap_audio_channels'](audio,case)[0]
    actual=out[[3,1,2]]/scale
    err=float(np.max(np.abs(actual-expected)));worst=max(worst,err);assert err<1e-12
    assert out[0]==1;count+=1
transforms=[]
for case in range(1,9):
    xyz=np.eye(3);wave=np.column_stack((np.ones(3),xyz[:,1],xyz[:,2],xyz[:,0]))
    out=ns['swap_audio_channels'](wave,case)[:,[3,1,2]];m=out.T
    transforms.append(dict(case=case,xyz_matrix=m.tolist(),determinant=int(round(np.linalg.det(m))),preserves_up=bool(np.allclose(m@np.array([0,0,1]),[0,0,1])),pure_upright_yaw=bool(np.linalg.det(m)>0 and np.allclose(m@np.array([0,0,1]),[0,0,1]))))
result=dict(transforms=transforms,status='passed',cases=count,max_error=worst,channel_order='W/Y/Z/X',coordinate_axes='right-handed +X azimuth0/front, +Y azimuth+90/left, +Z elevation+90/up',
 acs_identity_case=3,acs_changes_sh_normalization=False,
 source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.tools,a.labels_code)},
 method='Execute AST-extracted original swap_audio_channels and original CSV label conversion for every case; use integer angles to match original CSV quantization.')
a.output.write_text(json.dumps(result,indent=2));print(result)
