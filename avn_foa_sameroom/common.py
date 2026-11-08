"""Shared conventions; evaluation truth never enters the Agent API."""
import hashlib,json,math
from pathlib import Path
import numpy as np

RATE=16000
DURATION=10
VERSION='AVN-FOA-SameRoom-v0.1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def native_to_local(native,yaw):
    """RLR world N3D [W,+Hy,+Hz,+Hx] -> listener ACN/SN3D [W,Y,Z,X]."""
    from spatial_audio.foa import native_to_listener
    q=[0.,math.sin(yaw/2),0.,math.cos(yaw/2)]
    return native_to_listener(np.asarray(native,dtype=np.float32),q)


def local_unit(source,receiver,yaw):
    hx,hy,hz=np.asarray(source)-receiver
    x,y,z=-hz,-hx,hy
    c,s=math.cos(yaw),math.sin(yaw)
    u=np.array([c*x+s*y,-s*x+c*y,z],dtype=np.float64)
    return u/np.linalg.norm(u)


def angles(u):
    return float(np.degrees(np.arctan2(u[1],u[0]))),float(np.degrees(np.arcsin(np.clip(u[2],-1,1))))


def acoustic_config(spec):
    return {k:getattr(spec.acousticsConfig,k) for k in dir(spec.acousticsConfig)
            if not k.startswith('_') and isinstance(getattr(spec.acousticsConfig,k),(bool,int,float,str))}


def config_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()


def foa_contract():
    from spatial_audio import foa
    return dict(version=foa.CONTRACT_VERSION,module_sha256=sha(foa.__file__),
        native_order=list(foa.NATIVE_ORDER),native_normalization="N3D",output_order=list(foa.OUTPUT_ORDER),
        output_normalization="SN3D",axes="listener +X forward +Y left +Z up",quaternion="xyzw, local-to-Habitat active rotation",
        scope="Mapping for the locally calibrated Habitat/RLR build; another engine build must be recalibrated")
