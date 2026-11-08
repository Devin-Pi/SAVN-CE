"""Observed quaternion and calibrated world AmbiX conversions; no scene/target input."""
import math

CONTRACT_VERSION="savnce-rlr-foa-1.0"
# Applies to the locally calibrated Habitat/RLR build, not every Ambisonics implementation.
NATIVE_ORDER=("W","+HabitatY","+HabitatZ","+HabitatX")
OUTPUT_ORDER=("W","Y","Z","X")


def listener_basis(q):
    if len(q)!=4 or not all(math.isfinite(v) for v in q):raise ValueError('Invalid observed quaternion')
    x,y,z,w=q;n=sum(v*v for v in q)
    if abs(n-1)>1e-4:raise ValueError('Quaternion must be unit length')
    # Columns of the local->Habitat rotation matrix, then forward/left/up axes.
    right=[1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)]
    up=[2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)]
    back=[2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)]
    return [[-v for v in back],[-v for v in right],up]


def listener_to_habitat(vector,q):
    basis=listener_basis(q)
    return [sum(vector[j]*basis[j][i] for j in range(3)) for i in range(3)]


def world_foa_to_listener(wave,q,*,coordinate_frame="habitat_world",normalization="SN3D"):
    if coordinate_frame!="habitat_world" or normalization!="SN3D":
        raise ValueError("Expected calibrated world ACN/SN3D; refuse repeated listener rotation or normalization")
    import numpy as np
    if wave.ndim!=2 or wave.shape[1]!=4 or not np.isfinite(wave).all():raise ValueError('Invalid FOA')
    # Fixed AmbiX XYZ -> Habitat XYZ -> anchor listener forward/left/up.
    habitat=np.column_stack((-wave[:,1],wave[:,2],-wave[:,3]))
    local=habitat @ np.asarray(listener_basis(q),dtype=wave.dtype).T
    out=wave.copy();out[:,1]=local[:,1];out[:,2]=local[:,2];out[:,3]=local[:,0]
    return out


def native_to_world(native,*,native_order=NATIVE_ORDER,native_normalization="N3D"):
    if tuple(native_order)!=NATIVE_ORDER or native_normalization!="N3D":
        raise ValueError("Expected calibrated native N3D; refuse repeated SN3D normalization")
    import numpy as np
    if native.ndim!=2 or native.shape[1]!=4 or not np.isfinite(native).all():raise ValueError("Expected finite calibrated native FOA")
    output=native[:,[0,3,1,2]].copy()
    output[:,1:]*=np.asarray([-1,1,-1],dtype=output.dtype)/np.sqrt(3)
    return output


def native_to_listener(native,q):
    return world_foa_to_listener(native_to_world(native),q)
