"""Validate LOS on the geometry exported by the active acoustic engine itself."""
from pathlib import Path
import numpy as np
import trimesh
from .common import sha


class AcousticMesh:
    def __init__(self,sensor,path):
        path=Path(path)
        if sensor is not None and not sensor.writeSceneMeshOBJ(str(path)):raise RuntimeError('Acoustic mesh export failed')
        self.mesh=trimesh.load_mesh(path,process=False)
        if not isinstance(self.mesh,trimesh.Trimesh) or len(self.mesh.faces)==0:raise RuntimeError('Invalid acoustic geometry export')
        self.hash=sha(path)
        self.bounds=self.mesh.bounds.tolist()
        self.triangles=len(self.mesh.faces)
        self.ray=trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

    def clear(self,a,b,margin=.005):
        a,b=np.asarray(a),np.asarray(b);v=b-a;length=np.linalg.norm(v);d=v/length
        origins=np.array([a+d*margin,b-d*margin]);dirs=np.array([d,-d])
        loc,ids,_=self.ray.intersects_location(origins,dirs,multiple_hits=True)
        if not len(loc):return True
        distances=np.linalg.norm(loc-origins[ids],axis=1)
        return not bool(np.any(distances<length-2*margin))
