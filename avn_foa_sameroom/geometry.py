"""Strict .house floor polygons plus static render-mesh collision ray tests."""
from collections import defaultdict
import numpy as np
import habitat_sim as hs


def polygon_relation(point,polygon,margin=.04):
    """Boundary is unknown. Return True only strictly inside a simple polygon."""
    p=np.asarray(point);v=np.asarray(polygon)
    if len(v)<3:return None
    nxt=np.roll(v,-1,axis=0);d=nxt-v
    t=np.clip(np.sum((p-v)*d,axis=1)/(np.sum(d*d,axis=1)+1e-20),0,1)
    if np.min(np.linalg.norm(p-(v+t[:,None]*d),axis=1))<margin:return None
    x,y=p;inside=False
    for a,b in zip(v,nxt):
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return inside


class Rooms:
    def __init__(self,path):
        self.regions={};self.objects={};surfaces={};vertices=defaultdict(list);self.categories={}
        for line in open(path):
            t=line.split()
            if not t:continue
            if t[0]=='R':
                self.regions[int(t[1])]={'floor':float(t[11]),'ceiling':float(t[14]),'polygons':[],'label':t[5]}
            elif t[0]=='S':surfaces[int(t[1])]=(int(t[2]),t[4])
            elif t[0]=='V':vertices[int(t[2])].append(list(map(float,t[4:6])))
            elif t[0]=='O':self.objects[int(t[1])]={'room_id':int(t[2]),'category_index':int(t[3]),'house_center':list(map(float,t[4:7]))}
            elif t[0]=='C':self.categories[int(t[1])]=t[5]
        for sid,(rid,kind) in surfaces.items():
            if kind=='F' and rid in self.regions:self.regions[rid]['polygons'].append(np.asarray(vertices[sid]))
        for r in self.regions.values():
            if r['polygons']:
                v=np.concatenate(r['polygons']);r['bounds']=(v.min(axis=0)-.04,v.max(axis=0)+.04)
            else:r['bounds']=None

    def locate(self,habitat_point):
        x,height,hz=habitat_point
        matches=[];ambiguous=False
        for rid,r in self.regions.items():
            if not r['floor']+.04<height<r['ceiling']-.04:continue
            if r['bounds'] is None:continue
            lo,hi=r['bounds']
            if x<lo[0] or x>hi[0] or -hz<lo[1] or -hz>hi[1]:continue
            vals=[polygon_relation([x,-hz],p) for p in r['polygons']]
            if any(v is None for v in vals):ambiguous=True
            if any(v is True for v in vals):matches.append(rid)
        return matches[0] if len(matches)==1 and not ambiguous else None


def clear_segment(sim,a,b,endpoint_margin=.015):
    delta=np.asarray(b,dtype=np.float64)-a;length=float(np.linalg.norm(delta))
    if length<=endpoint_margin*2:return False
    direction=delta/length
    for start,end in ((np.asarray(a)+direction*endpoint_margin,np.asarray(b)-direction*endpoint_margin),
                      (np.asarray(b)-direction*endpoint_margin,np.asarray(a)+direction*endpoint_margin)):
        ray=hs.geo.Ray(start.astype(np.float32),((end-start)/np.linalg.norm(end-start)).astype(np.float32))
        result=sim.cast_ray(ray,max_distance=float(np.linalg.norm(end-start)))
        if any(h.ray_distance<float(np.linalg.norm(end-start))-.001 for h in result.hits):return False
    return True


def free_connected_point(sim,point,clearance=.04):
    """Conservative free space certificate, not a watertight mesh occupancy claim.

    A clear segment joins the point to air above a navigable floor. Reject near
    surfaces with six short, bidirectional casts. No point is moved on acceptance.
    """
    p=np.asarray(point,dtype=np.float64)
    base=np.asarray(sim.pathfinder.snap_point(p-[0,1.5,0]),dtype=np.float64)
    if not np.isfinite(base).all() or np.linalg.norm((base-p)[[0,2]])>.8 or not 0.15<p[1]-base[1]<2.8:return None
    anchor=base+[0,.15,0]
    if not clear_segment(sim,anchor,p):return None
    for axis in np.eye(3):
        if not clear_segment(sim,p-axis*clearance,p+axis*clearance,endpoint_margin=.001):return None
    return base
