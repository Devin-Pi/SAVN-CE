"""Experimental range evidence from signed FOA rays and observed receiver poses.

No scene assets, target metadata, navigation mesh, or depth surface assumptions.
Angular feasible regions are heuristic sensitivity bounds, not calibrated CIs.
"""
import copy
import math

FOA_STOP_METHODS = {'foa_multiview': 'foa-signed-rays-v1', 'foa_geometry': 'foa-signed-rays-v2'}
DEFAULT_FOA_TOLERANCE_DEG = {'foa_multiview': 15., 'foa_geometry': 5.}


class FoaArrival:
    VERSION=FOA_STOP_METHODS['foa_multiview']

    def __init__(self, *, rule='foa_multiview', tolerance_deg=None):
        import numpy as np
        self.np=np
        self.method=FOA_STOP_METHODS[rule]
        self.geometry=rule=='foa_geometry'
        tolerance_deg=DEFAULT_FOA_TOLERANCE_DEG[rule] if tolerance_deg is None else tolerance_deg
        if not 0.<tolerance_deg<=45.:
            raise ValueError('FOA tolerance must be in (0, 45] degrees')
        self.cos_tolerance=math.cos(math.radians(tolerance_deg))-1e-8
        self.history={};self.used=set();self.cache={};self.confirmation=None;self.episode=None
        self.config=dict(min_positions=3,position_spacing_m=.2,min_baseline_m=.5,
            min_parallax_deg=10.,angular_tolerance_deg=tolerance_deg,stop_radius_m=1.,
            confirmations=2,max_history_age_s=12.,confirmation_age_s=4.,source_stability_m=.5)

    def _report(self,anchor,checks,reason,**extra):
        ready=bool(checks and all(checks.values()))
        report=dict(method=self.method,ready=ready,reason=reason,
            evidence_id=anchor+':foa_stop',checks=checks,config=self.config.copy(),
            arrival=dict(source='foa_multiview',checks=checks),latest_fit_selection='signed_foa',
            calibration_status='heuristic angular bounds; not calibrated confidence',**extra)
        self.cache[anchor]=copy.deepcopy(report)
        return report

    def _candidate_grid(self, origin, direction):
        np=self.np
        tolerance=self.config['angular_tolerance_deg']
        seed=np.array([0.,1.,0.]) if abs(direction[1])<.9 else np.array([1.,0.,0.])
        u=np.cross(direction,seed);u/=np.linalg.norm(u);v=np.cross(direction,u)
        half_steps=math.ceil(tolerance/2.5)
        angles=np.radians(np.linspace(-tolerance,tolerance,2*half_steps+1))
        xy=np.asarray([(x,y) for x in angles for y in angles])
        rays=direction+np.tan(xy[:,0,None])*u+np.tan(xy[:,1,None])*v
        rays/=np.linalg.norm(rays,axis=1)[:,None]
        rays=rays[(rays@direction)>=self.cos_tolerance]
        radii=np.r_[np.arange(.025,2.001,.025),np.arange(2.1,15.001,.1),[20.,30.,50.,100.,1e6]]
        return (origin+radii[:,None,None]*rays[None]).reshape(-1,3)

    def _ray_support(self, candidates, p, d):
        np=self.np
        offsets=candidates[:,None]-p
        norms=np.linalg.norm(offsets,axis=-1)
        alignment=np.einsum('cni,ni->cn',offsets,d)/np.maximum(norms,1e-12)
        return (alignment>=self.cos_tolerance)&(norms>.001),alignment

    def _geometry_consensus(self, candidates, views, label):
        np=self.np
        support,alignment=self._ray_support(candidates,[v['origin'] for v in views],[v['direction'] for v in views])
        # Candidates must support the current observation. Class ranks a fit
        # within equally supported candidates but cannot shrink their range union.
        counts=np.where(support[:,0],support.sum(axis=1),0)
        best=np.flatnonzero(counts==counts.max())
        class_votes=support[best] @ np.array([int(v['label']==label) for v in views])
        class_preferred=best[class_votes==class_votes.max()]
        quality=np.where(support[class_preferred],alignment[class_preferred],0.).sum(axis=1)
        representative=class_preferred[quality.argmax()]
        selected=[v for v,keep in zip(views,support[representative]) if keep]
        return selected,candidates[best],candidates[representative]

    def _adjacent_pose(self, previous, start, origin, rotation):
        return (math.isclose(start,previous['end'],abs_tol=1e-7,rel_tol=0.)
            and self.np.allclose(origin,previous['origin'],rtol=0,atol=1e-7)
            and self.np.allclose(rotation,previous['rotation'],rtol=0,atol=1e-7))

    def update(self,result,*,audio_active):
        np=self.np
        episode=result.get('provenance',{}).get('episode_ref')
        if episode!=self.episode:
            self.history={};self.used=set();self.cache={};self.confirmation=None;self.episode=episode
        ids=result.get('provenance',{}).get('observation_ids',[])
        anchor=ids[-1] if ids else 'unavailable'
        if anchor in self.cache:
            return dict(copy.deepcopy(self.cache[anchor]),cache_hit=True)
        def unknown(reason):
            self.confirmation=None
            return self._report(anchor,{'current_usable_foa':False},reason)
        if result.get('status')!='ok' or not audio_active or len(ids)!=4 or len(set(ids))!=4:
            return unknown('No complete active 1s FOA window')
        if self.used.intersection(ids):return unknown('Overlapping audio cannot add independent support')
        poses=result.get('pose_refs',[])
        if [p['observation_id'] for p in poses]!=ids:return unknown('Pose support mismatch')
        origins=np.asarray([p['receiver_position_habitat_xyz_m'] for p in poses],float)
        rotations=np.asarray([p['robot_pose']['rotation_xyzw'] for p in poses],float)
        if not np.isfinite(origins).all() or not np.isfinite(rotations).all() or not np.allclose(origins,origins[-1],rtol=0,atol=1e-7) or not np.allclose(rotations,rotations[-1],rtol=0,atol=1e-7):
            return unknown('FOA must be collected at a fixed position and orientation')
        start,end=result['timing']['audio_start_s'],result['timing']['audio_end_s']
        if not math.isclose(end-start,1.,abs_tol=1e-7):return unknown('Expected 1s acquisition')
        previous=self.confirmation
        origin=origins[-1];rotation=rotations[-1]
        frames=result.get('events',[])
        if (self.geometry and frames==[[]] and previous and previous['count']==1
                and not previous['missed'] and self._adjacent_pose(previous,start,origin,rotation)):
            # Keep the first estimate across one missed window. The miss adds
            # no ray or vote and cannot satisfy the STOP gate.
            self.used.update(ids)
            self.confirmation=dict(previous,end=end,missed=True)
            return self._report(anchor,{'current_usable_foa':False},
                'Active confirmation window has no decoded event; collect one more stationary window',
                request_stationary_retry=True,confirmation_count=1,confirmation_reference_anchor=previous['anchor'])
        if len(frames)!=1 or len(frames[0])!=1:return unknown('No unique current event; identity is unverified')
        event=frames[0][0]
        if event.get('pose_ref')!=anchor or event.get('orientation_pose_ref')!=anchor:return unknown('Current ray uses an old pose')
        direction=np.asarray(event.get('direction_unit_habitat_xyz'),float)
        if direction.shape!=(3,) or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1)>1e-3:return unknown('Invalid signed world direction')
        direction/=np.linalg.norm(direction)
        self.used.update(ids)
        label=event['class_name']
        view=dict(anchor=anchor,origin=origin,direction=direction,end=end,label=label)
        key=None if self.geometry else label
        history=[v for v in self.history.get(key,[]) if 0<end-v['end']<=self.config['max_history_age_s']]
        history.append(view);self.history[key]=history[-12:]
        # Repeated headings at one position add temporal observations, not baseline.
        selected=[]
        for v in reversed(self.history[key]):
            if all(np.linalg.norm(v['origin']-x['origin'])>=self.config['position_spacing_m'] for x in selected):selected.append(v)
        if self.geometry and len(selected)>=3:
            candidates=self._candidate_grid(origin,direction)
            selected,feasible_points,representative=self._geometry_consensus(candidates,selected,label)
        metadata=dict(support_ids=[v['anchor'] for v in selected],predicted_class=label,
            association='signed-ray geometry, class tie-break' if self.geometry else 'same predicted class; instance identity unverified')
        if self.geometry:
            metadata['supporting_classes']=[v['label'] for v in selected]
        checks=dict(current_usable_foa=True,distinct_positions=len(selected)>=3)
        if not checks['distinct_positions']:
            self.confirmation=None
            return self._report(anchor,checks,'Need three spatially distinct, compatible signed rays',**metadata)
        p=np.asarray([v['origin'] for v in selected]);d=np.asarray([v['direction'] for v in selected])
        baseline=float(np.linalg.norm(p[:,None]-p,axis=-1).max())
        parallax=float(np.degrees(np.arccos(np.clip(np.abs(d@d.T),0,1))).max())
        checks.update(moving_baseline=baseline>=.5,parallax=parallax>=10.)
        projectors=np.eye(3)[None]-d[:,:,None]*d[:,None,:]
        a=projectors.sum(axis=0);eigenvalues=np.linalg.eigvalsh(a)
        checks['range_identifiable']=bool(eigenvalues[0]>1e-6 and eigenvalues[-1]/eigenvalues[0]<1e4)
        metadata.update(baseline_m=baseline,parallax_deg=parallax)
        if not all(checks.values()):
            self.confirmation=None
            return self._report(anchor,checks,'Insufficient geometry to infer range',**metadata)
        point=np.linalg.solve(a,np.einsum('nij,nj->i',projectors,p))
        fit_support,_=self._ray_support(point[None],p,d)
        checks['point_supported_by_signed_rays']=bool(fit_support.all())
        if self.geometry:
            if checks['point_supported_by_signed_rays']:
                feasible_points=np.vstack([feasible_points,point])
            else:
                point=representative
                checks['point_supported_by_signed_rays']=True
            # Diagnostic only, rounded for the interactive map. Range uses 3D points.
            metadata['feasible_xz_samples']=np.unique(np.round(feasible_points[:,[0,2]],2),axis=0).tolist()
        else:
            candidates=np.vstack([self._candidate_grid(origin,direction),point])
            support,_=self._ray_support(candidates,p,d)
            feasible_points=candidates[support.all(axis=1)]
        ranges=np.linalg.norm(feasible_points-origin,axis=1)
        checks['consistent_signed_rays']=bool(len(ranges))
        if not len(ranges):
            self.confirmation=None
            return self._report(anchor,checks,'Signed-ray cones have no consistent intersection',**metadata)
        lower,upper=float(ranges.min()),float(ranges.max())
        checks['entire_feasible_range_near']=upper<=1.+1e-6
        current_range=float(np.linalg.norm(point-origin))
        checks['point_estimate_near']=current_range<=1.+1e-6
        near=all(checks.values())
        stable=bool(previous and np.linalg.norm(point-previous['point'])<=.5)
        if stable and self.geometry:
            stable=self._adjacent_pose(previous,start,origin,rotation)
        elif stable:
            stable=previous['class']==label and 0<end-previous['end']<=4.
        count=(previous['count']+1 if stable else 1) if near else 0
        self.confirmation=dict(point=point,end=end,**{'class':label},count=count,
            origin=origin,rotation=rotation,anchor=anchor,missed=False) if near else None
        checks['fresh_stable_confirmation']=count>=2
        metadata.update(estimated_range_m=current_range,estimated_range_interval_m=[lower,upper],
            inferred_source_world_xyz=point.tolist(),confirmation_count=count,
            depth_requirement='none; front depth remains for collision avoidance only',
            note='3D Euclidean acoustic range, not guaranteed navigable distance; far checks are finite-grid sensitivity tests')
        if self.geometry:
            metadata['request_stationary_confirmation']=count==1
            if count>=2:metadata['confirmation_reference_anchor']=previous['anchor']
        return self._report(anchor,checks,'FOA multiview near evidence ready' if all(checks.values()) else 'Range remains far, uncertain, or unconfirmed',**metadata)


def policy_evidence(gate):
    """Expose STOP readiness only; never send inferred source coordinates/rays."""
    fields=('method','evidence_id','ready','reason','checks','estimated_range_interval_m','calibration_status')
    return {k:copy.deepcopy(gate[k]) for k in fields if k in gate}
