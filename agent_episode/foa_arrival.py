"""Experimental range evidence from signed FOA rays and observed receiver poses.

No scene assets, target metadata, navigation mesh, or depth surface assumptions.
Angular feasible regions are heuristic sensitivity bounds, not calibrated CIs.
"""
import copy
import math


class FoaArrival:
    VERSION='foa-signed-rays-v1'
    def __init__(self):
        import numpy as np
        self.np=np
        self.history={};self.used=set();self.cache={};self.confirmation=None;self.episode=None
        self.config=dict(min_positions=3,position_spacing_m=.2,min_baseline_m=.5,
            min_parallax_deg=10.,angular_tolerance_deg=15.,stop_radius_m=1.,
            confirmations=2,max_history_age_s=12.,confirmation_age_s=4.,source_stability_m=.5)

    def _report(self,anchor,checks,reason,**extra):
        ready=bool(checks and all(checks.values()))
        report=dict(method=self.VERSION,ready=ready,reason=reason,
            evidence_id=anchor+':foa_stop',checks=checks,config=self.config.copy(),
            arrival=dict(source='foa_multiview',checks=checks),latest_fit_selection='signed_foa',
            calibration_status='heuristic angular bounds; not calibrated confidence',**extra)
        self.cache[anchor]=copy.deepcopy(report)
        return report

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
        frames=result.get('events',[])
        if len(frames)!=1 or len(frames[0])!=1:return unknown('No unique current event; identity is unverified')
        event=frames[0][0]
        if event.get('pose_ref')!=anchor or event.get('orientation_pose_ref')!=anchor:return unknown('Current ray uses an old pose')
        direction=np.asarray(event.get('direction_unit_habitat_xyz'),float)
        if direction.shape!=(3,) or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1)>1e-3:return unknown('Invalid signed world direction')
        direction/=np.linalg.norm(direction)
        start,end=result['timing']['audio_start_s'],result['timing']['audio_end_s']
        if not math.isclose(end-start,1.,abs_tol=1e-7):return unknown('Expected 1s acquisition')
        self.used.update(ids)
        label=event['class_name'];origin=origins[-1]
        view=dict(anchor=anchor,ids=ids,origin=origin,direction=direction,end=end)
        history=[v for v in self.history.get(label,[]) if 0<end-v['end']<=self.config['max_history_age_s']]
        history.append(view);self.history[label]=history[-12:]
        # Repeated headings at one position add temporal observations, not baseline.
        selected=[]
        for v in reversed(self.history[label]):
            if all(np.linalg.norm(v['origin']-x['origin'])>=self.config['position_spacing_m'] for x in selected):selected.append(v)
        checks=dict(current_usable_foa=True,distinct_positions=len(selected)>=3)
        if not checks['distinct_positions']:
            self.confirmation=None
            return self._report(anchor,checks,'Need three spatially distinct signed-ray observations',support_ids=[v['anchor'] for v in selected])
        p=np.asarray([v['origin'] for v in selected]);d=np.asarray([v['direction'] for v in selected])
        baseline=float(np.linalg.norm(p[:,None]-p,axis=-1).max())
        parallax=float(np.degrees(np.arccos(np.clip(np.abs(d@d.T),0,1))).max())
        checks.update(moving_baseline=baseline>=.5,parallax=parallax>=10.)
        projectors=np.eye(3)[None]-d[:,:,None]*d[:,None,:]
        a=projectors.sum(axis=0);eigenvalues=np.linalg.eigvalsh(a)
        checks['range_identifiable']=bool(eigenvalues[0]>1e-6 and eigenvalues[-1]/eigenvalues[0]<1e4)
        metadata=dict(support_ids=[v['anchor'] for v in selected],predicted_class=label,
            baseline_m=baseline,parallax_deg=parallax,association='same predicted class; instance identity unverified')
        if not all(checks.values()):
            self.confirmation=None
            return self._report(anchor,checks,'Insufficient geometry to infer range',**metadata)
        point=np.linalg.solve(a,np.einsum('nij,nj->i',projectors,p))
        point_offsets=point-p
        checks['point_supported_by_signed_rays']=bool((np.einsum('ni,ni->n',point_offsets,d)/np.maximum(np.linalg.norm(point_offsets,axis=1),1e-12)>=math.cos(math.radians(15.))-1e-8).all())
        # Search the latest signed cone over near and far ranges, never its mirror.
        seed=np.array([0.,1.,0.]) if abs(direction[1])<.9 else np.array([1.,0.,0.])
        u=np.cross(direction,seed);u/=np.linalg.norm(u);v=np.cross(direction,u)
        angles=np.radians(np.arange(-15.,15.01,2.5));xy=np.asarray([(x,y) for x in angles for y in angles])
        rays=direction+np.tan(xy[:,0,None])*u+np.tan(xy[:,1,None])*v
        rays/=np.linalg.norm(rays,axis=1)[:,None]
        rays=rays[(rays@direction)>=math.cos(math.radians(15.))-1e-8]
        radii=np.r_[np.arange(.025,2.001,.025),np.arange(2.1,15.001,.1),[20.,30.,50.,100.,1e6]]
        candidates=(origin+radii[:,None,None]*rays[None]).reshape(-1,3)
        candidates=np.vstack([candidates,point])
        offsets=candidates[:,None]-p
        norms=np.linalg.norm(offsets,axis=-1)
        alignment=np.einsum('cni,ni->cn',offsets,d)/np.maximum(norms,1e-12)
        feasible=(alignment>=math.cos(math.radians(15.))-1e-8).all(axis=1)&(norms>.001).all(axis=1)
        ranges=np.linalg.norm(candidates[feasible]-origin,axis=1)
        checks['consistent_signed_rays']=bool(len(ranges))
        if not len(ranges):
            self.confirmation=None
            return self._report(anchor,checks,'Signed-ray cones have no consistent intersection',**metadata)
        lower,upper=float(ranges.min()),float(ranges.max())
        checks['entire_feasible_range_near']=upper<=1.+1e-6
        current_range=float(np.linalg.norm(point-origin))
        checks['point_estimate_near']=current_range<=1.+1e-6
        near=all(checks.values())
        previous=self.confirmation
        stable=bool(previous and previous['class']==label and 0<end-previous['end']<=4. and np.linalg.norm(point-previous['point'])<=.5)
        count=(previous['count']+1 if stable else 1) if near else 0
        self.confirmation=dict(point=point,end=end,**{'class':label},count=count) if near else None
        checks['fresh_stable_confirmation']=count>=2
        metadata.update(estimated_range_m=current_range,estimated_range_interval_m=[lower,upper],
            inferred_source_world_xyz=point.tolist(),confirmation_count=count,
            depth_requirement='none; front depth remains for collision avoidance only',
            note='3D Euclidean acoustic range, not guaranteed navigable distance; far checks are finite-grid sensitivity tests')
        return self._report(anchor,checks,'FOA multiview near evidence ready' if all(checks.values()) else 'Range remains far, uncertain, or unconfirmed',**metadata)


def policy_evidence(gate):
    """Expose STOP readiness only; never send inferred source coordinates/rays."""
    fields=('method','evidence_id','ready','reason','checks','estimated_range_interval_m','calibration_status')
    return {k:copy.deepcopy(gate[k]) for k in fields if k in gate}
