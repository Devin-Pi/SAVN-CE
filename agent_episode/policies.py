"""Transparent rule-based baselines, not trained localization or LLM policies."""
import numpy as np
from habitat_sim.utils.common import quat_from_coeffs,quat_rotate_vector

class AudioDepthAgent:
    def __init__(self,verify=False):self.verify=verify;self.history=[];self.turns=0
    def decide(self,o):
        a=o['binaural_audio'];rms=float(np.sqrt(np.mean(a*a)));power=np.mean(a*a,axis=0)+1e-12
        ild=float(10*np.log10(power[0]/power[1]));n=1<<int(np.ceil(np.log2(len(a)*2)))
        spectrum=np.fft.rfft(a[:,0],n)*np.conj(np.fft.rfft(a[:,1],n));spectrum/=np.abs(spectrum)+1e-12
        cc=np.fft.irfft(spectrum,n);lag_limit=10;small=np.concatenate([cc[-lag_limit:],cc[:lag_limit+1]])
        lag=int(np.argmax(np.abs(small))-lag_limit);peak=float(np.max(np.abs(small))/(np.mean(np.abs(small))+1e-12))
        audible=rms>1e-5;clipped=float(np.mean(np.abs(a)>=.999))
        # 18 cm nominal ear spacing; front/back ambiguity is deliberately retained.
        angle=float(np.arcsin(np.clip((lag/o['sample_rate'])*343/.18,-1,1)))
        q=quat_from_coeffs(np.array(o['pose']['rotation_xyzw']));p=np.array(o['pose']['position'])
        forward=quat_rotate_vector(q,np.array([0.,0.,-1.]));left=quat_rotate_vector(q,np.array([-1.,0.,0.]))
        front=np.cos(angle)*forward+np.sin(angle)*left;back=-np.cos(angle)*forward+np.sin(angle)*left
        ref=o['observation_id'];evidence=[ref+':audio',ref+':depth',ref+':pose']
        depth=o['depth_m'][...,0];h,w=depth.shape;front_clear=float(np.median(depth[h//3:2*h//3,2*w//5:3*w//5]));left_clear=float(np.median(depth[h//3:2*h//3,:w//3]));right_clear=float(np.median(depth[h//3:2*h//3,2*w//3:]))
        self.history.append({'audible':audible,'lag':lag,'ref':ref,'p':p,'direction':front})
        stable=len(self.history)>=3 and sum(x['audible'] for x in self.history[-3:])==3 and np.std([x['lag'] for x in self.history[-3:]])<=2
        checks={'audio_present':audible,'not_clipped':clipped<.01,'gcc_peak_ratio_gt_2':peak>2,'three_observations_stable_lag':bool(stable)}
        accepted=audible if not self.verify else all(checks.values())
        estimate={'world_position_estimate':None,'front_candidate_xyz':(p+2*front).tolist() if audible else None,'back_candidate_xyz':(p+2*back).tolist() if audible else None,'range_assumption_m':2.0,'range_measured':False,'method':'GCC-PHAT bearing with nominal 18cm ear spacing; two 2m hypotheses','front_back_ambiguous':True,'bearing_left_degrees':float(np.degrees(angle)) if audible else None,'gcc_lag_samples':lag if audible else None,'ild_db':ild if audible else None,'uncertainty':'uncalibrated HRTF/reverberation and unknown range','evidence_ids':evidence,'accepted_for_action':bool(accepted),'verification_checks':checks}
        rays=[x for x in self.history[-24:] if x['audible']]
        if len(rays)>=4:
            ps=np.array([x['p'][[0,2]] for x in rays]);ds=np.array([x['direction'][[0,2]] for x in rays]);normals=np.stack([-ds[:,1],ds[:,0]],axis=1)
            rhs=np.sum(normals*ps,axis=1);point,_,_,_=np.linalg.lstsq(normals,rhs,rcond=None);condition=float(np.linalg.cond(normals));residual=float(np.sqrt(np.mean((normals@point-rhs)**2)));baseline=float(np.max(np.linalg.norm(ps-ps[0],axis=1)))
            valid=baseline>.25 and condition<20 and residual<.75 and np.linalg.norm(point-p[[0,2]])<15
            estimate['triangulation']={'condition_number':condition if np.isfinite(condition) else None,'residual_m':residual,'baseline_m':baseline,'geometric_checks_passed':bool(valid),'evidence_ids':[x['ref']+':audio' for x in rays]}
            if valid:estimate['world_position_estimate']=[float(point[0]),None,float(point[1])]
        if front_clear<.65 or o['collided_previous_step'] or o.get('blocked_previous_step',False):
            action='turn_left' if left_clear>=right_clear else 'turn_right';reason='depth/collision blocks forward; turn toward greater clearance'
        elif accepted and abs(ild)>2.5 and self.turns<3:
            action='turn_left' if ild>0 else 'turn_right';reason='react to accepted binaural level asymmetry';self.turns+=1
        elif self.verify and audible and not accepted and self.turns<2:
            action='turn_left';reason='acquire another viewpoint because current audio evidence failed verification';self.turns+=1
        else:
            action='move_forward';reason='move into depth-clear space to collect a spatial baseline';self.turns=0
        decision={'policy':'evidence_gate' if self.verify else 'baseline','action':action,'reason':reason,'evidence_ids':evidence,'audio_rms':rms,'front_clearance_m':front_clear,'estimate':estimate}
        return decision
