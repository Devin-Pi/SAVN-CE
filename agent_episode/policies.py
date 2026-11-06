"""Transparent rule-based baselines, not trained localization or LLM policies."""
import numpy as np
from habitat_sim.utils.common import quat_from_coeffs,quat_rotate_vector

from .arrival import ArrivalEvidence
from .stop_probe import StopProbe, fractional_peak_lag

class AudioDepthAgent:
    # A close estimate needs agreement over distinct observations before stopping.
    STOP_RANGE_M = 1.0
    STOP_MAX_SPREAD_M = 0.5
    STOP_CONFIRMATIONS = 3
    STOP_MAX_BEARING_ERROR_DEG = 30.0
    STOP_MIN_SUPPORT = 4
    STOP_MIN_BASELINE_M = 0.25
    STOP_MIN_PARALLAX_DEG = 10.0
    STOP_RANGE_FIT_MARGIN_DEG = 3.0
    STOP_REPROBE_BASELINE_M = 0.5

    def __init__(self, verify=False):
        self.verify = verify
        self.history = []
        self.turns = 0
        self._stop_candidates = []
        self._stop_probe = None
        self._next_probe_step = 0
        self._last_probe_position = None
        self._arrival = None

    def _fit_stop_hypothesis(self, position, bearing, views):
        if len(views) < self.STOP_MIN_SUPPORT:
            return {'valid': False, 'reason': 'too few distinct audio observations'}

        # Fit range along one current ray. Each past view may independently
        # support the source with its front or back ray.
        ranges = np.arange(1, 151) / 10.0
        origin = position[[0, 2]]
        ray = bearing[[0, 2]]
        ray_norm = np.linalg.norm(ray)
        if ray_norm < 1e-6:
            return {'valid': False, 'reason': 'no horizontal bearing'}
        candidates = origin + ranges[:, None] * (ray / ray_norm)
        positions = np.array([view['p'][[0, 2]] for view in views])
        rays = np.array([
            [view['direction'][[0, 2]], view['back_direction'][[0, 2]]]
            for view in views
        ])
        rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-6)
        offsets = candidates[:, None, :] - positions
        directions = offsets / np.maximum(np.linalg.norm(offsets, axis=-1, keepdims=True), 1e-6)
        alignment = np.einsum('rvi,vhi->rvh', directions, rays).max(axis=2)
        angles = np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))
        errors = np.sqrt(np.mean(np.minimum(angles, 45.0) ** 2, axis=1))
        best = int(np.argmin(errors))
        # Competing ranges remain plausible even when their parallax is weak.
        plausible = ranges[errors <= errors[best] + self.STOP_RANGE_FIT_MARGIN_DEG]
        inliers = angles[best] <= self.STOP_MAX_BEARING_ERROR_DEG
        support_count = int(inliers.sum())
        baseline = 0.0
        parallax = 0.0
        if support_count:
            support_positions = positions[inliers]
            baseline = float(np.max(np.linalg.norm(support_positions[:, None] - support_positions, axis=-1)))
            support_directions = directions[best, inliers]
            # Acute angle between sight lines, excluding parallel/opposite
            # views that do not constrain range.
            cross = np.outer(support_directions[:, 0], support_directions[:, 1])
            parallax = float(np.degrees(np.arcsin(np.clip(np.max(np.abs(cross - cross.T)), 0.0, 1.0))))
        checks = {
            'enough_inliers': support_count >= self.STOP_MIN_SUPPORT and support_count / len(views) >= 0.6,
            'moving_baseline': baseline > self.STOP_MIN_BASELINE_M,
            'parallax': parallax >= self.STOP_MIN_PARALLAX_DEG,
        }
        return {
            'valid': all(checks.values()),
            'checks': checks,
            'estimated_horizontal_range_m': float(ranges[best]),
            'source_xz': candidates[best].tolist(),
            'plausible_range_m': [float(plausible[0]), float(plausible[-1])],
            'best_far_fit_error_deg': float(np.min(errors[ranges > self.STOP_RANGE_M])),
            'angular_fit_error_deg': float(errors[best]),
            'support_count': support_count,
            'observation_count': len(views),
            'baseline_m': baseline,
            'parallax_deg': parallax,
            'supporting_observation_ids': [view['ref'] for view, keep in zip(views, inliers) if keep],
        }

    def _stop_check(self, estimate, position, front_bearing, back_bearing, observation_id):
        views = list({
            view['ref']: view for view in self.history[-24:] if view['usable_for_stop']
        }.values())
        hypotheses = {
            name: self._fit_stop_hypothesis(position, bearing, views)
            for name, bearing in [('front', front_bearing), ('back', back_bearing)]
        }
        supported = [(name, fit) for name, fit in hypotheses.items() if fit['valid']]
        selected_name, selected = min(
            supported, key=lambda item: item[1]['angular_fit_error_deg'], default=(None, None)
        )
        checks = {
            'audio_verified': estimate['accepted_for_action'],
            'supported_hypothesis': selected is not None,
            'estimated_range_within_1m': (
                selected is not None and selected['estimated_horizontal_range_m'] <= self.STOP_RANGE_M
            ),
            'far_alternatives_rejected': selected is not None and all(
                'best_far_fit_error_deg' in fit
                and fit['best_far_fit_error_deg'] > selected['angular_fit_error_deg'] + self.STOP_RANGE_FIT_MARGIN_DEG
                for fit in hypotheses.values()
            ),
        }
        if not all(checks.values()):
            self._stop_candidates.clear()
        elif not self._stop_candidates or self._stop_candidates[-1][0] != observation_id:
            support = {
                ref for _, fit in supported for ref in fit['supporting_observation_ids']
            }
            self._stop_candidates.append((observation_id, selected['source_xz'], sorted(support)))
            self._stop_candidates = self._stop_candidates[-self.STOP_CONFIRMATIONS:]

        spread = None
        if self._stop_candidates:
            points = np.asarray([candidate[1] for candidate in self._stop_candidates])
            spread = float(np.max(np.linalg.norm(points[:, None] - points[None, :], axis=2)))
        checks['consecutive_consistent_estimates'] = (
            len(self._stop_candidates) == self.STOP_CONFIRMATIONS
            and spread <= self.STOP_MAX_SPREAD_M
        )
        return {
            'ready': all(checks.values()),
            'checks': checks,
            'hypotheses': hypotheses,
            'selected_hypothesis': selected_name,
            'estimated_horizontal_range_m': selected['estimated_horizontal_range_m'] if selected else None,
            'estimate_spread_m': spread,
            'confirming_observation_ids': [candidate[0] for candidate in self._stop_candidates],
            'supporting_observation_ids': sorted({
                ref for candidate in self._stop_candidates for ref in candidate[2]
            }),
        }

    def _probe_action(self, observation, estimate, position, forward, left, front, back, lag, usable):
        stop_check = estimate['stop_check']
        ref = observation['observation_id']
        if self._stop_probe is None:
            enough_new_motion = (
                self._last_probe_position is None
                or np.linalg.norm(position - self._last_probe_position) >= self.STOP_REPROBE_BASELINE_M
            )
            stop_check['enough_new_motion_for_probe'] = bool(enough_new_motion)
            if not stop_check['ready'] or observation['step'] < self._next_probe_step or not enough_new_motion:
                return None
            self._last_probe_position = position.copy()
            self._stop_probe = StopProbe(
                ref, position, forward, left, {'front': front, 'back': back},
                stop_check, observation['sample_rate'], self.STOP_RANGE_M,
            )
        action, log = self._stop_probe.observe(ref, position, forward, left, lag, usable)
        estimate['stop_probe'] = log
        if action == 'stop':
            side = log['selected_hypothesis']
            self._arrival = ArrivalEvidence(
                observation['step'], self._stop_probe.position, self._stop_probe.bearings[side],
                self._stop_probe.passive_hypotheses[side]['plausible_range_m'],
                log['supporting_observation_ids'],
            )
            action = None
            log['phase'] = 'direction_confirmed'
            log['reason'] = 'verified direction and range saved for the arrival check'
        if log['phase'] != 'probing':
            self._stop_probe = None
            self._stop_candidates.clear()
            self._next_probe_step = observation['step'] + 4
        return action

    def decide(self,o):
        a=o['binaural_audio'];rms=float(np.sqrt(np.mean(a*a)));power=np.mean(a*a,axis=0)+1e-12
        ild=float(10*np.log10(power[0]/power[1]));n=1<<int(np.ceil(np.log2(len(a)*2)))
        spectrum=np.fft.rfft(a[:,0],n)*np.conj(np.fft.rfft(a[:,1],n));spectrum/=np.abs(spectrum)+1e-12
        cc=np.fft.irfft(spectrum,n);lag_limit=10;small=np.concatenate([cc[-lag_limit:],cc[:lag_limit+1]])
        peak_index = int(np.argmax(np.abs(small)))
        lag = peak_index - lag_limit
        peak=float(np.max(np.abs(small))/(np.mean(np.abs(small))+1e-12))
        audible=rms>1e-5;clipped=float(np.mean(np.abs(a)>=.999))
        # 18 cm nominal ear spacing; front/back ambiguity is deliberately retained.
        angle=float(np.arcsin(np.clip((lag/o['sample_rate'])*343/.18,-1,1)))
        q=quat_from_coeffs(np.array(o['pose']['rotation_xyzw']));p=np.array(o['pose']['position'])
        forward=quat_rotate_vector(q,np.array([0.,0.,-1.]));left=quat_rotate_vector(q,np.array([-1.,0.,0.]))
        front=np.cos(angle)*forward+np.sin(angle)*left;back=-np.cos(angle)*forward+np.sin(angle)*left
        ref=o['observation_id'];evidence=[ref+':audio',ref+':depth',ref+':pose']
        depth=o['depth_m'][...,0];h,w=depth.shape;front_clear=float(np.median(depth[h//3:2*h//3,2*w//5:3*w//5]));left_clear=float(np.median(depth[h//3:2*h//3,:w//3]));right_clear=float(np.median(depth[h//3:2*h//3,2*w//3:]))
        self.history.append({
            'audible': audible, 'lag': lag, 'ref': ref, 'p': p, 'direction': front,
            'back_direction': back,
            'usable_for_stop': audible and clipped < 0.01 and peak > 2,
        })
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
        probe_action = None
        if self.verify:
            estimate['stop_check'] = self._stop_check(estimate, p, front, back, ref)
            fractional_lag = fractional_peak_lag(small, peak_index, lag_limit)
            probe_action = self._probe_action(
                o, estimate, p, forward, left, front, back, fractional_lag,
                self.history[-1]['usable_for_stop'],
            )
        arrival = None
        if self.verify and self._arrival is not None:
            arrival = self._arrival.check(o['step'], p, forward, left, o['depth_m'])
            estimate['arrival_check'] = arrival
            if not arrival['checks']['fresh_verified_sweep']:
                self._arrival = None
        if arrival is not None and arrival['ready'] and probe_action is None:
            action = 'stop'
            reason = 'verified acoustic interval is nearby and current depth supports access'
            arrival_evidence = [
                observation_id + ':' + modality
                for observation_id in arrival['supporting_observation_ids']
                for modality in ('audio', 'pose')
            ]
            evidence = list(dict.fromkeys(evidence + arrival_evidence))
        elif probe_action is not None:
            action = probe_action
            probe = estimate['stop_probe']
            reason = probe.get('reason', 'turn in place to measure ITD changes before stopping')
            probe_evidence = [
                observation_id + ':' + modality
                for observation_id in probe['supporting_observation_ids']
                for modality in ('audio', 'pose')
            ]
            evidence = list(dict.fromkeys(evidence + probe_evidence))
        elif front_clear<.65 or o['collided_previous_step'] or o.get('blocked_previous_step',False):
            action='turn_left' if left_clear>=right_clear else 'turn_right';reason='depth/collision blocks forward; turn toward greater clearance'
        elif accepted and abs(ild)>2.5 and self.turns<3:
            action='turn_left' if ild>0 else 'turn_right';reason='react to accepted binaural level asymmetry';self.turns+=1
        elif self.verify and audible and not accepted and self.turns<2:
            action='turn_left';reason='acquire another viewpoint because current audio evidence failed verification';self.turns+=1
        else:
            action='move_forward';reason='move into depth-clear space to collect a spatial baseline';self.turns=0
        decision={'policy':'evidence_gate' if self.verify else 'baseline','action':action,'reason':reason,'evidence_ids':evidence,'audio_rms':rms,'front_clearance_m':front_clear,'estimate':estimate}
        return decision
