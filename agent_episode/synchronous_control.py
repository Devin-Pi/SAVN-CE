"""Bounded synchronous action segments. No engine, model, or target truth imports."""
import copy
import hashlib
import json
import math
import time


def observation_state(obs):
    return {k: copy.deepcopy(obs[k]) for k in ('observation_id', 'step', 'timestamp_s', 'pose')}


def assert_frozen(expected, actual):
    if observation_state(expected) != observation_state(actual):
        raise RuntimeError('Simulator advanced while waiting for a high-level decision')


def interruption(action, data, obs):
    if obs['collided_previous_step'] or obs['blocked_previous_step']:
        return 'collision_or_blockage'
    if data['stop_gate']['ready']:
        return 'existing_arrival_evidence'
    center = data['summary']['depth'].get('center')
    if action == 'move_forward' and (center is None or center['p10_m'] < .45):
        return 'forward_depth_guard'
    return None


def horizontal_yaw_deg(pose, point):
    """Shortest signed yaw from observed forward to the estimated source."""
    x,y,z,w=pose['rotation_xyzw']
    fx,fz=-2*(x*z+y*w),2*(x*x+y*y)-1
    tx=point[0]-pose['position'][0]
    tz=point[2]-pose['position'][2]
    if not all(math.isfinite(v) for v in (fx,fz,tx,tz)) or min(math.hypot(fx,fz),math.hypot(tx,tz))<1e-6:
        return None
    return math.degrees(math.atan2(fz*tx-fx*tz,fx*tx+fz*tz))


def run_synchronous(initial, *, prepare, decide, advance, read_current,
                    before_action, log_event, max_steps, period_s=1., dt_s=.25,
                    listen_advance=None, after_listen=None, log_listening=None, face_on_stop=False):
    """Callbacks stay serialized; only advance() may move the simulator.

    One accepted primitive is held for <=4 native steps. The policy is told the
    exact action semantics. Sensor-only guards can interrupt; they never STOP.
    """
    ratio = period_s / dt_s
    if dt_s != .25 or period_s != 1. or not math.isclose(ratio, round(ratio)):
        raise ValueError('Synchronous mode requires 1s decisions and .25s native steps')
    if max_steps <= initial['step']:
        raise ValueError('No remaining native step budget')
    obs = initial
    data = prepare(obs)
    if listen_advance is not None and (after_listen is None or log_listening is None):
        raise ValueError('Stationary mode requires acquisition callbacks')

    def collect_listening(cycle, purpose):
        nonlocal obs, data
        start=observation_state(obs);window=[];done=False
        log_listening(dict(event='listening_started',cycle=cycle,purpose=purpose,state=start))
        for _ in range(min(4,max_steps-obs['step'])):
            previous=obs
            obs,done=listen_advance(cycle)
            if obs['step']!=previous['step']+1 or not math.isclose(obs['timestamp_s']-previous['timestamp_s'],dt_s):
                raise RuntimeError('Listening did not advance one native audio step')
            if math.dist(obs['pose']['position'],start['pose']['position'])>1e-7 or math.dist(obs['pose']['rotation_xyzw'],start['pose']['rotation_xyzw'])>1e-7:
                raise RuntimeError('Robot pose changed during stationary acquisition')
            window.append(observation_state(obs));data=prepare(obs)
            if done:break
        listening=dict(event='listening_finished',cycle=cycle,purpose=purpose,start_state=start,final_state=observation_state(obs),
            native_steps=len(window),advanced_sim_time_s=len(window)*dt_s,complete=len(window)==4,
            observation_ids=[x['observation_id'] for x in window],
            audio_interval_s=[window[0]['timestamp_s'],window[-1]['timestamp_s']+dt_s],unchanged_pose_verified=True)
        log_listening(listening)
        if len(window)==4:data=after_listen(obs,window,data)
        return listening,done or obs['step']>=max_steps

    cycle = 0
    while obs['step'] < max_steps:
        listening=None
        first_near_anchor=None
        missed_confirmation_anchor=None
        if listen_advance is not None:
            listening,exhausted=collect_listening(cycle,'decision')
            if exhausted:break
            if data['stop_gate'].get('request_stationary_confirmation'):
                first_near_anchor=obs['observation_id']
                listening,exhausted=collect_listening(cycle,'stop_confirmation')
                if exhausted:break
                if data['stop_gate'].get('request_stationary_retry'):
                    missed_confirmation_anchor=obs['observation_id']
                    listening,exhausted=collect_listening(cycle,'stop_confirmation_retry')
                    if exhausted:break
        before = observation_state(obs)
        assert_frozen(obs, read_current())
        cap = min(round(ratio), max_steps - obs['step'])
        control = dict(mode='synchronous_segments_v1', simulator_paused_during_decision=True,
                       nominal_period_s=period_s, native_step_s=dt_s, max_native_steps=cap,
                       max_segment_duration_s=cap*dt_s,
                       action_semantics=dict(move_forward=f'up to {cap} x 0.25m',
                                             turn_left=f'up to {cap} x 15deg',
                                             turn_right=f'up to {cap} x 15deg',
                                             stop='face estimated source in place then stop' if face_on_stop else 'immediate'),
                       latest_pose=copy.deepcopy(obs['pose']),
                       pose_timestamp_s=obs['timestamp_s'],
                       current_audio_interval_s=[obs['timestamp_s'], obs['timestamp_s']+dt_s],
                       acquisition='discrete pose blocks; historical audio has its own poses',
                       interruption='collision/blockage, forward depth p10 < .45m or unknown, existing arrival evidence')
        if listening is not None:
            control.update(mode='stationary_listen_v1',nominal_period_s=None,
                cadence='1s stationary acquisition plus <=1s movement, then pause for policy',
                listening_observation_ids=listening['observation_ids'],
                listening_audio_interval_s=listening['audio_interval_s'],
                listening_duration_s=1.,listening_pose_unchanged_verified=True)
            if first_near_anchor is not None:
                control.update(first_near_anchor=first_near_anchor,stop_confirmation_listen_s=1.)
            if missed_confirmation_anchor is not None:
                control.update(missed_confirmation_anchor=missed_confirmation_anchor,stop_confirmation_retry_listen_s=1.)
        decision_data = copy.deepcopy(data)
        decision_data['summary']['control'] = control
        started = time.time()
        log_event(dict(event='decision_started', cycle=cycle, state=before, wall_time_s=started))
        try:
            decision = decide(decision_data)
            assert_frozen(obs, read_current())
        except Exception:
            log_event(dict(event='decision_failed', cycle=cycle, state=before, wall_time_s=time.time()))
            raise
        if decision['observation_id'] != obs['observation_id']:
            raise ValueError('Decision refers to an old observation')
        if decision['action'] not in ('move_forward', 'turn_left', 'turn_right', 'stop'):
            raise ValueError('Invalid segment action')
        digest = hashlib.sha256(json.dumps(decision, sort_keys=True).encode()).hexdigest()
        log_event(dict(event='decision_returned', cycle=cycle, state=before,
                       wall_time_s=time.time(), frozen_verified=True, decision_sha256=digest,
                       max_native_steps=cap, decision=decision))
        executed = 0
        stop_facing = {}
        reason = 'period_complete'
        done = False
        while executed < cap:
            action = decision['action']
            if not executed and action == 'move_forward':
                center = data['summary']['depth'].get('center')
                if center is None or center['p10_m'] < .45:
                    raise ValueError('Unsafe forward action must be excluded before requesting a decision')
            guard = interruption(action, data, obs) if executed else None
            if guard:
                reason = guard
                break
            before_action(obs, decision)
            previous = obs
            gate = data['stop_gate']
            if face_on_stop and action=='stop' and not gate['ready']:
                raise ValueError('FOA STOP requires accepted arrival evidence')
            point = gate.get('inferred_source_world_xyz') if face_on_stop and action=='stop' and gate['ready'] else None
            yaw = horizontal_yaw_deg(previous['pose'],point) if point is not None else None
            if yaw is not None and abs(yaw)>1.:
                obs, done = advance(action, decision, stop_yaw_deg=yaw)
                residual = horizontal_yaw_deg(obs['pose'],point)
                if (obs['step']!=previous['step']+1 or not math.isclose(obs['timestamp_s']-previous['timestamp_s'],dt_s)
                        or math.dist(obs['pose']['position'],previous['pose']['position'])>1e-7
                        or residual is None or abs(residual)>1.):
                    raise RuntimeError('Terminal turn did not align the robot in place')
                executed=1
                stop_facing=dict(terminal_facing=dict(yaw_deg=yaw,residual_deg=residual,
                    status='aligned',evidence_id=gate['evidence_id']))
            else:
                obs, done = advance(action, decision)
                if action=='stop':
                    assert_frozen(previous, obs)
                    if face_on_stop:
                        status='already_aligned' if yaw is not None else 'direction_unavailable'
                        stop_facing=dict(terminal_facing=dict(yaw_deg=0.,residual_deg=yaw,status=status))
            if action == 'stop':
                if not done: raise RuntimeError('STOP did not terminate')
                reason = 'stop'
                break
            if obs['step'] != previous['step']+1 or not math.isclose(obs['timestamp_s']-previous['timestamp_s'], dt_s):
                raise RuntimeError('Unexpected native transition')
            executed += 1
            data = prepare(obs)
            if done:
                reason = 'step_budget'
                break
            guard = interruption(action, data, obs)
            if guard:
                reason = guard
                break
        log_event(dict(event='segment_finished', cycle=cycle, decision_anchor=before,
                       final_state=observation_state(obs), native_steps=executed,
                       advanced_sim_time_s=executed*dt_s, reason=reason, done=done,
                       decision_sha256=digest, wall_time_s=time.time(), **stop_facing))
        cycle += 1
        if done: break
    return obs, cycle
