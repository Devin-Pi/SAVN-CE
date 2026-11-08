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


def run_synchronous(initial, *, prepare, decide, advance, read_current,
                    before_action, log_event, max_steps, period_s=1., dt_s=.25):
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
    cycle = 0
    while obs['step'] < max_steps:
        before = observation_state(obs)
        assert_frozen(obs, read_current())
        cap = min(round(ratio), max_steps - obs['step'])
        control = dict(mode='synchronous_segments_v1', simulator_paused_during_decision=True,
                       nominal_period_s=period_s, native_step_s=dt_s, max_native_steps=cap,
                       max_segment_duration_s=cap*dt_s,
                       action_semantics=dict(move_forward=f'up to {cap} x 0.25m',
                                             turn_left=f'up to {cap} x 15deg',
                                             turn_right=f'up to {cap} x 15deg', stop='immediate'),
                       latest_pose=copy.deepcopy(obs['pose']),
                       pose_timestamp_s=obs['timestamp_s'],
                       current_audio_interval_s=[obs['timestamp_s'], obs['timestamp_s']+dt_s],
                       acquisition='discrete pose blocks; historical audio has its own poses',
                       interruption='collision/blockage, forward depth p10 < .45m or unknown, existing arrival evidence')
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
            obs, done = advance(action, decision)
            if action == 'stop':
                assert_frozen(previous, obs)
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
                       decision_sha256=digest, wall_time_s=time.time()))
        cycle += 1
        if done: break
    return obs, cycle
