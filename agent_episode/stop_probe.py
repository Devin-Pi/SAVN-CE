"""Resolve a pending STOP with an in-place binaural yaw sweep."""
import numpy as np


def fractional_peak_lag(correlation, peak_index, lag_limit):
    """Interpolate the local GCC-PHAT magnitude peak in samples."""
    lag = peak_index - lag_limit
    if 0 < peak_index < len(correlation) - 1:
        left, center, right = np.abs(correlation[peak_index - 1:peak_index + 2])
        curvature = left - 2 * center + right
        if curvature < -1e-12:
            lag += float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return float(lag)


class StopProbe:
    # 15-degree actions: left 30, right 30, then return to the start.
    TURNS = ('turn_left',) * 2 + ('turn_right',) * 4 + ('turn_left',) * 2
    MIN_ITD_SPAN = 0.5
    MIN_FIT_MARGIN = 0.2
    MAX_FIT_ERROR = 1.0

    def __init__(self, observation_id, position, forward, left, bearings, stop_check, sample_rate, near_range_m):
        self.start_observation_id = observation_id
        self.position = position.copy()
        self.forward = forward.copy()
        self.left = left.copy()
        self.bearings = bearings
        self.passive_hypotheses = stop_check['hypotheses']
        self.passive_support = stop_check['supporting_observation_ids']
        self.nominal_lag_scale = 0.18 * sample_rate / 343
        self.near_range_m = near_range_m
        self.samples = []

    def observe(self, observation_id, position, forward, left, lag, usable):
        yaw = float(np.degrees(np.arctan2(np.dot(forward, self.left), np.dot(forward, self.forward))))
        drift = float(np.linalg.norm(position - self.position))
        if not self.samples or self.samples[-1]['observation_id'] != observation_id:
            self.samples.append({
                'observation_id': observation_id,
                'yaw_from_start_deg': yaw,
                'itd_samples': lag if usable else None,
                'predicted_nominal_itd_samples': {
                    side: float(np.dot(left, direction) * self.nominal_lag_scale)
                    for side, direction in self.bearings.items()
                },
            })

        log = {
            'start_observation_id': self.start_observation_id,
            'samples': list(self.samples),
            'position_drift_m': drift,
            'supporting_observation_ids': sorted(set(self.passive_support).union(
                sample['observation_id'] for sample in self.samples if sample['itd_samples'] is not None
            )),
        }
        if drift > 0.02:
            return None, {**log, 'phase': 'rejected', 'reason': 'position changed during yaw sweep'}
        turn_index = len(self.samples) - 1
        if turn_index < len(self.TURNS):
            return self.TURNS[turn_index], {**log, 'phase': 'probing'}
        return self._finish(log)

    def _finish(self, log):
        valid = [sample for sample in self.samples if sample['itd_samples'] is not None]
        yaws = [sample['yaw_from_start_deg'] for sample in valid]
        if (
            len(valid) < 5 or min(yaws) > -25 or max(yaws) < 25
            or self.samples[0]['itd_samples'] is None or self.samples[-1]['itd_samples'] is None
            or abs(self.samples[-1]['yaw_from_start_deg']) > 5
        ):
            return None, {**log, 'phase': 'rejected', 'reason': 'missing usable audio or incomplete yaw sweep'}

        observed = np.array([sample['itd_samples'] for sample in valid])
        centered = observed - observed.mean()
        fits = {}
        for side in self.bearings:
            predicted = np.array([sample['predicted_nominal_itd_samples'][side] for sample in valid])
            predicted -= predicted.mean()
            # Allow an offset and positive gain for the observed lag curve,
            # while retaining the sign of the predicted response to yaw.
            gain = max(0.0, float(np.dot(predicted, centered) / max(np.dot(predicted, predicted), 1e-9)))
            rmse = float(np.sqrt(np.mean((centered - gain * predicted) ** 2)))
            passive = self.passive_hypotheses[side]
            fits[side] = {
                'lag_fit_rmse_samples': rmse,
                'lag_gain': gain,
                'passive_range_supported': passive['valid'],
                'estimated_horizontal_range_m': passive.get('estimated_horizontal_range_m'),
            }

        best, other = sorted(fits, key=lambda side: fits[side]['lag_fit_rmse_samples'])
        winner = fits[best]
        margin = fits[other]['lag_fit_rmse_samples'] - winner['lag_fit_rmse_samples']
        span = float(np.ptp(observed))
        confirmed = (
            span >= self.MIN_ITD_SPAN and margin >= self.MIN_FIT_MARGIN
            and winner['lag_fit_rmse_samples'] <= self.MAX_FIT_ERROR and winner['lag_gain'] > 0
        )
        near = confirmed and winner['passive_range_supported'] and winner['estimated_horizontal_range_m'] <= self.near_range_m
        log.update({
            'hypotheses': fits,
            'selected_hypothesis': best if confirmed else None,
            'direction_confirmed': confirmed,
            'observed_itd_span_samples': span,
            'fit_margin_samples': margin,
            'phase': 'confirmed_stop' if near else 'rejected',
            'reason': (
                'ITD change confirms a supported nearby source' if near
                else 'confirmed direction has no supported near range' if confirmed
                else 'yaw-induced ITD change does not distinguish front from back'
            ),
        })
        return ('stop' if near else None), log
