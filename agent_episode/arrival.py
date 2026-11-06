"""Short-lived arrival evidence built from a supported acoustic estimate."""
import numpy as np


class ArrivalEvidence:
    MAX_AGE_STEPS = 8
    MAX_RANGE_M = 0.9
    DEPTH_TOLERANCE_M = 0.2
    SURFACE_TOLERANCE_M = 0.25

    def __init__(self, step, position, bearing, range_interval, evidence_ids, source='yaw_probe',
                 depth_observation_id=None):
        ray = bearing[[0, 2]]
        ray = ray / np.linalg.norm(ray)
        self.endpoints = position[[0, 2]] + np.asarray(range_interval)[:, None] * ray
        self.step = step
        self.evidence_ids = evidence_ids
        self.source = source
        self.depth_observation_id = depth_observation_id

    @staticmethod
    def _depth_support(offset, forward, left, depth_m):
        # The online camera is aligned with the body and has a 90-degree HFOV.
        axial = float(np.dot(offset, forward[[0, 2]]))
        lateral = float(np.dot(offset, left[[0, 2]]))
        if axial < 0.05 or abs(lateral) >= axial * 0.9:
            return None
        height, width = depth_m.shape[:2]
        column = int(width * (1.0 - lateral / axial) / 2.0)
        half_window = max(1, width // 50)
        patch = depth_m[
            2*height//5:3*height//5,
            max(0, column-half_window):min(width, column+half_window+1),
        ]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if not valid.size or valid.size < patch.size * 0.8:
            return None
        # A lower quantile makes narrow obstacles in the ray patch count.
        return float(np.quantile(valid, 0.25)) - axial

    def check(self, step, position, forward, left, depth_m):
        offsets = self.endpoints - position[[0, 2]]
        upper_range = float(np.max(np.linalg.norm(offsets, axis=1)))
        clearances = [self._depth_support(offset, forward, left, depth_m) for offset in offsets]
        freshness_check = 'fresh_verified_sweep' if self.source == 'yaw_probe' else 'fresh_passive_estimate'
        checks = {
            freshness_check: 0 <= step - self.step <= self.MAX_AGE_STEPS,
            'entire_range_interval_near': upper_range <= self.MAX_RANGE_M,
            'visible_depth_does_not_block_interval': all(
                clearance is not None and clearance >= -self.DEPTH_TOLERANCE_M
                for clearance in clearances
            ),
        }
        if self.source == 'passive':
            # A clear ray can cross empty space. The observed surface must
            # also be close to the end of the proposed acoustic interval.
            checks['surface_corroborates_passive_range'] = (
                checks['visible_depth_does_not_block_interval']
                and min(clearances) <= self.SURFACE_TOLERANCE_M
            )
        return {
            'ready': all(checks.values()),
            'depth_observation_id': self.depth_observation_id,
            'source': self.source,
            'checks': checks,
            'age_steps': step - self.step,
            'estimated_upper_horizontal_range_m': upper_range,
            'depth_clearance_m': clearances,
            'supporting_observation_ids': self.evidence_ids,
        }
