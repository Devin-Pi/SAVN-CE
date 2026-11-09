"""Transform received RGB-D, binaural samples and pose into model evidence."""

import base64
import io
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from habitat_sim.utils.common import quat_from_coeffs, quat_rotate_vector

from .arrival import ArrivalEvidence
from .policies import AudioDepthAgent
from .stop_probe import fractional_peak_lag
from .foa_arrival import FoaArrival


ARRAYS = {'rgb', 'depth_m', 'binaural_audio', 'position', 'rotation_xyzw'}
DIRECTION_MARGIN_DEG = 3.0


def png64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def audio_features(audio, sample_rate):
    left, right = audio.astype(np.float64).T
    power = np.mean(audio.astype(np.float64)**2, axis=0)
    rms = np.sqrt(power)
    clipped = float(np.mean(np.abs(audio) >= .999))
    n_fft = 1 << (2 * len(audio) - 1).bit_length()
    spectrum = np.fft.rfft(left, n_fft) * np.conj(np.fft.rfft(right, n_fft))
    cc = np.fft.irfft(spectrum / np.maximum(np.abs(spectrum), 1e-12), n_fft)
    limit = int(np.ceil(sample_rate * .18 / 343))
    # L*conj(R) peaks at t_L-t_R. Display t_R-t_L so positive means left leads.
    curve = np.concatenate([cc[-limit:], cc[:limit+1]])[::-1]
    peak_index = int(np.argmax(np.abs(curve)))
    delay = float(fractional_peak_lag(curve, peak_index, limit))
    peak_ratio = float(np.max(np.abs(curve)) / (np.mean(np.abs(curve)) + 1e-12))
    active = bool(np.max(rms) > 1e-5)
    usable = active and clipped < .01 and peak_ratio > 2 and 0 < peak_index < 2 * limit
    # Signed bearing relative to forward and back respectively, towards the left.
    angle = np.arcsin(np.clip(delay / sample_rate * 343 / .18, -1, 1)) if usable else None
    scalars = {
        'active': active,
        'usable_for_bearing': usable,
        'rms_l_r': rms.round(9).tolist(),
        'clipped_fraction': round(clipped, 5),
        'gcc_peak_ratio': round(peak_ratio, 3) if active else None,
        'itd_us_positive_left_leads': round(1e6 * delay / sample_rate, 2) if usable else None,
        'ild_db_positive_left_louder': round(float(10 * np.log10((power[0] + 1e-12) /
                                                                (power[1] + 1e-12))), 2) if active else None,
        'bearing_candidates_deg_left_positive': {
            'front': round(float(np.degrees(angle)), 1),
            'back': round(float(180 - np.degrees(angle) if angle >= 0 else -180 - np.degrees(angle)), 1),
        } if angle is not None else None,
    }
    return scalars, angle, curve


def audio_image(audio, sr, scalars, curve):
    image = Image.new('RGB', (500, 350), '#111b29')
    draw = ImageDraw.Draw(image)
    ink, muted = '#eaf4ff', '#a6b9ce'
    draw.text((12, 8), 'BINAURAL EVIDENCE  |  one observed audio window', fill=ink)
    n_fft, hop = 256, 64
    window = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(audio.T, n_fft, axis=-1)[:, ::hop]
    mag = 2 * np.abs(np.fft.rfft(frames * window, axis=-1)) / window.sum()
    db = 20 * np.log10(np.maximum(mag, 1e-8))
    for channel, label in enumerate(('LEFT', 'RIGHT')):
        top = 32 + 83 * channel
        value = np.clip((db[channel].T[::-1] + 90) / 65, 0, 1)
        pixels = np.stack([12 + 238 * value**2, 24 + 210 * value, 43 + 127 * value], axis=-1)
        image.paste(Image.fromarray(pixels.astype(np.uint8)).resize((411, 63)), (71, top))
        draw.rectangle((70, top-1, 482, top+63), outline=muted)
        draw.text((12, top+21), label, fill=ink)
        draw.text((16, top), f'{sr/2000:g} kHz', fill=muted)
        draw.text((16, top+49), '0 kHz', fill=muted)
    draw.text((71, 182), f'0 to {len(audio)/sr*1000:g} ms   |   shared STFT scale -90 to -25 dBFS', fill=muted)
    draw.text((12, 208), 'GCC-PHAT   |   positive lag: left leads', fill=ink)
    x = np.linspace(71, 482, len(curve))
    y = 260 - curve / max(1e-6, np.max(np.abs(curve))) * 30
    draw.line((71, 260, 482, 260), fill='#54667a')
    draw.line((276, 228, 276, 291), fill='#54667a')
    draw.line(list(zip(x.tolist(), y.tolist())), fill='#56ead6', width=2)
    lag_limit_us = (len(curve)//2) * 1e6 / sr
    draw.text((71, 291), f'-{lag_limit_us:g} us', fill=muted)
    draw.text((270, 291), '0', fill=muted)
    draw.text((432, 291), f'+{lag_limit_us:g} us', fill=muted)
    bearing = scalars['bearing_candidates_deg_left_positive']
    detail = (f"ITD {scalars['itd_us_positive_left_leads']} us  |  front {bearing['front']} deg / back {bearing['back']} deg"
              if bearing else 'SILENT' if not scalars['active'] else 'BEARING UNRELIABLE')
    draw.text((12, 325), detail + '  |  range unmeasured by this frame', fill=ink)
    return png64(image)


def depth_summary(depth):
    h, w = depth.shape[:2]
    bands = np.array_split(depth[h*2//5:h*3//5, :, 0], 3, axis=1)
    result = {}
    for name, band in zip(('left', 'center', 'right'), bands):
        valid = band[band > 0]
        result[name] = {'p10_m': round(float(np.quantile(valid, .10)), 2),
                        'median_m': round(float(np.median(valid)), 2)} if valid.size else None
    return result


class PassiveArrival:
    """Track competing directions from normal navigation. Never requests motion."""

    def __init__(self):
        self.fitter = AudioDepthAgent(verify=True, stop_probe=False)
        self.views = []
        self.arrival = None

    def update(self, ident, step, p, forward, left, depth, angle):
        fits = {}
        selected = None
        if angle is not None:
            side = np.sin(angle) * left
            rays = {'front': np.cos(angle)*forward + side, 'back': -np.cos(angle)*forward + side}
            self.views.append({'p': p, 'direction': rays['front'], 'back_direction': rays['back'],
                               'ref': ident + ':audio'})
            self.views = self.views[-24:]
            separated = []
            for view in reversed(self.views):
                if all(np.linalg.norm(view['p'] - prior['p']) >= .1 for prior in separated):
                    separated.append(view)
            fits = {name: self.fitter._fit_stop_hypothesis(p, ray, separated)
                    for name, ray in rays.items()}
            supported = {name: fit for name, fit in fits.items() if fit['valid']}
            if supported:
                selected = min(supported, key=lambda name: supported[name]['angular_fit_error_deg'])
                fit = fits[selected]
                other = fits['back' if selected == 'front' else 'front']
                ceiling = fit['angular_fit_error_deg'] + DIRECTION_MARGIN_DEG
                unique_direction = other.get('angular_fit_error_deg', -1) > ceiling
                far_rejected = all(f.get('best_far_fit_error_deg', -1) > ceiling for f in fits.values())
                if unique_direction and far_rejected and fit['plausible_range_m'][1] <= 1.5:
                    arrival = ArrivalEvidence(step, p, rays[selected], fit['plausible_range_m'],
                                              fit['supporting_observation_ids'], source='passive',
                                              depth_observation_id=ident)
                    if arrival.check(step, p, forward, left, depth)['checks']['surface_corroborates_passive_range']:
                        self.arrival = arrival
                elif self.arrival is not None and len(separated) >= self.fitter.STOP_MIN_SUPPORT:
                    # Fresh contradictory evidence invalidates the carried estimate.
                    self.arrival = None
        checked = self.arrival.check(step, p, forward, left, depth) if self.arrival is not None else None
        return {'ready': bool(checked and checked['ready']), 'latest_fit_selection': selected,
                'fits': fits, 'arrival': checked}


def transform(obs, verifier):
    encoded = obs['sensor_payload']
    if encoded['encoding'] != 'npz_base64':
        raise ValueError('Expected inline NPZ observations')
    foa = 'foa_audio' in obs['shapes']
    audio_key = 'foa_audio' if foa else 'binaural_audio'
    expected = ARRAYS - {'binaural_audio'} | {audio_key}
    with np.load(io.BytesIO(base64.b64decode(encoded['data'], validate=True)), allow_pickle=False) as data:
        if set(data.files) != expected:
            raise ValueError('Unexpected observation arrays')
        rgb, depth, audio, p, q = [data[k] for k in ('rgb', 'depth_m', audio_key, 'position', 'rotation_xyzw')]
    sr, dt = obs['sample_rate'], obs['dt_s']
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or depth.shape != (*rgb.shape[:2], 1) or audio.shape != (round(sr*dt), 4 if foa else 2)
            or p.shape != (3,) or q.shape != (4,)
            or not all(np.isfinite(a).all() for a in (depth, audio, p, q))):
        raise ValueError('Invalid RGB-D, audio or pose')
    quat = quat_from_coeffs(q)
    forward = quat_rotate_vector(quat, np.array([0., 0., -1.]))
    left = quat_rotate_vector(quat, np.array([-1., 0., 0.]))
    if foa:
        audio_meta = {'format': 'FOA ACN/SN3D', 'rms_w': float(np.sqrt(np.mean(audio[:, 0]**2))),
                      'acoustic_receiver_position': obs['audio_receiver_position']}
        angle, acoustic_png = None, None
    else:
        audio_meta, angle, curve = audio_features(audio, sr)
        acoustic_png = audio_image(audio, sr, audio_meta, curve)
    ident = obs['observation_id']
    gate = verifier.update(ident, obs['step'], p, forward, left, depth, angle)
    summary = {'observation_id': ident, 'step': obs['step'], 'sim_time_s': obs['timestamp_s'],
               'xz_m': p[[0, 2]].round(3).tolist(), 'forward_xz': forward[[0, 2]].round(3).tolist(),
               'collided': obs['collided_previous_step'], 'forward_blocked': obs['blocked_previous_step'],
               'depth': depth_summary(depth), 'audio': audio_meta}
    # Expose the same passive fits to both model groups. The coordinator alone applies the STOP gate.
    summary['passive_acoustic_fits'] = {
        name: {k: fit[k] for k in ('valid', 'estimated_horizontal_range_m', 'plausible_range_m',
                                   'angular_fit_error_deg', 'best_far_fit_error_deg', 'support_count',
                                   'supporting_observation_ids') if k in fit}
        for name, fit in gate['fits'].items()
    }
    return {'summary': summary, 'rgb_png': png64(Image.fromarray(rgb)),
            'audio_png': acoustic_png, 'stop_gate': gate}


def stationary_summary(observations):
    """Aggregate four already acquired, fixed-pose binaural blocks; no gate update."""
    if len(observations)!=4 or len({o['observation_id'] for o in observations})!=4:
        raise ValueError('Expected four distinct stationary observations')
    first=observations[0];audio=[]
    for index,obs in enumerate(observations):
        if obs['sample_rate']!=16000 or obs['dt_s']!=.25 or obs['step']!=first['step']+index or not np.isclose(obs['timestamp_s'],first['timestamp_s']+index*.25):
            raise ValueError('Noncontiguous stationary audio')
        if not np.allclose(obs['pose']['position'],first['pose']['position'],rtol=0,atol=1e-7) or not np.allclose(obs['pose']['rotation_xyzw'],first['pose']['rotation_xyzw'],rtol=0,atol=1e-7):
            raise ValueError('Stationary aggregation received moving poses')
        with np.load(io.BytesIO(base64.b64decode(obs['sensor_payload']['data'],validate=True)),allow_pickle=False) as arrays:
            block=arrays['binaural_audio']
            if block.shape!=(4000,2) or not np.isfinite(block).all():raise ValueError('Invalid stationary binaural block')
            if not np.allclose(arrays['position'],obs['pose']['position'],rtol=0,atol=1e-7) or not np.allclose(arrays['rotation_xyzw'],obs['pose']['rotation_xyzw'],rtol=0,atol=1e-7):raise ValueError('Pose/payload mismatch')
            audio.append(block.copy())
    wave=np.concatenate(audio);features,_,curve=audio_features(wave,16000)
    features.update(observation_ids=[o['observation_id'] for o in observations],
        audio_interval_s=[first['timestamp_s'],observations[-1]['timestamp_s']+.25],duration_s=1.,
        pose=first['pose'],same_pose_verified=True,sample_rate_hz=16000,
        note='Same four observed blocks; aggregate is not additional independent evidence')
    return dict(stationary_audio=features,audio_png=audio_image(wave,16000,features,curve))


def main():
    verifier = PassiveArrival()
    foa_verifiers = {}
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('command')=='stationary_summary':
                result=stationary_summary(request['observations'])
            elif request.get('command')=='foa_stop_gate':
                config=(request.get('rule','foa_multiview'),request.get('tolerance_deg'))
                if config not in foa_verifiers:
                    foa_verifiers[config]=FoaArrival(rule=config[0],tolerance_deg=config[1])
                result={'gate':foa_verifiers[config].update(request['result'],audio_active=request['audio_active'])}
            else:result = transform(request, verifier)
            print(json.dumps({'ok': True, **result}, allow_nan=False), flush=True)
        except (ValueError, KeyError, TypeError) as exc:
            print(json.dumps({'ok': False, 'error': str(exc)}), flush=True)


if __name__ == '__main__':
    main()
