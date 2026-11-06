"""Resolve model tool arguments against previously delivered observations."""
import base64
import hashlib
import io
import json
import math
import select
import subprocess
import time
import zipfile

WINDOW_S = 10
MODEL_NAME = 'localization_tool'
MODEL_FORMAT = {
    'channels': ['W', 'Y', 'Z', 'X'], 'channel_order': 'ACN', 'ambisonic_order': 1,
    'normalization': 'SN3D', 'coordinate_frame': 'habitat_world',
    'ambix_positive_axes_in_habitat': {'X': '-Z', 'Y': '-X', 'Z': '+Y'},
}
LIMITATIONS = [
    'Closed set of sound event classes. Activity score is an ACCDOA magnitude.',
    'Direction is in the fixed Habitat world sound-field basis. No range or instance identity.',
    'All frames use the complete historical window. Motion and domain shift are uncalibrated.',
]
FUNCTION = {
    'type': 'function', 'name': MODEL_NAME,
    'description': 'Analyze a 10 s FOA window ending at an already observed ID. Use one anchor ID. '
                   'The same audio cannot count as additional independent evidence.',
    'parameters': {'type': 'object', 'properties': {'observation_ids': {
        'type': 'array', 'items': {'type': 'string'}}},
        'required': ['observation_ids'], 'additionalProperties': False},
    'strict': True,
}


class Worker:
    def __init__(self, command, log):
        self.log = log.open('x')
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, text=True)
        except OSError:
            self.log.close()
            raise

    def request(self, request):
        self.process.stdin.write(json.dumps(request, allow_nan=False)+'\n')
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 90)[0]:
            self.process.kill()
            raise RuntimeError('DSF worker request timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('DSF worker exited')
        return json.loads(line)

    def close(self):
        try:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        finally:
            self.process.stdout.close()
            self.log.close()


def worker_command(container):
    return ['docker', 'exec', '-i', '-e', 'CUDA_VISIBLE_DEVICES=2', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-w', '/workspaces/dsf_localization_runtime_v1', container, 'python', 'dsf_worker.py',
            '--source', '/workspaces/dsf_localization_runtime_v1', '--checkpoint',
            '/workspaces/av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt']


def finite_vector(value, size):
    return isinstance(value, list) and len(value) == size and all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in value)


class LocalizationTool:
    def __init__(self, request):
        self.request = request
        self.description = None
        self.reset_episode(None)

    def reset_episode(self, episode_ref):
        self.episode_ref = episode_ref
        self.observations = {}
        self.cache = {}

    def describe(self):
        if self.description is None:
            try:
                self.description = self.request({'command': 'describe'})
            except (OSError, ValueError, RuntimeError):
                self.description = {'status': 'unavailable', 'reason': 'DSF worker unavailable'}
        return {**self.description, 'tool': MODEL_NAME, 'input_format': MODEL_FORMAT, 'limitations': LIMITATIONS}

    def add_observation(self, observation):
        if self.episode_ref is None:
            raise ValueError('No active localization episode')
        ident = observation['observation_id']
        if ident in self.observations:
            return False
        step = observation['step']
        rate, dt = observation['sample_rate'], observation['dt_s']
        if step != len(self.observations) or not math.isclose(observation['timestamp_s'], step * dt, abs_tol=1e-7):
            raise ValueError('Noncontiguous observation sequence')
        fmt = observation.get('audio_format', {})
        valid_format = (
            rate in (16000, 24000) and dt == 0.25 and all(fmt.get(k) == v for k, v in MODEL_FORMAT.items())
            and observation['shapes'].get('foa_audio') == [round(rate * dt), 4]
        )
        data = None
        if valid_format:
            payload = observation['sensor_payload']
            if payload['encoding'] != 'npz_base64':
                raise ValueError('Expected inline NPZ')
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload['data'], validate=True))) as archive:
                info = archive.getinfo('foa_audio.npy')
                if info.file_size > 4*4*round(rate*dt) + 4096:
                    raise ValueError('Oversized audio member')
                data = archive.read(info)
        pose = observation['pose']
        receiver = observation.get('audio_receiver_position')
        if not finite_vector(pose['position'], 3) or not finite_vector(pose['rotation_xyzw'], 4):
            raise ValueError('Invalid observed pose')
        if valid_format and not finite_vector(receiver, 3):
            raise ValueError('FOA observation missing actual acoustic receiver pose')
        self.observations[ident] = {
            'id': ident, 'step': step, 'rate': rate, 'dt': dt, 'data': data,
            'receiver': receiver, 'pose': pose, 'time': observation['timestamp_s'],
        }
        return True

    def analyze(self, observation_ids):
        def result(status, reason, **fields):
            return {'status': status, 'reason': reason, 'limitations': LIMITATIONS, **fields}

        if (not isinstance(observation_ids, list) or not observation_ids
                or len(observation_ids) > 40 or any(not isinstance(i, str) for i in observation_ids)
                or len(set(observation_ids)) != len(observation_ids)
                or any(i not in self.observations for i in observation_ids)):
            return result('unsupported_input', 'Expected IDs observed in the active episode')
        last = self.observations[observation_ids[-1]]
        count = round(WINDOW_S / last['dt'])
        rows = list(self.observations.values())[max(0, last['step']-count+1):last['step']+1]
        if any(row['data'] is None or row['rate'] != last['rate'] for row in rows):
            return result('unsupported_input', 'Requires contiguous ACN/SN3D FOA at one supported sample rate')
        ids = [row['id'] for row in rows]
        if len(observation_ids) != 1 and observation_ids != ids:
            return result('unsupported_input', 'Expected one anchor or exactly its ordered support IDs')
        start, end = rows[0]['time'], last['time'] + last['dt']
        span = [round(start*last['rate']), round(end*last['rate'])]
        provenance = {'episode_ref': self.episode_ref, 'observation_ids': ids, 'support_sample_interval': span,
                      'cache_hit': False, 'new_physical_observation': False}
        if len(rows) < count:
            return result('not_ready', f'{len(rows)} of {count} windows acquired', provenance=provenance)
        backend = self.describe()
        if backend['status'] != 'ok':
            return result('unavailable', backend.get('reason'), provenance=provenance)
        provenance.update(model=backend['model_variant'], checkpoint_sha256=backend['checkpoint_sha256'],
                          source_sha256=backend['source_sha256'])
        content = hashlib.sha256()
        for row in rows:
            content.update(row['data'])
        key = (tuple(span), content.hexdigest(), backend['checkpoint_sha256'],
               tuple(sorted(backend['source_sha256'].items())), last['rate'])
        if key in self.cache:
            cached = self.cache[key]
            return {**cached, 'provenance': {**cached['provenance'], 'cache_hit': True},
                    'cache_read_unix_s': time.time()}
        try:
            prediction = self.request({'command': 'predict', 'sample_rate': last['rate'],
                                       'windows_npy_base64': [base64.b64encode(r['data']).decode() for r in rows]})
        except (OSError, RuntimeError, ValueError):
            return result('error', 'DSF worker request failed', provenance=provenance)
        if prediction['status'] != 'ok':
            return result(prediction['status'], prediction.get('reason'), provenance=provenance)
        times = []
        for index, events in enumerate(prediction['events_by_frame']):
            nominal = start + index/10
            row = rows[min(count-1, math.floor((index/10 + 1e-8)/last['dt']))]
            times.append(nominal)
            for event in events:
                x, y, z = event['direction_unit_xyz']
                event.update(direction_frame='habitat_world_ambix_xyz', pose_ref=row['id'],
                             direction_unit_habitat_xyz=[-y, z, -x])
        record = result(
            'ok', None, provenance=provenance,
            timing={'audio_start_s': start, 'audio_end_s': end, 'frame_times_s': times,
                    'result_available_at_audio_time_s': end, 'result_available_unix_s': time.time(),
                    **prediction['timing_s']},
            input_format={**MODEL_FORMAT, 'sample_rate_hz': last['rate'], 'model_sample_rate_hz': 24000,
                          'resampling': 'polyphase 3/2; original bandwidth <=8 kHz' if last['rate']==16000 else None},
            pose_refs=[{'observation_id': row['id'], 'timestamp_s': row['time'],
                        'receiver_position_habitat_xyz_m': row['receiver'], 'robot_pose': row['pose']} for row in rows],
            events=prediction['events_by_frame'], raw_accdoa=prediction['raw_accdoa'],
            peak_cuda_allocated_mib=prediction['peak_cuda_allocated_mib'],
            range_m=None, uncertainty=None, calibration_status='uncalibrated',
        )
        self.cache[key] = record
        return record


def compact(result):
    if result['status'] != 'ok':
        return {k: result[k] for k in ('status', 'reason')}
    end = result['timing']['audio_end_s']
    events = []
    for timestamp, frame in zip(result['timing']['frame_times_s'], result['events']):
        for event in frame:
            events.append({'time_s': timestamp, 'age_at_window_end_s': round(end-timestamp, 3),
                           **{key: event[key] for key in (
                               'class_name', 'azimuth_deg', 'elevation_deg', 'activity_score',
                               'pose_ref', 'direction_frame')}})
    recent = [event for event in events if event['age_at_window_end_s'] <= 1]
    evidence_id = result['provenance']['observation_ids'][-1] + ':localization'
    return {'status': 'ok', 'evidence_id': evidence_id,
            'audio_interval_s': [result['timing']['audio_start_s'], end],
            'cache_hit': result['provenance']['cache_hit'],
            'recent_events': recent[-8:], 'historical_events': [] if recent else events[-8:],
            'range_m': None,
            'summary': ('Supported events in the last second of the analyzed window.' if recent
                        else 'No supported event detected in the last second of the analyzed window. '
                             'Any listed earlier events are historical and share the same input window.'),
            'limitations': LIMITATIONS}
