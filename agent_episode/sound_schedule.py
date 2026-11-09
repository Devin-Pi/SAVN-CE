"""Explicit runtime-only sound schedule; never rewrite the original episode."""
import math

SOUND_SCHEDULES = ('episode', 'sustained', 'continuous')


def resolve_sound_schedule(mode,onset,offset,step_s,max_steps):
    if mode not in SOUND_SCHEDULES:raise ValueError('Unknown sound schedule')
    if not all(math.isfinite(x) for x in (onset,offset,step_s)) or step_s<=0 or onset<0 or offset<=onset or max_steps<=0:
        raise ValueError('Invalid sound schedule parameters')
    runtime_offset=offset if mode=='episode' else max(offset,math.ceil((max_steps+1)*step_s)+1)
    return dict(mode=mode,official_onset_s=onset,official_offset_s=offset,
                runtime_onset_s=0 if mode=='continuous' else onset,runtime_offset_s=runtime_offset,
                source_loop='existing native sample-index wrap; waveform and internal silence unchanged',
                episode_modified=False)
