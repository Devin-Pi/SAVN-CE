"""Explicit runtime-only sound schedule; never rewrite the original episode."""
import math


def resolve_sound_schedule(mode,onset,offset,step_s,max_steps):
    if mode not in ('episode','sustained'):raise ValueError('Unknown sound schedule')
    if not all(math.isfinite(x) for x in (onset,offset,step_s)) or step_s<=0 or onset<0 or offset<=onset or max_steps<=0:
        raise ValueError('Invalid sound schedule parameters')
    runtime_offset=offset if mode=='episode' else max(offset,math.ceil((max_steps+1)*step_s)+1)
    return dict(mode=mode,official_onset_s=onset,official_offset_s=offset,
                runtime_onset_s=onset,runtime_offset_s=runtime_offset,
                source_loop='existing native sample-index wrap; waveform and internal silence unchanged',
                episode_modified=False)
