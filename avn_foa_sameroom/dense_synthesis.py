"""Traceable high-duty source scheduling followed by one full-history FOA convolution."""
import numpy as np
from scipy.signal import fftconvolve
from .common import RATE, DURATION, angles
from .synthesis import rms, intervals

VERSION = 'AVN-FOA-SameRoom-v0.2-Dense15h'
MIN_ACTIVE_FRAMES = 92


def active_regions(dry):
    samples = RATE // 100
    length = len(dry) // samples
    power = np.sqrt(np.mean(dry[:length*samples].reshape(length, samples)**2, axis=1))
    threshold = max(float(power.max()) * .01, 1e-7)
    edges = np.flatnonzero(np.diff(np.r_[False, power > threshold, False]))
    regions = [(int(a*samples), int(b*samples)) for a,b in edges.reshape(-1,2) if b-a >= 5]
    if not regions:
        raise ValueError('No >=50ms genuinely active source region; refuse artificial labels')
    return regions, dict(source_selection_relative_db=-40, source_selection_threshold=threshold,
                         source_duration_s=len(dry)/RATE, source_active_region_count=len(regions),
                         source_original_10ms_active_fraction=float((power>threshold).mean()),
                         longest_selected_region_s=max(b-a for a,b in regions)/RATE)


def event_crop(dry, event):
    start, end = event['source_start_sample_16k'], event['source_end_sample_16k']
    crop = dry[start:end].copy()
    fade = event['fade_samples_16k']
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    crop[:fade] *= ramp
    crop[-fade:] *= ramp[::-1]
    return crop


def reconstruct_timeline(dry, events):
    timeline = np.zeros(RATE*DURATION, dtype=np.float32)
    for event in events:
        crop = event_crop(dry, event)
        onset = event['emission_start_sample_16k']
        timeline[onset:onset+len(crop)] += crop
    return timeline


def dense_timeline(dry, rng, regions=None):
    if regions is None:
        regions, _ = active_regions(dry)
    onset = int(rng.uniform(.04,.09)*RATE)
    stop = int(rng.uniform(9.80,9.88)*RATE)
    events = []
    while stop-onset > 160:
        remaining = stop-onset
        long = [(a,b) for a,b in regions if b-a >= remaining]
        available = long or regions
        weights = np.asarray([b-a for a,b in available], dtype=np.float64)
        lo,hi = available[int(rng.choice(len(available),p=weights/weights.sum()))]
        maximum = min(remaining, hi-lo)
        length = maximum if long or maximum<1600 else int(rng.uniform(max(800,maximum*.65),maximum+1))
        start = int(rng.integers(lo,hi-length+1))
        fade = min(80,length//4)
        events.append(dict(source_start_sample_16k=start,source_end_sample_16k=start+length,
            emission_start_sample_16k=onset,fade_samples_16k=fade,
            source_crop_interval_s=[start/RATE,(start+length)/RATE],
            emission_interval_s=[onset/RATE,(onset+length)/RATE],fade_duration_s=fade/RATE,
            looped=False,playback='one crop; consecutive crops crossfade at one fixed source'))
        if onset+length >= stop:
            break
        onset += length-fade
        if len(events)>1000:
            raise RuntimeError('Unexpected event count')
    return reconstruct_timeline(dry,events), events


def supervision(signal,audio,ir,unit,class_id,nclasses):
    power=np.sqrt(np.mean(signal.reshape(-1,RATE//100)**2,axis=1))
    threshold=max(float(power.max())*.00316227766,1e-8)
    dry_active=power>threshold
    support_power=np.max(np.abs(ir),axis=1)
    support=np.flatnonzero(support_power>max(float(support_power.max())*1e-5,1e-12))
    if not len(support):raise ValueError('Empty RIR')
    onset=float(support[0]/RATE)
    arriving=np.zeros(len(signal),bool)
    for start,end in intervals(dry_active,100):
        a=max(0,round((start+onset)*RATE));b=min(len(signal),round((end+onset)*RATE))
        arriving[a:b]=True
    occupancy=arriving.reshape(100,-1).mean(1)
    wet_rms=np.sqrt(np.mean(audio.reshape(100,RATE//10,4)**2,axis=(1,2)))
    floor=max(float(wet_rms.max())*1e-4,1e-9)
    audible=wet_rms>floor
    ignore=((occupancy>0)&(occupancy<.8))|((occupancy==0)&audible)|((occupancy>=.8)&(~audible))
    active=(occupancy>=.8)&(~ignore)
    adpit=np.zeros((100,6,4,nclasses),np.float32)
    adpit[active,0,0,class_id]=1
    for axis in range(3):adpit[active,0,axis+1,class_id]=unit[axis]
    xyz=np.zeros((100,3),np.float32);xyz[active]=unit
    az,el=angles(unit)
    labels=dict(adpit=adpit,activity=active.astype(np.uint8),ignore_mask=ignore.astype(np.uint8),direction_xyz=xyz,
        frame_index=np.arange(100),time_s=np.arange(100,dtype=np.float32)/10,
        class_id=np.where(active,class_id,-1),track_id=np.where(active,0,-1),
        azimuth_deg=np.where(active,az,0),elevation_deg=np.where(active,el,0))
    details=dict(dry_activity_intervals_s=intervals(dry_active,100),received_direct_activity_intervals_s=intervals(arriving,RATE),
        activity_threshold=dict(resolution_s=.01,relative_db=-50,absolute_rms_floor=1e-8,actual_threshold=threshold),
        activity_rule='100ms active iff causal dry-arrival occupancy >=80% and wet RMS above declared floor; partial/tail-only/inaudible frames ignored',
        activity_duration_s=float(arriving.sum()/RATE),supervised_active_frames=int(active.sum()),ignore_frames=int(ignore.sum()),
        wet_audible_frames=int(audible.sum()),wet_audibility_rms_floor=floor,
        rir_onset_s=onset,rir_support_end_s=float((support[-1]+1)/RATE),azimuth_deg=az,elevation_deg=el)
    return labels,details


def synthesize_dense(dry,ir,unit,class_id,nclasses,rng,negative=False):
    if negative:
        signal=np.zeros(RATE*DURATION,np.float32);events=[];profile={}
        audio=np.zeros((len(signal),4),np.float32);gain=1.;level=None;tail_energy=0.
    else:
        regions,profile=active_regions(dry)
        signal,events=dense_timeline(dry,rng,regions)
        full=fftconvolve(signal[:,None],ir,axes=0).astype(np.float32)
        audio=full[:len(signal)].copy()
        level=float(rng.uniform(-35,-20))
        gain=10**(level/20)/max(rms(audio[:,0]),1e-12)
        if float(np.abs(audio).max())*gain>.95:gain=.95/float(np.abs(audio).max())
        audio*=gain;tail_energy=float(np.sum(full[len(signal):]**2,dtype=np.float64))*gain*gain
    labels,details=supervision(signal*gain,audio,ir,unit,class_id,nclasses)
    details.update(profile,events=events,global_gain=float(gain),requested_w_level_dbfs=level,
        output_peak=float(np.abs(audio).max()),audio_rms=rms(audio),truncated_tail_energy=tail_energy,
        tail_rule='Entire scheduled mono stream convolved once; recorded tail retained, truncated energy reported',
        is_no_event=bool(negative),synthetic_background='none',
        source_streams=1,event_schedule_policy='Random actual active crops, per-event source offsets and 5ms crossfade; no whole-WAV periodic tiling or time stretch',
        temporal_distribution='deliberately dense training scenario, not natural event timing',
        independent_source_recordings_added=0,
        dcase_csv_note='Only active supervision frames listed; ignore_mask remains authoritative')
    if not negative and details['supervised_active_frames']<MIN_ACTIVE_FRAMES:
        raise ValueError(f"Insufficient real activity: {details['supervised_active_frames']}/100")
    return audio,labels,details
