"""Full-history single-source convolution and explicit activity/ignore supervision."""
import math
import numpy as np
from scipy.signal import fftconvolve,resample_poly
import soundfile as sf
from .common import RATE,DURATION,angles


def read_source(path):
    x,sr=sf.read(path,dtype='float32',always_2d=True);x=x.mean(axis=1)
    if sr!=RATE:
        d=math.gcd(sr,RATE);x=resample_poly(x,RATE//d,sr//d)
    if not np.isfinite(x).all():raise ValueError('Nonfinite dry source')
    return x


def intervals(mask,rate):
    idx=np.flatnonzero(np.diff(np.r_[False,mask,False]))
    return [[float(l/rate),float(r/rate)] for l,r in idx.reshape(-1,2)]


def rms(x):return float(np.sqrt(np.mean(np.square(x),dtype=np.float64)))


def synthesize(dry,ir,unit,class_id,nclasses,rng,negative=False):
    n=RATE*DURATION;signal=np.zeros(n,dtype=np.float32);events=[]
    if not negative:
        count=int(rng.choice([1,1,2]));slot=(DURATION-3)/count
        for j in range(count):
            duration=float(rng.uniform(1.1,min(3.2,slot)))
            length=min(int(duration*RATE),len(dry))
            if length<int(.25*RATE):raise ValueError('source too short')
            onset=int((.6+j*slot+rng.uniform(0,.35))*RATE)
            for _ in range(100):
                start=int(rng.integers(0,max(1,len(dry)-length+1)))
                crop=dry[start:start+length].copy()
                if rms(crop)>max(1e-6,rms(dry)*.015):break
            else:raise ValueError('No audible crop')
            # No time-stretch, looping, padding or gain per channel. Short fades are logged.
            fade=min(80,len(crop)//4);ramp=np.linspace(0,1,fade,dtype=np.float32)
            crop[:fade]*=ramp;crop[-fade:]*=ramp[::-1]
            signal[onset:onset+length]+=crop
            events.append(dict(source_crop_interval_s=[start/RATE,(start+length)/RATE],
                emission_interval_s=[onset/RATE,(onset+length)/RATE],fade_duration_s=fade/RATE,looped=False))
        # Preserve complete convolution history; retain the first real 10 seconds.
        full=fftconvolve(signal[:,None],ir,axes=0).astype(np.float32)
        audio=full[:n].copy()
        level_dbfs=float(rng.uniform(-35,-20))
        gain=10**(level_dbfs/20)/max(rms(audio[:,0]),1e-12)
        if np.max(np.abs(audio*gain))>.95:gain=.95/float(np.max(np.abs(audio)))
        audio*=gain;signal*=gain
    else:
        full=np.zeros((n+len(ir)-1,4),dtype=np.float32);audio=full[:n].copy();gain=1.;level_dbfs=None
    # Frame dry-source activity at 10 ms with an explicit floor. Original WAV silence is respected.
    micro=signal.reshape(-1,RATE//100)
    power=np.sqrt(np.mean(micro**2,axis=1))
    threshold=max(float(power.max())*.00316227766,1e-8) # -50 dB wrt maximum 10ms dry RMS
    dry_activity=power>threshold
    ir_power=np.max(np.abs(ir),axis=1)
    support=np.flatnonzero(ir_power>max(float(ir_power.max())*1e-5,1e-12))
    if not len(support):raise ValueError('Empty RIR')
    delay=float(support[0]/RATE);tail=float((support[-1]+1)/RATE)
    # Source-activity arrival uses the measured causal RIR onset, not the strongest echo.
    arriving=np.zeros(n,dtype=bool)
    for l,r in intervals(dry_activity,100):
        begin=max(0,int(round((l+delay)*RATE)));end=min(n,int(round((r+delay)*RATE)))
        arriving[begin:end]=True
    occupancy=arriving.reshape(100,-1).mean(axis=1)
    wet_rms=np.sqrt(np.mean(audio.reshape(100,RATE//10,4)**2,axis=(1,2)))
    audible_floor=max(float(wet_rms.max())*1e-4,1e-9)
    ignore=((occupancy>0)&(occupancy<.8)) | ((occupancy==0)&(wet_rms>audible_floor))
    active=(occupancy>=.8)&(~ignore)
    adpit=np.zeros((100,6,4,nclasses),dtype=np.float32)
    adpit[active,0,0,class_id]=1
    for axis in range(3):adpit[active,0,axis+1,class_id]=unit[axis]
    xyz=np.zeros((100,3),dtype=np.float32);xyz[active]=unit
    az,el=angles(unit)
    record=dict(events=events,dry_activity_intervals_s=intervals(dry_activity,100),
        received_direct_activity_intervals_s=intervals(arriving,RATE),
        activity_threshold={'resolution_s':.01,'relative_db':-50,'absolute_rms_floor':1e-8,'actual_threshold':threshold},
        activity_rule='100ms frame active iff measured-direct-arrival dry activity occupies >=80%; partial arrivals and audible reverb-only frames ignored',
        activity_duration_s=float(arriving.sum()/RATE),supervised_active_frames=int(active.sum()),ignore_frames=int(ignore.sum()),
        rir_onset_s=delay,rir_support_end_s=tail,global_gain=float(gain),requested_w_level_dbfs=level_dbfs,
        output_peak=float(np.max(np.abs(audio))),audio_rms=rms(audio),
        truncated_tail_energy=float(np.sum(full[n:]**2,dtype=np.float64))*gain*gain,
        tail_rule='Entire dry timeline convolved once; tail within recording retained; actual energy truncated beyond 10s reported',
        is_no_event=bool(negative),azimuth_deg=az,elevation_deg=el,synthetic_background='none',
        dcase_csv_note='Only supervised active frames are listed. Ignore frames MUST be excluded using ignore_mask.')
    labels=dict(adpit=adpit,activity=active.astype(np.uint8),ignore_mask=ignore.astype(np.uint8),
                direction_xyz=xyz,frame_index=np.arange(100),time_s=np.arange(100,dtype=np.float32)/10,
                class_id=np.where(active,class_id,-1),track_id=np.where(active,0,-1),azimuth_deg=np.where(active,az,0),elevation_deg=np.where(active,el,0))
    return audio,labels,record
