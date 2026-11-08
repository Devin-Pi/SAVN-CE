"""Real observed FOA/pose: deployed adapter vs frozen1s graph and retained replay."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'agent_episode'))
from short_candidate_worker import ShortCandidate
from spatial_audio.foa import world_foa_to_listener


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=Path('/workspaces/avn_short_windows_v1_20261108')
    runtime=Path('/workspaces/dsf_localization_runtime_v1')
    candidate=ShortCandidate(source/'code_snapshot',runtime,source/'runs/w100/run_seed7_full100/best.pt')
    old=source/'analysis_full100_20261108'
    wave,rate=sf.read(old/'recording_input/foa_ambix.wav',dtype='float32',always_2d=True)
    observations=[json.loads(l) for l in (old/'recording_input/observations.jsonl').read_text().splitlines()]
    index=11;sample=wave[(index-3)*4000:(index+1)*4000].copy();q=observations[index]['pose']['rotation_xyzw']
    with torch.inference_mode():
        result=candidate.predict(sample,rate,q)
        local=world_foa_to_listener(sample,q)
        feature=candidate.front(torch.from_numpy(local.T.copy())[None].cuda())
        direct=candidate.pool(candidate.model(feature),1.)[0].cpu().numpy()
    error=float(np.abs(direct-np.asarray(result['raw_accdoa'])).max());assert error<1e-6,error
    previous=next(json.loads(s) for s in (old/'online_replay/w100_predictions.jsonl').read_text().splitlines() if json.loads(s)['observation_id']=='obs_0011')
    replay_error=float(np.abs(direct[0]-np.asarray(previous['raw_accdoa'])).max());assert replay_error<1e-3,replay_error
    report=dict(status='passed',checkpoint_sha256=candidate.description['checkpoint_sha256'],
        real_observed_anchor='obs_0011',world_input_shape=list(sample.shape),sample_rate_hz=rate,
        feature_shape=list(feature.shape),output_shape=list(direct.shape),
        adapter_vs_frozen_graph_max_abs_error=error,adapter_vs_retained_replay_max_abs_error=replay_error,
        waveform_rescaling=False,resampling=False,output_scope='current250ms only',truth_used=False)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
