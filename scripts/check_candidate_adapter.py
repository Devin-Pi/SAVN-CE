"""Compare the exact deployed candidate frontend/output with the frozen native16k loader."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,soundfile as sf,torch
from candidate_worker import Candidate


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runtime=Path('/workspaces/dsf_localization_runtime_v1')
    candidate=Candidate(Path('/workspaces/avn_foa_dense15h_finetune_v02/code_snapshot'),runtime,
        Path('/workspaces/localization_candidates/avn_foa_dense15h_v02_epoch097/best.pt'))
    wave,rate=sf.read('/workspaces/avn_candidate_shadow_runtime_v1/parity_input.wav',dtype='float32',always_2d=True)
    observed=[]
    handle=candidate.model.audio_proj.register_forward_pre_hook(lambda module,args:observed.append(args[0].detach().clone()))
    with torch.inference_mode():
        result=candidate.predict(wave,rate,[0,0,0,1])
        feature=candidate.front(torch.from_numpy(wave.T.copy())[None].cuda())
        direct=candidate.model(feature).float().cpu().numpy()[0]
        expected=(feature-feature.mean(dim=2,keepdim=True))/feature.std(dim=2,keepdim=True).clamp(min=1e-8)
    handle.remove()
    error=float(np.max(np.abs(direct-np.asarray(result['raw_accdoa']))));assert error<1e-5,error
    normalization_error=float((expected-observed[-1]).abs().max());assert normalization_error<1e-6,normalization_error
    rejected=[]
    for bad,badrate in ((wave[:4000],16000),(wave,24000),(wave[:,:2],16000)):
        try:candidate.predict(bad,badrate,[0,0,0,1])
        except ValueError:rejected.append(True)
    assert len(rejected)==3
    report=dict(status='passed',checkpoint_sha256=candidate.description['checkpoint_sha256'],input_shape=list(wave.shape),sample_rate_hz=rate,
        frontend=candidate.description['frontend'],features_shape=list(feature.shape),output_shape=list(direct.shape),
        adapter_vs_frozen_direct_max_abs_error=error,model_time_normalization_max_abs_error=normalization_error,
        model_normalization='(feature-time_mean)/time_std.clamp(min=1e-8), PyTorch default sample standard deviation',
        external_scaler_used=False,waveform_rescaling=False,resampling=False,invalid_short_rate_channels_rejected=True,
        input_wav_sha256=hashlib.sha256(Path('/workspaces/avn_candidate_shadow_runtime_v1/parity_input.wav').read_bytes()).hexdigest())
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
