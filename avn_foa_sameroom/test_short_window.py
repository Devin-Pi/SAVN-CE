"""Contract checks for partial encoder cells, matched supervision and original DSF scoring."""
import argparse
import copy
import json
import sys
import types
from pathlib import Path
import numpy as np
import torch
from .short_window import ShortData,StepMetrics,current_step_weights,ceiling_pool_forward
from .labels50 import activity


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);sys.path.insert(0,str(a.runtime));from models.audio_encoder import ConvBlock
    original=ConvBlock(4,4).eval();patched=copy.deepcopy(original)
    patched.forward=types.MethodType(ceiling_pool_forward,patched)
    x=torch.randn(2,4,20,16)
    with torch.no_grad():np.testing.assert_array_equal(original(x).numpy(),patched(x).numpy())
    odd=torch.randn(2,4,5,16)
    try:original(odd)
    except RuntimeError:pass
    else:raise AssertionError('Original pool unexpectedly accepts odd time shape')
    assert patched(odd).shape==(2,4,3,8)
    np.testing.assert_allclose(current_step_weights(.25).numpy(),[.4,.4,.2],atol=1e-7)
    np.testing.assert_allclose(current_step_weights(1.)[-3:].numpy(),[.2,.4,.4],atol=1e-7)
    active,valid=activity(np.array([.9,.5,0.]),np.array([1.,1.,0.]),1e-9)
    np.testing.assert_array_equal(active,[True,False,False]);np.testing.assert_array_equal(valid,[True,False,True])
    datasets=[ShortData(a.data,a.labels,'train',w,7,True) for w in (.25,1.,1.5)]
    for epoch in (0,9):
        for d in datasets:d.epoch=epoch
        for index in range(16):
            samples=[d[index] for d in datasets]
            for sample in samples[1:]:
                assert sample['end_s']==samples[0]['end_s']
                np.testing.assert_array_equal(sample['adpit'],samples[0]['adpit'])
                np.testing.assert_array_equal(sample['valid_frames'],samples[0]['valid_frames'])
                np.testing.assert_allclose(sample['waveform'][:,-4000:],samples[0]['waveform'],atol=1e-7)
    classes=[k for k,v in sorted(json.loads((a.data/'class_map.json').read_text())['official_category_to_class_id'].items(),key=lambda item:item[1])]
    metric=StepMetrics(a.runtime,classes,4)
    pred=np.zeros((40,189),np.float32);target=np.zeros((40,6,4,21),np.float32)
    for c in range(21):pred[c].reshape(3,3,21)[:,0,c]=1;target[c,0,0,c]=target[c,0,1,c]=1
    metric.add(pred,target,np.ones(40,bool));perfect=metric.result()
    assert perfect['ER']==0 and perfect['F20']>1-1e-9 and perfect['SELD']<1e-5,perfect
    result=dict(status='passed',even_time_pooling_bitwise_unchanged=True,odd_partial_time_cell_supported=True,
        exactly_same_current250ms_waveform_target_and_yaw_across_three_contexts=True,
        tested_epochs=[0,9],tested_sample_ids_per_epoch=16,partial_activity_ignore_verified=True,
        current_step_pool_duration_weights_verified=True,original_DSF4Hz_perfect_self_test=perfect)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
