"""Read data, strictly restore old weights, one original and one new-head forward. No training."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from .common import write_json,sha
from .loader import Dataset,extractor,masked_adpit_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--adapter-dir',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.adapter_dir));from dsf_worker import DSF
    torch.manual_seed(7)
    dsf=DSF(a.runtime,a.checkpoint);extract=extractor(a.runtime)
    dataset=Dataset(a.root);sample=dataset[0];nclasses=sample['adpit'].shape[-1]
    with torch.inference_mode():
        feature=extract(sample['waveform'][None]);assert feature.shape==(1,7,1600,128) and torch.isfinite(feature).all()
        original=dsf.model(feature.cuda());assert original.shape==(1,100,117) and torch.isfinite(original).all()
        old=dsf.model.AVfnn_list[-1]
        dsf.model.AVfnn_list[-1]=torch.nn.Linear(old.in_features,3*3*nclasses).cuda()
        adapted=dsf.model(feature.cuda());assert adapted.shape==(1,100,9*nclasses) and torch.isfinite(adapted).all()
        loss=masked_adpit_loss(adapted,sample['adpit'][None].cuda(),sample['valid_frames'][None].cuda())
        assert torch.isfinite(loss)
    from utils.cls_feature_class import FeatureClass
    converter=FeatureClass.__new__(FeatureClass);converter._nb_unique_classes=nclasses
    import csv
    checked=0
    for row in dataset.rows[:8]:
        desc={}
        for fields in csv.reader((a.root/row['dcase_cartesian_csv']).open()):
            frame,cid,track=int(fields[0]),int(fields[1]),int(fields[2])
            desc.setdefault(frame,[]).append([cid,track,*map(float,fields[3:6])])
        converted=converter.get_adpit_labels_for_file(desc,100)
        with np.load(a.root/row['labels']) as z:np.testing.assert_allclose(converted,z['adpit'],atol=1e-7)
        checked+=1
    windows={}
    for duration in (.5,1,2):
        ds=Dataset(a.root,window_s=duration,hop_s=.5);item=ds[0]
        with torch.inference_mode():
            f=extract(item['waveform'][None]);pred=dsf.model(f.cuda())
        assert tuple(f.shape)==(1,7,round(duration*160),128)
        assert pred.shape[1]==round(duration*10) and item['adpit'].shape[0]==round(duration*10) and torch.isfinite(pred).all()
        windows[str(duration)]={'feature_shape':list(f.shape),'prediction_shape':list(pred.shape),'label_shape':list(item['adpit'].shape)}
    result=dict(status='passed',training_started=False,checkpoint_sha256=sha(a.checkpoint),strict_restore=dsf.description['strict_load'],
        waveform_shape=list(sample['waveform'].shape),feature_shape=list(feature.shape),target_shape=list(sample['adpit'].shape),
        original_prediction_shape=list(original.shape),random_new_head_prediction_shape=list(adapted.shape),masked_single_source_adpit_loss=float(loss),
        original_adpit_label_converter_checks=checked,window_tests=windows,original_frontend={'fs':24000,'n_fft':512,'win_length':512,'hop':150,'fmax':12000,'n_mels':128},
        native_frontend={'fs':16000,'n_fft':512,'win_length':342,'hop':100,'fmax':8000,'n_mels':128},
        compatibility='Original backbone restored strictly; 13-class head retained for baseline forward, then random official-class head used only for shape/loss smoke. No optimization. New head is not a trained expert.',
        caveats=['16k cannot retain >8k information','Mel frequency coverage differs from 24k; transfer requires adaptation','original intensity feature channel names XYZ actually process stored ACN directional channels in order YZX; kept unchanged for weight compatibility','valid_frames mask must be applied before loss reduction'])
    write_json(a.root/'dsf_forward_smoke.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
