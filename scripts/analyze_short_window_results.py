"""Audit saved test predictions; benchmark batch1 without rerunning the full test set."""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def distribution(values):
    x=np.asarray(values)
    return None if not x.size else dict(count=int(x.size),median=float(np.median(x)),
        q90=float(np.quantile(x,.9)),maximum=float(x.max()),above_threshold_fraction=float((x>.5).mean()))


def secondary_score(raw,rows,data,classes,runtime,StepMetrics):
    scorer=StepMetrics(runtime,classes,10)
    for pred,row in zip(raw,rows):
        weighted=np.zeros((100,189),np.float64);coverage=np.zeros(100,np.float64)
        for index in range(5,40):
            lo=index*.25;hi=lo+.25
            for j in range(int(math.floor(lo*10+1e-7)),min(100,int(math.ceil(hi*10-1e-7)))):
                width=max(0,min(hi,(j+1)*.1)-max(lo,j*.1))
                weighted[j]+=pred[index]*width;coverage[j]+=width
        mask=coverage>.099999;weighted[mask]/=coverage[mask,None]
        with np.load(data/row['labels'],allow_pickle=False) as z:
            scorer.add(weighted.astype(np.float32),z['adpit'],(~z['ignore_mask'].astype(bool))&mask)
    return scorer.result()


def audit(run,data,labels,config,StepMetrics,ShortData,runtime):
    dataset=ShortData(data,labels,'test',config['window_s'],7,False)
    classes=config['classes'];metric=StepMetrics(runtime,classes,4)
    source=run/'test_predictions.npz'
    with np.load(source,allow_pickle=False) as z:
        raw=z['accdoa'];target=z['targets'];valid=z['valid'];seen=z['seen'];ids=z['sample_id']
    assert raw.shape==(540,40,189) and target.shape==(540,40,6,4,21)
    assert np.isfinite(raw).all() and list(ids)==[r['sample_id'] for r in dataset.rows]
    assert seen[:,5:].all() and not seen[:,:5].any() and not valid[:,:5].any()
    counts=np.zeros((21,5),np.int64);clips=np.zeros(21,np.int64);scores=[[] for _ in classes]
    detected_all=np.zeros((540,40,21),bool);valid_background=background_alarm=0
    negative_clips=negative_alarm_clips=negative_steps=negative_alarm_steps=0
    ignored_seen=ignored_alarms=0;confusion=np.zeros((21,22),np.int64)
    decoded_categories=Counter();all_scores=[];negative_source_power_max=0.
    for parent,(r,t,v,row) in enumerate(zip(raw,target,valid,dataset.rows)):
        # Independently reconnect stored truth with immutable cache and metadata.
        for step in range(5,40):
            expected,mask=dataset.step_target(parent,(step+1)*.25)
            np.testing.assert_array_equal(t[step],expected[0].numpy())
            assert bool(v[step])==bool(mask[0])
        metric.add(r,t,v)
        truth=t[:,0,0]>.5;prediction=detected_all[parent]
        decoded=metric.decoder.decode(r)
        strength=np.linalg.norm(r.reshape(40,3,3,21),axis=2).max(1)
        all_scores.extend(strength[seen[parent]].reshape(-1).tolist())
        for frame,events in enumerate(decoded):
            for event in events:
                prediction[frame,event['class_id']]=True
                if seen[parent,frame]:decoded_categories[classes[event['class_id']]]+=1
            if v[frame] and truth[frame].any():
                actual=int(np.flatnonzero(truth[frame])[0])
                inferred=int(max(events,key=lambda e:e['activity_score'])['class_id']) if events else 21
                confusion[actual,inferred]+=1
        any_prediction=prediction.any(1);active=truth.any(1)&v;quiet=(~truth.any(1))&v
        valid_background+=int(quiet.sum());background_alarm+=int((quiet&any_prediction).sum())
        ignored=seen[parent]&(~v);ignored_seen+=int(ignored.sum());ignored_alarms+=int((ignored&any_prediction).sum())
        if row['is_no_event']:
            with np.load(labels/(row['sample_id']+'.npz'),allow_pickle=False) as z:
                negative_source_power_max=max(negative_source_power_max,float(z['power'].max()))
            negative_clips+=1;negative_alarm_clips+=int((any_prediction&v).any())
            negative_steps+=int(v.sum());negative_alarm_steps+=int((any_prediction&v).sum())
        for c in range(21):
            mask=truth[:,c]&v;clips[c]+=int(mask.any())
            counts[c]+=[int(mask.sum()),int((mask&prediction[:,c]).sum()),
                int((mask&(~any_prediction)).sum()),int((mask&any_prediction&(~prediction[:,c])).sum()),
                int((quiet&prediction[:,c]).sum())]
            scores[c].extend(strength[mask,c].tolist())
    assert negative_clips==54 and negative_source_power_max==0
    classwise=metric.scorer.compute_seld_scores()[5]
    perclass=[]
    for c,name in enumerate(classes):
        total,hit,miss,wrong,fa=map(int,counts[c])
        row=dict(class_id=c,category=name,positive_clips=int(clips[c]),active_valid_steps=total,
            correct_class_detected_steps=hit,frame_class_recall=hit/total if total else None,
            no_event_predicted_active_steps=miss,other_class_only_active_steps=wrong,
            valid_background_alarms_for_class=fa,raw_true_class_score=distribution(scores[c]))
        row.update(zip(('ER','F20','LE','LR','SELD'),map(float,classwise[:,c])))
        row['no_correct_class_detection']=hit==0
        perclass.append(row)
    expected=json.loads((run/'results.json').read_text())
    primary=metric.result();secondary=secondary_score(raw,dataset.rows,data,classes,runtime,StepMetrics)
    for key in ('ER','F20','LE','LR','SELD'):
        assert abs(primary[key]-expected['test'][key])<1e-8,(key,primary[key],expected['test'][key])
        assert abs(secondary[key]-expected['test_100ms'][key])<1e-8,(key,secondary[key],expected['test_100ms'][key])
    sumcounts=counts.sum(0)
    return dict(test=primary,test_100ms=secondary,per_class=perclass,
        activity=dict(active_valid_steps=int(sumcounts[0]),correct_class_detected_steps=int(sumcounts[1]),
            frame_class_recall=float(sumcounts[1]/sumcounts[0]),no_prediction_active_steps=int(sumcounts[2]),
            wrong_class_only_active_steps=int(sumcounts[3]),all_class_score=distribution(all_scores),
            decoded_class_step_events=dict(decoded_categories)),
        silence=dict(no_event_clips=negative_clips,no_event_clips_with_alarm=negative_alarm_clips,
            no_event_clip_alarm_rate=negative_alarm_clips/negative_clips,
            no_event_steps=negative_steps,no_event_steps_with_alarm=negative_alarm_steps,
            no_event_step_alarm_rate=negative_alarm_steps/negative_steps,
            valid_background_steps=valid_background,valid_background_steps_with_alarm=background_alarm,
            valid_background_step_alarm_rate=background_alarm/valid_background,
            ignored_observed_steps=ignored_seen,ignored_steps_with_prediction=ignored_alarms,
            zero_source_power_verified=True,note='No-event clips are exact zeros, not real environmental background; ignored tail/partial steps are excluded from false-alarm denominator'),
        top1_decoded_confusion=confusion.tolist(),confusion_columns=classes+['no_prediction'],
        test_predictions_sha256=sha(source),stored_test_metric_recomputation_passed=True,
        stored_ground_truth_cache_reconnection_passed=True)


@torch.inference_mode()
def latency_and_reload(run,config,data,labels,runtime,ShortData,patch_short_context,pool_current_step,load):
    model,front,embedded=load(run/'best.pt',runtime);patch_short_context(model)
    assert embedded['classes']==config['classes'] and embedded['window_s']==config['window_s']
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    dataset=ShortData(data,labels,'test',config['window_s'],7,False)
    with np.load(run/'test_predictions.npz',allow_pickle=False) as z:stored=z['accdoa']
    errors=[];agreement=[]
    # Fixed data-order spot check; no selection based on model scores.
    for index in (0,34,35,69):
        sample=dataset[index]
        wave=sample['waveform'][None].cuda()
        raw=pool_current_step(model(front(wave)),config['window_s'])[0,0].float().cpu().numpy()
        ref=stored[sample['parent_index'],sample['step_index']]
        errors.append(float(np.abs(raw-ref).max()))
        isactive=lambda x:np.linalg.norm(x.reshape(3,3,21),axis=1).max(0)>.5
        agreement.append(float((isactive(raw)==isactive(ref)).mean()))
    sample=dataset[0];cpu_wave=sample['waveform'][None].contiguous()
    import dsf_decode
    def one():
        feature=front(cpu_wave.cuda(non_blocking=False))
        raw=pool_current_step(model(feature),config['window_s'])[0].float().cpu().numpy()
        return dsf_decode.decode(raw)
    for _ in range(10):one()
    elapsed=[]
    for _ in range(100):
        torch.cuda.synchronize();start=time.perf_counter();one();torch.cuda.synchronize()
        elapsed.append((time.perf_counter()-start)*1000)
    result=dict(device=torch.cuda.get_device_name(0),torch=str(torch.__version__),batch=1,warmup_calls=10,measured_calls=100,
        p50_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)),max_ms=float(max(elapsed)),
        fraction_over_250ms=float((np.asarray(elapsed)>250).mean()),
        includes='CPU float32 tensor -> GPU, frozen16k frontend, model, current250ms pooling, CPU transfer and original decoder',
        excludes='file reading, FOA coordinate conversion, simulator step, JSONL/network, LLM and other scheduling overhead',
        earliest_context_ready_s=config['window_s'],update_period_s=.25,
        reload_spotcheck_max_abs_error=max(errors),reload_spotcheck_class_activity_agreement=min(agreement),
        checkpoint_all_tensors_finite=True)
    del model,front;torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.experiment/'code_snapshot'))
    from avn_foa_sameroom.short_window import ShortData,StepMetrics,patch_short_context,pool_current_step
    from avn_foa_sameroom.infer_finetuned import load
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    torch.manual_seed(7);torch.cuda.manual_seed_all(7);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    manifest=json.loads((a.experiment/'source_manifest.json').read_text())
    for name,value in manifest['source_sha256'].items():assert sha(a.experiment/'code_snapshot'/name)==value
    runs={}
    for label in ('w025','w100','w150'):
        root=a.experiment/'runs'/label/'run_seed7_full100'
        config=json.loads((root/'config.json').read_text());result=json.loads((root/'results.json').read_text())
        history=[json.loads(s) for s in (root/'history.jsonl').read_text().splitlines()]
        assert len(history)==100 and result['all_requested_epochs_completed'] and result['optimizer_steps']==27000
        assert sha(root/'best.pt')==result['best_checkpoint_sha256']
        best=min(history,key=lambda r:(r['validation']['SELD'],r['validation']['loss']))
        assert best['epoch']==result['best_epoch']
        audited=audit(root,a.data,a.experiment/'labels50',config,StepMetrics,ShortData,a.runtime)
        timing=latency_and_reload(root,config,a.data,a.experiment/'labels50',a.runtime,ShortData,patch_short_context,pool_current_step,load)
        runs[label]=dict(window_s=config['window_s'],completed_epochs=100,best_epoch=result['best_epoch'],
            best_validation=result['best_validation'],checkpoint_sha256=result['best_checkpoint_sha256'],
            training_hours=result['elapsed_s']/3600,latency=timing,
            history_diagnostics=dict(validation_SELD_min=min(r['validation']['SELD'] for r in history),
                validation_SELD_max=max(r['validation']['SELD'] for r in history),
                validation_epochs_with_F20_above_zero=sum(r['validation']['F20']>0 for r in history),
                final_train_loss=history[-1]['train_loss'],
                last10_validation_SELD=[r['validation']['SELD'] for r in history[-10:]]),**audited)
        with (a.output/f'{label}_per_class.csv').open('w',newline='') as f:
            fields=[k for k in audited['per_class'][0] if k!='raw_true_class_score']
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
            w.writerows({k:v for k,v in r.items() if k in fields} for r in audited['per_class'])
        print('AUDITED',label,json.dumps(dict(test=audited['test'],silence=audited['silence'],latency=timing)),flush=True)
    report=dict(status='completed',runs=runs,activity_threshold=.5,DSF_doa_threshold_deg=20,
        scorer='original DSF21class macro1s-block metrics; primary4Hz current-step; original100ms GT secondary',
        test_model_full_dataset_repeats=0,model_spotchecks_and_latency_only=True,
        existing_training_data_and_models_unchanged=True,source_snapshot_hashes_passed=True,
        no_threshold_or_checkpoint_tuning=True,no_navigation_control=True,
        limitations=['single seed per duration','known source families shared across scene-disjoint splits',
            'static single-source same-room LOS synthesized audio','zero-audio negative clips, not natural background',
            '1.5s failed run cannot prove all1.5s models are inferior','offline score/latency do not prove online navigation success'],
        analysis_source_sha256=sha(Path(__file__)))
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('SHORT_FULL100_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
