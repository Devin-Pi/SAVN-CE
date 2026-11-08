"""Fixed-checkpoint 2x2 evaluation and prediction-only activity diagnosis; no tuning."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch
from avn_foa_sameroom.common import sha,write_json
from avn_foa_sameroom.finetune import Metrics,TrainData,evaluate
from avn_foa_sameroom.infer_finetuned import load
from avn_foa_sameroom.evaluate_stopped_finetune import audit_predictions


def activity_diagnostics(predictions,data,runtime,classes):
    decoder=Metrics(runtime,classes).decoder
    counts=np.zeros((len(classes),4),dtype=np.int64)
    raw_scores=[[] for _ in classes]
    background=background_alarm=negative=negative_alarm=0
    with np.load(predictions,allow_pickle=False) as z:
        assert list(z['sample_id'])==[r['sample_id'] for r in data.rows]
        for raw,row in zip(z['accdoa'],data.rows):
            with np.load(data.root/row['labels'],allow_pickle=False) as label:
                target=label['adpit'][:,0,0]>.5;valid=~label['ignore_mask'].astype(bool)
            detected=np.zeros_like(target)
            for frame,events in enumerate(decoder.decode(raw)):
                for event in events:detected[frame,event['class_id']]=True
            any_detection=detected.any(1);active=target.any(1)&valid;quiet=(~target.any(1))&valid
            background+=int(quiet.sum());background_alarm+=int((quiet&any_detection).sum())
            if not active.any():
                negative+=1;negative_alarm+=int((any_detection&valid).any())
            scores=np.linalg.norm(raw.reshape(-1,3,3,len(classes)),axis=2).max(1)
            for c in range(len(classes)):
                mask=target[:,c]&valid
                counts[c]+=np.array([mask.sum(),(mask&detected[:,c]).sum(),(mask&(~any_detection)).sum(),(mask&any_detection&(~detected[:,c])).sum()])
                raw_scores[c].extend(scores[mask,c].tolist())
    perclass=[]
    for c,name in enumerate(classes):
        total,correct,none,wrong=map(int,counts[c]);values=np.array(raw_scores[c])
        perclass.append(dict(category=name,gt_active_frames=total,correct_class_detected_frames=correct,
            no_event_predicted_frames=none,other_class_only_frames=wrong,
            correct_class_frame_recall=correct/total if total else None,
            no_prediction_rate=none/total if total else None,
            raw_gt_class_score_median=float(np.median(values)) if len(values) else None,
            raw_gt_class_score_q90=float(np.quantile(values,.9)) if len(values) else None,
            raw_gt_class_score_max=float(values.max()) if len(values) else None))
    total,correct,none,wrong=map(int,counts.sum(0))
    return dict(activity_threshold=.5,diagnostic_note='Framewise decoded class detection without 20deg constraint; distinct from official block-level F20/LR; ignore frames excluded',
        total_gt_active_frames=total,correct_class_detected_frames=correct,correct_class_frame_recall=correct/total,
        active_frames_with_no_prediction=none,active_frames_with_other_class_only=wrong,
        valid_background_frames=background,valid_background_frames_with_prediction=background_alarm,
        valid_background_frame_alarm_rate=background_alarm/background if background else None,
        all_no_event_clips=negative,all_no_event_clips_with_false_alarm=negative_alarm,per_class=perclass)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--dense',type=Path,required=True)
    p.add_argument('--sparse',type=Path,required=True);p.add_argument('--old-run',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
    out=a.experiment/'analysis_full100_20261108';out.mkdir(exist_ok=True)
    newrun=a.experiment/'run_seed7_full100';result=json.loads((newrun/'results.json').read_text())
    config=json.loads((newrun/'config.json').read_text());oldresult=json.loads((a.old_run/'results.json').read_text())
    assert result['completed_epochs']==100 and result['all_requested_epochs_completed']
    assert sha(newrun/'best.pt')==result['best_checkpoint_sha256']
    assert sha(a.old_run/'best.pt')==oldresult['best_checkpoint_sha256']
    classes=config['classes'];dense=TrainData(a.dense,'test',window_s=10,hop_s=10);sparse=TrainData(a.sparse,'test',window_s=10,hop_s=10)
    started=time.monotonic()
    torch.manual_seed(7);torch.cuda.manual_seed_all(7);np.random.seed(7)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    pred=out/'old_model_dense_test_predictions.npz';scores=out/'old_model_dense_test_result.json'
    if not scores.exists():
        assert not pred.exists(),'Predictions exist: recover metrics instead of repeating model evaluation'
        model,front,embedded=load(a.old_run/'best.pt',a.runtime);assert embedded['classes']==classes
        value=evaluate(model,front,dense,a.runtime,classes,16,4,pred)
        write_json(scores,value);del model,front;torch.cuda.empty_cache()
    olddense=json.loads(scores.read_text())
    cases=[('old_sparse',a.old_run/'test_predictions.npz',sparse,oldresult['test']),
           ('old_dense',pred,dense,olddense),('new_sparse',newrun/'reference_test_predictions.npz',sparse,result['reference_test']),
           ('new_dense',newrun/'test_predictions.npz',dense,result['test'])]
    allcases={}
    for name,path,data,expected in cases:
        checked,perclass,diagnostics=audit_predictions(path,data,a.runtime,classes)
        for key in ('ER','F20','LE','LR','SELD'):assert abs(checked[key]-expected[key])<1e-8,(name,key)
        frame=activity_diagnostics(path,data,a.runtime,classes)
        allcases[name]=dict(metrics=checked,per_class=perclass,clip_diagnostics=diagnostics,activity_diagnostics=frame,
                            prediction_sha256=sha(path),records=len(data.rows))
        with (out/f'{name}_per_class.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(perclass[0]));writer.writeheader();writer.writerows(perclass)
        print('AUDITED',name,flush=True)
    history=[json.loads(s) for s in (newrun/'history.jsonl').read_text().splitlines()]
    tail=history[-10:]
    report=dict(status='passed',completed_epochs=100,optimizer_steps=result['optimizer_steps'],best_epoch=result['best_epoch'],
        best_validation=result['best_validation'],cases=allcases,classes=classes,
        old_checkpoint_sha256=oldresult['best_checkpoint_sha256'],new_checkpoint_sha256=result['best_checkpoint_sha256'],
        new_test_model_passes_in_this_analysis=0,old_model_dense_test_passes=1,all_stored_predictions_official_metric_check_passed=True,
        no_test_threshold_or_checkpoint_tuning=True,agent_success_evaluated=False,
        last10_dense_validation_SELD_range=[min(r['validation']['SELD'] for r in tail),max(r['validation']['SELD'] for r in tail)],
        last10_sparse_validation_SELD_range=[min(r['reference_validation']['SELD'] for r in tail),max(r['reference_validation']['SELD'] for r in tail)],
        evaluation_code_sha256=sha(Path(__file__)),elapsed_s=time.monotonic()-started)
    write_json(out/'comparison.json',report)
    print('CROSS_EVALUATION_COMPLETE',json.dumps({k:v['metrics'] for k,v in allcases.items()}),flush=True)

if __name__=='__main__':main()
