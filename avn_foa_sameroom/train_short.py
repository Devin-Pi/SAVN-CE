"""Three-context study: same initialization, current250ms targets, optimizer budget and splits."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import sha,write_json
from .loader import extractor,masked_adpit_loss
from .finetune import make_backend
from .short_window import ShortData,StepMetrics,patch_short_context,pool_current_step,PROTOCOL

INITIAL_SHA='85e380a6a8d2f1e545aa093e3f086f0a4d7830a15c41f7f79d047d7d9c1f9f90'


def worker_init(worker_id):
    torch.set_num_threads(1)


def loader_kwargs(workers):
    # CUDA/LLVM have already initialized in the trainer. Forking that state can
    # abort a worker during the second epoch; fresh spawned interpreters avoid it.
    return dict(num_workers=workers,worker_init_fn=worker_init,
        **({'multiprocessing_context':'spawn'} if workers else {}))


def parameter_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def evaluate(model,front,data,runtime,classes,batch,workers,prediction_path=None):
    model.eval();n=len(data.rows)
    raw=np.zeros((n,40,189),np.float32);target=np.zeros((n,40,6,4,21),np.float32)
    valid=np.zeros((n,40),bool);seen=np.zeros((n,40),bool)
    total_loss=0.;valid_total=0
    for sample in DataLoader(data,batch_size=batch,shuffle=False,pin_memory=True,**loader_kwargs(workers)):
        feature=front(sample['waveform'].cuda(non_blocking=True))
        step=pool_current_step(model(feature),data.window_s).float()
        truth=sample['adpit'].cuda(non_blocking=True);mask=sample['valid_frames'].cuda(non_blocking=True)
        assert step.shape==(len(truth),1,189) and torch.isfinite(step).all()
        loss=masked_adpit_loss(step,truth,mask);count=int(mask.sum())
        total_loss+=float(loss)*count;valid_total+=count
        for i,(parent,index) in enumerate(zip(sample['parent_index'].tolist(),sample['step_index'].tolist())):
            assert not seen[parent,index];seen[parent,index]=True
            raw[parent,index]=step[i,0].cpu().numpy()
            target[parent,index]=sample['adpit'][i,0].numpy();valid[parent,index]=sample['valid_frames'][i,0]
    assert seen[:,5:].all() and not seen[:,:5].any()
    primary=StepMetrics(runtime,classes,4);secondary=StepMetrics(runtime,classes,10)
    for parent,row in enumerate(data.rows):
        primary.add(raw[parent],target[parent],valid[parent])
        # Secondary DSF10Hz view uses original100ms truth, only fully observed bins.
        accumulated=np.zeros((100,189),np.float64);coverage=np.zeros(100,np.float64)
        for index in range(5,40):
            lo=index*.25;hi=lo+.25
            for j in range(int(math.floor(lo*10+1e-7)),min(100,int(math.ceil(hi*10-1e-7)))):
                width=max(0,min(hi,(j+1)*.1)-max(lo,j*.1))
                accumulated[j]+=raw[parent,index]*width;coverage[j]+=width
        covered=coverage>.099999
        accumulated[covered]/=coverage[covered,None]
        with np.load(data.root/row['labels'],allow_pickle=False) as z:
            secondary.add(accumulated.astype(np.float32),z['adpit'],(~z['ignore_mask'].astype(bool))&covered)
    result=primary.result();result['loss']=total_loss/max(valid_total,1)
    if prediction_path is not None:
        np.savez_compressed(prediction_path,accdoa=raw,sample_id=np.asarray([r['sample_id'] for r in data.rows]),
            targets=target,valid=valid,seen=seen)
    return result,secondary.result()


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--window-s',type=float,choices=[.25,1.,1.5],required=True)
    p.add_argument('--physical-gpu',type=int,required=True);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--encoder-lr',type=float,default=1e-5);p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert sha(a.checkpoint)==INITIAL_SHA
    label_config=json.loads((a.labels/'validation.json').read_text())
    assert label_config['all_original_100ms_activity_and_ignore_masks_reconstructed']
    assert label_config['dataset_seal_file_sha256']==sha(a.data/'SHA256SUMS')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    classes=[k for k,v in sorted(json.loads((a.data/'class_map.json').read_text())['official_category_to_class_id'].items(),key=lambda item:item[1])]
    assert len(classes)==21 and classes[0]=='chair'
    train=ShortData(a.data,a.labels,'train',a.window_s,a.seed,True)
    validation=ShortData(a.data,a.labels,'validation',a.window_s,a.seed,False)
    backend=make_backend(a.runtime,a.checkpoint,classes);model=patch_short_context(backend.model)
    initial_model_sha=parameter_sha(model)
    head=list(model.AVfnn_list[-1].parameters());ids={id(v) for v in head};encoder=[v for v in model.parameters() if id(v) not in ids]
    initial_head_sha=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in head)).hexdigest()
    front=extractor(a.runtime).cuda().eval()
    optimizer=torch.optim.Adam([dict(params=encoder,lr=a.encoder_lr,name='encoder'),dict(params=head,lr=a.head_lr,name='new21class_head')])
    config=dict(classes=classes,window_s=a.window_s,label_rate_hz=20,model_internal_bin_s=.1,step_s=.25,
        protocol=PROTOCOL,physical_gpu=a.physical_gpu,epochs=a.epochs,batch=a.batch,seed=a.seed,
        encoder_lr=a.encoder_lr,head_lr=a.head_lr,workers=a.workers,data_loader_start_method='spawn',optimizer='Adam',loss='standard',patience=0,
        precision='FP32 train/features/evaluation, TF32 high; no AMP',augmentation='train-only continuous yaw, same endpoint and yaw seed across contexts',
        train_records=len(train.rows),validation_records=len(validation.rows),train_examples_per_epoch=len(train),
        steps_per_epoch=math.ceil(len(train)/a.batch),total_optimizer_steps=a.epochs*math.ceil(len(train)/a.batch),
        train_target='one current250ms ADPIT target, >=80% real dry-arrival occupancy; historical context has no loss target',
        train_window_sampling='one deterministic common endpoint per source file per epoch, uniform over1.5:.25:10; no repeated zero-padding',
        pooling='ceil average pooling only resolves odd pre-pool time shape; no extra waveform samples; final current-step duration-weighted ACCDOA pooling',
        raw_sequence_frames=math.ceil(a.window_s*10),per_step_output_shape=[1,189],
        frontend=dict(sample_rate_hz=16000,n_fft=512,win_length=342,hop_length=100,mels=128,fmax=8000),
        waveform_shape=[4,round(a.window_s*16000)],features_shape=[7,round(a.window_s*160),128],
        initial_checkpoint_sha256=INITIAL_SHA,initial_model_state_sha256=initial_model_sha,initial_head_sha256=initial_head_sha,
        all_model_parameters_trainable_from_step1=all(v.requires_grad for v in model.parameters()),
        parameter_count=sum(v.numel() for v in model.parameters()),dataset_seal_sha256=sha(a.data/'SHA256SUMS'),
        label_cache_manifest_sha256=sha(a.labels/'manifest.jsonl'),label_cache_validation_sha256=sha(a.labels/'validation.json'),
        checkpoint_selection='minimum validation primary4Hz DSF21macro20deg1sblock SELD; tie by masked ADPIT loss',
        validation_protocol='every250ms update; only current250ms pooled output; common endpoints1.5:.25:10,35 updates/clip; identical target/eval coverage across contexts',
        secondary_metric='same step predictions resampled by time overlap to original100ms labels; requires full100ms coverage; not original full10s baseline protocol',
        metrics='original DSF scorer/decoder,21classes,macro,20deg,1s blocks,3tracks15deg merge,activity threshold0.5',
        no_test_selection=True,no_navigation_control=True,normalization='same existing per-window time mean/std; no waveform rescaling',
        comparison_caveat='old10s model used sequence loss and file-aligned windows; this controlled three-context study uses common current-step loss/random endpoints, so old/new difference is not window length alone',
        source_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('short_window.py'),Path(__file__).with_name('labels50.py'),Path(__file__).with_name('loader.py'))},
        torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0))
    write_json(a.output/'config.json',config)
    started=time.monotonic();global_step=0;best=float('inf');bestloss=float('inf');history=[]
    for epoch in range(a.epochs):
        train.epoch=epoch;model.train();generator=torch.Generator().manual_seed(a.seed+epoch)
        loader=DataLoader(train,batch_size=a.batch,pin_memory=True,shuffle=True,generator=generator,**loader_kwargs(a.workers))
        total=0.;count=0;epoch_start=time.monotonic()
        factor=.5*(1+math.cos(math.pi*epoch/a.epochs))
        for group,lr in zip(optimizer.param_groups,[a.encoder_lr,a.head_lr]):group['lr']=lr*factor
        for sample in loader:
            wave=sample['waveform'].cuda(non_blocking=True);truth=sample['adpit'].cuda(non_blocking=True);valid=sample['valid_frames'].cuda(non_blocking=True)
            with torch.no_grad():feature=front(wave)
            optimizer.zero_grad(set_to_none=True);raw_sequence=model(feature);raw=pool_current_step(raw_sequence,a.window_s)
            assert raw.shape==(len(wave),1,189)
            loss=masked_adpit_loss(raw,truth,valid);assert torch.isfinite(loss)
            loss.backward()
            if global_step==0:
                grads={n:float(v.grad.norm()) if v.grad is not None else None for n,v in model.named_parameters()}
                for prefix in ('audio_proj','mamba_blocks_1','AVfnn_list.2'):
                    assert any(g is not None and g>0 for n,g in grads.items() if n.startswith(prefix)),prefix
                before={n:v.detach().clone() for n,v in model.named_parameters() if n in ('audio_proj.weight','AVfnn_list.2.weight')}
                write_json(a.output/'step1_gradient_audit.json',dict(all_parameters_unfrozen=True,gradient_norms=grads,
                    waveform_shape=list(wave.shape),feature_shape=list(feature.shape),sequence_shape=list(raw_sequence.shape),step_shape=list(raw.shape)))
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            optimizer.step();global_step+=1
            if global_step==1:
                changes={n:float((dict(model.named_parameters())[n]-v).abs().max()) for n,v in before.items()}
                assert all(v>0 for v in changes.values());write_json(a.output/'step1_parameter_changes.json',changes)
            nvalid=int(valid.sum());total+=float(loss)*nvalid;count+=nvalid
            if global_step%25==0:
                print(json.dumps(dict(event='progress',epoch=epoch+1,step=global_step,loss=total/max(count,1),elapsed_s=time.monotonic()-started,peak_mib=torch.cuda.max_memory_allocated()/2**20)),flush=True)
            if a.smoke:
                model.eval()
                with torch.inference_mode():again=pool_current_step(model(front(wave)),a.window_s)
                assert torch.isfinite(again).all()
                validation.rows=validation.rows[:2]
                smoke_val,smoke_100=evaluate(model,front,validation,a.runtime,classes,a.batch,a.workers)
                assert all(math.isfinite(v) for v in (*smoke_val.values(),*smoke_100.values()))
                write_json(a.output/'smoke_result.json',dict(status='passed',loss=float(loss),gradient_norm=float(grad),
                    peak_mib=torch.cuda.max_memory_allocated()/2**20,parameter_changes=changes,
                    feature_shape=list(feature.shape),sequence_shape=list(raw_sequence.shape),step_shape=list(raw.shape),
                    initial_model_state_sha256=initial_model_sha,initial_head_sha256=initial_head_sha,
                    evaluation_4hz_and_original100ms_smoke_passed=True))
                return
        result,secondary=evaluate(model,front,validation,a.runtime,classes,a.batch,a.workers)
        row=dict(epoch=epoch+1,global_step=global_step,train_loss=total/max(count,1),validation=result,
            validation_100ms=secondary,epoch_wall_s=time.monotonic()-epoch_start,elapsed_s=time.monotonic()-started,
            learning_rates=[g['lr'] for g in optimizer.param_groups],evaluation_precision='FP32')
        history.append(row)
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(event='epoch',**row)),flush=True)
        state=dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),epoch=epoch+1,
            global_step=global_step,config=config,validation=result,validation_100ms=secondary,history=history,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        tmp=a.output/'last.tmp.pt';torch.save(state,tmp);tmp.replace(a.output/'last.pt')
        if result['SELD']<best-1e-9 or (abs(result['SELD']-best)<1e-9 and result['loss']<bestloss):
            best,bestloss=result['SELD'],result['loss'];tmp=a.output/'best.tmp.pt';torch.save(state,tmp);tmp.replace(a.output/'best.pt')
            write_json(a.output/'best_validation.json',row)
    saved=torch.load(str(a.output/'best.pt'),map_location='cpu');model.load_state_dict(saved['model_state_dict'],strict=True)
    test=ShortData(a.data,a.labels,'test',a.window_s,a.seed,False)
    test_result,test100=evaluate(model,front,test,a.runtime,classes,a.batch,a.workers,a.output/'test_predictions.npz')
    write_json(a.output/'results.json',dict(status='completed',best_epoch=saved['epoch'],completed_epochs=len(history),
        requested_epochs=a.epochs,all_requested_epochs_completed=len(history)==a.epochs,optimizer_steps=global_step,
        best_validation=saved['validation'],test=test_result,test_100ms=test100,test_used_for_selection=False,
        initial_checkpoint_sha256=INITIAL_SHA,best_checkpoint_sha256=sha(a.output/'best.pt'),
        dataset_seal_sha256=config['dataset_seal_sha256'],elapsed_s=time.monotonic()-started,agent_success_not_evaluated=True))
    print('SHORT_WINDOW_TRAINING_COMPLETE',a.window_s,flush=True)


if __name__=='__main__':main()
