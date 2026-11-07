"""Full-network native16k AVN finetuning. Only train receives waveform yaw augmentation."""
import argparse,copy,csv,hashlib,json,math,os,random,sys,time,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from .loader import Dataset,extractor,masked_adpit_loss
from .common import sha,write_json


def yaw(wave,target,angle):
    """Passive listener yaw: local direction azimuth decreases by angle."""
    c,s=math.cos(angle),math.sin(angle)
    output=wave.clone();x,y=wave[3].clone(),wave[1].clone()
    output[3]=c*x+s*y;output[1]=-s*x+c*y
    label=target.clone();x,y=target[:, :,1].clone(),target[:,:,2].clone()
    label[:,:,1]=c*x+s*y;label[:,:,2]=-s*x+c*y
    return output,label


class TrainData(Dataset):
    def __init__(self,*args,seed=7,augment=False,**kwargs):
        super().__init__(*args,**kwargs);self.epoch=0;self.seed=seed;self.augment=augment
        if self.augment and any(r['split']!='train' for r in self.rows):raise ValueError('Augmentation is train only')
    def __getitem__(self,index):
        sample=super().__getitem__(index)
        # Stateless angle remains reproducible across DataLoader worker counts.
        angle=float(np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index])).uniform(-np.pi,np.pi)) if self.augment else 0.
        if self.augment:sample['waveform'],sample['adpit']=yaw(sample['waveform'],sample['adpit'],angle)
        sample['yaw_rad']=angle
        return sample


def balanced_adpit_loss(prediction,target,valid_frames):
    """Same single-source ADPIT targets; equal means of active and background class entries."""
    if torch.any(target[:,:,1:]!=0):raise ValueError('Single-source targets only')
    b,t,_,_,c=target.shape
    pred=prediction.reshape(b,t,3,3,c)
    active=target[:,:,0,0,:]>.5
    truth=target[:,:,0,1:,:]*target[:,:,0,0:1,:]
    error=(pred-truth[:,:,None]).square().mean(dim=(2,3))
    positive=active & valid_frames[:,:,None]
    negative=(~active) & valid_frames[:,:,None]
    terms=[]
    if positive.any():terms.append(error[positive].mean())
    if negative.any():terms.append(error[negative].mean())
    return torch.stack(terms).mean() if terms else error.sum()*0


def foreground20_adpit_loss(prediction,target,valid_frames):
    if torch.any(target[:,:,1:]!=0):raise ValueError('Single-source targets only')
    b,t,_,_,c=target.shape;pred=prediction.reshape(b,t,3,3,c)
    active=target[:,:,0,0,:];truth=target[:,:,0,1:,:]*active[:,:,None,:]
    error=(pred-truth[:,:,None]).square().mean(dim=(2,3))
    weights=(1+19*active)*valid_frames[:,:,None]
    return (error*weights).sum()/weights.sum().clamp(min=1)


def make_backend(runtime,checkpoint,classes):
    sys.path.insert(0,str(runtime));from dsf_worker import DSF
    backend=DSF(runtime,checkpoint)
    old=backend.model.AVfnn_list[-1]
    backend.model.AVfnn_list[-1]=torch.nn.Linear(old.in_features,9*len(classes)).cuda()
    for param in backend.model.parameters():param.requires_grad_(True)
    return backend


class Metrics:
    def __init__(self,runtime,classes,threshold=.5):
        sys.path.insert(0,str(runtime));import dsf_decode as decoder
        from utils.SELD_evaluation_metrics import SELDMetrics
        from utils.cls_feature_class import FeatureClass
        decoder.CLASSES=tuple(classes);decoder.ACTIVITY_THRESHOLD=threshold
        self.decoder=decoder;self.scorer=SELDMetrics(doa_threshold=20,nb_classes=len(classes),average='macro');self.segmenter=FeatureClass
    def add(self,raw,target,valid):
        pred={i:[[e['class_id'],0,*e['direction_unit_xyz']] for e in events]
              for i,events in enumerate(self.decoder.decode(raw)) if valid[i] and events}
        gt={}
        for t in np.flatnonzero(valid):
            active=np.flatnonzero(target[t,0,0]>.5)
            if len(active):gt[int(t)]=[[int(k),0,*target[t,0,1:,k].tolist()] for k in active]
        segment=lambda d:self.segmenter.segment_labels(SimpleNamespace(_label_resolution=10),d,len(valid))
        self.scorer.update_seld_scores(segment(pred),segment(gt))
    def result(self):
        values=self.scorer.compute_seld_scores()
        return dict(zip(('ER','F20','LE','LR','SELD'),map(float,values[:5])))


def evaluate(model,front,data,runtime,classes,batch,workers,raw_output=None):
    model.eval();metric=Metrics(runtime,classes);losses=[];predictions=[];ids=[]
    loader=DataLoader(data,batch_size=batch,num_workers=workers,shuffle=False,pin_memory=True)
    with torch.inference_mode():
        for sample in loader:
            wave=sample['waveform'].cuda(non_blocking=True);target=sample['adpit'].cuda(non_blocking=True);valid=sample['valid_frames'].cuda(non_blocking=True)
            feature=front(wave)
            raw=model(feature) # FP32 evaluation, matching original DSF inference
            raw=raw.float();assert torch.isfinite(raw).all()
            losses.append((float(masked_adpit_loss(raw,target,valid)),len(wave)))
            arrays=raw.cpu().numpy();truth=target.cpu().numpy();mask=valid.cpu().numpy()
            for r,t,m in zip(arrays,truth,mask):metric.add(r,t,m)
            if raw_output is not None:predictions.extend(arrays);ids.extend(sample['sample_id'])
    result=metric.result();result['loss']=sum(v*n for v,n in losses)/sum(n for _,n in losses)
    if raw_output is not None:np.savez_compressed(raw_output,accdoa=np.stack(predictions),sample_id=np.array(ids))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--encoder-lr',type=float,default=1e-5);p.add_argument('--head-lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--resume',type=Path);p.add_argument('--loss',choices=('standard','balanced','foreground20'),default='standard');p.add_argument('--patience',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.set_float32_matmul_precision('medium') # original trainer: FP32 tensors with TF32 permitted
    cmap=json.loads((a.data/'class_map.json').read_text())['official_category_to_class_id'];classes=[k for k,v in sorted(cmap.items(),key=lambda x:x[1])]
    train=TrainData(a.data,'train',window_s=10,hop_s=10,seed=a.seed,augment=True);val=TrainData(a.data,'validation',window_s=10,hop_s=10)
    backend=make_backend(a.runtime,a.checkpoint,classes);model=backend.model;front=extractor(a.runtime).cuda().eval()
    head=list(model.AVfnn_list[-1].parameters());head_ids={id(v) for v in head};encoder=[v for v in model.parameters() if id(v) not in head_ids]
    optim=torch.optim.Adam([dict(params=encoder,lr=a.encoder_lr,name='encoder'),dict(params=head,lr=a.head_lr,name='new_21class_output')])
    total_steps=a.epochs*math.ceil(len(train)/a.batch);base_lr=[a.encoder_lr,a.head_lr]
    scaler=torch.cuda.amp.GradScaler(enabled=False)
    config=dict(**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},classes=classes,window_s=10,label_rate_hz=10,
        augmentation='train-only independent continuous passive yaw, applied before STFT; target XYZ rotates identically',
        train_records=len(train),validation_records=len(val),head_replacement='last linear128->189; all other encoder/MLP weights initialized from original',
        all_model_parameters_trainable_from_step1=all(v.requires_grad for v in model.parameters()),parameter_count=sum(v.numel() for v in model.parameters()),
        encoder_trainable_parameters=sum(v.numel() for v in encoder),head_trainable_parameters=sum(v.numel() for v in head),
        initial_checkpoint_sha256=sha(a.checkpoint),dataset_seal_sha256=sha(a.data/'SHA256SUMS'),split_manifest_sha256=sha(a.data/'split_manifest.json'),
        original_backend=backend.description,frontend=dict(rate=16000,n_fft=512,win_length=342,hop=100,mels=128,fmax=8000),
        optimizer='Adam',weight_decay=0,schedule='epoch cosine, T_max=epochs, eta_min=0; no warmup',loss_description='Original ADPIT MSE single-source specialization; all valid class entries uniformly weighted, no foreground reweighting. Ignore frames excluded.',
        metrics='original DSF scorer and segmenter,21classes,macro,20deg,1s blocks; original3track decoder with15deg merging; activity threshold fixed0.5',
        early_stop='disabled: complete all requested epochs' if a.patience==0 else str(a.patience),checkpoint_selection='minimum validation SELD; ties broken by validation loss',
        precision='FP32 train/evaluation/features, TF32 matmul permitted as in archived trainer; no AMP or GradScaler updates',torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0),
        scripts_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('loader.py'))})
    write_json(a.output/'config.json',config)
    # Metric self-test retains actual21class evaluation, not a reduced class count.
    raw=np.zeros((100,9*len(classes)),np.float32);target=np.zeros((100,6,4,len(classes)),np.float32)
    for k in range(len(classes)):raw[k].reshape(3,3,len(classes))[:,0,k]=1;target[k,0,0,k]=target[k,0,1,k]=1
    metric=Metrics(a.runtime,classes);metric.add(raw,target,np.ones(100,bool));perfect=metric.result();assert perfect['ER']==0 and perfect['F20']>1-1e-9 and perfect['LE']<.001 and perfect['SELD']<1e-5,perfect
    write_json(a.output/'metric_self_test.json',perfect)
    started=time.monotonic();step=0;best=float('inf');bestloss=float('inf');bad=0;history=[];start_epoch=0;prior_elapsed=0.
    if a.resume:
        prior=torch.load(str(a.resume),map_location='cpu');model.load_state_dict(prior['model_state_dict'],strict=True)
        optim.load_state_dict(prior['optimizer_state_dict']);scaler.load_state_dict(prior['scaler']);step=prior['global_step'];start_epoch=prior['epoch'];history=prior['history'];prior_elapsed=history[-1]['elapsed_s']
        torch.set_rng_state(prior['torch_rng']);torch.cuda.set_rng_state_all(prior['cuda_rng'])
        config['resume_checkpoint_sha256']=sha(a.resume);config['resume_epoch']=start_epoch;config['loss_schedule']={'resumed_from_epoch':start_epoch,'continuation_loss':a.loss}
        write_json(a.output/'config.json',config)
        for name in ('step1_gradient_audit.json','step1_parameter_changes.json'):shutil.copyfile(a.resume.parent/name,a.output/name)
        previous_best=torch.load(str(a.resume.parent/'best.pt'),map_location='cpu')
        model.load_state_dict(previous_best['model_state_dict'],strict=True)
        rescored=evaluate(model,front,val,a.runtime,classes,a.batch,a.workers)
        previous_best['validation']=rescored;previous_best['config']=config;torch.save(previous_best,a.output/'best.pt')
        best,bestloss=rescored['SELD'],rescored['loss'];write_json(a.output/'best_validation.json',dict(epoch=previous_best['epoch'],validation=rescored,reevaluated_in_fp32=True))
        model.load_state_dict(prior['model_state_dict'],strict=True)
        (a.output/'history.jsonl').write_text(''.join(json.dumps(dict(row,evaluation_precision='FP16 during original warmup'))+'\n' for row in history))
        print(json.dumps(dict(event='resume',epoch=start_epoch,step=step,previous_best_fp32=rescored)),flush=True)
    for epoch in range(start_epoch,a.epochs):
        train.epoch=epoch;model.train();generator=torch.Generator().manual_seed(a.seed+epoch)
        loader=DataLoader(train,batch_size=a.batch,shuffle=True,num_workers=a.workers,pin_memory=True,generator=generator)
        running=0;count=0;epoch_start=time.monotonic()
        for sample in loader:
            factor=.5*(1+math.cos(math.pi*epoch/a.epochs))
            for group,lr in zip(optim.param_groups,base_lr):group['lr']=lr*factor
            wave=sample['waveform'].cuda(non_blocking=True);target=sample['adpit'].cuda(non_blocking=True);valid=sample['valid_frames'].cuda(non_blocking=True)
            with torch.no_grad():feature=front(wave)
            optim.zero_grad(set_to_none=True)
            raw=model(feature)
            loss={'balanced':balanced_adpit_loss,'foreground20':foreground20_adpit_loss,'standard':masked_adpit_loss}[a.loss](raw.float(),target,valid);assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optim)
            if step==0:
                grads={n:float(v.grad.norm()) if v.grad is not None else None for n,v in model.named_parameters()}
                assert any(g is not None and g>0 for n,g in grads.items() if n.startswith('audio_proj'))
                assert any(g is not None and g>0 for n,g in grads.items() if n.startswith('mamba_blocks_1'))
                assert any(g is not None and g>0 for n,g in grads.items() if n.startswith('AVfnn_list.2'))
                before={n:v.detach().clone() for n,v in model.named_parameters() if n in ('audio_proj.weight','AVfnn_list.2.weight')}
                write_json(a.output/'step1_gradient_audit.json',dict(all_parameters_unfrozen=True,gradient_norms=grads))
            gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            scaler.step(optim);scaler.update()
            if step==0:
                changes={n:float((dict(model.named_parameters())[n]-v).abs().max()) for n,v in before.items()}
                assert all(v>0 for v in changes.values()),changes
                write_json(a.output/'step1_parameter_changes.json',changes)
            running+=float(loss)*len(wave);count+=len(wave);step+=1
            if step%25==0:print(json.dumps(dict(event='progress',epoch=epoch+1,step=step,loss=running/count,elapsed_s=time.monotonic()-started,peak_mib=torch.cuda.max_memory_allocated()/2**20)),flush=True)
            if a.smoke:
                write_json(a.output/'smoke_result.json',dict(status='passed',loss=float(loss),gradient_norm=float(gradnorm),peak_mib=torch.cuda.max_memory_allocated()/2**20,wall_s=time.monotonic()-started,parameter_changes=changes));return
        result=evaluate(model,front,val,a.runtime,classes,a.batch,a.workers)
        row=dict(epoch=epoch+1,global_step=step,train_loss=running/count,validation=result,epoch_wall_s=time.monotonic()-epoch_start,elapsed_s=prior_elapsed+time.monotonic()-started,evaluation_precision='FP32',learning_rates=[g['lr'] for g in optim.param_groups])
        history.append(row)
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(event='epoch',**row)),flush=True)
        state=dict(model_state_dict=model.state_dict(),optimizer_state_dict=optim.state_dict(),scaler=scaler.state_dict(),epoch=epoch+1,global_step=step,config=config,validation=result,history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        torch.save(state,a.output/'last.pt')
        if result['SELD']<best-1e-9 or (abs(result['SELD']-best)<1e-9 and result['loss']<bestloss):
            best,bestloss=result['SELD'],result['loss'];bad=0;torch.save(state,a.output/'best.pt');write_json(a.output/'best_validation.json',row)
        else:bad+=1
        if a.patience>0 and epoch+1>=10 and bad>=a.patience:break
    saved=torch.load(str(a.output/'best.pt'),map_location='cpu');model.load_state_dict(saved['model_state_dict'],strict=True)
    # Validation selects the checkpoint; only now open and score the internal test split.
    test=TrainData(a.data,'test',window_s=10,hop_s=10);test_result=evaluate(model,front,test,a.runtime,classes,a.batch,a.workers,a.output/'test_predictions.npz')
    write_json(a.output/'results.json',dict(status='completed',best_epoch=saved['epoch'],completed_epochs=len(history),optimizer_steps=step,best_validation=saved['validation'],test=test_result,elapsed_s=prior_elapsed+time.monotonic()-started,best_checkpoint_sha256=sha(a.output/'best.pt'),initial_checkpoint_sha256=sha(a.checkpoint),dataset_seal_sha256=sha(a.data/'SHA256SUMS'),test_used_for_selection=False,agent_success_not_evaluated=True,requested_epochs=a.epochs,all_requested_epochs_completed=len(history)==a.epochs))
    print('TRAINING_COMPLETE',json.dumps(test_result),flush=True)
if __name__=='__main__':main()
