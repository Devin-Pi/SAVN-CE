"""Report an already-completed run; reconstruct test metrics from saved predictions only."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .finetune import Metrics
from .common import sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--data',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
    result=json.loads((a.run/'results.json').read_text());config=json.loads((a.run/'config.json').read_text())
    rows={r['sample_id']:r for r in map(json.loads,(a.data/'metadata.jsonl').open())}
    metric=Metrics(a.runtime,config['classes'])
    with np.load(a.run/'test_predictions.npz',allow_pickle=False) as z:
        for raw,sid in zip(z['accdoa'],z['sample_id']):
            row=rows[str(sid)];assert row['split']=='test'
            with np.load(a.data/row['labels']) as target:metric.add(raw,target['adpit'],~target['ignore_mask'].astype(bool))
    checked=metric.result()
    for k in ('ER','F20','LE','LR','SELD'):assert abs(checked[k]-result['test'][k])<1e-8,(k,checked[k],result['test'][k])
    perclass=metric.scorer.compute_seld_scores()[5]
    with (a.run/'test_per_class.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['class_id','category','ER','F20','LE','LR','SELD'])
        for i,cat in enumerate(config['classes']):w.writerow([i,cat,*perclass[:,i]])
    gradient=json.loads((a.run/'step1_gradient_audit.json').read_text())['gradient_norms']
    text=['# AVN-FOA21 全网络微调结果','',
          f"完成 {result['completed_epochs']} 个epoch、{result['optimizer_steps']} 个batch更新；按内部validation SELD选择epoch **{result['best_epoch']}**。",'',
          '| split | ER↓ | F20↑ | LE°↓ | LR↑ | SELD↓ |','|---|---:|---:|---:|---:|---:|']
    for split,row in [('validation(best)',result['best_validation']),('test(final,once)',result['test'])]:text.append('| '+split+' | '+' | '.join(f'{row[k]:.6f}' for k in ('ER','F20','LE','LR','SELD'))+' |')
    test=result['test']; gap=test['SELD']-result['best_validation']['SELD']
    text += ['', f"最终测试集：F20={test['F20']:.4f}、方向误差LE={test['LE']:.2f}°、定位召回LR={test['LR']:.4f}；test与最佳validation的SELD差为{gap:+.4f}。LE仅描述匹配事件，必须结合F20/LR看漏检和误报，不能仅用较低LE证明专家可靠。", '已完整结束100轮后再汇总；本报告不根据测试结果回改配置或选择其他权重。']
    text+=['',f"旧权重严格加载后，只重建最后的128→189输出层（21类×3轨×XYZ）；其他旧MLP及encoder权重保留初始化，全部从第一步参与更新。{len(gradient)}个参数张量第一步全部非零梯度。两组Adam学习率：encoder=1e-5，新输出层=1e-4；weight_decay=0，每epoch余弦衰减，T_max=100，eta_min=0，无warmup。FP32训练/验证/推理，允许TF32矩阵乘（与归档trainer一致），无AMP，无梯度裁剪。没有冻结阶段，禁用提前停止。",'',
           '参考归档audio_bb_mamba_15_MB_RAW训练器：标准ADPIT MSE、Adam、按epoch余弦衰减、FP32。本次从旧预训练权重重新初始化独立实验，不恢复短程试跑；完整100轮使用固定配置，不根据逐轮指标改动超参数。采用原ADPIT单源特例，忽略帧不计入损失。此前短程调试及失败记录独立保留，没有测试集参与。',
           '训练数据为2880条10秒记录，每epoch完整覆盖8小时录音；连续随机水平yaw在波形层应用，标签XYZ同步旋转，不改变activity/ignore/class。未启用原ACS8的镜像或上下翻转。validation/test各360条、1小时，不增强。seed7；worker数4，batch16。',
           '前端为原生16k：FFT512，窗342，hop100，Mel128，fmax8k。模型内部保留原窗口时间标准化。输出189维，监督为原单源ADPIT的a0复制到3个输出轨；ignore帧在损失和评分中移除，时间索引保持原100ms网格。',
           '评分使用原DSF segmenter和SELD scorer，macro/21类/20°/1秒块；活动阈值固定0.5，原3轨合并规则与15°阈值保留。最佳权重由validation SELD选择，相同SELD时以validation loss打破平局。test在训练结束后对最佳模型评估一次；per-class报告只重算保存的预测，没有再次运行模型或用test调参。',
           '本数据是known-sound/unseen-scene协议，每类只有一个官方训练WAV家族。它不是unheard-sound评估；本结果不是导航Agent成功率，也不能直接与原论文DCASE13类/视听SELD=0.354比较。',
           f"最佳checkpoint SHA256：`{result['best_checkpoint_sha256']}`；原权重SHA256：`{result['initial_checkpoint_sha256']}`；数据seal SHA256：`{result['dataset_seal_sha256']}`。",'',
           '运行入口：`run_finetune.sh`。新权重`best.pt`、末轮与恢复状态`last.pt`、逐epoch日志`history.jsonl`、结果`results.json`、测试预测`test_predictions.npz`、分类结果`test_per_class.csv`、梯度/参数更新审计均保存在本目录。',
           '推理：`python -m avn_foa_sameroom.infer_finetuned --checkpoint <best.pt> --runtime <archived_runtime> --wav <16k_10s_AmbiX.wav> --output <prediction.json>`。推理只读取WAV，不读取类别真值、源位置或RIR。未修改Agent/STOP，也没有自动替换线上模型。','']
    (a.run/'README_zh-CN.md').write_text('\n'.join(text))
    write_json(a.run/'posthoc_test_metric_check.json',dict(status='passed',model_reexecuted=False,metrics=checked))
    print(result,flush=True)
if __name__=='__main__':main()
