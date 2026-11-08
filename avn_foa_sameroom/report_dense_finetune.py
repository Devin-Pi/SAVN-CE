"""Final dense/sparse report from saved predictions, without additional model passes."""
import argparse,csv,json
from pathlib import Path
from .common import write_json
from .finetune import TrainData
from .evaluate_stopped_finetune import audit_predictions


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference-data',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
    result=json.loads((a.run/'results.json').read_text());config=json.loads((a.run/'config.json').read_text())
    assert result['all_requested_epochs_completed'] and not result['test_used_for_selection']
    diagnostics={};text=['# Dense15h 全网络微调最终结果','','最佳权重由新版 dense validation SELD 选择；旧 sparse validation 仅作诊断。测试仅在全部训练完成后运行。','',
        '| split | SELD↓ | F20↑ | LE°↓ | LR↑ | ER↓ |','|---|---:|---:|---:|---:|---:|']
    for key in ('test','reference_test'):
        data=TrainData(a.data if key=='test' else a.reference_data,'test',window_s=10,hop_s=10)
        checked,rows,diagnostics[key]=audit_predictions(a.run/f'{key}_predictions.npz',data,a.runtime,config['classes'])
        for metric in ('SELD','ER','F20','LE','LR'):assert abs(checked[metric]-result[key][metric])<1e-8,(key,metric)
        with (a.run/f'{key}_per_class.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        text.append('| '+key+' | '+' | '.join(f'{checked[k]:.6f}' for k in ('SELD','F20','LE','LR','ER'))+' |')
    text+=['',f"完成 {result['completed_epochs']} epoch，{result['optimizer_steps']} 次更新，最佳 epoch {result['best_epoch']}。",
        '4320×10秒 train（12h）；新版 validation/test 各540×10秒（1.5h），旧版 reference validation/test 各360×10秒（1h）。',
        '与旧模型比较应使用 reference_test。新旧测试共享已知声音家族及各自保留的场景/RIR池；旧测试已提供过开发反馈，不是未触及的最终官方测试集。',
        '该协议为known-sound/unseen-scene、同房间LOS静态单源。新增数据主要增加活动密度和时间组合，没有新增独立音源或声学几何。',
        '沿用DSF的21类macro、20度阈值、1秒块、活动阈值0.5和三轨合并15度。无匹配事件的LE=180度是默认值；须同时看F20/LR和静音误报。',
        '10秒输入的离线结果不代表在线短窗口、NLOS、多源、移动声源或Agent导航成功。未自动替换Agent模型。',
        f"best.pt SHA256: {result['best_checkpoint_sha256']}",'']
    (a.run/'README_zh-CN.md').write_text('\n'.join(text))
    write_json(a.run/'posthoc_test_metric_check.json',dict(status='passed',model_reexecuted=False,diagnostics=diagnostics))
    print('DENSE_AND_REFERENCE_TEST_REPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
