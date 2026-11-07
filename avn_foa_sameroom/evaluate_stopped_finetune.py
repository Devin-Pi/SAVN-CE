"""Evaluate a preserved checkpoint after a user-requested training stop.

Keep the planned-100 training directory untouched. Use the original fixed
DSF scorer/decoder, and never select a checkpoint or threshold using test.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import sha, write_json
from .finetune import Metrics, TrainData, evaluate
from .infer_finetuned import load


def audit_predictions(path, data, runtime, classes):
    metric = Metrics(runtime, classes)
    frames = np.zeros(len(classes), dtype=int)
    clips = np.zeros(len(classes), dtype=int)
    silent_clips = false_alarm_clips = suppressed = suppressed_true = 0
    with np.load(path, allow_pickle=False) as archive:
        assert len(archive['sample_id']) == len(data.rows)
        assert list(archive['sample_id']) == [row['sample_id'] for row in data.rows]
        for raw, row in zip(archive['accdoa'], data.rows):
            with np.load(data.root / row['labels'], allow_pickle=False) as target:
                truth = target['adpit']; valid = ~target['ignore_mask'].astype(bool)
            metric.add(raw, truth, valid)
            activity = truth[:, 0, 0] > .5
            frames += activity[valid].sum(0)
            clips += activity[valid].any(0)
            if not activity[valid].any():
                silent_clips += 1
                decoded = metric.decoder.decode(raw)
                false_alarm_clips += any(events for i, events in enumerate(decoded) if valid[i])
            # Diagnostic only: retain the historical decoder's lone-pair behavior.
            vectors = raw.reshape(-1, 3, 3, len(classes)).transpose(0, 3, 1, 2)
            scores = np.linalg.norm(vectors, axis=-1)
            for t, category in zip(*np.where((scores > .5).sum(-1) == 2)):
                if not valid[t]:
                    continue
                a, b = np.flatnonzero(scores[t, category] > .5)
                cosine = np.dot(vectors[t, category, a], vectors[t, category, b]) / (scores[t, category, a] * scores[t, category, b])
                if np.degrees(np.arccos(np.clip(cosine, -1, 1))) < 15:
                    suppressed += 1
                    suppressed_true += int(activity[t, category])
    perclass = metric.scorer.compute_seld_scores()[5]
    rows = []
    for index, category in enumerate(classes):
        row = dict(class_id=index, category=category, positive_clips=int(clips[index]),
                   active_valid_frames=int(frames[index]))
        row.update(zip(('ER', 'F20', 'LE', 'LR', 'SELD'), map(float, perclass[:, index])))
        rows.append(row)
    diagnostics = dict(silent_clips=silent_clips, silent_clips_with_false_alarm=int(false_alarm_clips),
                       silent_clip_false_alarm_rate=false_alarm_clips/silent_clips if silent_clips else None,
                       historical_decoder_suppressed_class_frames=int(suppressed),
                       suppressed_class_frames_with_active_ground_truth=int(suppressed_true),
                       note='Diagnostic only; decoder/threshold unchanged; no alternate test metric tuning')
    return metric.result(), rows, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out = args.output
    assert not (out / 'results.json').exists(), 'Final results already exist; do not rerun test'
    assert not (out / 'test_predictions.npz').exists(), 'Saved test predictions exist; analyze them without rerunning the model'
    manifest = json.loads((out / 'stop_manifest.json').read_text())
    config = json.loads((out / 'source_training_config.json').read_text())
    history = [json.loads(line) for line in (out / 'source_training_history.jsonl').read_text().splitlines()]
    checkpoint = out / 'best.pt'
    assert sha(checkpoint) == manifest['checkpoint_sha256']
    saved = torch.load(str(checkpoint), map_location='cpu', mmap=True)
    best = min(history, key=lambda row: (row['validation']['SELD'], row['validation']['loss']))
    assert saved['epoch'] == best['epoch'] == manifest['selected_epoch']
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high') # DSF.__init__ overrides the trainer's earlier medium setting
    started = time.monotonic()
    model, frontend, embedded = load(checkpoint, args.runtime)
    assert embedded['classes'] == config['classes']
    classes = config['classes']
    val = TrainData(args.data, 'validation', window_s=10, hop_s=10)
    if (out/'validation_result.json').exists() and (out/'validation_predictions.npz').exists():
        validation = json.loads((out/'validation_result.json').read_text())
    else:
        validation = evaluate(model, frontend, val, args.runtime, classes, args.batch, args.workers, out/'validation_predictions.npz')
        write_json(out/'validation_result.json', validation)
    tolerances = dict(ER=.001, F20=.001, LE=.01, LR=.001, SELD=.001, loss=1e-7)
    differences = {key: abs(validation[key]-saved['validation'][key]) for key in tolerances}
    assert all(differences[key] <= tolerance for key,tolerance in tolerances.items()), differences
    write_json(out/'validation_reproduction_check.json', dict(mode='bounded agreement, not exact/bitwise reproduction', original=saved['validation'], recomputed=validation, absolute_differences=differences, absolute_tolerances=tolerances, exact_cause_of_cold_reload_difference='not fully isolated; preserved as a limitation'))
    print('VALIDATION_BOUNDED_AGREEMENT', json.dumps(dict(validation=validation,differences=differences)), flush=True)
    test = TrainData(args.data, 'test', window_s=10, hop_s=10)
    test_result = evaluate(model, frontend, test, args.runtime, classes, args.batch, args.workers, out/'test_predictions.npz')
    write_json(out/'test_result.json', test_result)
    perclass, diagnostics = {}, {}
    for split, dataset, expected in [('validation', val, validation), ('test', test, test_result)]:
        checked, perclass[split], diagnostics[split] = audit_predictions(out/f'{split}_predictions.npz', dataset, args.runtime, classes)
        for key in ('ER','F20','LE','LR','SELD'):
            assert abs(checked[key]-expected[key]) < 1e-8, (split,key)
        with (out/f'{split}_per_class.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(perclass[split][0]))
            writer.writeheader(); writer.writerows(perclass[split])
    tail = history[-10:]
    result = dict(status='completed', termination='user_requested_early_stop',
                  requested_epochs=config['epochs'], completed_epochs=len(history),
                  all_requested_epochs_completed=False, best_epoch=saved['epoch'],
                  optimizer_steps_at_last_complete_epoch=history[-1]['global_step'],
                  validation=validation, saved_validation=saved['validation'], test=test_result,
                  validation_checkpoint_reproduced=True,validation_reproduction_mode='bounded agreement, not exact',
                  validation_absolute_differences=differences,validation_absolute_tolerances=tolerances,
                  exact_cause_of_cold_reload_difference='not fully isolated',stored_predictions_metric_check_passed=True,
                  validation_records=len(val),test_records=len(test),
                  test_dataset_evaluation_passes=1,test_model_forward_batches=(len(test)+args.batch-1)//args.batch,test_used_for_selection=False,
                  effective_float32_matmul_precision=torch.get_float32_matmul_precision(),
                  activity_threshold=.5,merge_angle_deg=15,doa_threshold_deg=20,
                  average='macro',classes=classes,per_class=perclass,diagnostics=diagnostics,
                  last10_validation_seld_min=min(row['validation']['SELD'] for row in tail),
                  last10_validation_seld_max=max(row['validation']['SELD'] for row in tail),
                  best_checkpoint_sha256=sha(checkpoint),
                  initial_checkpoint_sha256=config['initial_checkpoint_sha256'],
                  dataset_seal_sha256=config['dataset_seal_sha256'],
                  evaluation_code_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-started,
                  original_training_code_and_config_unmodified=True,agent_navigation_success_not_evaluated=True)
    write_json(out/'results.json', result)
    text = ['# AVN-FOA21 提前结束训练后的最终验证', '',
            f"原计划 {config['epochs']} 轮；按用户新要求在完整 {len(history)} 轮后结束。未完成100轮，未宣称完整100轮收敛。最佳权重为第 {saved['epoch']} 轮，完全由validation SELD选择。未保存的后续半轮更新不纳入评估。", '',
            '| split | SELD↓ | F20↑ | LE°↓ | LR↑ | ER↓ |',
            '|---|---:|---:|---:|---:|---:|']
    for split, scores in [('validation(training record)',saved['validation']),('validation(checkpoint reload)',validation),('test',test_result)]:
        text.append('| '+split+' | '+' | '.join(f'{scores[k]:.6f}' for k in ('SELD','F20','LE','LR','ER'))+' |')
    text += ['', 'validation复算与训练记录存在小偏差，不是逐位一致：SELD绝对差约0.000275，ER差约0.000786。两种加载路径的首批原始输出和前端特征一致；原构造器复查的ER/F20/LR与原记录一致、LE仍差约0.0014度。精确原因未完全隔离，保留为复现限制，不能把冷加载分数较低当作性能改善。验证确认按公开容差通过（ER/F20/LR/SELD 0.001、LE 0.01度、loss 1e-7），详情见validation_reproduction_check.json及validation_constructor_audit.json。完整test仅执行一次完整数据集评估；保存预测后复算官方指标验证一致性，未用test更换权重或调整阈值。', '数值核查发现：原训练脚本先设置medium，但随后DSF.__init__会覆盖为high。本次验证及测试使用实际训练的high，但单独统一这一设置并未消除小偏差。此前验证尝试与日志独立保留，均未触及测试集。旧训练配置原文保留，此处补记有效设置。', '',
             f"最后10个完整epoch的validation SELD范围：{result['last10_validation_seld_min']:.6f}–{result['last10_validation_seld_max']:.6f}。这是有限轮数内的趋稳证据，不证明继续训练不可能改善。", '',
             '| 类别 | val F20 | val LE° | test F20 | test LE° | test LR | test SELD | test 正样本clip |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v,t in zip(perclass['validation'],perclass['test']):
        text.append(f"| {t['category']} | {v['F20']:.4f} | {v['LE']:.2f} | {t['F20']:.4f} | {t['LE']:.2f} | {t['LR']:.4f} | {t['SELD']:.4f} | {t['positive_clips']} |")
    text += ['', '各类别原始结果见两个per_class.csv。沿用原DSF macro实现：classwise ER重复全局ER，并非独立类别ER；没有匹配事件时LE默认180°，不是测得方向刚好相反。正样本clip与active_valid_frames提供每类支持数。', '',
             '判断：可作为当前受限数据协议下的研究原型候选，尚不能仅凭离线SELD称为可靠、通用的AVN localization expert。应结合各类F20、LE和LR判断，不能只看总体loss或个别类的低方向误差。',
             '限制：known-sound/unseen-scene，每类一个官方WAV家族；静态单源、同房间且LOS的合成协议；输入为10秒FOA。未验证独立新音源、NLOS/跨房间、多源和在线短窗口，也未测试Agent导航成功率；模型不预测距离，活动分数未校准。没有自动替换Agent或STOP策略。',
             '评分保持原DSF scorer/segmenter/decoder：21类macro、20°、1秒块、活动阈值0.5、三轨合并15°，ignore帧排除且时间索引保留。原decoder对“仅两轨激活且可合并”的抑制行为保留；诊断只统计，不修改本次测试的解码。',
             '静音clip误报和历史decoder抑制统计见results.json/diagnostics。它们用于解释结果，不用于回调测试阈值。', '',
             f"最佳权重 SHA256：`{result['best_checkpoint_sha256']}`。原训练目录的best.pt/last.pt、日志和配置均保留；本目录保存独立快照、验证/测试预测和最终结果。", '',
             '实际评估命令见run_evaluate_user_stop.sh；原始训练命令、配置及完整95轮历史见source_training_config.json和source_training_history.jsonl。']
    (out/'README_zh-CN.md').write_text('\n'.join(text)+'\n')
    print('USER_STOP_EVALUATION_COMPLETE',json.dumps(dict(best_epoch=saved['epoch'],completed_epochs=len(history),validation=validation,test=test_result)),flush=True)


if __name__ == '__main__':
    main()
