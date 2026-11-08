"""Generate a primary-new-test report and a separate sparse transfer diagnostic."""
import argparse,csv,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.comparison.read_text());cases=d['cases'];out=a.output;out.mkdir(parents=True,exist_ok=True)
    text=['# AVN Dense15h：100 epoch 最终分析','',
        '**主评估采用新版高活动测试集。** 540条10秒录音、1.5h，和新训练数据使用相同生成协议，场景/RIR按train/validation/test隔离；旧360条稀疏test仅作跨分布诊断。',
        '训练完整完成100 epoch、27000次更新。北京时间2026-11-08 04:06容器正常退出（code0，无OOM）。第97轮由新版validation SELD选为最佳；validation SELD=0.028729、F20=0.965293、LE=5.5782°。上一轮模型在用户要求下完成95轮，最佳为第95轮。','',
        '## 主结果：在同一个新版test上比较','',
        '| 模型 | SELD↓ | F20↑ | LE°↓ | LR↑ | ER↓ |','|---|---:|---:|---:|---:|---:|']
    for label,k in [('上一轮95epoch模型','old_dense'),('本轮100epoch模型','new_dense')]:
        m=cases[k]['metrics'];text.append('| '+label+' | '+' | '.join(f'{m[n]:.6f}' for n in ('SELD','F20','LE','LR','ER'))+' |')
    text+=['','这是同一新版test上的直接对照；没有把旧test分数和新test分数的差当作模型增益。新模型的测试模型前向已在训练结束后完成，本次仅重算其保存预测；旧模型在新test上补做一次固定检查点评估。四组保存预测均用原DSF评分器核对一致。','',
        '## 新版test：21类结果','',
        '| 类别 | 旧模型F20 | 新模型F20 | 旧模型LE° | 新模型LE° | 新模型LR | 新test正例数 |','|---|---:|---:|---:|---:|---:|---:|']
    for old,new in zip(cases['old_dense']['per_class'],cases['new_dense']['per_class']):
        assert old['category']==new['category']
        text.append(f"| {new['category']} | {old['F20']:.4f} | {new['F20']:.4f} | {old['LE']:.2f} | {new['LE']:.2f} | {new['LR']:.4f} | {new['positive_clips']} |")
    text+=['','此前弱类在新版test上：table/picture/cushion、bed、chest_of_drawers、towel、bathtub均需按上表与旧模型在同一新版test的结果比较。不能将新版连续/高活动事件的改善表述为已解决旧版短暂/间歇事件漏检。','',
        '在新test上，新模型table/picture/cushion/bed/chest_of_drawers/bathtub的正确类别帧召回均为100%，towel为99.01%；这些此前的严重漏检在新版分布下明显改善。新test剩余相对弱类为towel（F20=0.8696）、shower（0.9091）、bathtub（0.9130）和gym_equipment（0.9430）；shower/gym_equipment正确类别帧召回约91.41%/90.42%，仍有漏检。towel/bathtub的类别帧召回很高，较低F20还需结合方向阈值和额外预测解释，不宜全部叫作活动漏检。','',
        '## 静音与活动诊断','',
        '| 模型/数据 | 全无事件clip误报 | 有效无活动帧误报/帧数 | GT活动帧正确类别检出率 | 活动帧无任何输出 | 活动帧仅输出其他类 |',
        '|---|---:|---:|---:|---:|---:|']
    for k in ('old_dense','new_dense','old_sparse','new_sparse'):
        v=cases[k]['activity_diagnostics']
        text.append(f"| {k} | {v['all_no_event_clips_with_false_alarm']}/{v['all_no_event_clips']} | {v['valid_background_frames_with_prediction']}/{v['valid_background_frames']} | {v['correct_class_frame_recall']:.2%} | {v['active_frames_with_no_prediction']}/{v['total_gt_active_frames']} | {v['active_frames_with_other_class_only']}/{v['total_gt_active_frames']} |")
    text+=['','帧级诊断仅判断正确类别是否输出，不加20°约束，和官方1秒块F20/LR不同。排除了ignore帧；“无活动帧”不等于原始波形绝对静音（例如忽略规则之外的低能量尾部），所以另列真正全无事件clip。新版负例为全零波形，未覆盖现实背景噪声；新版正例活动接近连续，不能充分检验长静音间隙。全静音误报若前后均为0，只能说保持，不能称为改善。','',
        '关键区别：全无事件clip均为100帧、无ignore的全零录音。扣除这些负例后，新test正例仅剩6个有效无活动帧，旧模型0帧报警、新模型5帧报警；其余静音/尾部多被ignore，不能凭5406个总体无活动帧上的低误报率声称已充分验证事件间静音。旧sparse正例有20225个有效无活动帧，旧模型1帧报警（0.0049%），新模型760帧报警（3.7577%）；因此间隙误报没有改善，反而退化。这是按既有标签/ignore规则计算的背景帧诊断，不保证每一帧波形绝对为零。','',
        '## 旧版稀疏test：单独的跨分布诊断','',
        '| 模型 | SELD↓ | F20↑ | LE°↓ | LR↑ | ER↓ |','|---|---:|---:|---:|---:|---:|']
    for label,k in [('上一轮95epoch模型','old_sparse'),('本轮100epoch模型','new_sparse')]:
        m=cases[k]['metrics'];text.append('| '+label+' | '+' | '.join(f'{m[n]:.6f}' for n in ('SELD','F20','LE','LR','ER'))+' |')
    text+=['','| 类别 | 旧模型F20 | 新模型F20 | 旧模型LE° | 新模型LE° | 新模型LR |','|---|---:|---:|---:|---:|---:|']
    for old,new in zip(cases['old_sparse']['per_class'],cases['new_sparse']['per_class']):
        text.append(f"| {new['category']} | {old['F20']:.4f} | {new['F20']:.4f} | {old['LE']:.2f} | {new['LE']:.2f} | {new['LR']:.4f} |")
    text+=['','旧版test中table/picture仍无匹配；cushion虽出现极少匹配，但方向误差大、F20仍为0。bed/chest_of_drawers/towel/bathtub也未改善；原本表现较好的许多类别出现明显退化。该诊断说明模型适应了新生成分布，不能代表对各种事件时长通用。',
        f"最后10轮dense validation SELD范围：{d['last10_dense_validation_SELD_range']}；sparse validation范围：{d['last10_sparse_validation_SELD_range']}。训练中的sparse validation已经显示退化，非测试结束后才出现。只按dense validation选模是本轮预先固定规则；本次没有用test更换权重或降低阈值。",'',
        '## 如何解释，以及适用边界','',
        '新数据不只是增大样本数：旧版每段10秒只有1–2个事件，随机截取约1.1–3.2秒原片段，尊重原WAV内静音；新版从实际活动区间取片段并交叉淡化、多次触发，使正例约97/100帧活动，整体含负例活动占比87.42%。这改变了时序、瞬态/稳态结构和每类有效监督量。新旧沿用既有RIR与官方WAV家族，没有新增独立录音或声学几何。',
        '原模型在10秒前端特征上做时间维mean/std标准化，卷积模块还含BatchNorm；输入时序分布变化可能影响这些统计及后续表征。代码能证明这些机制存在，当前交叉评估能证明分布敏感，但没有消融证明哪个机制是主要原因。不能把所有增益或退化仅归因于活动比例、15h数据量、混响或绕射；训练轮数也从95变为100。',
        '判断：新权重在新版高活动、同房间LOS静态单源、已知WAV家族/未见场景协议下表现很强，可作为该协议的localization expert候选；不能据此宣称已验证在线AVN、短窗口、短暂/间歇声音、NLOS、多源、移动源或未见独立音源。旧test已提供开发反馈，应称内部迁移诊断，不是原封未触及的官方最终测试集。',
        '若后续AVN主任务采用新协议，可使用新版test作为主结果并保留该检查点。若在线输入包含大量短暂和间歇事件，建议下一轮让训练/验证同时覆盖高活动、稀疏、短事件及有噪背景，再固定独立test；本次未自动开始新训练、未修改数据/模型、未调整解码阈值，也未替换Agent。',
        'LE=180°在无匹配类别中是DSF默认值，不是测得方向恰好相反。classwise ER沿用原实现的全局ER，不能当成独立类别ER。固定21类macro、20°、1秒块、活动阈值0.5、三轨15°合并；原decoder的两轨合并抑制行为保留。','',
        '## 证据与复现','',
        '完整结构化结果：comparison.json；四组分类CSV：old_dense_per_class.csv、new_dense_per_class.csv、old_sparse_per_class.csv、new_sparse_per_class.csv。comparison.json还保存每类帧级漏检、错分类、原始GT类范数，以及全静音和无活动帧误报。',
        '实际补测及审计命令见avn_foa_sameroom/run_analyze_dense100.sh。其独立容器不修改训练代码快照、旧权重或原数据；旧模型在新test补测1次，其余3组仅分析保存预测。',
        f"新best.pt SHA256：`{d['new_checkpoint_sha256']}`；旧best.pt SHA256：`{d['old_checkpoint_sha256']}`。",'']
    (out/'ANALYSIS_zh-CN.md').write_text('\n'.join(text)+'\n')
    fields=['test_distribution','class_id','category','old_F20','new_F20','old_LE','new_LE','old_LR','new_LR','old_SELD','new_SELD']
    with (out/'class_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for split in ('dense','sparse'):
            for old,new in zip(cases[f'old_{split}']['per_class'],cases[f'new_{split}']['per_class']):
                row=dict(test_distribution=split,class_id=new['class_id'],category=new['category'])
                for metric in ('F20','LE','LR','SELD'):row['old_'+metric]=old[metric];row['new_'+metric]=new[metric]
                w.writerow(row)
    print(out/'ANALYSIS_zh-CN.md')

if __name__=='__main__':main()
