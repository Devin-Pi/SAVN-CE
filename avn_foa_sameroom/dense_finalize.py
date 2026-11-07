"""Publish actual dense-corpus statistics and seal all completed artifacts."""
import argparse
import collections
import json
from pathlib import Path
import shutil
from .common import sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--environment',type=Path,required=True);p.add_argument('--commands',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.root/'validation_report.json').read_text());assert report['status']=='passed' and report['records']==5400 and report['hours']==15
    assert report['active_grid_fraction']>=.8 and report['no_event_records']==540
    config=json.loads((a.root/'generation_config.json').read_text())
    rows=[json.loads(line) for line in (a.root/'metadata.jsonl').read_text().splitlines()]
    shutil.copy2(a.environment,a.root/'environment.json');shutil.copy2(a.commands,a.root/'RUN_COMMANDS.sh')
    testlog=a.environment.parent/'unit-tests-packaged.log'
    assert 'Ran 9 tests' in testlog.read_text() and 'OK' in testlog.read_text()
    shutil.copy2(testlog,a.root/'unit_tests.log')
    package=a.root/'code_snapshot'/'avn_foa_sameroom';package.mkdir(parents=True)
    for name in ('__init__.py','common.py','synthesis.py','loader.py','dense_synthesis.py','dense_generate.py','dense_validate.py','dense_finalize.py','test_dense_synthesis.py'):
        shutil.copy2(Path(__file__).with_name(name),package/name)
    write_json(a.root/'train_normalization_statistics.json',dict(train_only_scalar_rms=report['train_only_scalar_rms'],used_for_normalization=False))
    profile=json.loads((a.root/'source_activity_audit.json').read_text())
    text=['# AVN-FOA-SameRoom-v0.2-Dense15h','',
        f"验收通过：**5400条×10秒＝15小时**，16kHz、4通道ACN/SN3D W/Y/Z/X。监督活动帧占全部100ms时间网格的 **{report['active_grid_fraction']:.2%}**，包含纯无事件记录及ignore帧在分母内。",'',
        '| split | 条数 | 小时 | 正样本 | 无事件 | 全网格活动率 | 非ignore活动率 | 建筑 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split,s in report['split'].items():
        text.append(f"| {split} | {s['records']} | {s['hours']:.1f} | {s['positive_records']} | {s['negative_records']} | {s['active_grid_fraction']:.2%} | {s['active_nonignored_fraction']:.2%} | {s['buildings']} |")
    text+=['',f"全部正样本至少有{report['positive_minimum_active_frames']}/100个真实监督活动帧。纯无事件540条＝1.5小时，全部为真实零波形、零活动标签且无ignore。正样本录音时间13.5小时；按活动标签计的有效监督时间约{report['hours']*report['active_grid_fraction']:.3f}小时。",'',
        '## 与旧版的可比边界','',
        f"旧版总录音{report['baseline_hours']:.1f}小时，监督活动网格比例{report['baseline_supervised_active_grid_fraction']:.2%}；新版本为15小时和{report['active_grid_fraction']:.2%}。旧数据、旧95轮模型和评估结果均未覆盖。活动密度和总时长同时变化，之后不能把性能变化单独归因于其中一个因素。",
        '新版本采用原先通过几何、房间、LOS、NavMesh可达性和声学网格核查的1080条真实SoundSpaces原生16k RIR。逐条校验复制文件与原文件字节一致，源位置、接收位置、类别和物体实例与原验收记录一致；新渲染RIR数为0，没有新增独立声场。新录音变化来自声源片段编排、全局电平和接收器yaw，不能将5400条记录称为5400个独立声场。',
        '仍为同房间LOS、静态单声源、无额外干扰。RIR保留原仿真的反射、混响及传播设置；LOS不等于无混响。未移动官方声源、修改MP3D网格、重建NavMesh或改官方Episode。',
        '原33/7/7建筑划分不变，场景和RIR不跨split；每类仍只有一个官方训练WAV家族，原始独立录音数量未知，按known-sound/unseen-scene报告。没有引入官方验证或测试资源。新内部test不是新的封存官方AVN测试集。',
        '15小时是合成记录时间轴，包含重复的静音对照及有记录的来源片段再利用，不代表15小时独立原始录音。', '',
        '## 高活动音频怎样生成','',
        '先对原始干声以10ms RMS、相对最大值−40dB和绝对下限1e-7筛选真实活动区间，保留至少50ms的连续区间。长素材存在足够长的连续活动段时优先单段读取；短促素材使用随机真实片段重新触发，片段间做5ms交叉淡化，组成同一个固定位置的单声源流。没有把整条WAV机械tile、时间拉伸、填充静音后强行标活动，也没有生成新的独立声源家族。',
        '每次裁剪的原始16k采样区间、发声采样偏移、fade长度和秒级区间均在metadata.jsonl/events保存。短促类别的发声次数和节奏会发生显著变化；这是有意构建的高占空比训练条件，不能称为原始官方素材的自然时间分布。',
        '完整10秒干声时间线一次性与位置相符的4通道RIR卷积。没有逐0.25秒独立卷积后丢弃历史尾声；录音内混响保留，超出10秒的尾声能量显式记录。四通道使用同一全局增益，峰值限制0.95，不分别归一化通道。',
        '标签仍从真正发声的干声生成：10ms活动阈值为相对最大值−50dB及绝对下限1e-8，再加实际RIR起点延迟；100ms帧中到达活动占比≥80%且实际接收波形可听性RMS达标才计活动。部分边界、仅尾声和接收端低于可听性下限的帧设ignore。这里的单帧80%占用规则与全数据80%活动目标是两个不同定义。',
        f"正样本每条重新触发事件数范围为{report['event_count_range_positive']}。每条正样本活动不足92帧会拒绝并重新采样，拒绝次数和原因保留。",'',
        '## 独立验收与模型接口','',
        f"逐文件检查5400条WAV格式、有限数值、峰值、哈希、NPZ和CSV；从原始素材重建每条干声时间线，独立重算活动/ignore及方向标签。完整波形卷积重建{report['full_waveform_reconstruction_checks']}条，至少覆盖每个RIR空间配置一次。全部21类在每个完整split都有正样本，单类正样本活动率均≥92%，含该类关联无事件记录后的活动率也≥80%。", 
        '方向采用监听者局部+X前/+Y左/+Z上，W/Y/Z/X顺序和SN3D不变；源和接收器的几何及官方关联复用验收证书，明确记录在parent_certificates.json，不冒充新执行的场景重渲染。原始声音及Episode文件哈希重新检查。',
        '原生16k loader、10秒特征(1,7,1600,128)、21类ADPIT标签(100,6,4,21)及189维输出前向兼容检查通过，原13类权重严格加载和随机21类新头只用于形状/有限值/损失检查，不是新模型成绩。没有执行optimizer、重新训练encoder或修改Agent/STOP。', '',
        '## 原始声音的活动情况','',
        '| 类别 | 原WAV秒数 | 原10ms活动比例（−40dB） | 最长连续选中活动秒数 |','|---|---:|---:|---:|']
    for s in profile:text.append(f"| {s['category']} | {s['source_duration_s']:.2f} | {s['source_original_10ms_active_fraction']:.2%} | {s['longest_selected_region_s']:.2f} |")
    text+=['', '## 后续训练使用边界','',
        '模型读取方式保持原loader，valid_frames必须排除ignore。训练时仍可同步做水平yaw增强；此数据阶段没有应用会改变手性或上下轴的ACS8翻转。训练统计仅从train获取，没有用validation/test做归一化。',
        '短促类别的重触发节奏可能成为额外的人工分类线索；后续比较必须同时报告原始v0.1稀疏验证/测试表现，检查漏检改善和误报代价；仅在高活动新test上变好，不能证明自然AVN分布或在线导航也改善。原先已观察到的内部测试属于开发反馈，不将本版描述为独立的unheard-source或封存官方测试。',
        '有效活动比例提升不等于localization expert已经提升。仍未验证新音源、在线移动、短窗口、跨房间NLOS和多源导航。', '',
        '## 文件与复现','',
        '`audio/`为新FOA WAV，`labels/`为NPZ及Cartesian CSV，`metadata.jsonl`为源片段和空间溯源，`rir/`为校验复制的真实RIR；`class_coverage.csv`含每类每split样本及活动比例，`validation_report.json`含实际验收统计。',
        '`generation_config.json`、`environment.json`、`RUN_COMMANDS.sh`和`code_snapshot/`保存实际设置、Docker镜像ID、命令及代码；`SHA256SUMS`封存最终数据/报告。相同种子与同一缓存RIR/环境的任务定义可复现；没有承诺独立重渲染的RIR跨硬件逐位一致。',
        '原始SoundSpaces/MP3D资产及官方声音仍按外部依赖引用，数据中保留哈希与来源，不将其授权状态改写为可自由再分发。']
    (a.root/'README_zh-CN.md').write_text('\n'.join(text)+'\n')
    (a.root/'SHA256SUMS').write_text(''.join(sha(f)+'  '+str(f.relative_to(a.root))+'\n' for f in sorted(a.root.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
    print('DENSE15H_ACCEPTED_AND_SEALED',flush=True)


if __name__=='__main__':main()
