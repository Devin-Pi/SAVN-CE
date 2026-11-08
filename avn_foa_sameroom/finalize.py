"""Write the Chinese delivery report and seal actual artifacts with SHA256."""
import argparse,collections,csv,json,shutil
from pathlib import Path
from .common import write_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--assets',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.root/'validation_report.json').read_text());assert report['status']=='passed'
    if not (a.root/'checkpoint_format_audit.json').exists():
        write_json(a.root/'checkpoint_format_audit.json',dict(checkpoint_sha256='85e380a6a8d2f1e545aa093e3f086f0a4d7830a15c41f7f79d047d7d9c1f9f90',archived_configuration_audio_format='foa',checkpoint_hyperparameters_have_normalization_or_coordinate_field=False,exact_checkpoint_normalization_self_description='unverified',loader_feature_channel_processing='archived ACN storage order retained; IV component variable names do not reorder channels',new_dataset_format='independently calibrated listener-local ACN/SN3D W/Y/Z/X',evidence_boundary='Strict load and forward verify tensor/config compatibility, not source-domain normalization or transferred localization accuracy.'))
    records=[json.loads(l) for l in (a.root/'metadata.jsonl').read_text().splitlines()]
    repair=json.loads((a.root/'navmesh_repair.json').read_text()) if (a.root/'navmesh_repair.json').exists() else {}
    report['navmesh_repair']=repair
    report['render_rejections']=json.loads((a.root/'render_rejections.json').read_text())
    write_json(a.root/'validation_report.json',report)
    geometry=json.loads((a.assets/'geometry_audit.json').read_text())
    write_json(a.root/'geometry_audit.json',geometry)
    shutil.copyfile(a.assets/'geometry_rejections.jsonl',a.root/'geometry_rejections.jsonl')
    source_assets=json.loads((a.root/'source_assets.json').read_text())
    # Correct the distinction between an official WAV asset and verified independent original recordings.
    for row in source_assets:
        row.pop('original_recordings',None);row['original_wav_files']=1;row['original_recording_count']='unknown'
        row['acoustic_event_semantics']='official object/category-to-WAV mapping; independent perceptual event verification not performed'
    write_json(a.root/'source_assets.json',source_assets)
    audit=json.loads((a.root/'asset_audit.json').read_text());audit['independent_recording_counts']='unknown beyond one official train WAV family per category'
    audit['geometry_goal_rejections']=geometry['goal_rejections'];audit['selected_render_rejections']=json.loads((a.root/'render_rejections.json').read_text())
    write_json(a.root/'asset_audit.json',audit)
    # Exhaustive matrix over rooms reached by the legal geometry audit; no fictitious coverage.
    available=collections.Counter();selected=collections.Counter();room_ids=set()
    for l in (a.assets/'geometry_candidates.jsonl').open():
        g=json.loads(l);key=(g['split'],g['scene_id'],g['room_id'],g['class_id'],str(g['distance_bin']));available[key]+=1
        room_ids.add((g['split'],g['scene_id'],g['room_id']))
    for r in records:
        if not r['is_no_event']:selected[r['split'],r['scene_id'],r['room_id'],r['class_id'],str(r['distance_bin'])]+=1
    from .geometry import Rooms
    split_manifest=json.loads((a.root/'split_manifest.json').read_text())
    for split,scenes in split_manifest['scene_ids'].items():
        for scene in scenes:
            model=Rooms(Path('/datasets/mp3d')/scene/(scene+'.house'))
            for rid in model.regions:room_ids.add((split,scene,rid))
    cmap=json.loads((a.root/'class_map.json').read_text())['official_category_to_class_id']
    zero=0
    with (a.root/'coverage_combinations.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['split','scene_id','room_id','class_id','category','distance_bin_m','legal_candidate_configurations','positive_records','status'])
        for split,scene,room in sorted(room_ids):
            for cat,cid in sorted(cmap.items(),key=lambda x:x[1]):
                for bin_ in ([.5,1],[1,2],[2,4],[4,8]):
                    key=(split,scene,room,cid,str(bin_));count=selected[key];n=available[key]
                    status='selected' if count else 'not_selected' if n else 'no_qualified_configuration_found'
                    zero+=int(not n);w.writerow([split,scene,room,cid,cat,str(bin_),n,count,status])
    lines=['# AVN-FOA-SameRoom-v0.1 质量报告','',
        f"验收状态：**{report['status']}**。有效导出 {report['records']} 条，每条 10 秒，总计 **{report['hours']:.1f} 小时**（不乘声道数）。",'',
        '| 划分 | 条数 | 小时 | 建筑 | 房间 |','|---|---:|---:|---:|---:|']
    for s,n in report['records_by_split'].items():lines.append(f"| {s} | {n} | {report['hours_by_split'][s]:.1f} | {report['buildings'][s]} | {report['rooms'][s]} |")
    lines += ['',f"覆盖 {report['covered_classes']}/{report['official_class_count']} 个官方类别，{report['target_instances']} 个目标实例，{report['native_rirs']} 条独立静态配置 RIR。{report['record_spatial_configurations']} 个记录朝向配置不等同于独立声场：每条原生 RIR 最多派生三条不同朝向和声音片段的正样本；无事件记录可共享同一空间配置。",'',
        '## 声音来源和划分边界','',
        '仅使用官方训练 Episode、训练建筑和 `train/*.wav`。每类当前有一个官方训练 WAV，按完整 WAV 保守归为一个来源家族；其内部是否拼接了多条原始录音无法核验，独立录音数量标为 unknown。未将随机裁剪当作独立录音。',
        '采用 **known-sound / unseen-scene**：33/7/7 个建筑在内部 train/validation/test 之间不重叠；声音家族有意共享并在 split_manifest.json 明示。不宣称 unheard-sound 泛化。官方验证/测试资源未加入本批。',
        '扫描训练 WAV 未发现跨类别完全相同的规范化解码波形；无法据此排除同源录音或确认物体名称就是自然声事件语义。标签沿用官方物体—声音关联，未擅自重标。','',
        '## 几何和声学','',
        '房间由 .house 的实际地面多边形与高度范围判定；边界、无区域、多重归属拒绝。声源还须与目标实例所属房间相符。底盘位于原 NavMesh，声源和接收器以通向 NavMesh 空域的无遮挡线及六轴表面净空检查保守验证。此方法不是封闭体积占据的全场景证明。',
        '源位置严格保留官方位置：原始 goal + 原生 continuous Dataset 的 0.1m + renderer 的 1.5m，仅添加各一次。四条样本已通过原生 Dataset 和 config merger 核对。未移动声源、修改网格或重建导航网格。',
        '每个选中空间配置同时通过原 GLB 碰撞网格双向射线和声学引擎实际导出三角网格的双向射线 LOS。声学导出保存在 acoustic_geometry，完整建筑参与传播。几何LOS不等于直达声能量占比或最强到达方向的保证。',
        '独立空盒诊断证实当前安装版本 sourceIsVisible() 可误报，局部源码显示其射线方向反转；该 API 仅作诊断，未用于通过判定，未修改仿真引擎。',
        '与部署保持同一声学设置：16kHz、反射/绕射/透射开启、indirectRayCount=500，材质映射关闭，使用引擎默认材质，不能称为真实表面材质模型。每条独立 RIR 前 reset 销毁 RLR context，temporalCoherence=False；RIR 含真实仿真响应，非解析直达声或双耳补通道。全部实际参数见 generation_config.yaml。','',
        '## 音频与标签','',
        'float32 无损 WAV，16,000 Hz、ACN/SN3D `[W,Y,Z,X]`。原生 RLR 世界坐标 N3D 显式换序、变号、除 sqrt(3)，再逆接收器 yaw 旋转；不分声道归一化。28 项独立方向/朝向诊断通过，原生 RIR 和转换后世界坐标 RIR 都保留。',
        '每条最多一个目标声源，固定源与接收器。每条 1–2 个时间片段；无机械循环、额外发声干扰或独立 0.25s 卷积拼接。完整源时间线一次卷积，尾声保留在真实录音时间轴内，超出10秒的截断能量显式记录。对照原 SAVN 250ms overlap-add 数学实现，静态 RIR 一致性检查通过；另实际执行 SAVNCE_Simulator 12步音频，与独立累积卷积误差约3.7e-9，并验证关闭声源后的尾声。',
        '源素材静音由10ms RMS检出，阈值为相对最大值 -50dB 且绝对下限1e-8；接收标签按实际 RIR 起点延迟。100ms 帧中活动覆盖>=80%作为监督活动；部分边界和有尾声但无直达活动的帧设 ignore_mask。该阈值是工程约定，低于阈值的活动不等于物理绝对静音。',
        f"纯无事件 {report['no_event_records']} 条（{report['no_event_fraction']:.1%}）；监督活动帧 {report['supervised_active_frame_fraction']:.2%}；忽略帧 {report['ignore_frame_fraction']:.2%}；源活动到达时长合计 {report['direct_activity_duration_s']:.2f}s。十小时指总录音时间轴，并非十小时持续发声。",'',
        '方向采用真实声源相对接收器的单位几何向量，局部 +X 前、+Y 左、+Z 上，右手系。使用保存的 R_world_from_audio 独立复算。非活动帧目标为0，不预测距离；距离只供离线分组。',
        '标签 NPZ 含100帧的 activity、direction_xyz、ignore_mask 和原 ADPIT 六个辅助槽 `(100,6,4,C)`。单源只占 a0；实际三轨输出的复制/置换损失语义不变。CSV 为原 DSF Cartesian 格式，忽略帧需配合 NPZ mask，不能直接当作无事件。','',
        '## 实测分布和拒绝','',
        f"距离范围 {report['distance_range_m']}m；[0,0.5,1,2,4,8,100] 边界直方图 {report['distance_histogram']}。方位[-180,180]的12等分直方图 {report['azimuth_histogram_12_bins']}。",
        f"俯仰范围 {report['elevation_range_deg']}度；分布见 validation_report.json。源和机器人都接近固定高度，本批主要检验水平定位，不能称为均匀完整3D覆盖。上下方向仅在格式诊断中单独校验。",
        f"最终额外核验所有{report['native_rirs']}个空间配置的接收器均可从对应官方Episode起点到达；替换了{repair.get('replaced_configurations',0)}个孤立岛配置、{repair.get('replaced_records',0)}条草稿，草稿保留于数据目录以外的quarantine。候选不足时仅新增合法接收器位置，并实际重渲染，不移动声源。\n\n空间筛选接受 {geometry['accepted_configurations']} 个候选；目标级拒绝原因：{geometry['goal_rejections']}。接收点尝试级拒绝见 geometry_audit.json，逐目标拒绝见 geometry_rejections.jsonl。",
        f"类别×全部注释房间×距离档矩阵见 coverage_combinations.csv，{zero} 个组合没有合格配置。无目标实例、遮挡或房间大小限制均不强行填补；本轮未找到候选不等于证明绝对不可行。",'',
        '## DSF兼容检查与使用','',
        '旧 audio-only 权重严格恢复，无 missing/unexpected key。旧13类头前向输出 `(1,100,117)`；临时21类随机头输出 `(1,100,189)`，只做shape与损失检查，未训练或覆盖原权重。原 ADPIT 标签转换器与本批标签一致。检查点超参数不含声道归一化/坐标系的自描述字段，旧配置只声明FOA；精确历史归一化仍标为unverified，不能靠一次前向认定定位语义完全兼容。新数据的AmbiX约定由独立诊断确认。0.5/1/2/10秒窗口和0.5秒读取hop通过。',
        '16k 前端：FFT512、窗342样本（21.375ms）、hop100（6.25ms）、128 Mel、0–8kHz；输出10秒 `(1,7,1600,128)`，标签100帧。旧24k配置为窗512/24k、hop150、0–12kHz，不能直接照搬；16k的8kHz以上信息不可恢复，Mel覆盖变化需后续适配。',
        '仅统计 train RMS，不使用验证/测试统计做归一化。原模型在各输入窗口内做时间均值/方差归一化。masked_adpit_loss 提供单源 a0 的三轨复制和忽略帧处理，多实例时会拒绝。','',
        '```python','from avn_foa_sameroom.loader import Dataset, extractor, masked_adpit_loss',
        "dataset = Dataset('/path/to/AVN-FOA-SameRoom-v0.1', 'train', window_s=2, hop_s=0.5)",
        'item = dataset[0]  # waveform / adpit / valid_frames；真值只供离线监督',
        '```','',
        '实际运行命令见 RUN_COMMANDS.sh；代码、容器ID、配置和文件哈希见 environment.json、environment.freeze.json、SHA256SUMS。同种子下任务/几何/声音片段采样可复现；RLR未暴露内部随机种子，不承诺RIR跨硬件逐字节一致，实际文件哈希可追溯。','',
        '**本次未开始完整模型训练、LLM训练或Agent策略修改。生成和前向成功不代表模型已经学会定位，也不代表解决跨房间NLOS。**','']
    text='\n'.join(lines)
    if (a.root/'legacy_foa_format_audit.json').exists():
        text=text.replace('检查点超参数不含声道归一化/坐标系的自描述字段，旧配置只声明FOA；精确历史归一化仍标为unverified，不能靠一次前向认定定位语义完全兼容。新数据的AmbiX约定由独立诊断确认。', '后续溯源已确认旧加载器的ACN声道及前/左/上坐标约定与新数据一致，TAU RIR明确为SN3D；METU旧FOA文件缺失，仍不能把整个旧训练池统一确认为SN3D。新数据的AmbiX约定由独立诊断确认。')
        text+='\n## 后续核查与训练准备\n\n[旧权重、RIR与FOA制式核查](LEGACY_FOA_FORMAT_AUDIT_zh-CN.md)；[训练增强建议](TRAINING_AUGMENTATION_zh-CN.md)。建议训练时仅做音频与标签同步的水平yaw增强，不原样启用全部ACS8；未开始训练。\n'
    (a.root/'README_zh-CN.md').write_text(text)
    (a.root/'coverage_report.md').write_text(text)
    # Hash every finished data/report artifact. The seal itself is deliberately excluded.
    (a.root/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(a.root))+'\n' for p in sorted(a.root.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print('sealed',a.root,flush=True)
if __name__=='__main__':main()
