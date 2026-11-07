# AVN-FOA-SameRoom-v0.1

离线空间音频监督数据生成器。实际验收结果在数据目录的 `README_zh-CN.md` 和 `validation_report.json`，本文件说明复用接口。没有改动 Agent、STOP、官方导航 Episode、原始场景和模型权重。

本批使用官方训练资源，在建筑层面划分 33/7/7 个内部 train/validation/test。每类仅有一个官方训练 WAV 可用，独立原始录音数不可核验，故明确采用 **known-sound / unseen-scene**，不声称 unheard-sound 泛化。官方 21 类的 ID 来自仓库实际映射，通过官方 Episode 的物体和声音关联选源；类别不是本生成器硬编码的数字。

每条10秒，16kHz float32、接收器局部坐标 ACN/SN3D `[W,Y,Z,X]`；100ms标签，最多一个静态源。正样本有1–2个不循环的发声片段；约10%纯无事件；每个RIR最多三条不同朝向/裁剪的正样本。目标为2880/360/360条，即8/1/1小时。总时间包括实际静音和尾声，不乘声道数。

几何通过 `.house` 地面多边形和高度唯一归属判同房间；边界未知时拒绝。底盘必须可导航，源和接收器须连接合法空域。LOS同时检查原GLB碰撞网格与声学引擎实际导出的三角网格。当前引擎 `sourceIsVisible()` 在独立空盒中确认存在误报；只记录诊断值，使用独立射线结果，不修改引擎。材质映射关闭，保留默认材质反射/绕射/透射。独立配置 reset 声学 context，不带入历史混响状态。

`metadata.jsonl`、标签和RIR均含训练/离线评估真值，不能暴露给在线Agent。`acoustic_geometry/` 属于原建筑派生资产，不应随源码提交GitHub。

## 环境

云端代码：`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_foa_sameroom_worktree/`

数据：`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/research_outputs/AVN-FOA-SameRoom-v0.1/`

生成 Docker：`rdpi/avn-foa-sameroom:0.1`，Dockerfile见 `docker_avn_foa/`。基础为已有 `rdpi/savnce:render-v1`，只增加 Rtree1.4.1，离线构建。将 MP3D 挂载到 `/datasets/mp3d:ro`，项目以只读挂载，单独给 `research_outputs` 写权限。现有 `savnce_dev` 用于四张RGB审计图及原生Dataset检查。DSF前向用已有 `avseld_final_test` 的CUDA/Mamba环境。

完整命令记录在 `avn_foa_sameroom/RUN_COMMANDS.sh`；`run_batch.sh` 是本机实际执行的渲染入口，随后运行 `complete_dataset.sh` 完成全局可达性检查、孤立岛样本替换和最终验收。被拒绝的草稿移到独立 quarantine，最终仍为3600条；源、场景和官方 Episode 不变。路径是当前主机的约定，迁移时应显式调整。输出目录必须不存在，避免覆盖历史结果。

## 代码

- `fetch_assets.py`：只缓存官方 train Episode，不下载正式测试任务。
- `audit.py`：读取实际ID映射、WAV家族和官方目标，先建立建筑划分。
- `geometry.py`、`plan.py`、`parallel_plan.py`：独立筛选合法空间配置；场景种子由全局种子和场景ID哈希导出，线程数不改变任务定义。
- `calibrate.py`、`visibility_probe.py`、`acoustic_mesh.py`：FOA坐标/通道诊断、引擎可见性API诊断、实际声学网格LOS。
- `generate.py`：缓存真实原生FOA RIR，显式转成局部AmbiX。
- `synthesis.py`：完整时间线卷积、素材静音检测、传播延迟、尾声/边界mask和ADPIT标签。
- `loader.py`：原生16k前端、可配置读取窗口和hop、单源masked ADPIT损失。
- `forward_smoke.py`：严格恢复旧权重，检查原头和临时新头；不训练。
- `navcheck.py`、`repair_nav.py`：检查从官方起点的可达性，替换孤立NavMesh岛的接收器并保留拒绝草稿。
- `native_audio_probe.py`：实际执行原生SAVN音频并验证卷积历史和尾声。
- `native_check.py`：原生Dataset/高度偏移检查、静态分块overlap-add一致性和少量审计视图。
- `validate.py`、`tests.py`、`finalize.py`：验收、功能测试、中文报告和SHA256。

## DSF读取

```python
from avn_foa_sameroom.loader import Dataset, extractor, masked_adpit_loss

items = Dataset('/dataset/AVN-FOA-SameRoom-v0.1', split='train', window_s=2, hop_s=0.5)
sample = items[0]
# sample['waveform']: (4,32000)
# sample['adpit']: (20,6,4,C)，单源占原ADPIT的a0槽
# sample['valid_frames']: (20,)，尾声和不确定活动边界不参与损失
features = extractor('/path/to/original/dsf/runtime')(sample['waveform'][None])
```

真实16k前端的FFT512、窗342、hop100、128 Mel、0–8kHz，与旧24k的窗512、hop150、0–12kHz有明确区别；不是只改文件头。10秒特征形状仍为 `(1,7,1600,128)`，标签100帧。旧13类头输出117维，官方21类三轨头输出189维；旧encoder可严格加载，新类别头需要后续单独训练。检查点的超参数没有归一化/坐标系自描述字段，旧配置仅声明FOA，精确历史归一化仍为unverified；本批AmbiX是独立校准的标准格式，不能把前向shape通过解释为已验证定位语义或获得定位专家。

活动标签基于10ms素材RMS阈值和实际RIR起点延迟；100ms帧覆盖率>=80%为活动，混响尾声和不确定边界设ignore。无活动帧ACCDOA为0，真实距离不作为向量模长。CSV必须配合NPZ的ignore_mask使用。`masked_adpit_loss`为单源a0专用，拒绝多实例标签，保留三轨辅助复制语义。

本批主要是同房间、接近水平高度的定位数据；即使FOA格式上下校准通过，也不能据此宣称完整3D或NLOS覆盖。没有启动训练、LLM训练或策略修改。

## 后续制式核查与增强建议

已核实旧加载器的ACN声道与前/左/上坐标约定，以及TAU RIR的SN3D声明；METU旧FOA文件缺失，整个旧训练池仍为部分核实。参见 [制式溯源报告](docs/LEGACY_FOA_FORMAT_AUDIT_zh-CN.md)。

训练建议仅在线做音频与标签同步的水平yaw增强，保持竖直和原split；原ACS8包含镜像和上下翻转，另作消融。参见 [训练增强建议](docs/TRAINING_AUGMENTATION_zh-CN.md)。未启动训练、未扩写现有音频。
