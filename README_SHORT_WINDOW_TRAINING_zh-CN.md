# AVN 0.25／1／1.5秒上下文训练

这是完成[离线／在线差异诊断](README_LOCALIZATION_DIAGNOSIS_zh-CN.md)后，按用户新授权启动的独立研究实验。目的：比较每0.25秒更新一次定位时，最近0.25、1、1.5秒的历史上下文是否有帮助。模型不参与导航，原模型、原音频和原标签保留。

**状态更新：三组已完成100轮和最终测试。** 统一指标、各类别、静音、耗时及已有在线录音回放见[完整100轮分析](README_SHORT_WINDOW_FULL100_zh-CN.md)。下文保留实际训练协议与启动记录。

## 窗口与一步结果

输入窗长与更新间隔是两个参数。三组都预测**当前最后0.25秒**，长窗中更早的真实音频仅作上下文。没有读取窗口结束之后的声音；没有把短音频补成10秒。较长窗需要积累对应历史后才能首次输出。

|组|精确波形输入（16kHz FOA）|STFT特征时间帧|编码器内部预测帧|当前step输出|物理GPU|
|---|---|---:|---:|---|---:|
|w025|4×4000，即0.25秒|40|3，最后单元50ms|1×189|0|
|w100|4×16000，即1秒|160|10|1×189|1|
|w150|4×24000，即1.5秒|240|15|1×189|2|

保留原16k前端：FFT512、Hann342、hop100、128Mel、fmax8kHz，及原模型内部时间均值／标准差归一化。0.25秒会在卷积池化前产生5个时间单元；原池化的主支与残差分别向下／向上取整，形状不符，因此仅将此处平均池化改为ceil并排除补位计数。对原偶数时间维度的池化已验证逐位不变。这不增加真实音频或可学习参数。

按与最后0.25秒的实际重叠时长，对内部ACCDOA向量作加权汇总，再用原3轨／21类解码。0.25秒组的权重为0.4、0.4、0.2；1秒及1.5秒组最后三个内部单元的权重为0.2、0.4、0.4。训练损失对这个**单step结果**计算，三组的目标、时间端点和yaw角相同。该汇总针对当前单声源数据；不能未经验证外推到多实例、移动源或复杂遮挡场景。

## 50毫秒标签

独立缓存`labels50`由已保存的实际干声到达区间与原FOA湿声功率重建，采用原固定可闻阈值。不是把100毫秒标签复制两遍，也不声称知道原始标注之外的语义事件边界。将50毫秒的占比和功率重新聚合后，5400条录音的原100毫秒活动及忽略掩码全部恢复一致。

当前step覆盖5个50毫秒单元。先聚合真实到达占比和功率，再采用原规则：至少80%活动占比且可闻才是活动；部分到达、尾声或不可闻冲突忽略。由此处理0.25秒边界，不需要把0.25秒四舍五入成0.2或0.3秒。类别及局部方向来自原数据的监督信息，不送给在线Agent。

## 训练与可比性

- 继承Dense15h不可变划分：4320训练、540验证、540测试。chair等源文件家族在这些内部划分间共享；这是已多次分析的开发测试集，不是全新独立录音盲测。
- 三组均从与旧10秒基线相同的原始13类预训练权重初始化，换21类输出头；种子7。检查三个初始完整state和新头的SHA相同。
- 从第一步解冻全网络，Adam，编码器LR`1e-5`、新头`1e-4`，batch16，100 epoch、每轮270步，共27000步；epoch余弦调度，无早停、无AMP。
- 每文件每轮取一个随机端点，端点为1.5、1.75……10秒之一。三组按同一索引／epoch取相同端点及yaw；已检查不同epoch下末0.25秒波形与监督一致。
- 三组训练步数相同，但每次可见的上下文长度不同，因此累计输入秒数不同。不能称每组每轮都完整看过15小时音频。
- 三组统一当前step监督，而旧10秒基线使用整段序列损失／文件对齐窗口。因此新三组之间是受控比较；新旧表现差异不能全部归因于窗长。

验证也按导航方式，每0.25秒调用一次，只评分最后0.25秒。为公平，三组统一从1.5秒开始提供结果，每条录音35次；0.25秒模型原本可以更早开始，此处统一预热只用于比较。

主评分使用原DSF解码器和SELD scorer：21类、macro、20°、1秒分块，在4Hz step网格评分。以验证主SELD最小保存best，平手按验证loss；同时把这些step预测按时间重叠重采样到原100毫秒标签网格，保存`validation_100ms`／`test_100ms`辅助指标。4Hz主指标不能直接与旧完整10秒协议的分数等同。

测试模型评分只在100 epoch全部完成、验证集选好best后执行；标签格式核验可以提前检查各split，未用测试模型分数选权重或阈值。后续需要另做真实在线shadow验证，训练结束不会自动进入导航控制。

## 环境、执行和记录

云端实验根目录：
`/media/4tbssd/av_seld/workspaces/avn_short_windows_v1_20261108/`

- `labels50/`：独立50ms缓存、逐样本哈希和5400条回归核验。
- `label_source/`：实际生成缓存的源码。
- `code_snapshot/`、`source_manifest.json`：冻结训练／监控源码及镜像版本。
- `verification/contract_tests.json`：池化、共同目标、部分活动及DSF scorer自测。
- `runs/w025|w100|w150/smoke_batch16/`：实际梯度、参数变化及两套评分流程启动测试。
- `runs/w025|w100|w150/run_seed7_full100/`：config、history、best、last、最终测试预测与结果。
- `monitors/w025|w100|w150/`：独立SwanLab进程、运行URL、上传状态和脱敏日志。
- `attempt_01/`：首次启动的日志、源码和早期检查点；因LLVM fork工作进程异常与配置缺少类别列表而统一停止，未用于后续比较。修复后从相同原始权重重新启动；DataLoader改用spawn，避免继承已初始化的CUDA／LLVM状态。

复用现有PyTorch／Mamba Docker镜像；各训练容器只暴露自己的GPU，代码、原数据、原权重和标签缓存只读，仅当前run可写。训练容器关闭网络；SwanLab由独立宿主机进程读取history上传，沿用现有0600权限Key文件，Key不写入代码或参数。

实际阶段命令（在云端SAVN工作树中执行；已有输出不覆盖）：

```bash
python3 scripts/launch_short_windows.py prepare
python3 scripts/launch_short_windows.py test
python3 scripts/launch_short_windows.py smoke
# 三个smoke均通过且初始权重一致后：
python3 scripts/launch_short_windows.py train
# 三组config产生后启动独立采集器：
python3 scripts/launch_short_windows.py monitor
python3 scripts/launch_short_windows.py status
```

完整Docker参数在云端`label_launch.json`、`contract_test_launch.json`、`smoke_launch.json`和`train_launch.json`。SwanLab记录train loss、验证ER/F20/LE/LR/SELD、100ms辅助验证、学习率、耗时和硬件；主训练不依赖监控进程存活。

SwanLab实际运行链接：[0.25秒](https://swanlab.cn/@rdpi1995/AVN-Localization/runs/avn-short-w025-v1-95e35a3f5a88)、[1秒](https://swanlab.cn/@rdpi1995/AVN-Localization/runs/avn-short-w100-v1-50c20d335ad6)、[1.5秒](https://swanlab.cn/@rdpi1995/AVN-Localization/runs/avn-short-w150-v1-bef15e2f0352)。三组各18条指标序列已由SwanLab API读回并与本地history核对。

启动与监控的历史验证见`reports/short_window_training_20261108/startup_validation.json`；完整100轮最终结果见`reports/short_window_full100_20261108/`。离线指标与单次回放不等于完整在线导航成功。
