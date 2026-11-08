# 1秒权重：原生在线shadow测试结果

2026-11-08，episode `1000000008`，种子7，64步预算。本轮重新启动原生环境，由Codex实时选择动作；定位器在独立FOA环境中旁路记录，**未参与Agent决策、提示词、证据或STOP门控**。没有重训模型，也没有修改原始Episode、资产、旧权重或已有结果。

## 本轮结果

| 检查项 | 实际结果 |
| --- | --- |
| 原生动作／观测 | 64次移动，65次观测 |
| 定位状态 | 3次预热`not_ready`，62次`ok`；无重复计分、无缓存命中 |
| 原排程发声区间 | 2–8秒，共24个0.25秒时段 |
| 正确类别`chair`检出 | 15/24，62.5% |
| 正确类别且方向误差≤20° | 15/24 |
| 已检出`chair`的条件平均方向误差 | 6.999°，15次均≤20°，无前后翻转（>90°） |
| 其他类别输出 | `fireplace` 3次、`cabinet` 1次，均在原排程发声期间；不是额外真实声源 |
| 数值近静音时段误报 | 0/34；判定条件为当前0.25秒原始FOA RMS≤1e-7 |
| 首次完整定位输出 | 音频时间1.0秒，对应当前区间[0.75,1.0) |
| 首次`chair`输出 | 音频时间3.0秒，相对声源开始延迟1.0秒 |
| worker总处理时间 | 中位44.04ms，P95 53.08ms，最大1071.32ms |
| Codex调用 | `gpt-6.1-sol`，79次成功、0次失败；现有STOP门控拒绝15次提议 |
| 动作／位姿同步与Agent隔离检查 | 通过 |

`chair`漏检9个发声时段，另有4个错误类别事件（其中一次与`chair`同时输出），类别稳定性仍需改进。所有类别共19个输出方向的条件平均误差为7.342°；这不能替代类别正确性。

这些是单个Episode、实际轨迹上的事后诊断，**不是标准SELD/F20，也不是独立样本总体准确率**。方向误差只在有检出时计算；近静音不等同自然背景噪声。1秒输入窗口相互重叠，且本次场景／声音可能与训练素材有重合，不能据此声称跨场景或跨音源泛化。与先前录音回放的动作／位姿不同，不作受控前后增益结论。

原Agent本轮未STOP，以64步预算结束；最近测地距离0.315m、最终1.366m，第一次进入1m为第12步。这是原策略行为，**不是定位器导航成功或失败的评测**。距离只由环境关闭后的评估侧读取，没有提供给Agent或定位器。

## 输入、坐标与时序

- 权重：1秒组epoch83，原文件`best.pt`；SHA256为`e8c1539fb72aca7cdf40c40c9d90368dd19bffef93ca7824f00dc1bf52e38a6d`。
- 输入：16kHz、四通道ACN/SN3D W/Y/Z/X，连续4个已观测的0.25秒块，精确16000×4。前三块不补零、不重复、不提前使用未来音频。
- 采用现有共用转换模块`spatial_audio/foa.py`，版本`savnce-rlr-foa-1.0`。本机校准构建的原始世界FOA先转换为world AmbiX；worker再依当前已观测接收器姿态，把完整1秒历史转到当前监听器坐标。转换不读取声源位置，不重复旋转。
- `foa_shadow/foa_ambix.wav`保存的是**世界坐标**AmbiX，不能直接当作模型的监听器局部输入。其轴约定和版本在`audio_format.json`；局部输入由worker在线完成。
- 特征形状`1×7×160×128`，编码器原始输出`1×10×189`，沿用训练的时长加权方法汇总最新0.25秒为`1×1×189`，阈值0.5。没有增益、归一化或阈值临时调参。
- 每0.25秒更新一次当前结果，需先积累1秒上下文；这与每1秒才更新不同。方向与当前step姿态关联，不能把早期姿态应用到当前输出。
- 本轮声音排程仍为2–8秒；数值非零的0.25秒窗口起点为2.0至8.75秒，包含停止发声后的尾声。没有延长发声来替换原任务。

worker耗时统计覆盖62次真实调用，测量预处理＋同步模型推理＋后处理；P95取排序后`ceil(0.95*N)`项。首次`obs_0003`冷启动约1071ms（预处理239ms、推理832ms），其余61次均小于250ms。统计不含进程RPC、渲染、Codex调用。仿真0.25秒采样周期不等于墙钟实时闭环。

## 核验与实现

- 新增可选profile `avn_foa_short1s`和专用worker；原legacy及10秒candidate行为保留。
- 原有及新协议单元测试共14项通过，覆盖预热、连续窗口、缓存、重置、未来ID拒绝、当前姿态、采样率／长度错误。
- 实际保存FOA及姿态上，worker与冻结训练图的输出最大差0；与保留回放对照最大差约`1.32e-4`，不宣称跨路径逐位一致。
- 当前Habitat-Sim二进制与之前28方向校准构建的SHA相同。结论仅适用于本次本机构建，不推广到其他SoundSpaces版本。
- 两个独立原生环境逐步校验时间、观测ID、位置、姿态和执行动作；Agent动作在shadow前锁定并核对哈希。
- 独立非root只读Actor仅挂载Codex程序、认证／配置、证书和观测输入；不挂载仿真资产、评估真值、定位输出或Docker socket。定位工具未注册给Agent。
- 79次真实Codex API Key请求成功，模型参数为`gpt-6.1-sol`。不保存认证值，不更改宿主机默认provider。
- `checks/source_snapshot/`与`launch.json`保存启动时实际源码及SHA。结束后仅更新计分脚本以去重和导出CSV，另存`checks/postrun_source/`及SHA，未改在线策略／模型。
- 视频全量解码通过：512×512、4fps、65帧、16.25秒，H.264＋AAC；独立双耳及四通道WAV均为16kHz、FLOAT、260000采样点。下载后MP4 SHA一致。

## 运行命令

云端工作树：`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/`。

实际启动命令（仿真在`savnce_dev`，模型在`avseld_final_test`，Actor在独立容器）：

```bash
python3 -u scripts/candidate_codex_shadow.py \
  --profile avn_foa_short1s \
  --output outputs/avn_short1s_shadow_1000000008_codex_seed7_64_20261108 \
  --sim-workdir /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree \
  --inputs /media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/outputs/codex_actor_inputs_20261108 \
  --episode-id 1000000008 --steps 64 --seed 7
```

运行结束后的独立核查使用`checks/postrun_source/check_candidate_shadow.py`，参数为：

```bash
--run outputs/avn_short1s_shadow_1000000008_codex_seed7_64_20261108 \
--actor-audit outputs/short1s_shadow_checks_20261108/actor_environment.json
```

在`savnce_dev`内使用其Python执行，工作目录是上述工作树的容器映射路径。当前仓库的`scripts/check_candidate_shadow.py`为该事后核查版本。随机种子约束仿真与音源偏移，不保证远端LLM输出逐次相同。

云端原权重：`/media/4tbssd/av_seld/workspaces/avn_short_windows_v1_20261108/runs/w100/run_seed7_full100/best.pt`。

云端模型runtime：`/media/4tbssd/av_seld/workspaces/avn_short1s_shadow_runtime_v1/`。原训练源码快照与模型均保留。

## 人工检查文件

本地新目录是原人工检查文件夹下的`avn_short1s_shadow_1000000008_codex_seed7_64_20261108/`，旧结果保留。

| 文件 | 含义 |
| --- | --- |
| `avn_demo.mp4` | 原Agent导航RGB及对应双耳音频，未叠加定位结果，不是FOA混音 |
| `navigation/binaural.wav` | 视频对应的原始浮点双耳录音 |
| `foa_shadow/foa_ambix.wav`、`audio_format.json` | 同步世界坐标四通道FOA及完整格式约定 |
| `predictions.jsonl` | 每次定位的状态、1秒支持ID、当前输出区间、类别／方向、原始ACCDOA及耗时；包括前三次预热 |
| `per_step_localization.csv` | 62个有效当前step的RMS、类别分数、输出、动作及事后角误差；`source_on_evaluation_only`和角误差只用于评估 |
| `actions.jsonl` | 64次实际执行动作、前后位姿和原策略决策 |
| `shadow.jsonl` | 旁路输出摘要及已锁定动作哈希 |
| `model_observations.jsonl`、`model_decisions.jsonl`、`codex_calls.jsonl` | Agent输入摘要、提议／STOP门控和真实模型调用记录 |
| 两个环境的`observations.jsonl` | 同步观测索引及位姿，供对齐音频／动作 |
| `shadow_evaluation.json` | 环境关闭后的同步、隔离、方向和检测诊断，以及独立导航结果 |
| `runtime_timing.json`、`video_verification.json` | 处理耗时和全视频／音频解码核查 |
| `model.json`、`run_config.json`、`rollout_complete.json` | 模型与协议、运行参数及完成标志 |
| `checks/` | 启动源码、核查脚本、构建记录、适配器一致性及单元测试结果，无认证密钥 |
| `SHA256SUMS` | 本地交付文件校验值 |

逐帧RGB-D／原始FOA NPZ仍在云端同名输出目录中保留，本地包提供视频、完整音频、观测索引及日志，没有重复下载全部NPZ；私有声源／目标真值文件仍在独立权限目录，未复制到Agent输入。

本阶段仅完成旁路验收。仍需处理发声时段漏检、错误类别和首次推理延迟；没有开放定位器驱动导航，也没有自动训练或进入下一阶段。
