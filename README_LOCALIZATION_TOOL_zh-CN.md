# DSF-Net 音频定位工具

这条接入路径运行在云端 `research/dsf-localization` 独立工作树。规则控制器 R0 和默认双耳 LLM 路径可按原方式使用。普通和证据 LLM 都可以选同一个 FOA 工具配置。当前已验证声学模型、原域推理和真实仿真影子回放。真实 Responses 闭环仍缺应用凭证。

## 模型审计

公开 [avseld-mamba](https://github.com/Devin-Pi/avseld-mamba) 的实际 Git HEAD 为 `88cf2d55b5846728b18adf28f240e1d487d8a98f`。云端实验树 `/media/4tbssd/av_seld/workspaces/av-mamba` 的 HEAD 为 `2f61ab5`，有已有未提交改动。公开的 SELD 0.354 权重是音视频模型，SHA-256 为 `53d22e8049e306a0ddd928ee04f26c3e7fd82c8a84bcfda350052d1ac0348ba6`。

AV 检查点元数据的 `audio_weights_path` 指向 `pre_trained_weights/10s_epoch11_val_F0.8266.ckpt`。文件仍在，SHA-256 为 `1b602ed0be69f39939ada0a6b3f61594aea98d89ea618097fcff7751c3f8e855`。其状态字典含 7 通道输入层、两组时序模块和 `AVfnn_list.2.weight (117,128)`。2025-05-13 开始的训练日志记录外部归一化为 True，历史位置为 `/workspaces/audio_backbone/features/h5py_resnet/feature_scaler.h5`。检查到的 HDF5 副本属于 AV 评估或 3 秒归档，缺乏与这个 epoch 11 运行匹配的证据。另存的 `audio_mamba_10s.ckpt` 同样有 117 维头，但时序键使用单组 `mamba_blocks`，是不同的权重。

当前工具选用该项目的 **RAW audio-only epoch 17**。原档案 `/media/8tbhdd/rdpi/workspaces/audio_bb_mamba_15_MB_RAW/checkpoints/epoch17_val_F0.8022_val_seld_scr0.1549.ckpt` 与云端保留副本 `/media/4tbssd/av_seld/workspaces/av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt` 的 SHA-256 相同，均为 `85e380a6a8d2f1e545aa093e3f086f0a4d7830a15c41f7f79d047d7d9c1f9f90`。副本名称含 `3s`，但检查点记录 `num_video_frames=100`，RAW 配置及原训练数据入口均为 10 秒。

调用链是原 `dataloaders/dataloader.py::load_data_from_scratch` 使用 `utils.tools.load_audio_file`、验证 ACS 3 恒等变换和 `utils.extract_IV.LogmelIntensity_Extractor`。输入为 24 kHz、10 秒的 FOA。STFT 512 点 Hann、居中反射 padding、hop 150；128 Mel，0–12 kHz；4 路 log-Mel 和 3 路 W 与方向通道的实部互谱，经向量归一化和 Mel 投影。原提取器去除末帧，输出 `[1,7,1600,128]`。训练和验证使用 raw loader。模型开头对每一特征通道及 Mel 频带在时间轴做均值/标准差归一化，标准差下限 `1e-8`。这条入口不需要 HDF5 scaler。归档的另一个 `stage=test` 分支使用 HDF5，不能拿它的配置字段覆盖实际训练/验证路径。

`main_mamba_15.py::MyModel` 调用 `models.audio_encoder.AudioBackBone`。音频投影与 ResNet18 残差块产生每标签步 512 维特征，接两组各 8 层双向 Mamba，LayerNorm 和三层 MLP，最终输出 `[1,100,117]`。117 = 3 个轨道 × 3 个坐标 × 13 类。严格加载仅剥除 Lightning 的 `model.` 前缀，missing 和 unexpected 均为空。构造时跳过 ImageNet 下载，ResNet 留用部分的参数全部由检查点恢复。归档源码没有 Git 元数据，文件哈希见实际运行 `model.json`，包括提取器、模型、Mamba、配置和后处理。归档训练 YAML 的 SHA-256 为 `d1193bc597e44a6f6d73e0d286fd5bdae919226afb751e175f58b5676d6f2441`，训练入口 `main_mamba_15.py` 为 `6284d540c388b1f6ec2b06b82edbbbd28a611e87511dc005cdbf0a66ee2fea25`。

解码沿用活动模长严格大于 0.5、同类轨道方向夹角严格小于 15° 的合并逻辑。原验证代码在恰有一个重复对且第三轨不活跃时省略该对，当前保留这条行为。原始 100×117 张量完整保存，方便后续单独比较修正。输出方向为归一化 XYZ，方位角为 atan2(Y,X)，仰角向上为正。轨道只是该帧的输出槽。距离与不确定度均为 null。

## FOA 观测与时间

使用原生 `audio_mode=foa` 的 ACN/SN3D W/Y/Z/X 数据。世界 +X（前）、+Y（左）、+Z（上）的 AmbiX 轴分别对应 Habitat 世界 −Z、−X、+Y。可用 `scripts/calibrate_foa.py` 重现上一阶段的合成场景已知源方向和监听器旋转校准。当前实际接收器原点从音频传感器 node 的 world translation 读取，与 Habitat 在声学渲染时使用的原点一致。本场景接收器相对底盘向上约 1.5 m，RGB 相机和音频接收器各自有外参。世界方向转换到 Habitat XYZ 为 `[-Y,Z,-X]`，逐帧保留各自 pose_ref。

仿真每步 0.25 秒，实际 RIR 渲染和音频窗为 16 kHz，模型输入为 24 kHz。连续历史波形拼接后进行 3/2 多相重采样，不增加原始 8 kHz 以上信息。本次 chair 源文件是 44.1 kHz 单声道，原仿真通过 librosa 读入至渲染采样率。保留仿真输出的波形增益，然后应用原模型特征和归一化。现有场景渲染卷积与尾部缓冲按通道数分配。运动窗口存在跨位姿的卷积尾音和模型上下文，当前仅保存轨迹和时间关系。

首个结果需要 40 个已获得的 0.25 秒窗。结果在窗口结束时才可用于决策，10 Hz 帧时间是模型标注时间。之前的帧预测均依赖这个完整历史窗口。重复 observe 不追加样本，重复分析可命中缓存；重叠支持区间不能作为独立证据。STOP 不推进时钟。当前动作协议没有原地等待动作。

类别表见 [DCASE STARSS23](https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#sound-event-classes)。`offline_category_coverage.json` 列出 v1 的 20 条官方对象—声音关联，所有未取得独立声学标注的家具素材均标为未核验。预测的 `door` 等类别无法确认视觉实例。零激活表示未检测到支持类别活动。

## 运行与回滚

在工作树目录执行，当前专用模型进程使用 `avseld_final_test` 内的已有环境和 GPU 2。仿真依然使用 `savnce_dev`。前者的 Python 为 3.10.13、Torch 2.1.0+cu118、mamba-ssm 1.1.1、causal-conv1d 1.2.0.post2；后者为 Python 3.9.25、Torch 2.8.0+cu128。工作树的 `data` 是指向原项目数据的只读使用约定。服务由宿主机通过 stdin/stdout JSONL 启动，每个进程加载一次。无需安装新依赖或改仿真镜像。

若需要重建模型运行目录，先在宿主机新建 `/media/4tbssd/av_seld/workspaces/dsf_localization_runtime_v1`，从 RAW 归档复制 `models/`、`utils/`、`core/config.py`、`config/tools_cfg.py`，从工作树复制 `agent_episode/dsf_worker.py` 和 `agent_episode/dsf_decode.py`。只使用该目录内的 Python 源文件，历史 Lightning YAML 仅用于审计，运行时无需 Comet 凭证。

```bash
docker exec -i -e CUDA_VISIBLE_DEVICES=2 -e PYTHONDONTWRITEBYTECODE=1 \
  -w /workspaces/dsf_localization_runtime_v1 avseld_final_test \
  python dsf_worker.py --source /workspaces/dsf_localization_runtime_v1 \
  --checkpoint /workspaces/av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt
```

发送 `{"command":"describe"}` 检查 worker。可信协调器使用 `LocalizationTool.describe()`、`reset_episode(ref)`、`add_observation(obs)` 和 `analyze(["obs_0039"])`，只解析当前回合已获得的观测。工具输出见 [JSON schema](docs/localization_tool.schema.json) 和实测 `predictions.jsonl`，LLM 仅收到 `shadow.jsonl` 同形的紧凑摘要。工具不执行动作。与相同音频的 RMS 或其他特征共享支持区间。重分析只更新解释。

影子复跑使用过去规则策略记录的动作，FOA 分析不会影响轨迹。

```bash
python3 scripts/dsf_shadow.py \
  --actions ../generated_episode_G1_1000000008_64steps/actions.jsonl \
  --output outputs/dsf_shadow_1000000008_repeat --episode-id 1000000008 \
  --episode-file data/datasets/savnce_generated/mp3d/v1/train/content/17DRP5sb8fy.json.gz \
  --sim-workdir /workspace/SAVN-CE/outputs/localization_tool_worktree
```

已有 Responses 协调器可用 `--acoustic-backend dsf_audio_only` 注册真正的 function tool。网关返回 function call 后，协调器解析 ID、运行工具、写完整日志并回送紧凑结果，随后验证合法动作。当前旧被动 STOP 门控需要双耳方向/范围，FOA 工具只返回方向，证据模式会否决未经核验的 STOP。原有位置估计器继续供双耳路径使用。FOA 的逐帧 ray 和接收器位置已落盘，现有预算内没有足够非重叠窗口验收新的融合和 STOP 证据。

```bash
python3 agent_episode/llm_rollout.py --check-api
python3 agent_episode/llm_rollout.py --policy ordinary --episode-id 1000000008 \
  --output outputs/dsf_llm_1000000008 --acoustic-backend dsf_audio_only \
  --sim-workdir /workspace/SAVN-CE/outputs/localization_tool_worktree --seed 7 --steps 64
```

以上继续使用原客户端默认模型，也可按实验登记指定 `--model`。真实调用需在宿主机安全提供 `SAVNCE_API_KEY` 和 `SAVNCE_API_BASE_URL`。ChatGPT 登录的 Codex CLI 是另一条认证链路。401/403 会记录并退出。回滚只需省略 `--acoustic-backend` 或设置 `binaural`，主工作树中的现有服务可继续按原方式启动。

## 测试与边界

- 云端 6 项单元测试通过，见 `outputs/unit_tests_v2.log`。另有 `outputs/dsf_local_protocol_smoke_v1`，本地脚本化网关发起函数调用，真实 DSF 返回第 40 个观测窗的推理结果，校验后执行了 40 个实际仿真动作。该测试使用零次外部 LLM 调用。
- 最终影子回放见 `outputs/dsf_shadow_1000000008_seed7_v4`，47 条结果通过 JSON schema 校验。首个推理在音频 10 秒时可用，主要可测声能在 2 到 9 秒。活动窗口里的条件平均角误差为 11.27°（73 个预测，67 个不超过 20°，1 个超过 90°）。全部类别预测为 door，素材真实声事件类别未核验。数值静音窗中有 5 帧输出。两次同动作回放的浮点音频及模型输出有差异，v3/v4 均已保留。
- 原域保留验证录音 `fold2_room4_mix006.wav` 有 6 个标注事件，已跑满 60 秒的 6 个窗，逐窗比较原入口和适配输出，保存 WAV/标签哈希、张量、方向、标签和资源结果。另有一条保留的训练域录音作诊断。文件名一致的标签来自 SpatialScaper 备份。此处统计属于小样本测试。
- 仿真影子回放逐步核对旧轨迹位姿，归档包含窗口状态、实际预测、动作、类别覆盖、声学误差与独立可信导航评价。角误差使用渲染器实际声源点与音频接收器原点，真值仅用于离线报告。
- 当前 Responses 环境变量缺失，`outputs/gateway_health_v2.log` 记录本地检查失败，未发送真实请求。真实模型自主调用、共享工具的 v0/v1 对照和导航成功率仍待运行。
- 后续首个最小实验是在两组相同的 FOA 配置下选一个声音标签可核验、在 10 秒窗口后仍有声的开发 episode，取得一条实际工具调用和动作回执。独立验证方向跨域误差和融合阈值后再接入 FOA STOP 门控。


### 原域复跑和协议测试命令

```bash
docker exec -w /workspace/SAVN-CE/outputs/localization_tool_worktree \
  savnce_dev python tests/test_dsf_localization.py -v

docker exec -e CUDA_VISIBLE_DEVICES=2 -e PYTHONDONTWRITEBYTECODE=1 \
  -w /workspaces/dsf_localization_runtime_v1 avseld_final_test \
  python dsf_original_smoke.py --source /workspaces/dsf_localization_runtime_v1 \
  --checkpoint /workspaces/av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt \
  --wav /workspaces/av-mamba/fold2_room4_mix006.wav \
  --labels diagnostic_inputs/fold2_room4_mix006.csv --output original_smoke_repeat

python3 tests/dsf_local_protocol_smoke.py \
  --actions ../generated_episode_G1_1000000008_64steps/actions.jsonl \
  --output outputs/dsf_local_protocol_repeat \
  --sim-workdir /workspace/SAVN-CE/outputs/localization_tool_worktree
```

复跑前从工作树把 `scripts/dsf_original_smoke.py` 复制到模型运行目录。
原域标签副本来自 `/media/8tbhdd/rdpi/workspaces/SpatialScaper_/output/metadata_dev/dev-test/`。
同名保留录音见 `/workspaces/av-mamba/`。旧结果和中断运行保留在各自目录中。
