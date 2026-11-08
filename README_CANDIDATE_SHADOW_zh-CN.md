# AVN21 定位候选接入与 Codex 在线 shadow

本功能把 `avn_foa_dense15h_v02_epoch097` 接入现有 `LocalizationTool`，通过显式的 `profile='avn_foa_candidate'` 启用。默认仍为旧13类RAW模型；原模型、官方Episode、场景资产和既有Agent/STOP策略均保留。

转换实现现已收敛到`spatial_audio/foa.py`（`savnce-rlr-foa-1.0`）。生成器、公共FOA观测和新worker共用这一源码；新的生成配置记录其版本和SHA。已有冻结训练数据和源代码快照保持原样，通过固定28例及试跑8窗等价对照核对历史输入定义。

## 输入与坐标

- 原生16kHz、10秒、4声道AmbiX ACN/SN3D，顺序W/Y/Z/X；40个已经采集的0.25秒观测窗。未满40窗时返回`not_ready`，不补零、不重复observe、不重采样到24kHz。
- 在本次已实测的Habitat-Sim0.2.2/RLR构建中，原始顺序为W/+Habitat Y/+Habitat Z/+Habitat X；现有`foa.to_ambix`转换一次，三个一阶声道除以sqrt(3)，并把固定AmbiX前/左/上对齐到Habitat -Z/-X/+Y。
- 仿真导出的FOA是固定世界基底。新模型训练使用听者局部基底，因此每个10秒历史窗整体旋转到窗口末端**已观测**的监听器朝向。W和一阶能量保持，SN3D不再缩放。
- 输出XYZ/方位角属于这个统一的监听器anchor基底；`orientation_pose_ref`指向窗口末端观测，`pose_ref`指向对应历史音频帧的接收器位置。`direction_unit_habitat_xyz`用于世界方向，禁止误用历史朝向对anchor方向再旋转一次。
- 同一窗口中的移动、跨位置混响尾音及方向变化仍属未校准输入域。本阶段核对转换链，不据此宣称在线方向准确性已验证。

原生16k前端与冻结训练入口完全一致：FFT512、Hann窗342、hop100、128 Mel、fmax8k，得到1×7×1600×128特征；21类三轨输出为100×189。保留波形增益，无外部HDF5 scaler；网络内部沿时间轴做mean/std标准化，std下限1e-8。活动阈值0.5、原DSF三轨15度合并规则保留，不返回距离或已校准置信度。

## Codex 与 shadow 的分离

导航端复用原RGB-D/双耳特征、原提示词、原evidence STOP门控，不做专门扫转。模型明确指定`gpt-6.1-sol`，以Codex CLI调用现有网关和API Key。

协调器启动两个独立的原生`agent_episode.serve`环境：主环境提供双耳导航观测；FOA环境执行完全相同的动作。逐步核验ID、时间、位置和四元数同步。候选只读FOA环境的公共观测及机器人位姿，无法读取场景目标真值。

每次先让Codex选定并验证动作，再执行shadow分析；动作在shadow前后校验哈希一致。`choose()`不注册候选function tool，不传递候选输出，模型可引用的证据也不包含`:localization`。满10秒后每1秒分析一次，结束时再记录一个窗口；重叠窗不构成独立证据。

Codex在单独的Ubuntu容器中运行：只挂载native CLI、当前两张观测图片/JSON schema、CA证书和只读私有认证文件；不挂载SAVN-CE代码、场景、数据集、Docker socket、shadow结果或评估文件。只读root、非root用户、cap-drop ALL，关闭shell、子Agent、插件、浏览器等工具。每步仅通过stdin和当前图片取得允许的观测。

API Key从用户已有的`~/.codex/auth.json`复制到宿主机权限700的私有目录，文件权限600；不会放进镜像层、仓库、运行参数、公开日志或视频。Codex实际使用当前第三方网关`code.ppchat.vip`，本阶段不更改用户默认provider/model。

本次网关模型清单为gpt-5.6、gpt-5.6-sol、gpt-5.6-terra、gpt-6-astra、gpt-6-sol和gpt-6.1-sol。原默认gpt-5.4请求返回503“没有可用通道”；用户明确选择gpt-6.1-sol，实际API Key请求成功。清单声明不等于逐一验证其他模型可调用。

## 云端路径与运行

源工作树：`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree`。
模型worker：`/media/4tbssd/av_seld/workspaces/avn_candidate_shadow_runtime_v1`，运行于原`avseld_final_test`环境/GPU2。
仿真：原`savnce_dev`；独立Codex容器：`savnce_codex_actor_v1`。

数据链接使用相对路径`data -> ../../data`，兼容宿主机和`/workspace/SAVN-CE`容器挂载。

```bash
cd /media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree
# 首次创建独立actor；读取当前私有认证，仅创建本实验的独立配置。
python3 scripts/provision_codex_actor.py

python3 -u scripts/candidate_codex_shadow.py \
  --output outputs/avn_candidate_shadow_1000000008_codex_seed7_64_20261108 \
  --sim-workdir /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree \
  --inputs /media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/outputs/codex_actor_inputs_20261108 \
  --episode-id 1000000008 --steps 64 --seed 7
```

本次RGB-D分辨率为512×512。未做另一条不含shadow的实时LLM反事实复跑，不宣称两次随机模型调用的动作逐位一致；这里验证的是候选不进入策略输入、动作先确定、主/影子环境逐步同步。

已有结果不会被覆盖。seed7控制仿真；远端模型采样不保证逐次动作相同。若Agent在10秒前停止，shadow仍如实记为未就绪，不得为了凑窗否决其合法STOP。当前evidence协调器本身的STOP规则保持原实现。

## 核验命令

```bash
docker exec -w /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree savnce_dev \
  python scripts/calibrate_foa.py --output outputs/candidate_calibration_20261108/foa_physical.json

docker exec -w /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree savnce_dev \
  python -m unittest discover -s tests -p 'test_*localization.py'

docker exec -e CUDA_VISIBLE_DEVICES=2 -w /workspaces/avn_candidate_shadow_runtime_v1 avseld_final_test \
  python check_candidate_adapter.py --output /workspaces/avn_candidate_shadow_runtime_v1/adapter_parity.json

# 完成后才读取privileged truth，绝不把该结果送回Agent。
docker exec -w /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree savnce_dev \
  python scripts/check_candidate_shadow.py --run outputs/avn_candidate_shadow_1000000008_codex_seed7_64_20261108
```

24个实际直接声RIR校准用例的系数最大误差约7.25e-8；11项新旧边界/兼容测试通过。使用同一16k WAV，适配器与冻结入口原始输出差为0；捕获audio_proj前输入，模型内时间归一化与公式差为0。错误采样率、声道和短窗被拒绝。

## 输出文件

- `navigation/avn_demo.mp4`：主环境512×512 RGB加原生双耳音频，人眼/人耳检查的主视频。
- `navigation/binaural.wav`、`navigation/actions.jsonl`、`navigation/observations.jsonl`：原始音频、实际动作与观测索引；`observations/*.npz`保留原始RGB-D/音频/机器人位姿。
- `foa_shadow/foa_ambix.wav`：同步FOA环境的四通道浮点AmbiX；`audio_format.json`描述格式。其`rgb.mp4`仅为无声预览。
- `model_decisions.jsonl`：Codex动作提议、证据、STOP否决/接受与执行结果；`codex_calls.jsonl`：实际CLI调用状态、耗时、token计数，不含密钥。
- `predictions.jsonl`：窗口、观测引用、原始100×189输出、逐帧方向及坐标转换；`shadow.jsonl`是紧凑摘要，明确记录动作在定位前已经确定。
- `model.json`、`run_config.json`、`rollout_complete.json`：权重哈希、前端/格式、运行配置和完成状态。
- `shadow_evaluation.json`：完成后的独立真值评估与输入隔离/同步校验。条件角误差只在有预测且数值有声的帧统计，重叠窗会重复计数，不能冒充SELD、F20、独立样本或导航成功率。

最终结果与本地下载记录在该README的运行结果段及独立交付目录中；本功能没有更改默认Agent模型或让候选参与STOP。

## 已执行结果与验收状态（2026-11-08）

单episode工程试跑已完成64步、65个观测；72次gpt-6.1-sol实际调用成功。39次not_ready后获得8个完整10秒结果。初次试跑在共享模块重构前执行，准确源码位于下载目录source_snapshot；全部8个真实输入窗在重构后逐位一致，未伪称另一次自主运行。

候选在线效果**不通过**：chair输出0，只有数值静音窗上的2个cushion和1个counter输出；有声帧目标最大ACCDOA范数0.4318，低于固定0.5。数值有声约2–9秒，首个10秒结果在10秒，故该episode也不是实时辅助导航的合适正例。模型保留为候选，未启用导航/STOP控制，未调阈值或训练新权重。

导航由Codex和原evidence门控完成：8次STOP提议被原门控否决，64步未执行STOP。离线最小geodesic距离0.365m，最终6.038m；这些真值未发送给策略，结果不能归因于未参与决策的候选。

共享模块：冻结28例回归及新渲染28例通过；16k冻结前端/模型内归一化对照误差0；真实8窗转换差0。混合12步L L F R F R L F R F L F实际平移1.2443m，既有卷积和局部转换误差约2.98e-8。它不证明真实运动历史混响的物理准确性。

sourceIsVisible：本机源码封装的射线反向，6例查询错3例，但6例direct渲染均与独立几何吻合。支持本次封装/受测几何属于外部查询问题；不声称穷尽预编译SDK内部。1080条独立RIR的保存配置temporalCoherence=0且全部history_reset=true；A-B-A reset直接声差0。在线原生配置仍为True，可能带来快速运动平滑模糊，两者不可混同。

详细步骤、控制变量、验收范围和未解决问题见[实验设计](EXPERIMENT_DESIGN_CANDIDATE_SHADOW_zh-CN.md)；结构化证据见[报告目录](reports/candidate_shadow_20261108/)。历史12步测试确有12次turn_left、底盘固定；新增测试另行记录平移和姿态，不混写历史日志。

共享生成环境用`bash docker_avn_foa/build_shared.sh`构建`rdpi/avn-foa-sameroom:dense15h-sharedfoa-v1`，只打包和检查模块，不启动造数/训练。历史`run_dense15h.sh`保留原实验固定路径，不应拿来启动本轮评估。

Codex认证参考：[官方认证说明](https://learn.chatgpt.com/docs/auth)。本轮结论依据实际CLI请求和动作日志；第三方网关清单不能当成全部模型均已实测可用。
