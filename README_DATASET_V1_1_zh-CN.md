# SAVN-CE Agent Dataset v1.1

v1.1 是原20条开发任务的研究元数据层和离线审计，不替换官方 Episode 格式、不创建新物理场景。保留原始 `.json.gz`、索引、物体—声音关联以及 SAVNCE_Dataset / Habitat Env 原生加载。任务基于官方关联生成，不是作者发布的官方 benchmark。

## 已实现和实测

运行位置：`polyu_serve:/media/4tbssd/00_PROJECT_CODE/SAVN-CE`；容器 `savnce_dev`，镜像 `rdpi/savnce:render-v1`。Habitat-Sim 0.2.2、Python 3.9.25，Bullet编译支持存在，但当前任务 `enable_physics=False`，无独立刚体。

- 20/20 原生加载、序列化往返、RGB-D/双耳音频、参考动作可达性验证通过。参考控制器不是 Agent，不能据此报告 Agent 成功率。
- 120个特选合法离线观察位姿的语义像素交叉核验。没有预生成全部固定轨迹的RGB-D/音频序列。
- 20/20 独立合法转向探针覆盖发声前、期间及结束后至少1秒，确认真实双耳音频能量与原生发声门控。参考轨迹本身19条未覆盖结束后窗口，此限制仍原样记录。
- G0简单候选1条：`1000000016`；G1同类别视觉候选2条：`1000000000`、`1000000008`；17条未分组。均为仪器筛选、待人工复核，尚未确认感知冲突。
- 8条任务的目标与其他官方同类别实例共用同一声音文件和声源点；在匹配时间表、波形相位和听者状态下，当前点声源音频无法区分这些实例。相应任务排除用于精确发声实例验证。
- 种子20261105重新生成的20条任务内容压缩字节与原始文件一致。测试副本私有保存，没有新增正式开发任务。
- 外部接口 `reset / observe / step / stop / close` 实际会话测试通过；重复observe不推进时间，要求未来step的observe也只返回当前观测。
- 12个原始数据/官方来源/场景文件的审计前后SHA256一致。未修改或加载预训练权重。

已额外运行无项目/数据挂载的非root只读客户端容器探针；它只证明该启动模板看不到私有文件，不是完整的认证网关。

详见独立输出目录中的 `validation_v1_1.json` 和 `coverage_report.md`。该结果不是已完成的人类感知标注或算法比较。

## 文件位置

原始数据：`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/data/datasets/savnce_generated/mp3d/v1/`，保持不变。

研究者侧输出：`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/research_outputs/agent_dataset_v1_1/`：

| 文件 | 内容 |
|---|---|
| `episode_audit.jsonl` | 每条关联、起点/终点/动作/距离、音频窗口、离线视角、限制和适用性 |
| `scenario_metadata.jsonl` | episode_id关联的独立研究标签，全部必填字段和权限分类 |
| `coverage_report.md` | 汇总、逐条表格、G0/G1选择与证据边界 |
| `validation_v1_1.json` | 实际断言结果与未解决限制 |
| `environment.json` | 实际安装版本、资产路径、绑定能力、声学网格导出摘要 |
| `engine_source_evidence.json` | 本机C++参考源码行号及SHA256；二进制精确提交对应未确认 |
| `study_selection.json` | G0/G1/未分组ID；没有修改原生索引 |
| `input_hashes_before.json` / `input_hashes_after.json` | 原始输入完整性 |
| `private/` | 离线语义映射、特选RGB/实例图、完整时间探针标量、可复现性测试副本 |
| `TEST_COMMANDS.txt` 和 `*_run.log` | 实际命令与输出 |

`dataset_v1_1/` 包包含审计、时间窗探针、报告整合、验证与最小Scene Variant契约。未来能力分析见 [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md)，接口和优先级见 [SCENE_VARIANT_DESIGN.md](SCENE_VARIANT_DESIGN.md)。

## 标注含义

`target_visibility` 记录对齐语义渲染中的像素、视锥代理和遮挡状态。实际有语义像素不等于RGB可识别；零像素不等于已证明遮挡。没有像素但包围盒进入视锥时，遮挡原因记unknown。

`visible_same_class_candidates` 是同类别语义实例且至少20/16384像素的候选，不是RGB相似度标注，也不证明其正在发声。`visual_ambiguity.appearance_similarity` 和实际发声实例的观测归因保持unverified。原生声音由 `goal.position + [0,1.5,0]` 的点声源产生；不是声音物体动画，声源不保证位于目标包围盒内。

`verification_opportunity` 将合法平移后候选集合改变与真正发声实例验证分开。前者仅证明能够取得新证据。离线视角可以使用参考动作找合法位姿，但不能把这些位姿、真值或未来画面提供给Agent。

`scenario_group` 的G0/G1是待复核开发筛选。G0也可能目标很小；G1同类候选可能仅露出局部。当前没有任何任务标为已确认视听冲突基准。

## Agent边界与在线使用

可信协调器在容器中运行：

```bash
cd /media/4tbssd/00_PROJECT_CODE/SAVN-CE
docker exec -i -w /workspace/SAVN-CE savnce_dev \
  /opt/conda/envs/ss/bin/python -m agent_episode.serve --inline-observations
```

stdin每行一条JSON；例如由可信协调器选定任务和输出目录后reset：

```json
{"command":"reset","output":"outputs/my_agent_run_001","episode_id":"1000000008","episode_file":"data/datasets/savnce_generated/mp3d/v1/train/content/17DRP5sb8fy.json.gz","resolution":256,"max_steps":80,"seed":7}
{"command":"observe"}
{"command":"step","action":"turn_left","decision":{"evidence_ids":["obs_0000:audio"],"reason":"request another permitted viewpoint"}}
{"command":"stop"}
{"command":"close"}
```

动作由Agent在线选择。只有选择执行后产生的新观测可用。inline响应的 `sensor_payload` 为base64压缩NPZ，仅含 `rgb / depth_m / binaural_audio / position / rotation_xyzw`；时间戳、采样率、动作碰撞状态和自身位姿可用。不包含语义ID、目标位置、类别、声音文件、最优动作、评估距离、研究标签或未选择的未来观测。合法step为0.25m/转向15度，每步0.25s；原始双耳音频16kHz，每窗4000×2。

新运行的 `evaluation_truth.json`、`resolved_simulator.yaml` 以及评估文件写入 `research_outputs/agent_private/<run_token>/`，私有目录0700；公共输出保留实际已选择观测和动作证据。旧历史运行保持不变，可能含历史truth，必须视为研究者侧资料。

可信模拟服务必然访问官方任务真值。**不要给不可信Agent授予SSH/root、整个项目/数据盘挂载或服务stderr**。inline模式消除文件路径依赖，适合把Agent放到无项目挂载的独立客户端；本阶段没有实现抵御恶意请求或root访问的安全网关。reset的任务路径、输出路径、预算应由可信协调器固定，Agent只控制observe/step/stop。`evaluation.json` 只供研究者评估，不能作为决策输入。

## 可选 FOA 音频

在 JSONL 的 `reset` 请求里添加 `"audio_mode":"foa"`。省略该字段时继续使用双耳。
FOA 模式的每步 NPZ 使用 `foa_audio` 字段（16 kHz、每窗 4000 × 4），同一目录的
WAV 保存同样的浮点样本。发送 `close` 后，服务按观测顺序把各窗拼接成
`foa_ambix.wav`。STOP 不产生新的窗口。WAV 的声道语义见 `audio_format.json`，
导入播放器或分析工具时选择一阶 ACN/SN3D。现有基于双耳 ITD/ILD 的策略使用默认模式。

本容器的原生 RLR 输出采用固定 Habitat 世界轴，声道为
`[W, +Habitat Y, +Habitat Z, +Habitat X]`，一阶分量相对 W 的尺度为 N3D。
导出转换为 `[W, -c3/√3, c1/√3, -c2/√3]`（原生通道从 0 编号）。
得到的 ACN 顺序为 W/Y/Z/X，AmbiX 的前、左、上分别对应 Habitat 世界的 −Z、−X、+Y。
WAV 保持固定世界轴，消费端可结合当前观测的位姿转换到机器人局部坐标。

标定使用合成房间、仅直达声、2 米距离、6 个正负主轴和 2 个斜方向，
分别测试 0°、90°、180° 的监听器偏航。可用以下命令验证当前 Habitat-Sim 构建
（需要容器中的 `trimesh`）：

```bash
docker exec -w /workspace/SAVN-CE savnce_dev \
  python -m scripts.calibrate_foa --output /tmp/foa_calibration.json
```

## 实际验证与复跑

已在现有容器运行 `dataset_v1_1.audit`、`finalize`、`timing_probe`、`report`、`tests`。最后测试汇总见JSON。没有重跑会覆写原始v1验证文件的旧 `episode_tools.validate`。

完整新审计使用新的输出目录；以下在宿主机执行，`OUT` 为容器和宿主机共同挂载项目下的相对路径：

```bash
cd /media/4tbssd/00_PROJECT_CODE/SAVN-CE
OUT=research_outputs/agent_dataset_v1_1_repeat
for module in snapshot audit finalize timing_probe report tests; do
  docker exec -w /workspace/SAVN-CE savnce_dev \
    /opt/conda/envs/ss/bin/python -m dataset_v1_1.$module --output "$OUT" || break
done
```

snapshot/audit/timing_probe拒绝覆盖已有主结果。tests使用独立随机命名输出，可再次运行。生成器复现固定任务定义，不保证GPU光线追踪音频浮点值逐位相同。审计相机128×128/90°，高度1.25m，和该次原生Env审计对齐；更高分辨率Agent运行必须单独复核可见性标签。

## 当前边界

本阶段仅实现未改动静态场景的研究层。新增障碍物、家具重排、动态声源、组合物理变体均没有实现；传入这些变更时 `ExistingSceneBackend` 抛出NotImplementedError。仅改JSON声源点不被认定为新物理场景。原始扫描资产中的家具无法直接用独立刚体API移动。

优先后续工作是人工复核三条筛选任务、固定允许传感器协议和评估规则，再运行普通Agent/证据验证Agent对比。复杂场景需另行实施，本交付没有自动开始。
