# Scene Variant contract and implementation priorities

这是未来接口设计。v1.1唯一已实现的backend是 `ExistingSceneBackend`，仅接受 `base:17DRP5sb8fy`、geometry_epoch=0、所有变更列表为空。其余请求抛NotImplementedError；伪造新场景名也被拒绝。没有增删家具、障碍物、动态源或原始资产写入。

## 数据分层

1. 官方 `.json.gz`：继续保留scene_id、goals、oracle_actions、onset/offset等原生字段和索引，不加入研究标签。
2. 研究者侧 `scenario_metadata.jsonl`：按episode_id绑定group、scene_variant_id、可见性、歧义、观测质量、验证机会、方法和状态。它含离线真值衍生信息，默认privileged。
3. 未来 `scene_variants/<variant_id>/manifest.json`：资产副本、变更声明、版本、几何/声学缓存及一致性证据，只有可信协调器可读。base scene ID是原始场景的引用，不代表修改它。
4. Agent传感器流：仅执行到当前时间的RGB-D/原始双耳音频/自身位姿/实际动作反馈；不得附加任务组、目标实例、源轨迹、参考动作、未来审计视角或成功距离。
5. `research_outputs/agent_private/`：评估真值、每源轨迹、oracle、离线验证位置；Agent动作和基于允许证据形成的belief可以单独记入公共运行日志。

当前可执行的dataclass见 `dataset_v1_1/variants.py`。下面是**未来更完整的manifest设计草案，不是当前backend可执行配置**：

```json
{
  "schema_version": "scene_variant/0.1-proposal",
  "scene_variant_id": "proposal:17DRP5sb8fy:static-001",
  "base_scene_id": "17DRP5sb8fy",
  "dataset_version": "1.1",
  "seed": 20261105,
  "geometry_epoch": 1,
  "coordinate_frame": {"units": "metres", "up": "+Y", "forward_local": "-Z", "quaternion": "xyzw"},
  "base_asset_sha256": {},
  "asset_overrides": {"visual": null, "collision": null, "semantic": null, "navmesh": null, "acoustic": null},
  "static_objects": [],
  "furniture_edits": [],
  "acoustic_material_overrides": [],
  "source_trajectories": [],
  "validation": {
    "visual_geometry": "unverified",
    "collision_geometry": "unverified",
    "navmesh": "unverified",
    "semantic_instances": "unverified",
    "acoustic_geometry": "unverified",
    "temporal_audio": "unverified"
  },
  "access_class": "researcher_only_privileged_not_agent_input"
}
```

每个static_object需要稳定instance_uuid、资产内容哈希、视觉/碰撞网格、STATIC运动类型、世界变换、尺寸、独立语义ID和声学材质映射。furniture_edits需指定对资产副本中的哪些三角面/实例执行移除/替换；不得通过叠放新家具而保留旧家具的不可见碰撞或声学残影。

每个source_trajectory需要source_uuid、waveform哈希、发声时间表、波形相位策略、时间戳位置/姿态、插值模式、点声源/指向性模型、坐标高度约定、延迟模型和时变RIR算法版本。轨迹位置应为实际声学世界坐标，避免与原生goal+1.5m重复叠加。源运动与发声/静音分别定义；目标实例的运动标签只保存在评估器。

## 拟议生命周期（未实现）

`resolve_variant -> validate_assets -> apply_visual_collision_semantics -> rebuild_navmesh -> initialize_acoustics -> crosscheck_geometry -> reset_episode -> online_step -> finalize_private_evaluation`

官方加载器仍解析同一Episode schema。未来非base变体需要可信协调器/配置解析器把canonical scene引用解析为独立资产副本，明确处理原生reconfigure覆盖SCENE的行为。仅写sidecar不能改变原生渲染；该解析钩子当前没有实现。不得把未应用sidecar的任务标为新物理场景。

每次几何变动采用事务式geometry_epoch：视觉、碰撞、语义、NavMesh、声学导出与缓存全部记录相同epoch和资产哈希；任何一层unverified即禁止发布物理一致性结论。新布局失败时回退旧epoch，不覆盖原始数据。声学缓存键至少包含场景资产哈希、epoch、材料、源/听者变换、声学求解配置；不能只用episode_id。

在线step拟议顺序：确认当前时间/已选动作 -> 更新独立源时钟/位姿 -> 同步允许的物理变化 -> 更新听者/IR -> 按统一波形时钟生成当前音频并保留正确历史尾部 -> 获取同一时间约定的RGB-D和自身位姿 -> 返回当前观测。不得为了决策提前生成/暴露未来观测。内部采样或插值不能被误认为新一次Agent观测。

## 发布门禁与未来测试

| 门禁 | 必须的证据 |
|---|---|
| 视觉 | 指定位姿能观察新增/移除几何；深度与已知距离一致 |
| 碰撞 | 穿越测试/射线与碰撞网格一致；不存在残影或穿透 |
| 导航 | include_static_objects=True、相同agent参数重建，合法起点和参考可达性重新算；不可复用旧oracle冒充有效 |
| 语义 | 新实例像素ID与实例登记一致，移除物体无残留像素，不能只改标签表 |
| 声学 | 导出网格确实包含变体，阻挡/材质的对照IR实验产生可解释差异；固定随机设置并估计采样噪声 |
| 时序 | 静止源回归；分段移动/连续移动对照；无非物理跳变、源相位重置或旧位置尾部丢失；延迟/Doppler边界明确 |
| 隔离 | Agent协议只含允许数组和当前观测；无项目挂载、私有stderr或真值读取能力 |
| 可复现 | manifest、种子、版本、资产/配置哈希、命令齐全；区分定义重现与随机声学浮点输出 |

## 实施优先级

- P0（本阶段已做）：原20条审计、sidecar、权限与协议数据流分离、完整声音时窗验证、base-only拒绝未实现变体。还需人工复核候选感知歧义和真实Agent对比。
- P1（未做）：仅在原场景筛选更好视觉候选，必要时改变合法起点/朝向/时间表；保留官方关联，不引入物理变化。
- P2（未做）：一个独立STATIC障碍物的离线固定布局试验；完成六层一致性门禁后再扩展新家具。优先可预烘焙/初始化前固定的方案。
- P3（未做）：移动点声源调度与时变音频；先明确物理近似和精度指标，再决定是否修改原生桥接/求解器。
- P4（未做）：资产副本家具重排和多因素组合，运行中几何同步需要引擎桥接工作。

本交付到接口与能力分析为止，不自动实施P1–P4。当前8条同类同声同点实例别名必须显式排除精确实例识别研究，不能用看似复杂的画面掩盖没有可辨识声学证据的问题。
