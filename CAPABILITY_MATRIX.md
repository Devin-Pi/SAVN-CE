# Installed engine capability audit

核验对象是本机容器 `savnce_dev` 的 Habitat-Sim 0.2.2 / SoundSpaces音频绑定和当前SAVN-CE包装器。结论区分运行验证、源码/API证据与设计推断。没有添加家具、障碍物或移动声源。

## 一致性现状

| 层 | 实际状态 | 证据及限制 |
|---|---|---|
| 视觉 | 原始 `17DRP5sb8fy.glb`；RGB/深度正常 | 20条原生Env与120个语义对齐审计视角；像素不是物体发声证据 |
| 碰撞 | stage collision asset也是GLB；当前physics=False、独立刚体0 | Bullet编译支持为True，不等于当前任务启用了动力学；合法移动基于navmesh |
| 导航 | 单独 `.navmesh`，20条参考轨迹0碰撞且可达 | `recompute_navmesh(...,include_static_objects=False)` 存在；显式True才包含STATIC对象；未做新增障碍物重建实验 |
| 语义 | 独立 `_semantic.ply` / `.house` | 实际 `SemanticObject.semantic_id` 与实例像素交叉核对；不是给新刚体自动生成MP3D实例语义的证明 |
| 声学 | 原生RLRA基础几何导出成功：191977顶点、215763面 | 导出SHA256 `4e5bd8a37f1b7bc5d50e3f9dd97317efea4fb5bc1a69e3eccddf715113723fe5`；保留统计后删除临时OBJ；未证明与视觉/碰撞/语义表面完全等价 |
| 时序音频 | 原生固定点声源、每步RIR卷积叠加未来尾部、双耳16kHz | 20条完整声音时间窗探针；没有验证运动声源的传播延迟、Doppler或历史声场正确性 |

当前基础场景有可用的视觉、导航和声学，但这些是不同数据表示。不能以“刚体可见/碰撞”推断“声传播已经考虑刚体”。

## A–E 支持矩阵

| 能力 | 当前可用程度 | 所需工作 / 一致性要求 | 本次测试状态 |
|---|---|---|---|
| A 复用现有静音家具作视觉干扰 | 可用基础能力；语义筛选已实现 | 保持扫描布局；候选必须实际可见。只从配置知道非声源对象未被调度，不可据此说Agent有静音证据。同类不等于外观相似 | 已发现G1候选2条；感知冲突/实际发声归因未验证 |
| B 新增独立静态障碍物 | 视觉、刚体、射线和NavMesh API存在；完整集成未实现 | 独立资产/STATIC变换，显式更新碰撞、include_static_objects=True重建navmesh，分配一致语义；在音频第一次几何上传前加入或完整重建音频上下文；比较声学导出/IR变化 | API/源码检查；未添加实物或宣称阻声生效 |
| C 新家具/布局改变 | 新独立家具可走B；原扫描家具重排不直接支持 | 扫描家具烘焙在stage，不是可移动刚体。须创建资产副本并分割/移除旧几何、修复纹理语义碰撞，或采用模块化场景；新位置重新构建导航与声学 | 设计阶段；没有改原始MP3D资产 |
| D 移动声源轨迹及真实空间音频 | source transform和单时刻IR接口存在；连续移动音频未实现 | 统一源时钟、音频相位、世界坐标/高度约定、每步源/听者更新、缓存失效、时变RIR与历史尾响；验证延迟、切换连续性和物理近似 | 固定声源音频已测；没有移动源实验，不可宣称真实连续运动音频 |
| E 布局+遮挡+声学+移动源组合 | 设计阶段 | 事务式geometry_epoch、全层一致性门禁、缓存版本化、重建/同步生命周期、组合回归。不能把单项API能力拼成已可用复杂场景 | 仅设计接口；不支持的变更明确拒绝 |

## 源码和绑定依据

完整行号与源码SHA256保存在 `research_outputs/agent_dataset_v1_1/engine_source_evidence.json`。参考源码位于 `/media/4tbssd/00_PROJECT_CODE/soundspace_gen/habitat-sim_old/`。**旁边源码与安装二进制的精确build commit尚未确认**，下列源码行为是风险依据，必须在任何扩展中进行运行验证。

- `src/esp/sensor/AudioSensor.cpp:132`：`newInitialization_` 分支进行几何上传，随后清为False；不是每帧从Habitat场景图重取所有几何。
- `AudioSensor.cpp:497`：无语义材质路径调用 `sim.getJoinedMesh(true)`。`Simulator.cpp:839` 的joined mesh从stage开始，includeStaticObjects时加入STATIC刚体/关节物体碰撞网格；不能推断kinematic/dynamic物体自动入声学。
- `AudioSensor.cpp:380`：语义材质路径调用 `getJoinedSemanticMesh`；`Simulator.cpp:937` 使用stage语义资产。新增刚体不因此自动出现在语义声学网格中。
- 当前 `enableMaterials=False`，走非语义材质几何路径。以后启用语义材质时需要重新评估新增物体遗漏问题；材质配置须在上传前完成。
- `AudioSensor.cpp:75` 的reset销毁RLRA上下文；重新初始化涉及源、听者、材质、几何与外部卷积缓冲，不能只调用reset后继续用旧缓存。
- `AudioSensor.cpp:95` 的source transform调用RLRA_SetSourcePosition。它仅证明设定源位置的入口，不证明时变卷积、传播延迟或源波形历史正确。
- 实际Python AudioSensor绑定有 `runSimulation / reset / setAudioSourceTransform / setAudioListenerTransform / getIR / writeSceneMeshOBJ / setAudioMaterialsJSON`；没有逐对象声学网格或对象位姿更新接口。
- 低层 `RLRAudioPropagation.h:352–383` 有Add/ClearObject、SetObjectPosition/Orientation、SetObjectMeshOBJ/PLY；因此不能说RLRA绝对不支持对象更新。当前缺失的是Habitat桥接/绑定和多系统同步实现，实际连续声场能力仍须实验确认。
- `savnce/sims/savnce_simulator.py` 在reconfigure中将源设为官方goal+[0,1.5,0]；`_compute_audiogoal`只在onset/offset门控内更新卷积，向未来缓冲累加尾部。原包装器没有移动源轨迹调度或几何epoch管理。

## 哪些必须改引擎

1. **离线固定布局变体**：可先使用预烘焙资产副本、现有STATIC接口、重建NavMesh、在音频初始化前固定几何。这条路线不必先修改RLRA求解器，但需要资产管线/协调器/语义集成和运行验证，当前未实现。
2. **运行中变更家具/障碍物并增量更新声学**：现有Python绑定不足。需修改Habitat AudioSensor及其C++/Python桥接，暴露对象/几何同步、明确缓存失效和生命周期，或采取成本较高的完整音频重初始化。NavMesh与语义更新同样需要协调。低层存在接口不代表桥接已完成。
3. **连续运动发声**：至少修改SAVN-CE音频调度、缓存与时变卷积实现。是否必须改RLRA求解器取决于传播延迟、Doppler、动态遮挡和历史混响的目标精度；当前未验证，不能承诺仅改source transform即可达到物理正确。
4. **完整组合场景**：要先满足上述每层验证，不在本阶段实施。
