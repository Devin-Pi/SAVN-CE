# FOA近距验证与STOP门控（实验模式）

`--stop-rule foa_multiview`只替换STOP的证据来源与审核。旧`legacy`仍为默认，原逐步、移动采音和纯shadow实验可继续复现。

`--stop-rule foa_geometry`按有向射线的几何一致性关联观测，类别只用于在支持数量相同的候选中辅助选择。该模式的射线容差默认为 2°，可用 `--foa-tolerance-deg 5` 复现此前的 5° 配置。10° 最低视差和 1 米距离门槛各自独立。下文的 15° 和同类别关联描述早期 `foa_multiview` 模式。

49项测试和64步在线验收已完成。[实际轨迹与未停原因](reports/foa_stop_20261109/README_zh-CN.md)：8轮没有满足近距证据门槛，未执行STOP，尚未证明实际停止效果改善。

本模式将FOA**用于STOP**，因此不再是纯shadow。Agent看到精简的`stop_verification`（准备状态、检查项、启发式距离区间、证据ID），仍使用原RGB-D／双耳信息选择移动方向；不会获得FOA方向向量、拟合声源坐标、完整模型输出或任何目标真值。协调器不自动发STOP；只有Agent提出STOP且新门控ready才执行。旧双耳前后拟合即使认为接近，也不能代替新STOP证据。

## 证据规则

1. 每个完整原地1秒窗口最多贡献一条来自实际FOA模型的**有符号世界坐标方向射线**，起点为实际观测到的声学接收器位置。不同朝向或重叠音频不会增加空间基线。当前无活动、无检测、多事件、方向/位姿不合规时，状态保持未验证。
2. 使用最近12秒、同预测类别的历史观测。至少3个相距0.2m的位置、0.5m移动基线和10°视差；不同类别或多事件不强行合并。相同预测类别不证明实例相同，本版本针对单静态声源实验。
3. 以各条有符号射线进行3D交会拟合，再采样最新方向15°锥内的候选点，要求与历史方向一致。使用近距细网格和远距候选，存在1米外可行解、几何退化或拟合点不获支持时不放行。没有把同一FOA方向镜像成前后两个假设。
4. 整个采样可行区间须在1米内，并且连续两个新完整窗口确认，拟合点变化不超过0.5m。重复读取旧结果不增加确认数。没有用活动分数、RMS或单条方向射线直接估距离。

前视深度仍用于前进避障，但不要求声源方向处于相机视野内，也不要求声学点与某个深度表面重合。声音在后方并不会单独导致拒停。任何时候证据不足都明确记为未验证。

这里估计的是接收器到声学点的3D欧氏距离，**不是可导航测地距离**。15°与采样可行区间是固定启发式，不是校准置信区间；同类多实例、遮挡反射、未知墙体和运动源仍会影响可靠性。实际成功与否由关停后的独立评估使用真值核查，线上算法不读场景资产、NavMesh、Oracle或真实距离。

## 云端复现

工作树`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/`：

```bash
python3 -u scripts/candidate_codex_shadow.py \
  --profile avn_foa_short1s --decision-period-s 1 --listen-before-decide-s 1 \
  --sound-schedule sustained --stop-rule foa_multiview \
  --output outputs/avn_foastop1s_1000000008_codex_seed7_64_20261108 \
  --sim-workdir /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree \
  --inputs /media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/outputs/codex_actor_inputs_20261108 \
  --episode-id 1000000008 --seed 7 --steps 64
```

保留原1秒权重、固定姿态监听、64步总预算和持续发声条件。`foa_stop_evidence.jsonl`记录完整门控诊断；`derived_evidence_registry.jsonl`登记每个Agent可引用的STOP证据及其实际音频窗口来源；`model_observations.jsonl`保存Actor真正收到的精简证据。所有原始预测仍单独记录。前后环境同步、状态冻结和真值隔离由事后核查验证。

测试命令在现有容器内使用Python：`python -m unittest discover -s tests -p test_foa_arrival.py -v`与`python -m unittest discover -s tests -p test_foa_stop_protocol.py -v`；同时保留35项旧协议、声音排程及调度回归测试。无需重新训练或修改场景／Episode。
