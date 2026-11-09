# FOA停止规则：实现与原生验收

2026-11-09核查。运行`avn_foastop1s_1000000008_codex_seed7_64_20261108`，实现提交`36ca767`。保留episode `1000000008`、种子7、64步总预算、固定姿态监听1秒、原1秒epoch83权重与2–18秒运行时持续发声。原Episode、源WAV和检查点SHA均与持续发声旧规则对照一致。

新增可选`--stop-rule foa_multiview`。该模式在不同观测位置交会有符号FOA方向，检查距离可行区间与连续确认；不使用双耳前后双假设或前视深度表面匹配做STOP硬门槛。深度仍用于前进避障。Actor仅收到精简`stop_verification`及证据ID，FOA方向与估计声源坐标不用于选择移动方向。门控只审核Agent提出的STOP，不自动终止。

## 测试结果

49项测试通过：35项既有协议／调度／排程回归、10项FOA几何／来源测试、4项Actor与STOP协议测试。合成几何验证了后方声源在没有前视深度时仍能生成有效到达证据；同位姿、平行射线、远源、重复／重叠窗口、类别不一致、错误方向和跨Episode缓存不能错误增加确认。通过这些单元与协议测试不等同于真实在线成功STOP。

| 指标 | 持续发声＋旧门控 | 持续发声＋FOA门控 |
| --- | --- | --- |
| 原生步数／高层决策 | 64／8 | 64／8 |
| 移动／监听步数 | 29／35 | 30／34 |
| Codex成功请求 | 12 | 8 |
| STOP提议／拒绝／实际执行 | 4／4／0 | 0／0／0 |
| 最近测地距离 | 0.398m | 0.309m |
| 最终测地距离 | 0.626m | 2.762m |
| 成功STOP | 否 | 否 |

新模式8个完整监听窗口都没有达到`ready`，没有提出STOP。**功能与隔离验收通过，实际停止效果尚未改善。** 该表是单Episode两次独立在线轨迹，远端LLM采样不由种子固定；不能据此宣称门控导致更差导航或得到一般效果结论。

## 已进入1米，却无法确认近距的原因

环境关闭后，用保存的机器人位姿与实际运行目标在同一NavMesh中复算距离，初始／最终结果与原生评估一致。真值没有提供给Actor或门控。

| 决策窗口 | 真实测地距离 | 新门控 |
| --- | ---: | --- |
| `obs_0042` | 0.401m | 有方向交会，启发式区间 **0.10–1.375m**，仍包含1米外解 |
| `obs_0050` | 0.806m | 少于3个满足空间间隔的同预测类别方向观测 |
| `obs_0058` | 1.772m | 启发式区间1.50–1.95m，未到达 |

定位在7个发声锚点输出4次chair、3次cabinet；已输出方向都在20°以内，但类别波动和空间基线不足会打断同源射线积累。这体现了“方向较准”仍不能直接转换成可靠距离。另用旧持续发声轨迹的已保存观测回放新规则，全程不读目标真值：`obs_0052`的采样区间0.60–0.975m，但线拟合点未同时得到所有有向角锥支持；`obs_0060`上界1.025m。两处均未满足稳定确认。

回放中也复核了旧报告措辞：旧`obs_0020`同时报chair与fireplace，旧`obs_0028`漏检；fireplace是额外输出，不是漏检时刻的替代类别。已更正文字，原始数据和汇总数值保持原样。

新规则不再要求后方声源必须有前视深度。仍需解决观测位置不足、跨窗口同源关联、距离不确定性及采样拟合保守性。该距离是3D欧氏声学估计，无法保证可导航测地距离；遮挡／反射／多实例尚未完成验证。没有用真值、放宽阈值或自动STOP制造成功。

## 复现和交付

在云端工作树`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/`运行：

```bash
python3 -u scripts/candidate_codex_shadow.py \
  --profile avn_foa_short1s --decision-period-s 1 --listen-before-decide-s 1 \
  --sound-schedule sustained --stop-rule foa_multiview \
  --output outputs/avn_foastop1s_1000000008_codex_seed7_64_20261108 \
  --sim-workdir /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree \
  --inputs /media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/outputs/codex_actor_inputs_20261108 \
  --episode-id 1000000008 --steps 64 --seed 7
```

默认`--stop-rule legacy`及旧结果保留。事后核查通过：两环境同步、等待时实时位姿冻结、固定姿态窗口、真实音频来源与过去锚点关联、Actor只接收允许的STOP摘要。`checks/launch.json`及`source_snapshot/`保留实际运行源码；事后核查器单独存在`checks/postrun_source/`。`run_config.launch.json`为原启动记录，修订后的`run_config.json`仅澄清FOA使用范围及实际调用频率，未改在线行为。

本地同名子目录包含16.25秒带双耳音频视频、决策／动作与定位日志、完整证据登记；旧结果保留。`foa_stop_evidence.jsonl`为门控诊断，`derived_evidence_registry.jsonl`为证据来源，`stop_distance_audit.json`为隔离后的实际距离复算，`comparison.json`为旧规则对照。原始双耳／FOA WAV与逐帧NPZ完整保留在云端同名目录，未在此轻量人工检查包中重复下载。
