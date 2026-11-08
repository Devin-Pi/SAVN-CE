# 1秒定位候选：原生在线shadow测试

本轮已完成，实际结果见[在线测试报告](reports/short1s_online_shadow_20261108/README_zh-CN.md)：24个发声时段检出15个`chair`，条件平均方向误差6.999°，34个数值近静音时段零误报。全程仅旁路记录，没有用于Agent判断；以下为运行协议。

本轮实际运行SAVN-CE／Habitat环境，由现有Codex Agent在线选择动作，独立FOA环境镜像执行这些动作。**定位结果只记录，不注册到Agent工具列表，也不进入提示词、历史、可引用证据或STOP门控。** 这不是把旧视频／旧动作重新回放，也不把定位改进解释为导航成功。

## 固定设置

- episode `1000000008`，沿用既有generated官方格式和物体—声音关联；原声音排程2–8秒保持不变。
- 种子7、64步预算、512×512、原生双耳导航、原evidence策略与被动STOP门控；不启用扫转。种子控制仿真和源偏移，不控制远端LLM采样。
- 现有API Key／网关，Codex请求`gpt-6.1-sol`；独立非root、只读Actor容器不挂载仿真资产、评估真值或shadow输出。
- 定位模型为1秒组epoch83，SHA256 `e8c1539fb72aca7cdf40c40c9d90368dd19bffef93ca7824f00dc1bf52e38a6d`。不改权重、特征、活动阈值0.5或统一增益。
- 16kHz、四通道ACN/SN3D W/Y/Z/X。原生传感器先经共用模块转成world AmbiX，再由worker把完整已观测的1秒历史旋转到最新监听器坐标；不逐块重复旋转，不用声源位置校正。

## 每step数据与输出

每步取得0.25秒真实FOA。缓存满4个连续、不同的已观测ID后，输入精确16000×4；前三次返回`not_ready`，不补零、不重复旧音频、不能指定未观测的未来ID。

前端得到`1×7×160×128`，原编码器内部输出`1×10×189`；按训练时相同的时长权重汇总当前最后0.25秒，得到一个189维ACCDOA结果。因此这里是**1秒上下文、每0.25秒更新当前step**，不是每1秒才产生结果。

输入支持区间例如`[0,1)`；首份输出对应`[0.75,1)`，直到音频时间1秒才能取得。输出方向的`pose_ref`和`orientation_pose_ref`都指向当前step；支持区间中更早的位姿不用于再次旋转当前方向。

Agent先决定并锁定动作，协调器再运行shadow，检查动作JSON哈希没有变化，随后给两个环境执行同一个动作。两个原生环境逐步检查观测ID、时间、位置和旋转一致。terminal观测另存定位结果，不额外制造导航动作。

每0.25秒是仿真音频更新周期；整个闭环还包含LLM和渲染等待，不声称实现250毫秒墙钟实时控制。ACCDOA大小不是校准概率，模型不输出距离或目标实例。

## 接入核验

- 原有及新增协议测试14项通过，包括旧10秒profile兼容、连续采集、1秒预热、缓存、无未来数据、当前step位姿以及错误采样率／长度拒绝。
- 实际FOA＋观测姿态：适配器与冻结训练图输出最大差0；与先前保留回放最大差约`1.32e-4`，不声称跨路径逐位一致。
- 本次加载的Habitat-Sim二进制SHA与之前28方向校准使用的构建一致；共用转换版本`savnce-rlr-foa-1.0`。这不是对所有SoundSpaces实现的断言。
- 已实际通过Codex API Key健康请求。认证值不写入日志；不修改宿主机默认模型／provider配置。

## 实际命令与产物

在云端工作树`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/`执行：

```bash
python3 -u scripts/candidate_codex_shadow.py \
  --profile avn_foa_short1s \
  --output outputs/avn_short1s_shadow_1000000008_codex_seed7_64_20261108 \
  --sim-workdir /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree \
  --inputs /media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/outputs/codex_actor_inputs_20261108 \
  --episode-id 1000000008 --steps 64 --seed 7
```

新profile必须显式选择；原legacy和10秒candidate默认行为保留。GPU模型worker仍在现有`avseld_final_test`容器中运行，原生模拟器在`savnce_dev`中运行。

- `navigation/`：原生RGB-D、双耳音频、动作日志和带音频视频。
- `foa_shadow/`：实际FOA观测、四通道WAV、对应位姿及镜像动作。
- `predictions.jsonl`：每次定位状态、4个支持ID、输入区间、当前step输出、局部／世界方向、原始ACCDOA及耗时。
- `shadow.jsonl`：摘要和在定位前已锁定的动作哈希。
- `model_observations.jsonl`、`model_decisions.jsonl`、`codex_calls.jsonl`：Agent输入摘要、动作提议／STOP门控和真实Codex调用。
- `model.json`、`run_config.json`、`rollout_complete.json`：权重、预处理、协议和结束状态。
- `shadow_evaluation.json`：两个环境关闭后，独立读取评估侧真值，检查同步／隔离并统计检测和方向误差。

真值仍在单独的`research_outputs/agent_private`权限目录，未放入Agent可见观测。最终结果及下载索引在本次run的README中记录；没有修改原始Episode、资产、既有模型、旧视频或训练数据。
