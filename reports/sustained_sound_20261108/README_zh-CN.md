# 持续发声对照结果

2026-11-08，episode `1000000008`、种子7、64个原生step总预算，固定位置／朝向监听1秒后决策。实现提交`9b187e8`，原地监听与1秒epoch83定位权重不变，定位结果仍完全旁路。

原任务并非只允许播放一次源WAV：引擎已循环读取源样本，但只在2–8秒注入声音。此次只将**运行时**排程延长为2–18秒，保留2秒起声，覆盖16.25秒视频。原Episode、物体—声音关联、WAV、场景和权重没有修改；Episode与源WAV SHA均与原地监听基线一致。源WAV内部的响度变化／静音不剪除，不使用恒幅信号代替声音。

## 实际对照

| 项目 | 原排程原地监听版 | 持续排程原地监听版 |
| --- | --- | --- |
| 发声排程 | 2–8秒 | 2–18秒，测试结束时仍启用 |
| 总原生step／高层决策 | 64／8 | 64／8 |
| 移动／监听step | 30／34 | 29／35 |
| 9秒后实际音频 | 原版该阶段已静音 | 29/29个0.25秒窗口有数值有效声音 |
| 完全处于9秒后的完整监听窗 | 无有效声音 | 3/3有有效声音 |
| 仍在发声排程内的定位锚点 | 3 | 7 |
| 正确chair输出 | 2/3 | 6/7，另一次误分类fireplace |
| 正确chair的条件平均方向误差 | 6.991° | 8.497°，6次均≤20° |
| 最近测地距离 | 0.309m | 0.398m |
| 最终测地距离 | 1.772m | **0.626m** |
| 真正执行STOP／成功STOP | 否／否 | **否／否** |
| LLM提出但门控拒绝的STOP | 1次 | **4次** |

持续版的初始静音窗口没有误报；静音样本只有1个，不能说明一般背景噪声鲁棒性。新旧定位样本数、时刻和实际轨迹不同，不能把2/3与6/7或两个条件方向误差当作受控准确率增益，也不是标准SELD/F20。远端LLM采样不由仿真种子固定；本次只证明持续声音实际存在，并报告各自真实导航结果，不宣称导航变化全由声音排程导致。

## 为何仍未STOP

Codex实际请求12次，0次失败；8个接受的导航提议、4个被拒绝的STOP提议。拒绝时刻为`obs_0028 / obs_0036 / obs_0044 / obs_0052`。这是原有被动停止门控的行为，未使用定位器或真实目标距离。

用保存的Agent可见音频、位姿、深度及当时记录的被动拟合复算（不读目标真值、不重新渲染）：

- `obs_0028`和`obs_0044`：前后假设误差差距未超过3°门槛，方向不唯一；所拟合的0.1m位置也没有得到相应深度表面佐证。
- `obs_0036`和`obs_0052`：选中后方假设，前视深度无法佐证该射线；前者的估计区间上界1.1m也超过门控0.9m上限。
- 另发现数值边界：`obs_0052`上界计算为`0.9000000000000001`，严格比较也判为超过0.9m。但该帧同时缺少深度佐证，单独修正浮点容差仍不能使这次STOP通过。本次保留原门控以维持对照，没有修改策略。

因此，声音供给确实曾限制后半程听觉观测，但**持续发声没有自动解决停止问题**。本次机器人最终进入1m内仍未STOP，剩余瓶颈包括到达证据、前后歧义与深度佐证；没有将接近目标解释为成功导航。

## 验证与命令

35项测试通过：4项排程、9项同步、6项原地调度、14项定位协议、2项音频汇总。初次原生加载发现官方配置合并会冻结参数，修正覆盖顺序后重做原生对照；原始失败与重试日志均保留，没有未解决的加载错误。

固定初始位置、相同种子，分别运行原排程和持续排程各40个原地step：9–10秒的5个采集窗口，原排程0/5有声音，持续排程5/5有声音；源采样索引一致，原Episode SHA未变。这一对照不依赖LLM导航。

正式运行验证两个原生环境动作／姿态／时间同步，8个完整监听窗口各4个新FOA块位置／朝向不变，Agent等待期间实际引擎状态冻结，定位结果未进入Agent输入、工具、证据或门控。最后只剩3步，采集0.75秒后预算耗尽，不补零、不再调用定位器或Agent。

工作树`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/`中实际命令：

```bash
python3 -u scripts/candidate_codex_shadow.py \
  --profile avn_foa_short1s --decision-period-s 1 --listen-before-decide-s 1 \
  --sound-schedule sustained \
  --output outputs/avn_sustained1s_shadow_1000000008_codex_seed7_64_20261108 \
  --sim-workdir /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree \
  --inputs /media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/outputs/codex_actor_inputs_20261108 \
  --episode-id 1000000008 --seed 7 --steps 64
```

## 文件

云端输出为上述工作树下同名`outputs/`子目录；本地在原人工检查父目录下新增同名子目录，旧结果保留。

- `avn_demo.mp4`：512×512、4fps、16.25秒，原生双耳音轨；完整解码通过。
- `navigation/binaural.wav`、`foa_shadow/foa_ambix.wav`：完整16kHz浮点双耳／世界坐标ACN/SN3D FOA。
- 两个环境的`sound_schedule.json`：官方2–8秒、运行时2–18秒排程，不提供给Agent。
- `comparison.json`：新旧结果和9秒后实际活动统计，包含Episode／WAV哈希一致性核验。
- `listening_windows.csv`、`predictions.jsonl`：8个固定位置窗口的区间、RMS、类别、条件角误差及原始模型结果。
- `stop_diagnostics.json`：4次STOP提议和当时门控记录；`stop_checks_recomputed.json`：仅用保存观测进行的离线复算。
- `listening_cycles.jsonl`、`decision_cycles.jsonl`、`actions.jsonl`：原地采音、模型等待、29次移动／35次监听和中断记录。
- `model_observations.jsonl`、`model_decisions.jsonl`、`codex_calls.jsonl`：实际Agent输入与请求，供检查定位结果没有暴露。
- `shadow_evaluation.json`、`validation.json`、`video_verification.json`、`checks/`：独立核查、35项测试、原生声音对照、实际命令与源码快照。

逐帧RGB-D／原音频NPZ和私有目标真值留在云端。没有重新训练、启用定位器驱动导航、修改STOP门控或自动进入下一阶段。
