# 原地监听1秒，再请求Agent决策

使用`--listen-before-decide-s 1`启用：**动作段结束 → 固定位置和朝向采集1秒新音频 → 汇总观测／旁路定位 → 暂停仿真等待Codex → 执行下一段动作**。启动后也先原地听1秒，再进行第一次决策。原移动采音版与旧逐步模式保留。

已完成云端64步原生验收：30步移动、34步监听、8轮决策；固定姿态、等待冻结及定位隔离检查通过。[实际结果与文件说明](reports/stationary_listen_20261108/README_zh-CN.md)。

## 真正采音与等待模型

监听阶段由协调器调用`agent_episode.serve`的`listen`请求；该请求内部给Habitat注册独立的零位移动作（`move_forward`执行量为0），沿用原生`SAVNCE_Simulator.step()`推进音源采样、卷积缓冲及step时间。它不是STOP，不是重复读取缓存，不用转过去再转回来，也不改原始场景资产或官方Episode。

每次监听连续执行4个0.25秒原生step，位置和四元数都必须保持不变。4次返回的观测ID必须不同、连续；每个音频块都由引擎新渲染。渲染历史混响尾声保留，不清空卷积状态，因而不宣称移动后原地录音与从一开始就静态卷积的训练样本逐位相同。

监听完成后，实时读取引擎位姿并暂停串行执行。Codex返回前后再次核验实际位置、旋转、step和时间未变。等待模型的墙钟时间不推进音源播放，也不会额外采集声音。

## Agent与定位器各收到什么

- Agent：监听结束处最新RGB-D、完整位姿、已发生的动作／观测历史，以及这4块双耳音频的1秒特征汇总和声学图。`current.stationary_audio`明确记录窗口ID、区间和固定姿态；`current.audio`仍是最新0.25秒特征。被动STOP门控仍按原先逐步观测更新，汇总不会重复添加门控证据。
- 定位器：只在完整原地监听窗口结束时推理，输入严格等于对应4块新FOA，共16000×4、16kHz；使用既有共用ACN/SN3D转换和1秒epoch83权重。输出仍是当前最后0.25秒的类别／方向，不是整秒平均方向。
- 隔离：定位结果只写日志，不注册为Agent工具，不进入提示、证据、STOP或动作段中断。原地采集的双耳特征是Agent本来允许使用的传感器信息，不是定位器输出。

相同4块音频的汇总与其组成块不是额外独立证据。移动阶段的FOA仍保存到云端，但本模式不会把混合移动／静止窗口当作固定位置输入进行定位推理。

## 时间、预算与声音排程

`--steps 64`继续表示**全部原生step预算**，监听与移动都消耗它；每步0.25秒，总推进最多16秒。两种时间分别记录，转向归入移动动作步。动作段仍最多4步，碰撞／前方深度／原有到达证据可提前结束，因此高层决策间隔通常是1秒监听加最多1秒动作，不能再称固定每1秒决策。

原生观测原有约定不变：位姿时间为`step*0.25`，返回音频区间为`[timestamp,timestamp+0.25)`。例如到达时位姿时间2.0秒，4个新监听观测的位姿时间为2.25、2.5、2.75、3.0秒；输入音频区间为[2.25,3.25)。原地位置时间推进1秒，音频可用区间也推进1秒。日志分别保存位姿时间和音频区间，不能把音频区间末尾伪装成一次新的位置更新。初始观测还有一块音频，不纳入首个监听窗口。

原Episode声音排程2–8秒保留，监听会消耗发声时间，因此某些监听窗口可能静音。每窗口记录整秒和最新0.25秒的活动，评估以真正采到的音频为准，不延长声音来制造成功。预算末尾不足4步时，只记录未完成监听，不补零、不调用定位器或Agent；若恰好听满但预算耗尽，可以记录定位结果，不再发起无法执行的导航决策。

STOP立即终止，不继续监听；API失败、位姿漂移、时间不连续等情况会在下一条动作执行前失败退出。

## 云端使用

工作树：`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/`。

```bash
python3 -u scripts/candidate_codex_shadow.py \
  --profile avn_foa_short1s --decision-period-s 1 --listen-before-decide-s 1 \
  --output outputs/avn_stationary1s_shadow_1000000008_codex_seed7_64_20261108 \
  --sim-workdir /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree \
  --inputs /media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/outputs/codex_actor_inputs_20261108 \
  --episode-id 1000000008 --seed 7 --steps 64
```

GPU定位worker仍在`avseld_final_test`，两个原生环境仍在`savnce_dev`，Codex仍使用独立只读Actor及现有API Key，不改变全局provider／模型配置。

## 验证与日志

已验证原生连续12步原地监听：位置／朝向不变，音源索引与时间推进，发声后音频样本持续更新。31项测试通过：原9项同步调度、6项原地阶段调度、14项定位协议、2项固定位置音频汇总。测试命令：

```bash
python -m unittest discover -s tests -p test_synchronous_control.py -v
python -m unittest discover -s tests -p test_stationary_control.py -v
python -m unittest discover -s tests -p 'test_*localization.py' -v
# 以下在savnce_dev中执行
python -m unittest discover -s tests -p test_stationary_audio.py -v
```

新增`listening_cycles.jsonl`逐轮记录监听前后位姿、4个新观测ID、原生时间推进、音频区间及完整性；`decision_cycles.jsonl`仍记录Agent等待与动作段。`actions.jsonl`中的`listen`来自协调器，不是Agent候选动作。`rollout_complete.json`与独立评估分别统计`movement_steps / listening_steps / total_native_steps`。视频包含监听时的静止画面和真正新音频，不包含等待LLM的墙钟停顿。

事后核查命令在`savnce_dev`对应工作树内执行：

```bash
python scripts/check_candidate_shadow.py \
  --run outputs/avn_stationary1s_shadow_1000000008_codex_seed7_64_20261108 \
  --actor-audit outputs/stationary1s_control_checks_20261108/actor_environment.json
```

核查验证每份模型输入都关联完整固定位置窗口、每次定位恰好使用对应4块FOA、两个环境同步、模型等待时冻结，以及定位输出没有提供给Agent。声源位置、正确实例和导航距离只在环境关闭后的独立评估使用。
