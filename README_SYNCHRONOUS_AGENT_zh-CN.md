# 同步高层决策：执行、暂停、观测、决策

需要移动后固定位置采音时，使用新增的[原地监听1秒模式](README_STATIONARY_LISTEN_zh-CN.md)。下文保留移动采音版本作为对照，不代表原地监听流程。

新增可选`--decision-period-s 1`模式，供现有Codex Agent使用。底层仍为0.25秒一个原生step；高层一次指令最多执行4步，然后暂停仿真，使用最新RGB-D／位姿和已采集历史请求下一条指令。模型等待期间不执行step、不推进音频时间。1秒定位器始终只做shadow记录，没有加入Agent输入、工具、STOP或中断规则。

已完成原生64步验收：17轮高层决策，23项测试通过，全部等待期间实时位姿核验通过。[实际结果与文件说明](reports/synchronous_control_20261108/README_zh-CN.md)。本模式的音频来自动作执行过程，没有新增原地听满1秒阶段。

## 时间和动作约定

| 层次 | 本模式行为 |
| --- | --- |
| 原生仿真 | 0.25秒一步，每步更新RGB-D、双耳、FOA及各自位姿 |
| 高层决策 | 名义周期1秒仿真时间；提前中断时可以更早重新决策 |
| 等待Codex | 串行阻塞；调用前后读取两个引擎的实时机器人位姿，核验ID、step、时间、位置和旋转均未变化 |
| 定位器 | 过去1秒FOA上下文，每0.25秒输出当前step，前三块预热；仅记录 |
| step预算 | 仍按原生step计；64步对应16秒位置更新时间，不是64秒或64轮LLM决策 |

原生动作空间不变：`move_forward / turn_left / turn_right / stop`。高层选择一个primitive，协调器最多执行4次；模型提示和当前观测中的`control.action_semantics`明确告知：完整前进段最多1m，完整转向段最多60°。不能把新的高层动作当成旧的单次0.25m／15°动作。最后一段按剩余预算缩短。STOP立即执行一次，不等待满1秒，不推进位置和仿真时间；原有被动STOP门控保留。

执行段不是不检查传感器的盲重复。每个原生step之后，协调器根据实际已采集信息检查：

- 碰撞或前进被阻挡；
- 当前前进方向深度中心带`p10 < 0.45m`，或该深度无效；
- 原有被动到达证据已ready。

满足条件时结束本段，在最新位置请求新的高层决策，不自动宣布到达，也不代替Agent执行STOP。前进方向深度不足时，在决策前就移除前进候选，避免同一位置无限重复空段。深度阈值是避障启发式，不能保证目标距离，也不等于精确碰撞体规划。

本模式不强制每段都走满1秒。安全中断与STOP优先于名义周期。当前版本没有新增WAIT或原地持续监听动作。

## 最新位姿与历史音频

假设上一段从A移动到B：下一次输入的`current.control.latest_pose`是B的实际位姿，当前RGB-D也来自B。最近1秒音频由已发生的多个位置块组成，保留各自观测ID、时间和位姿，不能全部标成B采集。`past_steps_oldest_first`包含已执行底层动作及其原观测时间／二维位姿。

当前原生音频块的时间区间为`[observation.timestamp_s, timestamp_s+0.25)`，块内采用对应的离散位姿；它不是连续运动插值录音。`control.pose_timestamp_s`和`current_audio_interval_s`分别记录这两个约定，不把音频区间末尾伪装成新的位置更新。每段前进4步是位置时间推进1秒，初始观测还包含一块音频。

`agent_episode.serve`新增只读请求`{"command":"state"}`，直接读取当前引擎机器人位姿，不依赖`observe`缓存，不生成新观测、不推进音频、不包含目标真值。协调器用它核验等待期间状态，两个引擎也互相核验。查询无效、状态漂移、API失败或旧观测ID会在执行新动作前失败退出。

## 实现和日志

- `agent_episode/synchronous_control.py`：与引擎／模型解耦的串行调度、执行上限、状态核验及中断。
- `scripts/candidate_codex_shadow.py`：接入现有两个原生环境、Codex、被动STOP门控和1秒FOA shadow。新模式显式启用；原0.25秒逐步决策模式仍保留。
- `agent_episode/codex_backend.py`：仅在新模式下向模型解释动作段、暂停状态、当前位姿及历史音频语义；认证和模型调用方式保留。
- `agent_episode/environment.py`、`serve.py`：只读实时位姿查询。
- `scripts/check_candidate_shadow.py`：结束后核查一个高层提议对应多个实际底层动作；旧逐步评估兼容。
- `tests/test_synchronous_control.py`：9项测试；同时运行原14项定位协议测试。

新增`decision_cycles.jsonl`逐轮保存：`decision_started → decision_returned → segment_finished`，包括最新位姿、墙钟等待时间、冻结校验、指令哈希、实际步数、实际推进仿真时间和中断原因。`sensor_summaries.jsonl`保留每个底层观测；`model_observations.jsonl`仅记录真正提供给Agent的高层决策观测。`actions.jsonl`中的连续多步可以引用同一个高层决策anchor，这是动作段的正常表示，不代表Agent每步重做了决策。

## 云端命令

工作树为`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/`：

```bash
python3 -u scripts/candidate_codex_shadow.py \
  --profile avn_foa_short1s --decision-period-s 1 \
  --output outputs/avn_sync1s_shadow_1000000008_codex_seed7_64_20261108 \
  --sim-workdir /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree \
  --inputs /media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/outputs/codex_actor_inputs_20261108 \
  --episode-id 1000000008 --seed 7 --steps 64
```

在`savnce_dev`内对应工作树运行事后核查：

```bash
python scripts/check_candidate_shadow.py \
  --run outputs/avn_sync1s_shadow_1000000008_codex_seed7_64_20261108 \
  --actor-audit outputs/sync1s_control_checks_20261108/actor_environment.json
```

测试命令（模型容器使用工作树的`repository`副本）：

```bash
python -m unittest discover -s tests -p test_synchronous_control.py -v
python -m unittest discover -s tests -p 'test_*localization.py' -v
```

旧Episode、资产、权重、训练集与既有输出均不修改；新结果使用独立目录。这里实现的是同步仿真调度，不保证真实机器人在LLM等待时自动保持位置，实机需要独立的速度归零／制动控制。
