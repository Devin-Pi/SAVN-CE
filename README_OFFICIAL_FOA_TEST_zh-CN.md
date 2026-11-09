# 官方 SAVN-CE Episode 的 FOA 测试准备

首批使用官方 clean test 的原始 10 个分片。官方 episode、起点、目标实例、导航坐标、WAV 和发声时间表由原生加载器与模拟器处理。索引以分片和 episode ID 共同标识记录，因为不同分片可以复用同一个 ID。

## 生成验收数据

在云端 SAVN-CE 工作树执行。将分片放在 `--episode-dir` 指定的目录，目录内的文件名为 `test-split_0.json.gz` 到 `test-split_9.json.gz`。

```bash
docker exec -e HABITAT_SIM_LOG=quiet -e MAGNUM_LOG=quiet \
  -w /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree \
  savnce_dev python -u scripts/prepare_official_foa.py \
  --episode-dir outputs/avn_author_3990_inputs_20261109 \
  --output outputs/official_foa_test_pilot_20261109 --seed 7
```

脚本用原生 Dataset 加载 1,000 条记录并检查场景和声音文件。首批固定选取 `3990`，再按种子 7 选择四个不同场景和类别的样本。各样本在官方起点固定监听，从起点观测采集至发声开始后 2 秒，输出 16 kHz、ACN/SN3D、W/Y/Z/X 的世界坐标 AmbiX 波形和逐步 RGB-D、位姿及音频。每 0.25 秒有 4,000 个四通道采样，另保存发声前及发声开始后的 1 秒 WAV。

- `official_index.jsonl` 列出所有可用记录及原始分片校验和。
- `pilot_plan.json` 和 `validation.json` 保存首批选择及采集验收结果。
- `captures/` 保存验收波形和观测。实际导航时，仿真根据 Agent 在线选择的动作重新渲染。
- `evaluation_only/` 保存源坐标与目标真值，访问权限与 Actor 输入分开。

## 开始在线首批测试

在云端工作树执行以下命令。它会依次运行首批五条，使用单线程声学渲染、种子 7、固定位置监听 1 秒及 2° FOA STOP 判据。RGB-D 和双耳仍用于移动决策，FOA 几何证据用于 STOP。每条默认 500 个 0.25 秒原生步，监听也占预算。需要缩短调试时可用 `STEPS` 覆盖。

```bash
SIM_WORKDIR=/workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree \
ACTOR_INPUTS="$PWD/outputs/codex_actor_inputs_20261108" \
RUN_ROOT=outputs/official_foa_agent_runs \
bash outputs/official_foa_test_pilot_20261109/run_agent_pilot.sh
```

成功判定使用 Agent 的 STOP 和距官方导航目标的测地距离。首批五条用于接口和流程验收，汇总成功率应注明这是小样本试跑。报告与作者 MAGNet 基准的对照时，注明这里的 1 秒监听与原生步预算。
