# 持续发声对照：保留2秒起声，延后运行时停止

原episode `1000000008`只有一段2–8秒的发声区间，之后停止注入新的源信号，约9秒后只剩数值静音。原生模拟器已使用循环音源索引读取WAV，因此限制来自排程关断，不是只允许播放一次WAV。

新增`--sound-schedule sustained`，保留官方起声时刻，只把运行时停止时刻延后到本次预算的音频末尾之后。64个0.25秒step对应的运行时排程为2–18秒，覆盖最长16.25秒的导出音频。默认`episode`仍严格使用2–8秒。

该条件不改原始`.json.gz`、物体—声音关联、音源WAV、模型、场景资产或播放增益。沿用原生采样索引循环和空间卷积，不剪除源WAV内部静音，不把源信号替换成恒幅声。`navigation/sound_schedule.json`和`foa_shadow/sound_schedule.json`记录官方及运行时排程；私有评估保留原始Episode并按运行时排程计分，不把额外播放时段误标成静音误报。

## 云端命令

工作树：`/media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/`。

```bash
python3 -u scripts/candidate_codex_shadow.py \
  --profile avn_foa_short1s --decision-period-s 1 --listen-before-decide-s 1 \
  --sound-schedule sustained \
  --output outputs/avn_sustained1s_shadow_1000000008_codex_seed7_64_20261108 \
  --sim-workdir /workspace/SAVN-CE/outputs/avn_candidate_shadow_worktree \
  --inputs /media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree/outputs/codex_actor_inputs_20261108 \
  --episode-id 1000000008 --seed 7 --steps 64
```

其余设置沿用[原地监听版](README_STATIONARY_LISTEN_zh-CN.md)：固定位置／朝向采集4个新块后决策，等待模型时冻结；监听与移动共同消耗64步预算；1秒epoch83定位器只做shadow，既有Codex策略与STOP门控保留。

## 对照与验证

新增4项排程测试，与原31项协议／调度／音频测试共同运行。另在同一初始位置、同一种子下分别执行原排程与持续排程各40个原地step，核验9秒以后的实际FOA：原排程应静音，持续排程仍有新音频；源采样索引和原Episode文件SHA保持一致。

原生源码和新参数传入情况记录在每次run的`checks/launch.json`及快照中，两个环境排程必须一致，模型请求不接收排程真值、目标类别或目标位置。事后核查验证原Episode SHA未改变。

这是运行时声音排程的研究对照，不是新增物理场景或官方榜单结果。相同仿真种子不保证远端LLM生成完全相同的动作，因此新旧导航轨迹不能视作严格受控的确定性因果比较；实际持续发声与否另外用固定位置原生对照核验。
