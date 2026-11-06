# SAVN-CE LLM Agent 接入设计

## 控制链路

保留 agent_episode.serve、官方 Dataset 加载及现有规则控制器。宿主机的可信协调器通过 stdin/stdout 启动容器中的 serve。协调器选择 episode、输出位置、种子和预算，从 JSONL 接收观测，在宿主机生成模型输入，再提交经过校验的一步动作。模型只接收协调器明确传入的内容。

    docker exec -i savnce_dev python -m agent_episode.serve --inline-observations

| 请求 | 协调器处理 |
| --- | --- |
| reset | 传入可信实验配置，检查 ok，记录可执行动作并接收 obs_0000。 |
| observe | 读取同一帧。重复读取既不推进仿真，也不能当成独立证据。 |
| step | 提交一个动作和结构化 decision，接收 observation 与 done。 |
| close | 释放环境。在 done、超时或中断后都清理进程。 |

动作集合是 move_forward、turn_left、turn_right、stop。当前配置下前进 0.25 m、转向 15°。非 STOP 动作推进一个 0.25 秒的仿真步，哪怕受阻也消耗一步。STOP 立即结束并返回同一帧。done 后不再请求模型决策。模型推理耗时和仿真时间分别记录，等待模型时仿真不推进。

协调器每次只保留一个待响应的环境请求。模型超时或输出无效时，可以在同一观测上重试一次。仍失败则以单独的失败原因结束，不能把失败转写为 STOP。若发送 step 后连接中断，当前协议没有请求去重机制，执行结果未知时结束该次运行，避免重复执行动作。

## 模型输入与决策契约

- 协调器从 sensor_payload 解码当前帧，使用 NumPy 的 allow_pickle=False，校验数组白名单、形状和有限性。数组为 rgb、depth_m、binaural_audio、position 和 rotation_xyzw。当前 256 像素配置下，RGB-D 与 16 kHz 的 4,000×2 双耳采样按步同步。
- 模型输入包含当前 RGB、分区米制深度、位姿变化、碰撞或受阻、短窗口内已执行动作及其观测 ID。协调器从原始双耳音频提取每耳 RMS、削波比例、GCC-PHAT 双耳时差、峰质量及前后两组候选。RMS 用于判断有无有效信号，深度用于评估视线内表面或通行，声源方向与距离保留不确定性。静音或低质量时相应线索标为不可用。
- 记忆只由已经收到的观测和已经执行的动作构成。首版用固定长度的近期窗口和显式的前后候选历史。引用格式沿用 obs_XXXX:rgb、obs_XXXX:depth、obs_XXXX:audio、obs_XXXX:pose。
- 模型输入不携带 episode 配置、目标类别、目标实例、原始声源文件、目标坐标、真实距离、Oracle 轨迹、离线观察位姿或观测文件路径。模型后端不开放工具、文件系统、Docker 或评估目录；JSONL 的普通观测本身不能替代进程权限隔离。

模型每次返回以下字段。协调器在真正执行前校验当前观测 ID、动作枚举、理由长度及证据引用是否属于本次实际发给模型的输入。证据 ID 通过校验只代表引用存在。

    {
      "observation_id": "obs_0000",
      "action": "move_forward",
      "reason": "当前前方通行空间充足。",
      "evidence_ids": ["obs_0000:rgb", "obs_0000:depth"]
    }

## 两个实验组

两组使用同一模型、输入格式、动作集合和实验配置。首版沿用关闭专门扫转的设置。普通左右转向仍然是导航动作，移动后自然取得的观测可用于更新前后假设。

| 实验组 | 决策与 STOP |
| --- | --- |
| 普通 LLM Agent | 通过协议校验的单步动作直接执行。保留模型提出 STOP 所依据的观测。 |
| 证据验证 LLM Agent | 共用普通组的协议校验。执行 STOP 前，从已收到的观测核查是否有跨位姿的音频支持，是否仍有竞争的方向或远距解释，深度是否与该方向及近距假设相容。若证据不足，把缺项交给模型在同一观测上重选，并分别记录提议和执行动作。 |

两组都保留前后假设。看见物体只能证明它在画面里，近处深度只能证明该方向有近处表面。判断候选是否有发声证据，需要观察音频线索随位姿变化是否与候选方位一致。静音或几何退化时可以维持未知。实现时可复用规则基线中可观测的 GCC-PHAT 和前后拟合计算，验证模块独立于 AudioDepthAgent.decide 的动作选择。

## 评估与实现顺序

1. 实现宿主机协调器、观测适配器与可替换的模型后端。先用固定响应测试合法动作、过期观测 ID、虚假证据 ID、超时和终止。Codex CLI 已验证可做接入测试。批量实验前要单独验证可直接调用、可返回结构化视觉结果且不开放工具的推理后端；CLI 连通结果不验证这条 API 路径。
2. 在可信侧接上 episode 评估。当前 serve 的 step/close 不返回评价，close 也不写 evaluation.json。正式运行需在可信侧持有的环境结束时单独调用现有 evaluate 并私下落盘，处理超时和中断，JSONL 响应继续仅包含观测和完成状态。沿用现有评估口径，主动 STOP 且终点测地距离小于 1 m 为成功；另报最小距离和进入 1 m 后继续行动的次数。轨迹指标只由评估侧计算。
3. 用开发任务 1000000016、1000000000 和 1000000008 先跑两组，仿真种子 7、64 步上限。三条任务源自 v1.1 筛选，使用官方物体与声音关联。G1 标签表示待复核的视觉候选歧义。记录主动 STOP、误停、成功率、预算耗尽、步数、碰撞、推理时延及前后判断与评估侧相对方位的差异。区分模型随机性与仿真种子，保存模型版本、提示词版本、观测 ID 和模型提议与执行动作。

## 云端接入验证

在 polyu_serve 上使用 Codex CLI 0.167.0 和 gpt-5.6-sol。纯文本请求返回 CODEX_CLOUD_OK，退出码为 0。文本和下述图像请求均未出现 model metadata 或 fallback 警告。

    codex exec -m gpt-5.6-sol --sandbox read-only \
      'Reply with exactly CODEX_CLOUD_OK. Do not call any tools.'

在 savnce_dev 中对 1000000008 运行 reset、observe、step、close。使用种子 7、64 步上限和 256×256 分辨率。重复 observe 返回同一观测。宿主机从 inline 观测解码首帧，向 Codex CLI 传入该帧 RGB、前方深度中位数、双耳 RMS、位姿和运动标志，以 --output-schema 约束返回结构。模型返回 move_forward，引用 obs_0000:rgb 和 obs_0000:depth。协调器校验后执行，收到 obs_0001，仿真时间推进 0.25 秒，正常关闭。首帧音频为静音，模型请求约 18.4 秒。

公共记录位于 outputs/llm_interface_smoke_d7d5b2a9f6/actions.jsonl。这次测试覆盖一次真实观测到模型动作的往返。以下是后续实现和首轮对照的记录。

## 已实现的运行入口

宿主机运行 agent_episode/llm_rollout.py。它启动容器内的 JSONL 环境与观测处理进程，使用现有网关的 Responses 接口调用模型。每次请求设置 tools 为空、store 为 false，并要求返回严格 JSON Schema。可在运行环境中提供 SAVNCE_API_BASE_URL 和 SAVNCE_API_KEY。密钥和数据路径不进入提示词或公开日志。

    python3 agent_episode/llm_rollout.py --policy ordinary \
      --episode-id 1000000016 --model gpt-5.6-sol --seed 7 --steps 64 \
      --output outputs/my_llm_ordinary_1000000016

将策略参数换为 evidence 运行证据验证组，为每次运行指定新的相对输出路径。仿真随机种子固定为 7，推理 effort 为 low。模型调用本身没有固定随机种子。

LLM 每步收到两张 PNG。第一张是当前 256×256 RGB，第二张是从同一步的 16 kHz 双耳观测计算出的 500×350 声学证据图。图中是共享 dBFS 标尺的左右 STFT 和带符号的 GCC-PHAT 延迟曲线。并行的数值字段包含双耳 RMS、ILD、ITD、削波、互相关峰质量和前后两组候选。正 ITD 表示左耳先到。图和数值只来自当前与先前实际收到的观测。模型不接收 WAV、干净声源或评估真值。左右声谱图便于检查是否有有效信号，时延数值用于方位推断，单步线索保留前后歧义。

证据组对模型提出的 STOP 执行被动复核。它使用最近 24 条有效音频观测，筛选位置至少相隔 0.1 m 的视点并比较前后假设。现有拟合复用多视点支持、移动基线和视差阈值。验证还要求竞争方向和远距离拟合的角误差高出至少 3°，并由当前视向内的深度支持近处表面。已取得的证据最长保留 8 个仿真步，STOP 时要求整个估计水平距离区间不超过 0.9 m。若证据不足，由模型在同一帧选择一个移动或转向动作。两组均关闭专门扫转。

serve 的 --finalize-rollout 选项在可信侧记录测地距离，在 close 时输出双耳导航视频和私有 evaluation.json。JSONL 返回结构保持原样。已验证静音无方位、人工延迟 3 个样本为 +187.5 µs、普通 STOP、STOP 否决后的重选、通过核验的 STOP 以及拒绝过期观测 ID。两步真实运行也通过了模型请求、动作执行和视频导出验证。

首轮运行位于 outputs/llm_stop_comparison_seed7_64steps。普通组在 1000000016 主动 STOP，测地距离 1.319 m。证据组在同一任务拒绝过一次 STOP，继续行动。其余五个运行被网关 HTTP 401 Invalid token 中断，保留了截至中断的输出，尚无法计算两组在三条任务上的成功率。
