# 当前 100 epoch 训练的 SwanLab 独立监控

已接入私有实验：[FOA21-full100-seed7](https://swanlab.cn/@rdpi1995/AVN-Localization/runs/avnfoa-full100-seed7-70ba503aa507)。账号需要有项目访问权限。

监控程序只读取原始训练的 `history.jsonl`、`config.json`，训练结束后读取 `results.json`。它不导入 Torch、不加载权重、不访问训练音频，不修改训练配置、代码、优化器或数据，也不向训练进程发送信号。训练保持完整 100 epoch 和原验证集最佳 SELD 选择规则。

## 实际部署

- 主机：`polyu_serve`。
- 训练容器：`avseld_final_test`；训练主进程 PID：`978563`。
- 独立采集环境：宿主机 Python 3.10 虚拟环境，SDK `swanlab==0.10.4`。
- 采集根目录：`/media/4tbssd/av_seld/workspaces/avn_swanlab_monitor_v1/`。
- 只读训练目录：`/media/4tbssd/av_seld/workspaces/avn_foa_full_finetune_v1/run_seed7_full100/`。
- 凭据：`/home/rdpi/.config/avn-swanlab/api_key`，目录权限 `700`、文件权限 `600`，位于 Git 和训练输出目录之外。

启动命令（已在云端执行，不要重复启动）：

```bash
nohup bash /media/4tbssd/av_seld/workspaces/avn_swanlab_monitor_v1/run_swanlab_monitor.sh \
  > /media/4tbssd/av_seld/workspaces/avn_swanlab_monitor_v1/supervisor.log 2>&1 < /dev/null &
```

`run_swanlab_monitor.sh` 通过独立 PID 和锁文件守护采集器。异常退出会在 30 秒后尝试恢复同一个 SwanLab run；正常完成、主动停止或数据/进程身份验证错误会退出。它由 nohup 保持 SSH 断开后运行，不依赖登录会话，也不更改服务器的 systemd linger 配置。主机重启后的训练恢复需要人工核验新 PID 后再接入。

## 指标和补录方式

原始历史按 epoch 补录，之后每 10 秒检查新记录。横轴 step 就是原 epoch，既不重新计算指标，也不评价逐轮表现。

| 分组 | 记录内容 |
|---|---|
| `train` | epoch 平均 loss |
| `val` | ER、F20、LE、LR、SELD、验证 loss |
| `lr` | encoder 和新头各自学习率 |
| `training` | epoch 和 global step |
| `timing` | 单轮耗时、原训练累计耗时 |

共 13 条自定义指标。文件末尾未写完的 JSON 行会在下一轮重试；相同 epoch 去重、冲突或缺轮报错。重连使用固定 run ID，并从完整历史重放，避免把“本地已入队”当作“云端已收到”；同一指标以原 epoch 对齐。

训练完整结束后才上传最终 test 指标和验证集最佳 epoch，并正常结束 SwanLab 实验。如果训练进程消失且 120 秒内没有完整结果，会把采集实验标为错误，保留日志，不尝试重启训练。

GPU 硬件数据从采集器启动时开始记录，不能补回以前没有采集的 GPU 历史。训练使用物理 GPU 2；自动硬件监控可能显示宿主机其他 GPU。独立采集器的进程 CPU/内存不是训练进程的 CPU/内存。

## 已执行核验

- 4 项无网络单元测试通过：末行部分写入、重复/冲突记录、缺轮拒绝、原值映射及非有限值拒绝。
- 从 SwanLab 服务端读取 13 条曲线，与原始 JSONL 逐点对照；首次核验覆盖 46 个 epoch，包括最初补录的 45 个 epoch 和之后新增的 1 个 epoch。
- 原训练 PID、进程启动标识、命令、配置 SHA256 和训练代码 SHA256 前后一致。
- 扫描采集输出，未发现凭据原文；凭据未写入源码、Git、训练日志或采集日志。

这些核验只确认监控接入和训练未被改动，不构成对尚未完成的 100 epoch 实验的性能分析。

云端验证：

```bash
/media/4tbssd/av_seld/workspaces/avn_swanlab_monitor_v1/venv/bin/python \
  /media/4tbssd/av_seld/workspaces/avn_swanlab_monitor_v1/verify_swanlab_monitor.py \
  --monitor-root /media/4tbssd/av_seld/workspaces/avn_swanlab_monitor_v1 \
  --owner rdpi1995
```

`verify_swanlab_monitor.py` 用于训练仍运行期间核验服务端补录、增量更新和原训练进程身份；训练结束后应以完整的 `results.json` 和采集结束状态核验。

## 查看状态

```bash
ssh polyu_serve
cat /media/4tbssd/av_seld/workspaces/avn_swanlab_monitor_v1/state/status.json
tail -n 30 /media/4tbssd/av_seld/workspaces/avn_swanlab_monitor_v1/monitor.log
```

`last_enqueued_epoch` 只代表已经交给 SDK 的轮数。`verification.json` 记录服务端实际回读验证覆盖的轮数。诊断文件及 SDK 日志保存在采集目录，不混入训练目录。

## 安全更换凭据

在云端交互终端输入，输入内容不会显示，也不作为命令参数：

```bash
umask 077
mkdir -p /home/rdpi/.config/avn-swanlab
chmod 700 /home/rdpi/.config/avn-swanlab
read -r -s -p 'SwanLab API Key: ' AVN_SWANLAB_KEY
printf '\n'
printf '%s\n' "$AVN_SWANLAB_KEY" > /home/rdpi/.config/avn-swanlab/api_key
unset AVN_SWANLAB_KEY
chmod 600 /home/rdpi/.config/avn-swanlab/api_key
```

更换后需要仅重启独立采集器。密钥不写入代码或公开命令；重启仍复用同一个 SwanLab run ID。当前密钥已经配置完成，不需要再次提供。

## 环境复用

实际安装记录为采集根目录的 `install.log`，精确依赖列表为 `requirements.lock.txt`。宿主机缺少 ensurepip，本次使用现有系统 pip 引导独立虚拟环境，未安装或升级训练容器中的任何包：

```bash
monitor_root=/media/4tbssd/av_seld/workspaces/avn_swanlab_monitor_v1
python3 -m venv --without-pip "$monitor_root/venv"
PYTHONPATH=/usr/lib/python3/dist-packages "$monitor_root/venv/bin/python" \
  -m pip install --ignore-installed pip swanlab==0.10.4
```

以上是环境重建记录，不需在当前运行中重复执行。
