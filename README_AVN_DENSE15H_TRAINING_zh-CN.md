# AVN Dense15h v0.2：新一轮全网络微调

本轮在 `polyu_serve` 上独立启动。原10h数据、原13类预训练权重、原95轮训练结果均保留。

- 数据：`research_outputs/AVN-FOA-SameRoom-v0.2-Dense15h`，5400×10秒，15h，16kHz，ACN/SN3D W/Y/Z/X；整体活动网格占比87.42%。train4320条（12h），validation/test各540条（1.5h）。保留10%全静音负例。
- 初始化：原13类检查点严格加载，替换最后线性层为21类×3轨×XYZ，共189维；从第一步解冻整个encoder。**从旧预训练初始化，不接续原95轮模型。**
- Adam：encoder学习率1e-5，新输出头1e-4；batch16、workers4、seed7；每epoch余弦衰减；标准单源ADPIT MSE；不加类别重加权、warmup、AMP或梯度裁剪。有效matmul精度明确为high，与原DSF构造器实际设置一致。
- 100 epoch，每epoch270次更新，总计27000步；禁用自动早停。保持配置固定，全部结束后统一分析。
- 仅训练时做连续水平yaw增强，FOA及XYZ方向同步旋转；validation/test不增强。
- **只用新版dense validation SELD选择最佳权重**，同分时按validation loss。每轮另记录原v0.1 sparse validation指标，作为时间先验变化的诊断，不参与选模。
- 训练完成后，最佳模型分别对新版dense test（540条）和原sparse test（360条）进行一次模型评估。保存原始预测，再计算分类指标和静音误报；测试不选模、不调阈值。

## 环境与可复现命令

容器 `avseld_dense15h_train_v02`，镜像 `rdpi/avseld_final:latest`，GPU2（RTX3090），`--runtime=nvidia --gpus all -e CUDA_VISIBLE_DEVICES=2 --shm-size=4g`。

镜像ID：`sha256:997f3dd6336af585a688ab93d6dbba8d97c30e5e89dbec0a0a8aef08ef33631a`。
数据seal SHA256：`c4d77d6ea96d125aca67e7ae4a5c19de4f90676055a9c4234f4162eb8b827c65`。
初始化权重SHA256：`85e380a6a8d2f1e545aa093e3f086f0a4d7830a15c41f7f79d047d7d9c1f9f90`。

宿主机实验根目录：

```text
/media/4tbssd/av_seld/workspaces/avn_foa_dense15h_finetune_v02/
  code_snapshot/           # 启动前冻结的实际训练源代码
  launch_manifest.json     # 代码/数据/初始化权重哈希、镜像及预算
  smoke_batch16/           # 独立启动检查，正式训练重新初始化
  run_seed7_full100/       # config/history/best.pt/last.pt及最终结果
  monitor/                # 独立SwanLab读取进程状态
  logs/smoke.log
```

数据、旧runtime、旧权重和训练源代码均以只读挂载；新结果写入实验目录。启动脚本拒绝覆盖已存在的正式运行目录。

实际执行的入口（宿主机）：

```bash
experiment=/media/4tbssd/av_seld/workspaces/avn_foa_dense15h_finetune_v02
bash "$experiment/code_snapshot/avn_foa_sameroom/launch_dense_finetune.sh" smoke
bash "$experiment/code_snapshot/avn_foa_sameroom/launch_dense_finetune.sh" train
```

实际容器内命令及完成后验证入口见 [run_dense_finetune_inside.sh](avn_foa_sameroom/run_dense_finetune_inside.sh)。正式训练命令为：

```bash
PYTHONPATH=/training_source python -u -m avn_foa_sameroom.finetune \
  --data /datasets/main --reference-data /datasets/reference \
  --runtime /workspaces/dsf_localization_runtime_v1 \
  --checkpoint /workspaces/av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt \
  --output /experiment/run_seed7_full100 \
  --epochs 100 --batch 16 --workers 4 --seed 7 \
  --encoder-lr 1e-5 --head-lr 1e-4 --loss standard --patience 0
```

查看运行状态，不修改训练：

```bash
ssh polyu_serve 'docker logs --tail 5 avseld_dense15h_train_v02'
ssh polyu_serve 'docker inspect avseld_dense15h_train_v02 --format "{{.State.Status}} {{.State.ExitCode}}"'
```

## SwanLab

[FOA21-Dense15h-full100-seed7](https://swanlab.cn/@rdpi1995/AVN-Localization/runs/avnfoa-dense15h-seed7-18f0d9abba1f)

复用宿主机独立SDK环境及权限600的私有key文件，不将凭证写入代码、训练参数或Git。只读追踪`history.jsonl`，监控进程断线会重试，不向训练进程发信号。

`val/*`为选模用dense validation；`reference_val/*`为旧sparse validation。训练结束后的`test/*`与`reference_test/*`分别为两套测试指标。epoch不等于optimizer step；SwanLab横轴为epoch。自动硬件历史从监控启动时开始，没有补造历史GPU数据。

监督脚本 [run_dense_swanlab.sh](avn_foa_sameroom/run_dense_swanlab.sh) 以真实训练Python的宿主机PID启动，而非容器init/shell或DataLoader worker的PID。在线读回校验脚本为 [verify_dense_swanlab.py](avn_foa_sameroom/verify_dense_swanlab.py)。

## 验证与边界

启动检查已在同镜像和同挂载下运行：5项yaw/损失测试、4项JSONL/监控映射测试；batch16前向/反向、有限梯度、encoder与新输出层均实际更新，峰值分配显存约15.2GiB。训练用独立初始化，不接续该检查的参数。

另有3项完整预算记账回归检查已通过：27000步接受、旧18000步拒绝、非整batch向上取整。

结束后自动执行完整100轮记账检查（根据数据量计算27000步，拒绝旧的18000步预算）、保存预测重算DSF分类指标，以及仅输入WAV的推理检查。运行中不会宣称这些完成检查已经通过。

新增数据沿用原场景/RIR和21个官方WAV家族，主要增加活动密度及时间组合；不是新的独立录音或声学几何。两版reference测试已提供过开发反馈，不是原封未触及的官方最终测试集。协议为known-sound/unseen-scene、同房间LOS静态单源、10秒输入；不代表在线短窗口、NLOS、多源、移动源或导航Agent成功。训练完成前不宣称已获得可靠localization expert，也不自动替换Agent模型。

## 100 epoch 已完成（2026-11-08）

完整完成27000步，按新版validation SELD选择第97轮。主结果采用新版540条test：SELD=0.018637、F20=0.982210、LE=4.9149°。上一轮模型在同一新test补测：SELD=0.805789、F20=0.146912、LE=124.6751°。

旧sparse test只作跨分布诊断；新模型在该集明显退化。整段全静音负例误报保持为0，事件录音中的有效无活动帧误报仍需关注。完整21类、2×2交叉评估及限制见[最终分析](reports/avn_foa_dense15h_full100/ANALYSIS_zh-CN.md)，结构化结果和CSV同目录。没有自动开始新训练或替换Agent模型。
