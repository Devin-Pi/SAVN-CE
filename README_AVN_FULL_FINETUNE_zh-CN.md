# AVN-FOA 21 类 audio encoder 全网络微调

本实验遵循完整 100 epoch、固定配置、训练结束后统一分析的协议。它使用 AVN-FOA-SameRoom-v0.1 的现有 21 类声音数据，从旧 audio-only 检查点初始化，替换最后分类/方向联合输出层，从第一个 batch 解冻全部网络。仅训练数据使用水平 yaw 增强。原模型、数据和 Agent 接口保持独立，不自动切换线上定位工具。

## 云端运行位置和命令

SSH：`polyu_serve`；Docker：`avseld_final_test`；GPU：物理 GPU 2。

宿主机实验根目录：

```text
/media/4tbssd/av_seld/workspaces/avn_foa_full_finetune_v1
```

正式 100 轮输出：`run_seed7_full100/`；日志：`train_full100.log`；命令记录：`run_finetune_full100.sh`；完成后的核验日志：`final_checks_full100.log`。

实际训练命令（由宿主机执行；输出目录必须尚不存在，以免覆盖旧实验）：

```bash
bash avn_foa_sameroom/run_finetune.sh
```

脚本中的完整命令：

```bash
docker exec \
  -e PYTHONPATH=/workspaces/avn_foa_full_finetune_v1/code \
  -e CUDA_VISIBLE_DEVICES=2 avseld_final_test \
  python -u -m avn_foa_sameroom.finetune \
  --data /workspaces/avn_foa_full_finetune_v1/data \
  --runtime /workspaces/dsf_localization_runtime_v1 \
  --checkpoint /workspaces/av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt \
  --output /workspaces/avn_foa_full_finetune_v1/run_seed7_full100 \
  --loss standard --batch 16 --workers 4 --epochs 100 --seed 7 \
  --encoder-lr 1e-5 --head-lr 1e-4 --patience 0
```

训练正常结束后，`finish_finetune.sh` 核验 100 轮和 18,000 次更新的完整性、按验证集选择权重、生成结果报告、执行纯 WAV 推理，并检查原始数据及权重的完整性。控制脚本只监测进程是否存在，不按逐轮指标改动训练。出现运行错误会保留日志并失败退出，不冒充训练完成。

## 参考的原训练方式

参考云端归档 `/media/8tbhdd/rdpi/workspaces/audio_bb_mamba_15_MB_RAW/`：

- `main.py:208`：Adam，原学习率 1e-4；CosineAnnealingLR，T_max 为 max_epochs，eta_min=0，按 epoch 更新。
- `config/config_mamba_15_raw.yaml`：FP32，batch 16，归档 max_epochs=30，13 类，macro/20°。
- `models/AVmodel.py:18`：标准 ADPIT MSE。
- `dataloaders/dataloader.py`：原训练集有 8 种 ACS 变换。

本次采用标准 ADPIT MSE、Adam、FP32、按 epoch 余弦衰减，无 warmup、无 weight decay、无梯度裁剪、无提前停止。保留原 trainer 的 `torch.set_float32_matmul_precision('medium')`，即 FP32 张量并允许硬件 TF32 矩阵乘。没有使用 AMP。

与原实验的必要差异：完整 100 epoch；21 类；最终 Linear 从 128→117 换为 128→189（3 轨 × XYZ × 21 类）；旧 encoder 学习率 1e-5，新输出层 1e-4；原生 16 kHz 前端；ignore 帧不监督；水平连续 yaw 替代原 8 种包含反射/翻转的 ACS。没有将 yaw 增强宣称为扩增了独立录音时长。

`legacy_training_reference.json` 保存归档文件 SHA256 和已核查参数，不复制旧日志服务密钥。当前损失是原 ADPIT 对单声源标签的等价简化；12 组原实现数值与梯度对照通过。它会拒绝多源标签，不应直接用于未来多源训练。

## 数据与评估

数据：3600 条 10 秒、16 kHz、ACN/SN3D W/Y/Z/X 音频，总计 10 小时。训练 2880 条/8 小时；验证及测试各 360 条/1 小时。每个 epoch 完整读取训练集一次，不对验证/测试增强。

监听者坐标 +X 前、+Y 左、+Z 上。被动左转 yaw α 时：X′=cosα X+sinα Y，Y′=−sinα X+cosα Y。波形 X/Y 与标签 XYZ 同步旋转，W/Z、类别、活动与 ignore 不变；随机角由 seed、epoch、样本索引决定。

每轮仅用验证集计算原 DSF macro SELD，阈值 20°、1 秒块、活动阈值固定 0.5；原三轨合并逻辑和 15° 阈值保留。按验证 SELD 最小保存 `best.pt`，SELD 平局按验证 loss。`last.pt` 保存末轮与优化器状态。训练 100 轮全部结束后，才使用 `best.pt` 在 test 上运行一次模型；分类报告从已保存预测复算，不重复运行模型或用 test 调参。

结果文件由真实运行生成，不能把任务启动当作结果：

| 文件 | 含义 |
|---|---|
| `config.json` | 固定超参数、旧权重/数据/代码哈希 |
| `history.jsonl` | 完整逐 epoch 训练与验证记录 |
| `step1_gradient_audit.json` | 解冻与第一步梯度核查 |
| `step1_parameter_changes.json` | encoder 与新头第一步实际更新 |
| `best.pt`, `last.pt` | 验证集最佳及末轮检查点 |
| `results.json` | 完成状态、最佳轮、验证及最终 test 指标 |
| `test_predictions.npz`, `test_per_class.csv` | 最终 test 原始预测及分类指标 |
| `completed_run_validation.json` | 100 轮、18,000 步及选择逻辑核验 |
| `posthoc_test_metric_check.json` | 从保存预测复算指标的一致性 |
| `inference_smoke.json` | 只读验证 WAV 的推理测试 |
| `README_zh-CN.md` | 训练结束后自动生成的结果分析 |

本数据采用 known-sound/unseen-scene 协议，每类只有一个官方训练 WAV 家族。分割场景不能证明对未见独立音源泛化。指标也不能直接与原 13 类 DCASE 预训练或论文的视听 SELD=0.354 比较，更不是导航 Agent 成功率。

此前短程试跑保留为开发记录，正式 full100 不从它们恢复。正式实验过程中只检查进程、GPU、非有限数值和错误退出，不据逐轮表现改超参数。结果好坏均等待完整 100 轮结束再分析。

## 测试及推理

```bash
python -m unittest avn_foa_sameroom.test_finetune -v
python -m avn_foa_sameroom.verify_legacy_loss \
  --source /tmp/avn_original_ADPIT_source.py --output /tmp/adpit_equivalence.json
python -m avn_foa_sameroom.infer_finetuned \
  --checkpoint <best.pt> --runtime /workspaces/dsf_localization_runtime_v1 \
  --wav <10s_16k_4ch_AmbiX.wav> --output <prediction.json>
```

推理仅读取 WAV，不读取目标物体、正确方向、RIR 或参考轨迹。模型输出活动、类别、三维方向，不预测距离。旧 encoder、归档架构、Mamba CUDA 扩展和 Docker 环境仍是运行依赖；本任务没有将这些归档打包成新的通用 Docker 镜像。
