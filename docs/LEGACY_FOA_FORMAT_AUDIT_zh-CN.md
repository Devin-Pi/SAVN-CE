# 旧 audio encoder 的 FOA 制式核查

本次核查在 polyu_serve 执行。未训练模型，未修改声音、RIR、权重或 Agent 策略。后续按用户要求停止继续追索缺失的旧 RIR。

结论：**旧加载器的 ACN `[W,Y,Z,X]` 声道约定，以及“+X前、+Y左、+Z上”的方向标签约定，与新数据一致。TAU RIR 的 SN3D 已有明确证据；整个历史训练池仍不能统一确认为 SN3D，因为 METU 的旧 FOA 文件缺失。没有证据要求把新 AmbiX 数据额外乘 √3 或重新换序。**

## 证据链

1. 当前使用的 epoch17 权重与 `/media/8tbhdd/rdpi/workspaces/audio_bb_mamba_15_MB_RAW/checkpoints/epoch17_val_F0.8022_val_seld_scr0.1549.ckpt` SHA256 完全一致：`85e380a6a8d2f1e545aa093e3f086f0a4d7830a15c41f7f79d047d7d9c1f9f90`。原模型代码与当前严格加载的模型代码 SHA256 也一致。
2. 检查点超参数指向 `/workspaces/SpatialScaper/output/`，epoch17、global_step=517860；归档日志的 epoch17 记录 F=0.802172、LE=9.785631°、SELD=0.154884，与权重名称相符。这些是历史日志结果，不是此次重新评估。
3. 原加载器读取 `list_dataset/`，不是旁边的 `list_dataset_raw/`。归档的实际解析清单为 train960条：SC203180、SE203184、METU596；validation400条：SC203100、SE203100、METU200。1559条的扩展清单含 ARNI，但不在该加载器解析的960条清单中。清单文件是归档快照，检查点没有逐数据文件校验和，因此不能证明每条音频的历史消费状态。
4. TAU 原始 `rirdata.mat` 明确保存 `foa_norm='sn3d'`、`foa_channel_order='wyzx'`、`fs=24000`，元数据 SHA256 为 `9b64199f6fe16234208576ebc8f896972cf07992070384990d019e468e13dde8`。来源为[官方 TAU-SRIR 发布库](https://zenodo.org/records/6408611)。4个房间162条 RIR 的独立直达声窗口检查也支持该声明：方向分量/W的相干增益中位数约0.96–0.98，符合SN3D附近的实测响应，而非理想N3D的√3。实测阵列滤波、近场和早期反射会影响数值，不能要求每条都精确等于1。
5. 归档准备代码直接保留 TAU 的 FOA 通道。`IR_normalizer` 对同一RIR的四通道使用共同能量因子，事件电平与最终峰值缩放也为共同因子；没有观察到 SN3D→N3D 的逐阶增益转换。`SpatialScaper_` 副本缺少 `sofa_utils.py`，相应辅助代码从另一个 SpatialScaper 归档核查；历史处理后的 SOFA 已不在，无法逐字节核对旧 SOFA。
6. 原 `swap_audio_channels` 与原标签 CSV 转换完成1952项对照，最大误差约5×10⁻¹⁶。确认存储顺序、角度正负、XYZ语义及8种增强的一致性。中间三个 intensity-vector 变量的名字不等于实际重新换序；读取顺序保持原ACN顺序。
7. 保留的 TAU 训练 WAV `fold1_room1_mix001.wav` 与原训练清单、标签匹配。277个单源内部帧的相干方向测量，在48种有符号排列中，最优映射是 `XYZ=stored[3,1,2]`，无变号，方位误差中位数约7.76°。混响会降低相干增益，此结果用于声道/方向佐证，不能拿整段混响音频的幅度比当作归一化证明。

## METU 未核实的具体原因

`metu_foa.sofa` 在云端搜索范围内未找到。归档 `prepare_metu()` 只生成原始阵列的MIC格式；FOA当时依赖[SpatialScaper 官方 README 提供的独立编码文件](https://github.com/iranroman/SpatialScaper#preparing-rir-datasets)，其旧 Google Drive 链接目前返回404。现存32路原始胶囊IR不能自动等同于当年的FOA编码结果。

保留的 `fold2_room4_mix006.wav` 只有6个有标签的100ms帧，其中满足前后邻帧均单源的仅4帧。该角度范围接近X/Y对称，无法唯一判定通道排列或SN3D/N3D；不能用这几帧宣布整个 METU 库已确认。需要恢复当时的 `metu_foa.sofa`、明确的编码配置或更充分的带标签旧音频才可完成这一部分。

## 对新数据与后续训练的影响

| 项目 | 旧 encoder 证据 | 新数据 | 判断 |
|---|---|---|---|
| 声道顺序 | 原加载器及TAU训练音频为W/Y/Z/X | 已独立校准W/Y/Z/X | 接口约定一致 |
| 方向坐标 | +X前、+Y左、+Z上；方位+90°向左 | 接收器局部同样约定 | 接口约定一致；新数据已从世界系转到接收器系 |
| TAU归一化 | 原始元数据明确SN3D，转换保留 | SN3D | 一致 |
| METU归一化 | 缺历史FOA文件 | SN3D | 未能完成全库确认 |
| 采样率/前端 | 24kHz，Mel上限12kHz | 16kHz，Mel上限8kHz | 仍需使用已交付的16k前端 |
| 输出类别 | 原13类声事件 | AVN官方21类 | 必须重建输出头并重新适配 |

控制实验把方向声道共同乘√3、W保持不变，原模型输出平均绝对变化约9.65×10⁻⁵、最大约0.00413。原模型的窗口内特征标准化可消除很多常量增益，但并非完全数值不变；这项实验不能反推旧训练制式，也不能作为额外改写新数据的理由。

按最新任务方向，保留新数据的标准 ACN/SN3D。训练增强的建议另见 `TRAINING_AUGMENTATION_zh-CN.md`。类别改变要求重建21类输出头；全encoder重新适配可采用旧权重初始化后全网络微调，无需把“重新训练”理解为必须从随机权重开始。

详细文件、哈希、测量结果和限制保存在 `legacy_foa_format_audit.json` 与 `legacy_format_evidence/`。
