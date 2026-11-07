# AVN-FOA-SameRoom-v0.2-Dense15h

当前状态：新生成器和Docker镜像已实现，云端9项测试通过，45条覆盖21类的试验样本已生成。后台流程已启动；随后本机会话到云端SSH连接超时，**全量5400条是否完成及最终验收状态尚未核验，不宣称15小时交付完成**。恢复连接后以实际validation_report.json和SHA256SUMS为准。

## 本次改动

目标为5400条10秒FOA记录，总计15小时：train 4320条/12小时，validation和test各540条/1.5小时。21个官方类别，16kHz、ACN/SN3D W/Y/Z/X、同房间LOS、静态单声源的协议保持一致。

保留10%真无事件记录，正样本要求至少92/100个真实监督活动帧，故全网格活动率理论下限为82.8%。分母包含纯无事件和ignore帧，不通过排除困难帧把比例做高。每个完整split和每类都独立检查比例。

原始素材本身含大量静音，例如table/picture/cushion连续活动段最长不足0.5秒。新生成器从原WAV的真实非静音区间随机选片段、逐次重新触发并作5ms交叉淡化，记录精确源区间和发声偏移；不把静音标成活动，不把整WAV机械tile，也不做时间拉伸。高密度调度会改变原始节奏，不能称为自然事件时间分布。

完整干声时间线一次性与位置对应的真实FOA RIR卷积。保留传播延迟和录音内尾声，标签由源活动到达、100ms占用率和实际波形RMS共同重算，边界和仅尾声采用ignore。

先固定已验收声场来检验活动密度和总时长改动，复用1080条经哈希核对的原生16k SoundSpaces RIR。新增独立RIR数为0，原33/7/7建筑划分不变。15小时指合成记录时间轴，包含片段再利用及1.5小时真静音对照，不是新增15小时独立原始声音或5400个独立声场。

每类仍只有一个官方训练WAV家族，独立原始录音数未知；known-sound/unseen-scene协议不变。官方验证/测试资源没有加入训练。旧10小时数据和第95轮权重保留，后续需同时在原稀疏验证/测试上检查性能及误报代价，不能仅根据高活动新test下结论。

## 云端目录

```text
项目：/media/4tbssd/00_PROJECT_CODE/SAVN-CE
代码：outputs/avn_foa_dense15h_worktree
试验：research_outputs/AVN-FOA-SameRoom-v0.2-Dense15h_pilot
目标：research_outputs/AVN-FOA-SameRoom-v0.2-Dense15h
日志：research_outputs/AVN-FOA-SameRoom-v0.2-Dense15h_assets
```

生成镜像：`rdpi/avn-foa-sameroom:dense15h-v0.2`；基础镜像为现有`rdpi/avn-foa-sameroom:0.1`。新Dockerfile及专用dockerignore避免修改原镜像的构建范围。CPU生成采用4进程、4CPU及12GiB限制；旧项目和资产只读挂载，仅新输出目录可写。

独立CUDA前向使用`rdpi/avseld_final:latest`，此主机需要显式`--runtime=nvidia --gpus all -e CUDA_VISIBLE_DEVICES=2`；已通过CUDA可用性探测。该检查仅读数据、恢复旧权重并测试形状，不执行optimizer或新训练。

## 执行和验收

实际后台脚本：`avn_foa_sameroom/run_dense15h.sh`。流程为镜像构建、9项测试、45条试验、DSF前向、独立试验验收、全量生成、全量前向与验收、报告和哈希封存。

脚本已在云端启动。**恢复连接后先查看状态，不要重复启动已有流程或覆盖已有输出。**

```bash
ssh polyu_serve
cat /media/4tbssd/00_PROJECT_CODE/SAVN-CE/research_outputs/AVN-FOA-SameRoom-v0.2-Dense15h_assets/pipeline.log
```

只有最后出现`DENSE15H_ALL_STAGES_COMPLETE`，且目标目录的`validation_report.json`为passed、records=5400、hours=15、active_grid_fraction>=0.8并生成SHA256SUMS，才能报告交付完成。

独立检查逐文件格式、有限值、峰值、哈希、NPZ/CSV、每条真实干声时间线及活动/ignore标签，并对每个RIR至少重建一条完整卷积波形。几何和官方关联明确复用旧版验收证书，不冒充新渲染。

本次未启动新的完整模型训练，没有修改Agent或STOP。数据比例达标不等于localization expert性能已提升。
