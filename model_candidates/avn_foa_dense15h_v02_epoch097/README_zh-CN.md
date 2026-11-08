# 定位候选：avn_foa_dense15h_v02_epoch097

状态：**已归档的定位候选，尚未激活为Agent默认模型**。

云端权重：`/media/4tbssd/av_seld/workspaces/localization_candidates/avn_foa_dense15h_v02_epoch097/best.pt`。
归档为完整的原best.pt副本，保留原文件；SHA256见candidate.json。Git仅保存候选说明、指标与归档脚本，不包含权重。

输入：10秒、16kHz、4声道ACN/SN3D W/Y/Z/X、听者局部坐标FOA。输出：21类三轨ACCDOA，100ms一帧；方向属于听者坐标，不输出距离，活动分数未校准。沿用原DSF阈值0.5及15度轨道合并。

完成100 epoch，使用新版validation SELD选择第97轮。主测试集为新版540条（1.5h）：SELD=0.018637，F20=0.982210，LE=4.9149度，LR=0.995445。

适用于当前高活动、同房间LOS静态单源、已知声音家族/未见场景的研究协议。旧稀疏test明显退化（SELD=0.950408），不能直接视为短暂/间歇事件的通用定位专家。全静音负例零误报不代表能可靠处理事件间隙或现实背景。

下一步建议先在独立的FOA适配层核验输入窗、坐标、时间戳和输出接口，再做在线shadow评估；仅把Agent允许取得的音频和位姿用于推理，真值留在评估侧。是否替换Agent或用于STOP，应在上述验证后另行决定。本次仅归档，不自动进行该阶段。

完整21类和2×2交叉比较见[评估报告](../../reports/avn_foa_dense15h_full100/ANALYSIS_zh-CN.md)。复现推理入口：`avn_foa_sameroom.infer_finetuned`，使用checkpoint、runtime、WAV、output四项参数；不向模型提供目标真值。

## 在线shadow补查

2026-11-08在1000000008完成64步、8个10秒窗：未检出chair，在线效果未通过，候选不进入导航控制。共享转换、接口与版本范围核验详见[实验设计](../../EXPERIMENT_DESIGN_CANDIDATE_SHADOW_zh-CN.md)，本目录online_shadow_assessment_20261108.json记录结果。原离线指标与权重不变；不把该结果写成全协议失效或归因于单一因素。
