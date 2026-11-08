#!/usr/bin/env bash
set -euo pipefail
experiment=/media/4tbssd/av_seld/workspaces/avn_foa_dense15h_finetune_v02
project=/media/4tbssd/00_PROJECT_CODE/SAVN-CE
docker run -d --name avseld_dense15h_analysis_v02 --runtime=nvidia --gpus all --shm-size=4g --init \
  -e CUDA_VISIBLE_DEVICES=2 -e PYTHONPATH=/training_source -e PYTHONDONTWRITEBYTECODE=1 \
  -v /media/4tbssd/av_seld/workspaces:/workspaces:ro \
  -v "$project/research_outputs/AVN-FOA-SameRoom-v0.2-Dense15h:/datasets/main:ro" \
  -v "$project/research_outputs/AVN-FOA-SameRoom-v0.1:/datasets/reference:ro" \
  -v "$experiment:/experiment:rw" -v "$experiment/code_snapshot:/training_source:ro" \
  --entrypoint python rdpi/avseld_final:latest -u /experiment/analyze_dense_full100.py \
  --experiment /experiment --dense /datasets/main --sparse /datasets/reference \
  --old-run /workspaces/avn_foa_full_finetune_v1/evaluation_user_stop_epoch095 \
  --runtime /workspaces/dsf_localization_runtime_v1
