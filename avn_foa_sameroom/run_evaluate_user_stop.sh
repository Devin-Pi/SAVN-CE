#!/usr/bin/env bash
set -euo pipefail
docker exec -e PYTHONPATH=/workspaces/avn_foa_full_finetune_v1/code -e CUDA_VISIBLE_DEVICES=2 \
  avseld_final_test python -u -m avn_foa_sameroom.evaluate_stopped_finetune \
  --output /workspaces/avn_foa_full_finetune_v1/evaluation_user_stop_epoch095 \
  --data /workspaces/avn_foa_full_finetune_v1/data \
  --runtime /workspaces/dsf_localization_runtime_v1 --batch 16 --workers 4
