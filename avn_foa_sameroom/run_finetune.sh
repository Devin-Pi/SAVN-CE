#!/usr/bin/env bash
set -euo pipefail
docker exec -e PYTHONPATH=/workspaces/avn_foa_full_finetune_v1/code -e CUDA_VISIBLE_DEVICES=2 \
  avseld_final_test python -u -m avn_foa_sameroom.finetune \
  --data /workspaces/avn_foa_full_finetune_v1/data \
  --runtime /workspaces/dsf_localization_runtime_v1 \
  --checkpoint /workspaces/av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt \
  --output /workspaces/avn_foa_full_finetune_v1/run_seed7_full100 \
  --loss standard --batch 16 --workers 4 --epochs 100 --seed 7 --encoder-lr 1e-5 --head-lr 1e-4 --patience 0
