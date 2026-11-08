#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/training_source
run_dir=/experiment/run_seed7_full100
runtime=/workspaces/dsf_localization_runtime_v1
python -u -m avn_foa_sameroom.finetune \
  --data /datasets/main --reference-data /datasets/reference \
  --runtime "$runtime" \
  --checkpoint /workspaces/av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt \
  --output "$run_dir" --epochs 100 --batch 16 --workers 4 \
  --encoder-lr 1e-5 --head-lr 1e-4 --seed 7 --loss standard --patience 0
python -m avn_foa_sameroom.validate_completed_finetune "$run_dir"
python -m avn_foa_sameroom.report_dense_finetune "$run_dir" \
  --data /datasets/main --reference-data /datasets/reference --runtime "$runtime"
python -m avn_foa_sameroom.infer_finetuned --checkpoint "$run_dir/best.pt" \
  --runtime "$runtime" --wav /datasets/main/audio/validation/dense15h_validation_000001.wav \
  --output "$run_dir/inference_smoke.json"
echo 'DENSE15H_TRAINING_AND_FINAL_CHECKS_COMPLETE'
