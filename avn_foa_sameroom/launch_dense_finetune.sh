#!/usr/bin/env bash
# Execute on polyu_serve from an immutable experiment/code_snapshot directory.
set -euo pipefail
experiment=/media/4tbssd/av_seld/workspaces/avn_foa_dense15h_finetune_v02
project=/media/4tbssd/00_PROJECT_CODE/SAVN-CE
image=rdpi/avseld_final:latest
mode=${1:-smoke}
common=(--runtime=nvidia --gpus all --shm-size=4g --init
  -e CUDA_VISIBLE_DEVICES=2 -e PYTHONPATH=/training_source -e PYTHONDONTWRITEBYTECODE=1
  -v /media/4tbssd/av_seld/workspaces:/workspaces:ro
  -v "$project/research_outputs/AVN-FOA-SameRoom-v0.2-Dense15h:/datasets/main:ro"
  -v "$project/research_outputs/AVN-FOA-SameRoom-v0.1:/datasets/reference:ro"
  -v "$experiment:/experiment:rw"
  -v "$experiment/code_snapshot:/training_source:ro" --entrypoint bash)
case "$mode" in
 smoke)
  docker run --name avseld_dense15h_smoke_v02 "${common[@]}" "$image" -c '
    set -euo pipefail
    python -m unittest avn_foa_sameroom.test_finetune avn_foa_sameroom.test_swanlab_monitor
    python -u -m avn_foa_sameroom.finetune --data /datasets/main --reference-data /datasets/reference \
      --runtime /workspaces/dsf_localization_runtime_v1 \
      --checkpoint /workspaces/av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt \
      --output /experiment/smoke_batch16 --epochs 100 --batch 16 --workers 4 \
      --encoder-lr 1e-5 --head-lr 1e-4 --seed 7 --loss standard --patience 0 --smoke
  '
  ;;
 train)
  test -f "$experiment/smoke_batch16/smoke_result.json"
  test ! -e "$experiment/run_seed7_full100"
  docker run -d --name avseld_dense15h_train_v02 "${common[@]}" "$image" /training_source/avn_foa_sameroom/run_dense_finetune_inside.sh
  ;;
 *) echo 'Usage: launch_dense_finetune.sh smoke|train' >&2; exit 2;;
esac
