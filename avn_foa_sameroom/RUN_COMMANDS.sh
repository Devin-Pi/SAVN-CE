#!/usr/bin/env bash
# Recorded commands, for review/reproduction. Do not execute over an existing dataset.
# Environment: SSH polyu_serve; /media/4tbssd/00_PROJECT_CODE/SAVN-CE
# Source: outputs/avn_foa_sameroom_worktree ; Python module avn_foa_sameroom
# 1) Only official TRAIN definitions were downloaded using the existing official index:
python3 -m avn_foa_sameroom.fetch_assets --index audit/official-file-index.json \
  --output research_outputs/AVN-FOA-SameRoom-v0.1_assets/official_train
# 2) In savnce_dev, with PYTHONPATH=/workspace/SAVN-CE/outputs/avn_foa_sameroom_worktree:
python -m avn_foa_sameroom.audit --repo /workspace/SAVN-CE \
  --episodes research_outputs/AVN-FOA-SameRoom-v0.1_assets/official_train \
  --sounds data/sounds/semantic_splits --output research_outputs/AVN-FOA-SameRoom-v0.1_assets/audit
python -m avn_foa_sameroom.calibrate --output research_outputs/AVN-FOA-SameRoom-v0.1_assets/foa_calibration.json
python -m avn_foa_sameroom.parallel_plan \
  --assets research_outputs/AVN-FOA-SameRoom-v0.1_assets/audit \
  --output research_outputs/AVN-FOA-SameRoom-v0.1_assets/audit_parallel --workers 4
# Pilot used an earlier, partial geometry inventory, same qualification logic:
python -m avn_foa_sameroom.generate --smoke \
  --assets research_outputs/AVN-FOA-SameRoom-v0.1_assets/audit \
  --output research_outputs/AVN-FOA-SameRoom-v0.1_pilot_v2
python -m avn_foa_sameroom.native_check research_outputs/AVN-FOA-SameRoom-v0.1_pilot_v2
# 3) Original CUDA/Mamba image, native16k data copied to its existing /workspaces mount:
docker exec -e PYTHONPATH=/workspaces/avn_foa_sameroom_checks_v01 -e CUDA_VISIBLE_DEVICES=2 \
  avseld_final_test python -m avn_foa_sameroom.forward_smoke \
  /workspaces/avn_foa_sameroom_checks_v01/pilot \
  --runtime /workspaces/dsf_localization_runtime_v1 \
  --checkpoint /workspaces/av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt \
  --adapter-dir /workspaces/dsf_localization_runtime_v1
# Copy dsf_forward_smoke.json back to pilot, then run validation (see run_batch.sh):
python -m avn_foa_sameroom.validate research_outputs/AVN-FOA-SameRoom-v0.1_pilot_v2 --pilot
# 4) Host: offline Docker build (wheel acquired by pip download rtree==1.4.1):
docker build --network=none -t rdpi/avn-foa-sameroom:0.1 docker_avn_foa
# Batch runner waits for the complete spatial audit and requires passed pilot acceptance.
bash avn_foa_sameroom/run_batch.sh
# 5) Full-corpus acceptance and report (in savnce_dev):
python -m avn_foa_sameroom.validate research_outputs/AVN-FOA-SameRoom-v0.1
python -m avn_foa_sameroom.finalize research_outputs/AVN-FOA-SameRoom-v0.1 \
  --assets research_outputs/AVN-FOA-SameRoom-v0.1_assets/audit_parallel
# 6) Focused functional tests, also run in the new Docker image:
python -m avn_foa_sameroom.tests
# No optimizer/training command was run.
