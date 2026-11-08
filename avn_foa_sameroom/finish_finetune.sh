#!/usr/bin/env bash
set -euo pipefail
run_host=/media/4tbssd/av_seld/workspaces/avn_foa_full_finetune_v1/run_seed7_full100
while [ ! -f "$run_host/results.json" ]; do
  if ! pgrep -f '^python -u -m avn_foa_sameroom.finetune .*run_seed7_full100' >/dev/null; then
    echo 'Training process exited before results.json; inspect train_full100.log' >&2
    exit 1
  fi
  sleep 10
done
docker exec -e PYTHONPATH=/workspaces/avn_foa_full_finetune_v1/code avseld_final_test \
  python -m avn_foa_sameroom.validate_completed_finetune /workspaces/avn_foa_full_finetune_v1/run_seed7_full100
docker exec -e PYTHONPATH=/workspaces/avn_foa_full_finetune_v1/code avseld_final_test \
  python -m avn_foa_sameroom.summarize_finetune /workspaces/avn_foa_full_finetune_v1/run_seed7_full100 \
  --data /workspaces/avn_foa_full_finetune_v1/data --runtime /workspaces/dsf_localization_runtime_v1
# This reads a fixed validation WAV without passing any target annotations to the model.
docker exec -e PYTHONPATH=/workspaces/avn_foa_full_finetune_v1/code -e CUDA_VISIBLE_DEVICES=2 avseld_final_test \
  python -m avn_foa_sameroom.infer_finetuned \
  --checkpoint /workspaces/avn_foa_full_finetune_v1/run_seed7_full100/best.pt \
  --runtime /workspaces/dsf_localization_runtime_v1 \
  --wav /workspaces/avn_foa_full_finetune_v1/data/audio/validation/validation_000001.wav \
  --output /workspaces/avn_foa_full_finetune_v1/run_seed7_full100/inference_smoke.json
original_data=/media/4tbssd/00_PROJECT_CODE/SAVN-CE/research_outputs/AVN-FOA-SameRoom-v0.1
cd "$original_data"
sha256sum --check --quiet SHA256SUMS
python3 - "$run_host" <<'PY'
import pathlib,json,hashlib,sys
r=pathlib.Path(sys.argv[1]);result=json.loads((r/'results.json').read_text())
assert result['initial_checkpoint_sha256']=='85e380a6a8d2f1e545aa093e3f086f0a4d7830a15c41f7f79d047d7d9c1f9f90'
(r.parent/'original_assets_preservation.json').write_text(json.dumps(dict(status='passed',original_dataset_seal_verified=True,original_checkpoint_sha256=result['initial_checkpoint_sha256']),indent=2))
PY
echo 'FINETUNING_AND_FINAL_CHECKS_COMPLETE'
