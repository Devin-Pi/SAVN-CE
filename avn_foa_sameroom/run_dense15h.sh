#!/usr/bin/env bash
# Build -> all-class pilot -> compatibility/independent acceptance -> 15h corpus.
# Old project/data are mounted read-only; each call mounts only its new output RW.
set -euo pipefail
project_root=/media/4tbssd/00_PROJECT_CODE/SAVN-CE
code_root=$project_root/outputs/avn_foa_dense15h_worktree
base_name=AVN-FOA-SameRoom-v0.2-Dense15h
asset_root=$project_root/research_outputs/${base_name}_assets
image=rdpi/avn-foa-sameroom:dense15h-v0.2
mkdir -p "$asset_root"
docker build --network=none -f "$code_root/docker_avn_foa/Dockerfile.dense15h" -t "$image" "$code_root" > "$asset_root/docker-build.log" 2>&1
python3 - "$asset_root" "$image" <<'PY'
import pathlib,json,sys,subprocess
root=pathlib.Path(sys.argv[1])
d={name:json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]['Id'] for name in [sys.argv[2],'rdpi/avn-foa-sameroom:0.1','rdpi/avseld_final:latest']}
(root/'environment.json').write_text(json.dumps(dict(images=d,generation='CPU4 processes,4CPU limit,12GiB memory limit',old_assets_mounted_read_only=True,forward_gpu='physical GPU2',forward_runtime='nvidia, explicitly required on this host',network='none'),indent=2))
PY
docker run --rm --network=none --cpus=4 --memory=12g \
  "$image" python -m unittest avn_foa_sameroom.test_dense_synthesis avn_foa_sameroom.tests -v \
  > "$asset_root/unit-tests-packaged.log" 2>&1
cpu_task() {
  local run_name=$1 output_name=$2 module_name=$3
  shift 3
  mkdir -p "$project_root/research_outputs/$output_name"
  docker run --name "$run_name" --network=none --cpus=4 --memory=12g \
    -v "$project_root:/workspace/SAVN-CE:ro" \
    -v "$project_root/research_outputs/$output_name:/workspace/SAVN-CE/research_outputs/$output_name:rw" \
    "$image" python -u -m "$module_name" "$@"
}
forward_task() {
  local run_name=$1 output_name=$2
  docker run --name "$run_name" --network=none --runtime=nvidia --gpus all \
    -e CUDA_VISIBLE_DEVICES=2 -e PYTHONPATH=/workspace/SAVN-CE/outputs/avn_foa_dense15h_worktree \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$project_root:/workspace/SAVN-CE:ro" \
    -v "$project_root/research_outputs/$output_name:/workspace/SAVN-CE/research_outputs/$output_name:rw" \
    -v /media/4tbssd/av_seld/workspaces:/workspaces:ro -w /workspace/SAVN-CE \
    rdpi/avseld_final:latest python -u -m avn_foa_sameroom.forward_smoke \
    "research_outputs/$output_name" --runtime /workspaces/dsf_localization_runtime_v1 \
    --checkpoint /workspaces/av-mamba/pre_trained_weights/smallsamll_3s_epoch17_val_F0.8022_val_seld_scr0.1549.ckpt \
    --adapter-dir /workspaces/dsf_localization_runtime_v1
}
if [ ! -f "$project_root/research_outputs/${base_name}_pilot/generation_summary.json" ]; then
cpu_task avn_foa_dense15h_pilot_v02 "${base_name}_pilot" avn_foa_sameroom.dense_generate \
  --source research_outputs/AVN-FOA-SameRoom-v0.1 --output "research_outputs/${base_name}_pilot" \
  --pilot --seed 20261107 --workers 4 > "$asset_root/pilot-generate.log" 2>&1
fi
forward_task avn_foa_dense15h_pilot_forward_v02 "${base_name}_pilot" > "$asset_root/pilot-forward.log" 2>&1
cpu_task avn_foa_dense15h_pilot_validate_v02 "${base_name}_pilot" avn_foa_sameroom.dense_validate \
  "research_outputs/${base_name}_pilot" --workers 4 \
  --forward-report "research_outputs/${base_name}_pilot/dsf_forward_smoke.json" > "$asset_root/pilot-validate.log" 2>&1
cpu_task avn_foa_dense15h_v02 "$base_name" avn_foa_sameroom.dense_generate \
  --source research_outputs/AVN-FOA-SameRoom-v0.1 --output "research_outputs/$base_name" \
  --acceptance "research_outputs/${base_name}_pilot/validation_report.json" \
  --seed 20261107 --workers 4 > "$asset_root/full-generate.log" 2>&1
forward_task avn_foa_dense15h_forward_v02 "$base_name" > "$asset_root/full-forward.log" 2>&1
cpu_task avn_foa_dense15h_validate_v02 "$base_name" avn_foa_sameroom.dense_validate \
  "research_outputs/$base_name" --workers 4 \
  --forward-report "research_outputs/$base_name/dsf_forward_smoke.json" > "$asset_root/full-validate.log" 2>&1
cpu_task avn_foa_dense15h_finalize_v02 "$base_name" avn_foa_sameroom.dense_finalize \
  "research_outputs/$base_name" --environment "research_outputs/${base_name}_assets/environment.json" \
  --commands "outputs/avn_foa_dense15h_worktree/avn_foa_sameroom/run_dense15h.sh" > "$asset_root/finalize.log" 2>&1
echo DENSE15H_ALL_STAGES_COMPLETE
