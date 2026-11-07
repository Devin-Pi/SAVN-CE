#!/usr/bin/env bash
set -euo pipefail
project_root=/media/4tbssd/00_PROJECT_CODE/SAVN-CE
code_root=$project_root/outputs/avn_foa_sameroom_worktree
asset_root=$project_root/research_outputs/AVN-FOA-SameRoom-v0.1_assets
output_name=AVN-FOA-SameRoom-v0.1
trap 'printf "batch failed at line %s\n" "$LINENO" >&2' ERR
while [ ! -f "$asset_root/audit_parallel/geometry_audit.json" ]; do sleep 10; done
python3 - "$project_root/research_outputs/AVN-FOA-SameRoom-v0.1_pilot_v2/validation_report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['status']=='passed' and r['native_association']=='passed' and r['dsf_forward']=='passed' and r['records']>=30
PY
docker run --name avn_foa_sameroom_generator_v01 --ipc=host \
    -v "$project_root:/workspace/SAVN-CE:ro" \
    -v "$project_root/research_outputs:/workspace/SAVN-CE/research_outputs:rw" \
    -v /media/4tbssd/DATA/mp3d/v1/tasks/mp3d:/datasets/mp3d:ro \
    -e PYTHONPATH=/workspace/SAVN-CE/outputs/avn_foa_sameroom_worktree \
    rdpi/avn-foa-sameroom:0.1 \
    python -u -m avn_foa_sameroom.generate \
    --assets research_outputs/AVN-FOA-SameRoom-v0.1_assets/audit_parallel \
    --output "research_outputs/$output_name" \
    --acceptance research_outputs/AVN-FOA-SameRoom-v0.1_pilot_v2/validation_report.json
docker exec -i savnce_dev python - "$output_name" <<'PY_COPY'
import sys,shutil,pathlib
r=pathlib.Path('research_outputs')/sys.argv[1]
shutil.copyfile(pathlib.Path('research_outputs/AVN-FOA-SameRoom-v0.1_pilot_v2/dsf_forward_smoke.json'),r/'dsf_forward_smoke.json')
for source in ('visibility_api_probe.json','native_audio_pipeline_check.json','geometry_reproducibility.json'):
 shutil.copyfile(pathlib.Path('research_outputs/AVN-FOA-SameRoom-v0.1_assets')/source,r/source)
PY_COPY
AVN_WAIT_RENDER_DONE=0 bash "$code_root/avn_foa_sameroom/complete_dataset.sh"
