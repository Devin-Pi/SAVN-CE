#!/usr/bin/env bash
set -euo pipefail
project_root=/media/4tbssd/00_PROJECT_CODE/SAVN-CE
code_root=$project_root/outputs/avn_foa_sameroom_worktree
asset_root=$project_root/research_outputs/AVN-FOA-SameRoom-v0.1_assets
output_name=AVN-FOA-SameRoom-v0.1
# Finish the currently running render/baseline acceptance before independent final checks.
if [ "${AVN_WAIT_RENDER_DONE:-1}" = 1 ]; then
  while pgrep -f '^bash outputs/avn_foa_sameroom_worktree/avn_foa_sameroom/run_batch.sh$' >/dev/null; do sleep 5; done
fi
test -f "$project_root/research_outputs/$output_name/generation_summary.json"
run_module() {
    docker exec -e PYTHONPATH=/workspace/SAVN-CE/outputs/avn_foa_sameroom_worktree:/workspace/SAVN-CE/outputs/avn_foa_sameroom_worktree/vendor \
      -e HABITAT_SIM_LOG=quiet -e MAGNUM_LOG=quiet savnce_dev python -u -m "$@"
}
if ! run_module avn_foa_sameroom.navcheck "research_outputs/$output_name"; then
    run_module avn_foa_sameroom.repair_nav "research_outputs/$output_name" \
      --assets research_outputs/AVN-FOA-SameRoom-v0.1_assets/audit_parallel \
      --quarantine research_outputs/AVN-FOA-SameRoom-v0.1_assets/rejected_isolated_navmesh
    run_module avn_foa_sameroom.navcheck "research_outputs/$output_name"
fi
run_module avn_foa_sameroom.native_check "research_outputs/$output_name"
run_module avn_foa_sameroom.validate "research_outputs/$output_name"
run_module avn_foa_sameroom.finalize "research_outputs/$output_name" \
    --assets research_outputs/AVN-FOA-SameRoom-v0.1_assets/audit_parallel
printf 'FINAL_DATASET_ACCEPTED_AND_SEALED\n'
