#!/usr/bin/env bash
# Package the shared FOA contract; never launches generation or training.
set -euo pipefail
source_root=$(cd "$(dirname "$0")/.." && pwd)
image=rdpi/avn-foa-sameroom:dense15h-sharedfoa-v1
docker build --network=none -f "$source_root/docker_avn_foa/Dockerfile.dense15h" -t "$image" "$source_root"
docker run --rm --network=none "$image" python -c 'from avn_foa_sameroom.common import foa_contract; import json; print(json.dumps(foa_contract()))'
