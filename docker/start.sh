#!/usr/bin/env bash
set -euo pipefail
cd /media/4tbssd/00_PROJECT_CODE/SAVN-CE
docker build -t rdpi/savnce:render-v1 -f docker/Dockerfile .
if docker container inspect savnce_dev >/dev/null 2>&1; then
  docker start savnce_dev >/dev/null
else
  docker run -d --name savnce_dev --runtime=nvidia --gpus device=0 --shm-size=2g \
    --mount type=bind,src=/media/4tbssd/00_PROJECT_CODE/SAVN-CE,dst=/workspace/SAVN-CE \
    --mount type=bind,src=/media/4tbssd/DATA/mp3d/v1/tasks/mp3d,dst=/datasets/mp3d,readonly \
    --mount type=bind,src=/media/4tbssd/DATA/vgg_assets/audio,dst=/datasets/source_audio,readonly \
    rdpi/savnce:render-v1
fi
