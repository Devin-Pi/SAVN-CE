#!/usr/bin/env bash
set -euo pipefail
context_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$context_dir/wheelhouse"
python3 -m pip download --only-binary=:all: --no-deps --dest "$context_dir/wheelhouse" rtree==1.4.1
docker build --network=none --tag rdpi/avn-foa-sameroom:0.1 "$context_dir"
