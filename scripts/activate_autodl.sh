#!/usr/bin/env bash
# Source this file: source /root/autodl-tmp/scripts/activate_autodl.sh
FCL_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Migrate only the old in-project defaults; preserve custom external roots.
if [[ "${DATA_ROOT:-}" == "${FCL_PROJECT_ROOT}/data" ]]; then
    export DATA_ROOT="${FCL_PROJECT_ROOT}/dataset"
fi
if [[ "${WEIGHTS_ROOT:-}" == "${FCL_PROJECT_ROOT}/weights" ]]; then
    export WEIGHTS_ROOT="${FCL_PROJECT_ROOT}/dataset/weights"
fi
export DATA_ROOT="${DATA_ROOT:-${FCL_PROJECT_ROOT}/dataset}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${FCL_PROJECT_ROOT}/outputs}"
export WEIGHTS_ROOT="${WEIGHTS_ROOT:-${DATA_ROOT}/weights}"
export CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${WEIGHTS_ROOT}}"
export PATH="/root/miniconda3/bin:${PATH}"
export PYTHONUNBUFFERED=1
cd "${FCL_PROJECT_ROOT}" || return 1
