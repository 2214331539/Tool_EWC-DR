#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_DIRECTORY" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHON_BIN=${PYTHON_BIN:-python}
source "${SCRIPT_DIR}/_common.sh"
cd "${EWC_ROOT}"

GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB:-8000}
GPU_MAX_UTIL_PERCENT=${GPU_MAX_UTIL_PERCENT:-60}
GPU_ID=$(select_ewc_gpu)
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

OUTPUT_DIR=$(realpath -m "$1")
"${EWC_PYTHON}" -m toolhcl_ewcdr_pure.evaluate \
  --config "${OUTPUT_DIR}/config.json" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_dir "${OUTPUT_DIR}/checkpoints"
