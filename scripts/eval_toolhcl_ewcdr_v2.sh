#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_common.sh"
cd "${EWC_ROOT}"
export GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB:-8000}
export GPU_MAX_UTIL_PERCENT=${GPU_MAX_UTIL_PERCENT:-20}
GPU_ID=$(select_ewc_gpu)
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_DIRECTORY" >&2
  exit 2
fi
OUTPUT_DIR=$(realpath -m "$1")
"${EWC_PYTHON}" -m toolhcl_ewcdr_pure.evaluate \
  --config configs/toolhcl_ewcdr_v2.yaml \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_dir "${OUTPUT_DIR}/checkpoints"
