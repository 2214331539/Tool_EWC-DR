#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_pure_common.sh"
cd "${PURE_ROOT}"
prepare_pure_storage
GPU_ID=$(select_pure_gpu)
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
OUTPUT_DIR=${OUTPUT_DIR:-$(cat "${PURE_ARTIFACT_ROOT}/latest_run.txt")}
LOG_PATH="${OUTPUT_DIR}/logs/full_pipeline.log"

"${PURE_PYTHON}" -m toolhcl_ewcdr_pure.evaluate \
  --config configs/toolhcl_ewcdr_pure.yaml \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_dir "${OUTPUT_DIR}/checkpoints" \
  "${@}" 2>&1 | tee -a "${LOG_PATH}"
