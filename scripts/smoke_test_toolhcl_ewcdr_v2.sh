#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_common.sh"
cd "${EWC_ROOT}"
prepare_ewc_storage
export GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB:-8000}
export GPU_MAX_UTIL_PERCENT=${GPU_MAX_UTIL_PERCENT:-20}
GPU_ID=$(select_ewc_gpu)
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
RUN_NAME=${RUN_NAME:-toolhcl_ewcdr_v2_smoke_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=$(prepare_ewc_run_dir "${RUN_NAME}")
LOG_PATH="${OUTPUT_DIR}/logs/full_pipeline.log"

{
  echo "gpu=${GPU_ID} output=${OUTPUT_DIR}"
  "${EWC_PYTHON}" -m toolhcl_ewcdr_pure.train \
    --config configs/toolhcl_ewcdr_v2.yaml \
    --output_dir "${OUTPUT_DIR}" \
    --method ewc_dr \
    --stages base,task1 \
    --epochs 1 \
    --importance_max_samples 128
  "${EWC_PYTHON}" -m toolhcl_ewcdr_pure.evaluate \
    --config configs/toolhcl_ewcdr_v2.yaml \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoint_dir "${OUTPUT_DIR}/checkpoints" \
    --stages base,task1
} 2>&1 | tee -a "${LOG_PATH}"

echo "EWC_DR_V2_SMOKE_OUTPUT=${OUTPUT_DIR}"
