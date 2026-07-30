#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHON_BIN=${PYTHON_BIN:-python}
source "${SCRIPT_DIR}/_common.sh"
cd "${EWC_ROOT}"

GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB:-30000}
GPU_MAX_UTIL_PERCENT=${GPU_MAX_UTIL_PERCENT:-25}
GPU_ID=$(select_ewc_gpu)
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

RUN_NAME=${RUN_NAME:-transaction_ewcdr_calibrated_currentce_smoke_$(date +%Y%m%d_%H%M%S)}
TRANSACTION_ARTIFACT_ROOT=${TRANSACTION_ARTIFACT_ROOT:-${EWC_ROOT}/artifacts/transaction}
OUTPUT_DIR="${TRANSACTION_ARTIFACT_ROOT}/runs/${RUN_NAME}"
[[ ! -e "${OUTPUT_DIR}" ]] || exit 1
mkdir -p "${OUTPUT_DIR}/logs"
LOG_PATH="${OUTPUT_DIR}/logs/full_pipeline.log"

{
  echo "gpu=${GPU_ID} output=${OUTPUT_DIR} protocol=calibrated_currentce_smoke"
  "${EWC_PYTHON}" -m toolhcl_ewcdr_pure.train \
    --config configs/toolhcl_ewcdr_transaction_calibrated_currentce_5epoch.yaml \
    --output_dir "${OUTPUT_DIR}" --method ewc_dr --smoke
  "${EWC_PYTHON}" -m toolhcl_ewcdr_pure.evaluate \
    --config "${OUTPUT_DIR}/config.json" \
    --output_dir "${OUTPUT_DIR}" --checkpoint_dir "${OUTPUT_DIR}/checkpoints"
} 2>&1 | tee -a "${LOG_PATH}"

echo "EWC_DR_CALIBRATED_CURRENTCE_SMOKE_OUTPUT=${OUTPUT_DIR}"
