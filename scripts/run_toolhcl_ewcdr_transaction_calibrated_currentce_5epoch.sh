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

RUN_NAME=${RUN_NAME:-transaction_ewcdr_calibrated_currentce_5epoch_seed42_$(date +%Y%m%d_%H%M%S)}
TRANSACTION_ARTIFACT_ROOT=${TRANSACTION_ARTIFACT_ROOT:-${EWC_ROOT}/artifacts/transaction}
REAL_OUTPUT="${TRANSACTION_ARTIFACT_ROOT}/runs/${RUN_NAME}"
VISIBLE_OUTPUT="${EWC_ROOT}/${RUN_NAME}"
[[ ! -e "${REAL_OUTPUT}" && ! -e "${VISIBLE_OUTPUT}" && ! -L "${VISIBLE_OUTPUT}" ]] || exit 1
mkdir -p "${REAL_OUTPUT}/logs"
ln -s "${REAL_OUTPUT}" "${VISIBLE_OUTPUT}"
LOG_PATH="${REAL_OUTPUT}/logs/full_pipeline.log"

{
  echo "gpu=${GPU_ID} output=${REAL_OUTPUT} protocol=decontaminated_calibrated_currentce_5epoch_fp32eval"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
  "${EWC_PYTHON}" -m toolhcl_ewcdr_pure.train \
    --config configs/toolhcl_ewcdr_transaction_calibrated_currentce_5epoch.yaml \
    --output_dir "${REAL_OUTPUT}" --method ewc_dr
  "${EWC_PYTHON}" -m toolhcl_ewcdr_pure.evaluate \
    --config "${REAL_OUTPUT}/config.json" \
    --output_dir "${REAL_OUTPUT}" --checkpoint_dir "${REAL_OUTPUT}/checkpoints"
} 2>&1 | tee -a "${LOG_PATH}"

printf '%s\n' "${REAL_OUTPUT}" > "${TRANSACTION_ARTIFACT_ROOT}/latest_calibrated_currentce_5epoch_run.txt"
echo "EWC_DR_CALIBRATED_CURRENTCE_5EPOCH_OUTPUT=${VISIBLE_OUTPUT}"
