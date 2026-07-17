#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_pure_common.sh"
cd "${PURE_ROOT}"
prepare_pure_storage
GPU_ID=$(select_pure_gpu)
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
RUN_NAME=${RUN_NAME:-toolhcl_ewcdr_runs_pure_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=$(prepare_pure_run_dir "${RUN_NAME}")
LOG_PATH="${OUTPUT_DIR}/logs/full_pipeline.log"

echo "${OUTPUT_DIR}" > "${PURE_ARTIFACT_ROOT}/latest_run.txt"
{
  echo "gpu=${GPU_ID} output=${OUTPUT_DIR}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
  "${PURE_PYTHON}" -m toolhcl_ewcdr_pure.train \
    --config configs/toolhcl_ewcdr_pure.yaml \
    --output_dir "${OUTPUT_DIR}" \
    --method ewc_dr \
    "${@}"
} 2>&1 | tee -a "${LOG_PATH}"

echo "PURE_RUN_OUTPUT=${OUTPUT_DIR}"
