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

RUN_NAME=${RUN_NAME:-toolhcl_ewcdr_v2_$(date +%Y%m%d_%H%M%S)}
VISIBLE_OUTPUT="${EWC_ROOT}/${RUN_NAME}"
VISIBLE_SELECTION="${EWC_ROOT}/${RUN_NAME}_selection"
REAL_OUTPUT="${EWC_ARTIFACT_ROOT}/runs/${RUN_NAME}"
REAL_SELECTION="${EWC_ARTIFACT_ROOT}/runs/${RUN_NAME}_selection"
for path in "${VISIBLE_OUTPUT}" "${VISIBLE_SELECTION}" "${REAL_OUTPUT}" "${REAL_SELECTION}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to overwrite existing path: ${path}" >&2
    exit 1
  fi
done
mkdir -p "${REAL_OUTPUT}/logs"
ln -s "${REAL_OUTPUT}" "${VISIBLE_OUTPUT}"
ln -s "${REAL_SELECTION}" "${VISIBLE_SELECTION}"
LOG_PATH="${REAL_OUTPUT}/logs/full_pipeline.log"
echo "${VISIBLE_OUTPUT}" > "${EWC_ARTIFACT_ROOT}/latest_v2_run.txt"

{
  echo "gpu=${GPU_ID} output=${VISIBLE_OUTPUT} selection=${VISIBLE_SELECTION}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
  "${EWC_PYTHON}" -m toolhcl_ewcdr_pure.validated_pipeline \
    --config configs/toolhcl_ewcdr_v2.yaml \
    --output_dir "${REAL_OUTPUT}" \
    --selection_dir "${REAL_SELECTION}" \
    --method ewc_dr
} 2>&1 | tee -a "${LOG_PATH}"

if [[ ! -e "${REAL_SELECTION}" && -L "${VISIBLE_SELECTION}" ]]; then
  rm "${VISIBLE_SELECTION}"
fi
echo "EWC_DR_V2_OUTPUT=${VISIBLE_OUTPUT}"
