#!/usr/bin/env bash

EWC_ROOT=${EWC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EWC_PYTHON=${PYTHON_BIN:-python}
EWC_ARTIFACT_ROOT=${ARTIFACT_ROOT:-${EWC_ROOT}/artifacts}
EWC_ARTIFACT_ROOT=$(realpath -m "${EWC_ARTIFACT_ROOT}")

select_ewc_gpu() {
  if [[ -n "${GPU_ID:-}" ]]; then
    printf '%s\n' "${GPU_ID}"
    return 0
  fi
  local selected
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for automatic GPU selection. Set GPU_ID explicitly after configuring CUDA." >&2
    return 1
  fi
  local min_free=${GPU_MIN_FREE_MIB:-26000}
  local max_util=${GPU_MAX_UTIL_PERCENT:-10}
  selected=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | \
    awk -F',' -v min_free="${min_free}" -v max_util="${max_util}" '{gsub(/ /,"",$1);gsub(/ /,"",$2);gsub(/ /,"",$3); if ($2 >= min_free && $3 <= max_util) print $1,$2,$3}' | \
    sort -k2,2nr | head -n1 | awk '{print $1}')
  if [[ -z "${selected}" ]]; then
    echo "No GPU currently satisfies free_memory>=${min_free} MiB and utilization<=${max_util}%. Set GPU_ID explicitly only after checking competing jobs." >&2
    return 1
  fi
  printf '%s\n' "${selected}"
}

prepare_ewc_storage() {
  mkdir -p "${EWC_ARTIFACT_ROOT}/cache" "${EWC_ARTIFACT_ROOT}/cache_smoke" "${EWC_ARTIFACT_ROOT}/runs"
  if [[ ! -e "${EWC_ROOT}/toolhcl_ewcdr_cache" && ! -L "${EWC_ROOT}/toolhcl_ewcdr_cache" ]]; then
    ln -s "${EWC_ARTIFACT_ROOT}/cache" "${EWC_ROOT}/toolhcl_ewcdr_cache"
  fi
  if [[ ! -e "${EWC_ROOT}/toolhcl_ewcdr_cache_smoke" && ! -L "${EWC_ROOT}/toolhcl_ewcdr_cache_smoke" ]]; then
    ln -s "${EWC_ARTIFACT_ROOT}/cache_smoke" "${EWC_ROOT}/toolhcl_ewcdr_cache_smoke"
  fi
}

prepare_ewc_run_dir() {
  local run_name=$1
  local visible_dir="${EWC_ROOT}/${run_name}"
  local real_dir="${EWC_ARTIFACT_ROOT}/runs/${run_name}"
  if [[ -e "${visible_dir}" || -L "${visible_dir}" || -e "${real_dir}" ]]; then
    echo "Refusing to overwrite existing run: ${run_name}" >&2
    return 1
  fi
  mkdir -p "${real_dir}/logs"
  ln -s "${real_dir}" "${visible_dir}"
  printf '%s\n' "${visible_dir}"
}
