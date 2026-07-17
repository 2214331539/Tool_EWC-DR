#!/usr/bin/env bash

PURE_ROOT=${PURE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PURE_PYTHON=${PYTHON_BIN:-python}
PURE_ARTIFACT_ROOT=${ARTIFACT_ROOT:-${PURE_ROOT}/artifacts}
PURE_ARTIFACT_ROOT=$(realpath -m "${PURE_ARTIFACT_ROOT}")

select_pure_gpu() {
  if [[ -n "${GPU_ID:-}" ]]; then
    printf '%s\n' "${GPU_ID}"
    return 0
  fi
  local selected
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for automatic GPU selection. Set GPU_ID explicitly after configuring CUDA." >&2
    return 1
  fi
  selected=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | \
    awk -F',' '{gsub(/ /,"",$1);gsub(/ /,"",$2);gsub(/ /,"",$3); if ($2 >= 26000 && $3 <= 10) print $1,$2,$3}' | \
    sort -k2,2nr | head -n1 | awk '{print $1}')
  if [[ -z "${selected}" ]]; then
    echo "No GPU currently satisfies free_memory>=26000 MiB and utilization<=10%. Set GPU_ID explicitly only after checking competing jobs." >&2
    return 1
  fi
  printf '%s\n' "${selected}"
}

prepare_pure_storage() {
  mkdir -p "${PURE_ARTIFACT_ROOT}/cache" "${PURE_ARTIFACT_ROOT}/cache_smoke" "${PURE_ARTIFACT_ROOT}/runs"
  if [[ ! -e "${PURE_ROOT}/toolhcl_ewcdr_pure_cache" && ! -L "${PURE_ROOT}/toolhcl_ewcdr_pure_cache" ]]; then
    ln -s "${PURE_ARTIFACT_ROOT}/cache" "${PURE_ROOT}/toolhcl_ewcdr_pure_cache"
  fi
  if [[ ! -e "${PURE_ROOT}/toolhcl_ewcdr_pure_cache_smoke" && ! -L "${PURE_ROOT}/toolhcl_ewcdr_pure_cache_smoke" ]]; then
    ln -s "${PURE_ARTIFACT_ROOT}/cache_smoke" "${PURE_ROOT}/toolhcl_ewcdr_pure_cache_smoke"
  fi
}

prepare_pure_run_dir() {
  local run_name=$1
  local visible_dir="${PURE_ROOT}/${run_name}"
  local real_dir="${PURE_ARTIFACT_ROOT}/runs/${run_name}"
  if [[ -e "${visible_dir}" || -L "${visible_dir}" || -e "${real_dir}" ]]; then
    echo "Refusing to overwrite existing run: ${run_name}" >&2
    return 1
  fi
  mkdir -p "${real_dir}/logs"
  ln -s "${real_dir}" "${visible_dir}"
  printf '%s\n' "${visible_dir}"
}
