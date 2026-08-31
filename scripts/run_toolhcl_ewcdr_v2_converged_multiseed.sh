#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_common.sh"
cd "${EWC_ROOT}"
prepare_ewc_storage

SEEDS=${SEEDS:-"42 43 44"}
GPU_IDS=${GPU_IDS:-""}
RUN_PREFIX=${RUN_PREFIX:-toolbench_v2_converged5pct}
CONFIG=${CONFIG:-configs/toolhcl_ewcdr_v2_converged.yaml}
mkdir -p "${EWC_ARTIFACT_ROOT}/runs" "${EWC_ARTIFACT_ROOT}/multiseed_logs"

read -r -a seed_values <<< "${SEEDS}"
read -r -a gpu_values <<< "${GPU_IDS}"
if [[ ${#gpu_values[@]} -gt 0 && ${#gpu_values[@]} -ne ${#seed_values[@]} ]]; then
  echo "GPU_IDS must contain one GPU index per seed." >&2
  exit 1
fi

pids=()
run_names=()
for index in "${!seed_values[@]}"; do
  seed=${seed_values[$index]}
  run_name="${RUN_PREFIX}_seed${seed}"
  output_dir="${EWC_ARTIFACT_ROOT}/runs/${run_name}"
  log_path="${EWC_ARTIFACT_ROOT}/multiseed_logs/${run_name}.log"
  for path in "${output_dir}" "${EWC_ROOT}/${run_name}"; do
    if [[ -e "${path}" || -L "${path}" ]]; then
      echo "Refusing to overwrite existing path: ${path}" >&2
      exit 1
    fi
  done
  ln -s "${output_dir}" "${EWC_ROOT}/${run_name}"
  run_names+=("${run_name}")

  if [[ ${#gpu_values[@]} -gt 0 ]]; then
    gpu_id=${gpu_values[$index]}
  else
    GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB:-8000} \
      GPU_MAX_UTIL_PERCENT=${GPU_MAX_UTIL_PERCENT:-20} \
      gpu_id=$(select_ewc_gpu)
  fi

  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    echo "seed=${seed} gpu=${gpu_id} output=${output_dir}"
    "${EWC_PYTHON}" -m toolhcl_ewcdr_pure.train \
      --config "${CONFIG}" \
      --output_dir "${output_dir}" \
      --method ewc_dr \
      --seed "${seed}"
    "${EWC_PYTHON}" -m toolhcl_ewcdr_pure.evaluate \
      --config "${CONFIG}" \
      --output_dir "${output_dir}" \
      --checkpoint_dir "${output_dir}/checkpoints"
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  echo "started seed=${seed} gpu=${gpu_id} pid=${pids[-1]} log=${log_path}"
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed ${run_names[$index]}"
  else
    echo "failed ${run_names[$index]}" >&2
    status=1
  fi
done
exit "${status}"
