#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${TOOLHCL_DATA_ROOT:?Set TOOLHCL_DATA_ROOT to the ToolHCL data directory}"
: "${TOOLHCL_MODELS_ROOT:?Set TOOLHCL_MODELS_ROOT to the directory containing Meta-Llama-3-8B}"

link_asset() {
  local target=$1
  local link=$2
  if [[ ! -e "${target}" ]]; then
    echo "Missing asset target: ${target}" >&2
    return 1
  fi
  if [[ -L "${link}" ]]; then
    rm "${link}"
  elif [[ -e "${link}" ]]; then
    echo "Refusing to replace non-symlink path: ${link}" >&2
    return 1
  fi
  ln -s "$(realpath "${target}")" "${link}"
  echo "${link} -> $(readlink "${link}")"
}

mkdir -p "${ROOT}/toolhcl_links"
link_asset "${TOOLHCL_DATA_ROOT}" "${ROOT}/toolhcl_links/data"
link_asset "${TOOLHCL_MODELS_ROOT}" "${ROOT}/toolhcl_links/models"

if [[ -n "${TOOLHCL_ROOT:-}" ]]; then
  link_asset "${TOOLHCL_ROOT}" "${ROOT}/toolhcl_links/ToolHCHL"
fi

model_dir="${ROOT}/toolhcl_links/models/Meta-Llama-3-8B"
if [[ ! -f "${model_dir}/config.json" || ! -f "${model_dir}/tokenizer.json" ]]; then
  echo "Expected a complete Meta-Llama-3-8B checkout under ${model_dir}" >&2
  exit 1
fi

for required in \
  train/raw/retrieval_train.json \
  train/raw/retrieval_eval.json \
  train/raw/train_tools_with_id.json \
  task1/raw/retrieval_train.json \
  task1/raw/retrieval_eval.json \
  task1/raw/task1_tools_with_id.json \
  task2/raw/retrieval_train.json \
  task2/raw/retrieval_eval.json \
  task2/raw/task2_tools_with_id.json \
  task3/raw/retrieval_train.json \
  task3/raw/retrieval_eval.json \
  task3/raw/task3_tools_with_id.json; do
  if [[ ! -f "${ROOT}/toolhcl_links/data/${required}" ]]; then
    echo "Missing ToolHCL input: ${required}" >&2
    exit 1
  fi
done

echo "ToolHCL data and model links are ready."
