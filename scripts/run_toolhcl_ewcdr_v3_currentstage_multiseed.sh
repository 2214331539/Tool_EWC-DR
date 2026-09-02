#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${CONFIG:-configs/toolhcl_ewcdr_v3_currentstage.yaml}
RUN_PREFIX=${RUN_PREFIX:-toolbench_v3_currentstage_20260902}
SEEDS=${SEEDS:-"42 43 44"}
GPU_IDS=${GPU_IDS:-""}
CONFIG="${CONFIG}" RUN_PREFIX="${RUN_PREFIX}" SEEDS="${SEEDS}" GPU_IDS="${GPU_IDS}" \
  bash "${SCRIPT_DIR}/run_toolhcl_ewcdr_v2_converged_multiseed.sh"
