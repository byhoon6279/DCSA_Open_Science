#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROBE_SCRIPT="${SCRIPT_DIR}/run_density_conditioned_import_intervention_probe.py"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_density_conditioned_import_intervention_probe_results.py"
CONFIG_ROOT="${SECTION_DIR}/configs/LightGBM"
OUT_ROOT="${SECTION_DIR}/results/LightGBM/block3_probe_lightgbm_imports_parallel"
MERGED_OUT="${SECTION_DIR}/results/LightGBM/block3_probe_lightgbm_imports_parallel_merged"

run_seed() {
  local seed="$1"
  local config="${CONFIG_ROOT}/block3_probe_lightgbm_imports_seed${seed}.json"
  local out_dir="${OUT_ROOT}/seed_${seed}"
  echo "[block3-lightgbm] start seed=${seed}"
  "${PYTHON_BIN}" "${PROBE_SCRIPT}" --config "${config}" --output-dir "${out_dir}"
  echo "[block3-lightgbm] done seed=${seed}"
}

mkdir -p "${OUT_ROOT}"

# Safer default: run at most two seeds at a time to avoid memory pressure.
run_seed 13 &
run_seed 42 &
wait

run_seed 77 &
run_seed 123 &
wait

run_seed 2024

"${PYTHON_BIN}" "${MERGE_SCRIPT}" \
  --input-root "${OUT_ROOT}" \
  --output-dir "${MERGED_OUT}"

echo "[block3-lightgbm] merged results written to ${MERGED_OUT}"
