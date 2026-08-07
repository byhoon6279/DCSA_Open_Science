#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROBE_SCRIPT="${SCRIPT_DIR}/run_density_conditioned_import_intervention_probe.py"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_density_conditioned_import_intervention_probe_results.py"
CONFIG_ROOT="${SECTION_DIR}/configs/LightGBM"
TMP_CONFIG_ROOT="${SECTION_DIR}/configs/LightGBM/.random_tmp"
OUT_ROOT="${SECTION_DIR}/results/LightGBM/block3_probe_lightgbm_imports_random_parallel"
MERGED_OUT="${SECTION_DIR}/results/LightGBM/block3_probe_lightgbm_imports_random_parallel_merged"

mkdir -p "${TMP_CONFIG_ROOT}" "${OUT_ROOT}"

make_random_config() {
  local seed="$1"
  local src="${CONFIG_ROOT}/block3_probe_lightgbm_imports_seed${seed}.json"
  local dst="${TMP_CONFIG_ROOT}/block3_probe_lightgbm_imports_seed${seed}_random.json"
  sed 's/"random_trials": 0/"random_trials": 20/' "${src}" > "${dst}"
}

run_seed() {
  local seed="$1"
  local config="${TMP_CONFIG_ROOT}/block3_probe_lightgbm_imports_seed${seed}_random.json"
  local out_dir="${OUT_ROOT}/seed_${seed}"
  echo "[block3-random] start seed=${seed}"
  "${PYTHON_BIN}" "${PROBE_SCRIPT}" --config "${config}" --output-dir "${out_dir}"
  echo "[block3-random] done seed=${seed}"
}

for seed in 13 42 77 123 2024; do
  make_random_config "${seed}"
done

# Safer default: run at most two seeds at a time.
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

echo "[block3-random] merged results written to ${MERGED_OUT}"
