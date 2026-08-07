#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/../scripts/run_class_conditional_density_conditioned_fragility.py" --config "${SCRIPT_DIR}/d4_lr_controlled_main.json" --output-dir "${SCRIPT_DIR}/../results/LR/density_fragility_controlled_main"
"${PYTHON_BIN}" "${SCRIPT_DIR}/../scripts/run_class_conditional_density_conditioned_fragility.py" --config "${SCRIPT_DIR}/d4_lightgbm_controlled_main_permutation.json" --output-dir "${SCRIPT_DIR}/../results/LightGBM/density_fragility_controlled_main_lightgbm_permutation"
"${PYTHON_BIN}" "${SCRIPT_DIR}/../scripts/run_class_conditional_density_conditioned_fragility_with_rf.py" --config "${SCRIPT_DIR}/d4_rf_controlled_main.json" --output-dir "${SCRIPT_DIR}/../results/RF/rf_full_wild_b"
