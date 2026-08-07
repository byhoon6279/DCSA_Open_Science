#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/../scripts/run_class_conditional_density_stratified_reliability.py" --config "${SCRIPT_DIR}/d3_lr_controlled_main.json" --output-dir "${SCRIPT_DIR}/../results/LR/density_reliability_controlled_main"
"${PYTHON_BIN}" "${SCRIPT_DIR}/../scripts/run_class_conditional_density_stratified_reliability.py" --config "${SCRIPT_DIR}/d3_lightgbm_controlled_main.json" --output-dir "${SCRIPT_DIR}/../results/LightGBM/density_reliability_controlled_main_lightgbm"
"${PYTHON_BIN}" "${SCRIPT_DIR}/../scripts/run_class_conditional_density_stratified_reliability_with_rf.py" --config "${SCRIPT_DIR}/d3_rf_controlled_main.json" --output-dir "${SCRIPT_DIR}/../results/RF/rf_full_wild_b"
