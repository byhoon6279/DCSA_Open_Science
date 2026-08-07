#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${BASE_DIR}/pe_inspired_feature_intervention/scripts/run_pe_feature_intervention_validation_with_rf.py"
CFG_DIR="${BASE_DIR}/configs/RF"
OUT_DIR="${BASE_DIR}/results/RF"

"${PYTHON_BIN}" "${RUNNER}" \
  --config "${CFG_DIR}/pe_intervention_balanced_main_random_forest.json" \
  --output-dir "${OUT_DIR}/pe_intervention_balanced_main_random_forest"

"${PYTHON_BIN}" "${RUNNER}" \
  --config "${CFG_DIR}/pe_intervention_unpacked_balanced_random_forest.json" \
  --output-dir "${OUT_DIR}/pe_intervention_unpacked_balanced_random_forest"

echo "RQ2-4 RF runs completed under ${OUT_DIR}"
