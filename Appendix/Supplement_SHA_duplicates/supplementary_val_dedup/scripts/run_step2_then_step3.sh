#!/bin/bash
set -e
cd "$(dirname "$0")"
LOG_DIR=../logs
until grep -q "STEP1_DONE" "$LOG_DIR/step1.log" 2>/dev/null; do
  if grep -q "Traceback" "$LOG_DIR/step1.log" 2>/dev/null; then
    echo "STEP1_FAILED, aborting chain"
    exit 1
  fi
  sleep 20
done
echo "step1 confirmed done, starting step2"
python3 step2_masking_reeval.py > "$LOG_DIR/step2.log" 2>&1
echo "step2 done, starting step3"
python3 step3_hashed_import_intervention.py > "$LOG_DIR/step3.log" 2>&1
echo "ALL_STEPS_DONE"
