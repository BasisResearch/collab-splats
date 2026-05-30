#!/usr/bin/env bash
# Run all 10 reconstruction datasets sequentially with vggt_omega.
# Usage: bash docs/run_all_datasets.sh [--overwrite]
set -euo pipefail

PYTHON=/opt/conda/envs/reconstruction/bin/python
SCRIPT=$(dirname "$0")/reconstruct.py
OVERWRITE="${1:-}"

DATASETS=(
  birds_c0043
  birds_c0065
  birds_c0067
  birds_gh010070
  birds_gh010097
  birds_gh010105
  birds_gh010164
  birds_pxl_20231105
  ants_gh010210
  rats_c0119
)

LOG_DIR=/workspace/outputs/logs
mkdir -p "$LOG_DIR"

for DS in "${DATASETS[@]}"; do
  echo ""
  echo "================================================================"
  echo "START: $DS  $(date)"
  echo "================================================================"
  LOG="$LOG_DIR/${DS}.log"
  if $PYTHON "$SCRIPT" --dataset "$DS" $OVERWRITE 2>&1 | tee "$LOG"; then
    echo "DONE: $DS  $(date)"
  else
    echo "ERROR: $DS  (exit $?)  $(date)" >&2
  fi
done

echo ""
echo "All datasets complete: $(date)"
