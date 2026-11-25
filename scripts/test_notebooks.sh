#!/bin/bash
set -euo pipefail

# Constrain threading to avoid resource limits in CI/containers
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

# Run notebook tests in single worker to prevent resource issues
CI=1 pytest --nbval-lax --nbval-cell-timeout=600 -n 1 docs/source/tutorials/
