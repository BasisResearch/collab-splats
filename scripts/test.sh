#!/bin/bash
set -euxo pipefail

# Constrain threading to avoid resource limits in CI/containers
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

# Run tests in a single worker to prevent spawning many processes
pytest tests/ -n 1