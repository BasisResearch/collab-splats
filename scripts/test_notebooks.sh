#!/bin/bash

# Constrain threading to avoid resource limits in CI/containers
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

INCLUDED_NOTEBOOKS="docs/"

# Array of notebooks to exclude
EXCLUDED_NOTEBOOKS=(
    "docs/data/gcloud_data_interface.ipynb"
    "docs/splats/derive_splats.ipynb"
    "docs/splats/create_mesh.ipynb"
    "docs/splats/visualization.ipynb"
)

# Build the ignore flags
IGNORE_FLAGS=""
for notebook in "${EXCLUDED_NOTEBOOKS[@]}"; do
    IGNORE_FLAGS="$IGNORE_FLAGS --ignore $notebook"
done

# Run notebook tests in single worker to prevent resource issues
CI=1 pytest --nbval-lax --nbval-cell-timeout=600 -n 1 $INCLUDED_NOTEBOOKS $IGNORE_FLAGS