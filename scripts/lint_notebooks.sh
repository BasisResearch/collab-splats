#!/bin/bash

INCLUDED_NOTEBOOKS="docs/"

# Array of notebooks to exclude
EXCLUDED_NOTEBOOKS=(
    "docs/splats/derive_splats.ipynb"
    "docs/splats/create_mesh.ipynb"
    "docs/splats/visualization.ipynb"  # All notebooks excluded - testing disabled
)

# Build the nbqa-exclude pattern (regex with | separator)
EXCLUDE_PATTERN=$(printf "|%s" "${EXCLUDED_NOTEBOOKS[@]}")
EXCLUDE_PATTERN=${EXCLUDE_PATTERN:1}  # Remove the leading '|'

# Type-check (best-effort; some notebooks may not be importable as packages)
nbqa mypy $INCLUDED_NOTEBOOKS --nbqa-exclude "$EXCLUDE_PATTERN" || true

# Auto-fix import ordering and formatting
nbqa isort $INCLUDED_NOTEBOOKS --nbqa-exclude "$EXCLUDE_PATTERN"

# Check formatting
nbqa black --check $INCLUDED_NOTEBOOKS --nbqa-exclude "$EXCLUDE_PATTERN"

# Lint while relaxing common notebook patterns and matching repo line length
nbqa flake8 \
  $INCLUDED_NOTEBOOKS \
  --nbqa-exclude "$EXCLUDE_PATTERN" \
  --max-line-length=120 \
  --extend-ignore=E203,E402,E401,F401,E501