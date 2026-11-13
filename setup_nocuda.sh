#!/bin/bash
# Standalone setup w/o CUDA: the semantics extra (no CUDA build) + dev tools into <repo>/.venv
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
uv sync --locked --extra semantics --extra dev
