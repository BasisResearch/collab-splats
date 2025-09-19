#!/bin/bash
set -euxo pipefail

SRC_FILES="tests/ collab_splats/"
SRC_PKG="collab_splats"

mypy -p "$SRC_PKG" --follow-imports=skip --incremental --cache-dir .mypy_cache
ruff check $SRC_FILES
ruff format --diff $SRC_FILES