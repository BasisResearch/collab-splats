#!/bin/bash
set -euxo pipefail

SRC="collab_splats tests"
ruff check --fix $SRC || true
ruff format $SRC