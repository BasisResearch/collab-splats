#!/usr/bin/env bash
# setup/loger.sh — vendor LoGeR into third_party/ for the `loger` feedforward backend.
#
# LoGeR = Pi3 backbone + TTT fast-weight memory + sliding-window inference.
# Used by collab_splats/pointcloud/feedforward/loger.py via a sys.path insert
# inside _load_model (the tree is not pip-installed).
#
# Upstream is pinned: github.com/Junyi42/LoGeR @ 7685b7a is the commit that adds
# "load images to cpu to reduce VRAM usage", which is load-bearing under our
# 46.6 GB cgroup cap and is absent from the PolyCam fork.
#
# Usage:
#   bash setup/loger.sh
set -e

LOGER_COMMIT="7685b7a"
LOGER_DIR="$(dirname "$0")/../third_party/LoGeR"

if [ ! -d "$LOGER_DIR" ]; then
    echo "=== Cloning LoGeR into $LOGER_DIR ==="
    git clone https://github.com/Junyi42/LoGeR.git "$LOGER_DIR"
else
    echo "=== $LOGER_DIR already exists, skipping clone ==="
fi

echo "=== Pinning to $LOGER_COMMIT ==="
git -C "$LOGER_DIR" fetch --all --quiet
git -C "$LOGER_DIR" checkout --quiet "$LOGER_COMMIT"

echo ""
echo "=== LoGeR vendored at $(git -C "$LOGER_DIR" rev-parse --short HEAD) ==="
echo "Checkpoint weights download on first use from HF Junyi42/LoGeR."
echo "No pip install needed — zero new dependencies."
