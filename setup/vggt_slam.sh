#!/usr/bin/env bash
# setup/vggt_slam.sh — create isolated uv venv for VGGT-SLAM 2.0
#
# VGGT-SLAM requires torch==2.3.1 which conflicts with our reconstruction env
# (torch 2.5.1+cu121). This script creates a dedicated venv that is safe to
# run alongside reconstruction without any interference.
#
# Usage:
#   bash setup/vggt_slam.sh
#
# After setup, run VGGT-SLAM evals via:
#   python evals/runners/run_vggt_slam.py \
#       --image_dir /path/to/images \
#       --output /path/to/out.tum \
#       --python /opt/venv/vggt_slam/bin/python
set -e

VENV_DIR="/opt/venv/vggt_slam"
VGGTSLAM_DIR="$(dirname "$0")/third_party/VGGT-SLAM"
CUDA_TAG="cu121"  # matches our CUDA 12.1 install

echo "=== Creating uv venv: $VENV_DIR (python 3.11) ==="
uv venv "$VENV_DIR" --python 3.11

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

echo "=== Installing torch 2.3.1 + cu121 ==="
"$PIP" install \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "=== Installing VGGT-SLAM base requirements (sans torch) ==="
# Install requirements.txt but skip torch/torchvision (already installed above)
"$PIP" install \
    numpy Pillow open3d huggingface_hub einops safetensors \
    pytorch_metric_learning pytorch-lightning termcolor \
    viser==0.2.23 tqdm omegaconf opencv-python scipy requests \
    trimesh matplotlib lz4 ftfy regex \
    gtsam-develop

echo "=== Installing Salad (DINO-SALAD retrieval) ==="
if [ ! -d "$VGGTSLAM_DIR/third_party/salad" ]; then
    git clone https://github.com/Dominic101/salad.git \
        "$VGGTSLAM_DIR/third_party/salad"
fi
"$PIP" install -e "$VGGTSLAM_DIR/third_party/salad"

echo "=== Installing VGGT SPARK fork (provides compute_similarity API) ==="
if [ ! -d "$VGGTSLAM_DIR/third_party/vggt" ]; then
    git clone https://github.com/MIT-SPARK/VGGT_SPARK.git \
        "$VGGTSLAM_DIR/third_party/vggt"
fi
"$PIP" install -e "$VGGTSLAM_DIR/third_party/vggt"

echo "=== Installing VGGT-SLAM itself ==="
"$PIP" install -e "$VGGTSLAM_DIR"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Python binary for --python arg:"
echo "  $PYTHON"
echo ""
echo "Example eval run:"
echo "  python evals/runners/run_vggt_slam.py \\"
echo "      --image_dir data/7scenes/chess/seq-01 \\"
echo "      --output evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \\"
echo "      --max_loops 1 --max_frames 200 \\"
echo "      --python $PYTHON"
