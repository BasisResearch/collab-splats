#!/bin/bash

set -e  # Exit on error

# Install dependencies
pip install -e .

# Install collab-data
pip install git+https://github.com/BasisResearch/collab-data.git

# Optional: Install FastVGGT for accelerated structure-from-motion (~4x faster than standard VGGT)
# This enables the --sfm-tool fastvggt option in ns-process-data
# Uncomment the section below to install:
#
# echo ""
# echo "Installing FastVGGT for accelerated SfM..."
#
# # Clone FastVGGT if not already present
# if [ ! -d "FastVGGT" ]; then
#     echo "Cloning FastVGGT repository..."
#     git clone https://github.com/mystorm16/FastVGGT.git
# else
#     echo "FastVGGT directory already exists, skipping clone..."
# fi
#
# # Create pyproject.toml if it doesn't exist (for proper pip installation)
# if [ ! -f "FastVGGT/pyproject.toml" ]; then
#     echo "Creating pyproject.toml for FastVGGT..."
#     cat > FastVGGT/pyproject.toml << 'EOF'
# [project]
# name = "vggt"
# version = "0.0.2"
# description = "FastVGGT - Accelerated Visual Geometry Grounded Transformer"
# authors = [{name = "FastVGGT Contributors"}]
# dependencies = [
#     "numpy<2",
#     "Pillow",
#     "huggingface_hub",
#     "einops",
#     "safetensors",
#     "opencv-python",
#     "torch",
#     "torchvision",
# ]
# requires-python = ">= 3.10"
#
# [build-system]
# requires = ["setuptools>=61.0", "wheel"]
# build-backend = "setuptools.build_meta"
#
# [tool.setuptools.packages.find]
# where = ["."]
# include = ["vggt*", "merging*"]
# EOF
# fi
#
# # Install FastVGGT (this upgrades vggt 0.0.1 -> 0.0.2 with acceleration support)
# echo "Installing FastVGGT package..."
# pip install -e FastVGGT/ --force-reinstall --no-deps
#
# echo ""
# echo "✅ FastVGGT installation complete!"
# echo "You can now use: ns-process-data video --data video.mp4 --output-dir out --sfm-tool fastvggt"
# echo ""