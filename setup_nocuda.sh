#!/bin/bash

python -m pip install --upgrade pip setuptools wheel scikit-build-core pybind11
pip install "numpy<2.0"
pip install torch==2.1.2+cpu torchvision==0.16.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Install dependencies
BUILD_NO_CUDA=1 FORCE_CUDA=0 pip install --no-build-isolation -e ".[dev]"

