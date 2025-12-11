#!/bin/bash

# Install dependencies
pip install -e .

# Install hloc
git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git /opt/hloc
cd /opt/hloc
git checkout v1.4
git submodule update --init --recursive
pip install -e . --no-cache-dir
git checkout v1.5 # Bump upwards to get latest features

# Install collab-data
pip install git+https://github.com/BasisResearch/collab-data.git