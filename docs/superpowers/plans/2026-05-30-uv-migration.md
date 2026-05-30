# uv Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Miniconda with uv in the Dockerfile and all setup scripts, eliminating the conda dependency while keeping the same Python 3.11 + CUDA 12.1 + torch 2.5.1 environment.

**Architecture:** The builder stage installs uv, creates `/opt/venv/reconstruction` via `uv venv`, and uses CUDA_HOME=/usr/local/cuda (already present in the devel base image). The runtime stage COPYs the venv from builder. All shell scripts update pip/python paths from `/opt/conda/envs/reconstruction/bin/` to `/opt/venv/reconstruction/bin/`. The `vggt_slam` isolated env switches from `conda create` to `uv venv`.

**Tech Stack:** uv 0.7+, Python 3.11 (uv-managed), CUDA 12.1 (system `/usr/local/cuda` from `nvidia/cuda:12.1.1-devel`), torch 2.5.1+cu121

---

## Files Modified

- `Dockerfile` — remove Miniconda bootstrap, add uv, replace conda env with uv venv, update paths
- `setup.sh` — update PIP path and CUDA_HOME for bae build
- `setup/feedforward.sh` — update PYTHON and PIP paths
- `setup/hloc.sh` — update PYTHON and PIP paths
- `setup/vggt_slam.sh` — replace `conda create` with `uv venv`, update paths and docs
- `CLAUDE.md` — update Python env path reference

---

### Task 1: Rewrite Dockerfile builder stage (remove conda, add uv)

**Files:**
- Modify: `Dockerfile:14-68`

The builder currently installs Miniconda (~500MB), creates a conda env, then runs `conda install cuda-toolkit` to populate CUDA_HOME. The `nvidia/cuda:12.1.1-devel-ubuntu22.04` base image already has nvcc + headers at `/usr/local/cuda`, so the conda CUDA install is redundant. Replace with: install uv, use `uv python install` + `uv venv`.

- [ ] **Step 1: Replace the Dockerfile builder stage**

Replace lines 12–68 in `Dockerfile` with the following. (Keep lines 1–11 `ARG` declarations unchanged.)

```dockerfile
##################################################
# Stage 1: Builder — uv + Python 3.11 + torch + gsplat-rade
##################################################

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder
ARG PYTHON_VERSION
ARG TORCH_ARCH_LIST

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl ca-certificates gnupg build-essential git \
        unzip xz-utils cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# uv — fast Python package manager (replaces conda + pip)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:${PATH}

# Python 3.11 + isolated venv (uv manages the Python install)
RUN uv python install ${PYTHON_VERSION} \
 && uv venv /opt/venv/reconstruction --python ${PYTHON_VERSION}

# CUDA_HOME = system path (nvidia/cuda devel image ships nvcc + headers at /usr/local/cuda)
ENV CUDA_HOME=/usr/local/cuda \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    PATH=/opt/venv/reconstruction/bin:/root/.local/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    LIBRARY_PATH=/opt/venv/reconstruction/lib:/usr/local/cuda/lib64:${LIBRARY_PATH} \
    CPATH=/opt/venv/reconstruction/include:/usr/local/cuda/include:${CPATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_ARCH_LIST}

# torch 2.5.1 + cu121
RUN pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Verify torch reachable
RUN python -c 'import torch; print(f"[Builder] torch={torch.__version__}, cuda={torch.version.cuda}")'

# gsplat-rade fork (compiles CUDA kernels — slow step)
# --no-build-isolation: setup.py imports torch at top-level to query CUDA ABI; env already has torch
RUN pip install --no-cache-dir --no-build-isolation \
        setuptools wheel ninja && \
    pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/brian-xu/gsplat-rade.git

# rclone
RUN curl https://rclone.org/install.sh | bash
```

Note: `pip` and `python` resolve to the venv's binaries because `/opt/venv/reconstruction/bin` is first on PATH. No `conda activate` needed.

- [ ] **Step 2: Commit**

```bash
cd /workspace/collab-splats/.worktrees/uv-migration
git add Dockerfile
git commit -m "refactor(docker): replace conda builder with uv venv"
```

---

### Task 2: Rewrite Dockerfile runtime stage (remove conda-source, COPY venv)

**Files:**
- Modify: `Dockerfile:73-151`

The runtime stage currently pulls `continuumio/miniconda3` as `conda-source` and COPYs the full `/opt/conda/` tree (~1GB+) plus the reconstruction env. Replace with COPYing just `/opt/venv/reconstruction/` from builder.

- [ ] **Step 1: Remove conda-source stage and update runtime stage**

Replace lines 73–151 in `Dockerfile` with the following:

```dockerfile
##################################################
# Pre-built sources for runtime stage
##################################################

FROM colmap/colmap:20240213.23 AS colmap-source

##################################################
# Stage 2: Runtime
##################################################

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-filesystem1.74.0 libboost-program-options1.74.0 \
        libc6 libceres2 libfreeimage3 libgcc-s1 \
        libgl1 libglew2.2 libgoogle-glog0v5 \
        libqt5core5a libqt5gui5 libqt5widgets5 \
        libgl1-mesa-glx libhdf5-dev xvfb \
        build-essential ffmpeg \
        wget curl unzip xz-utils git vim htop tmux less \
        openssh-server gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://rclone.org/install.sh | bash

# uv — needed post-build for setup/vggt_slam.sh to create isolated venv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:${PATH}

# Copy pre-built venv from builder (replaces full conda copy)
COPY --from=builder /opt/venv/reconstruction/ /opt/venv/reconstruction/
# Copy uv-managed Python install so the interpreter is present at its canonical path
COPY --from=builder /root/.local/share/uv/ /root/.local/share/uv/

# Colmap binary
COPY --from=colmap-source /usr/local/bin/colmap /usr/local/bin/
COPY --from=colmap-source /usr/local/lib/libcolmap* /usr/local/lib/

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda \
    PATH=/opt/venv/reconstruction/bin:/root/.local/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/venv/reconstruction/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    TORCH_HOME=/workspace/models \
    HF_HOME=/workspace/models

# Smoke test — verifies torch + venv importable after copy across stages
RUN python -c 'import torch; print(f"[Runtime] torch={torch.__version__}, cuda={torch.version.cuda}")' && \
    echo '[Runtime] env verified'

# SSH
RUN echo "PermitRootLogin yes"        >> /etc/ssh/sshd_config && \
    echo "PermitTTY yes"              >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no"  >> /etc/ssh/sshd_config

# Bashrc: activate venv in interactive sessions (replaces conda activate)
RUN { \
    echo 'export TORCH_HOME="/workspace/models"'; \
    echo 'export HF_HOME="/workspace/models"'; \
    echo 'export CUDA_HOME=/usr/local/cuda'; \
    echo 'export CUDA_ROOT=/usr/local/cuda'; \
    echo 'export PATH="/usr/local/cuda/bin:${PATH}"'; \
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"'; \
    echo 'export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc'; \
    echo 'source /opt/venv/reconstruction/bin/activate'; \
    echo 'export PATH="/opt/venv/reconstruction/bin:${PATH}"'; \
    } >> /root/.bashrc

WORKDIR /workspace

CMD bash -c "\
mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && \
service ssh start && \
sleep infinity"
```

- [ ] **Step 2: Verify Dockerfile is syntactically valid**

```bash
docker buildx build --check /workspace/collab-splats/.worktrees/uv-migration/ 2>&1 | head -20
```

If `docker` is unavailable, at minimum verify the file has no obvious syntax issues:
```bash
grep -n "^FROM\|^RUN\|^ENV\|^COPY\|^ARG" /workspace/collab-splats/.worktrees/uv-migration/Dockerfile
```

Expected: 4 `FROM` lines (builder, colmap-source, runtime, plus the implicit first), matching `ARG` declarations.

- [ ] **Step 3: Commit**

```bash
cd /workspace/collab-splats/.worktrees/uv-migration
git add Dockerfile
git commit -m "refactor(docker): runtime stage uses uv venv, drops conda-source"
```

---

### Task 3: Update setup.sh

**Files:**
- Modify: `setup.sh:5,12`

Two changes: pip path and CUDA_HOME for the bae build.

- [ ] **Step 1: Update PIP path and CUDA_HOME**

In `setup.sh`, make these two changes:

Change line 5:
```bash
PIP=/opt/conda/envs/reconstruction/bin/pip
```
to:
```bash
PIP=/opt/venv/reconstruction/bin/pip
```

Change line 12:
```bash
CUDA_HOME=/opt/conda/envs/reconstruction \
```
to:
```bash
CUDA_HOME=/usr/local/cuda \
```

The full updated file should be:

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIP=/opt/venv/reconstruction/bin/pip
export PIP_ROOT_USER_ACTION=ignore

echo "=== Pre-step a: install cuDSS (must land on disk before bae builds; bae find_cudss_root() scans site-packages/nvidia/cu12 at build time, so it cannot share bae's pip transaction) ==="
$PIP install --no-build-isolation "nvidia-cudss-cu12==0.6.0.5"

echo "=== Pre-step b: build bae CUDA extension (--no-build-isolation required; PIP_NO_BUILD_ISOLATION=1 does not propagate to dependency builds) ==="
CUDA_HOME=/usr/local/cuda \
    $PIP install --no-build-isolation \
        "bae @ git+https://github.com/pypose/bae.git@0.2.4"

echo "=== Step 1: install collab-splats ==="
$PIP install -e "$SCRIPT_DIR"

echo "=== Step 2: install collab-data (private) ==="
$PIP install git+https://github.com/BasisResearch/collab-data.git

echo "=== Step 3: install co3d eval dependency (--no-deps required) ==="
$PIP install git+https://github.com/facebookresearch/co3d.git --no-deps
```

- [ ] **Step 2: Commit**

```bash
cd /workspace/collab-splats/.worktrees/uv-migration
git add setup.sh
git commit -m "refactor(setup): update pip path and CUDA_HOME for uv venv"
```

---

### Task 4: Update setup/feedforward.sh

**Files:**
- Modify: `setup/feedforward.sh:13-14`

- [ ] **Step 1: Update PYTHON and PIP paths**

Change lines 13–14:
```bash
PYTHON=/opt/conda/envs/reconstruction/bin/python
PIP=/opt/conda/envs/reconstruction/bin/pip
```
to:
```bash
PYTHON=/opt/venv/reconstruction/bin/python
PIP=/opt/venv/reconstruction/bin/pip
```

Also update the comment on line 4:
```bash
# Must run with nerfstudio env active, or use the full python path.
```
to:
```bash
# Must use the full python/pip path or have the venv activated.
```

- [ ] **Step 2: Commit**

```bash
cd /workspace/collab-splats/.worktrees/uv-migration
git add setup/feedforward.sh
git commit -m "refactor(setup): feedforward.sh use venv pip/python paths"
```

---

### Task 5: Update setup/hloc.sh

**Files:**
- Modify: `setup/hloc.sh:5-6`

- [ ] **Step 1: Update PIP and PYTHON paths**

Change lines 5–6:
```bash
PIP="/opt/conda/envs/reconstruction/bin/pip"
PYTHON="/opt/conda/envs/reconstruction/bin/python"
```
to:
```bash
PIP="/opt/venv/reconstruction/bin/pip"
PYTHON="/opt/venv/reconstruction/bin/python"
```

- [ ] **Step 2: Commit**

```bash
cd /workspace/collab-splats/.worktrees/uv-migration
git add setup/hloc.sh
git commit -m "refactor(setup): hloc.sh use venv pip/python paths"
```

---

### Task 6: Update setup/vggt_slam.sh

**Files:**
- Modify: `setup/vggt_slam.sh` (full rewrite of env creation section)

Replace `conda create` with `uv venv`. uv must be installed in the runtime image (done in Task 2).

- [ ] **Step 1: Replace conda env creation with uv venv**

Replace the file header + env creation block. The full updated file:

```bash
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
```

- [ ] **Step 2: Commit**

```bash
cd /workspace/collab-splats/.worktrees/uv-migration
git add setup/vggt_slam.sh
git commit -m "refactor(setup): vggt_slam.sh use uv venv instead of conda create"
```

---

### Task 7: Update CLAUDE.md developer docs

**Files:**
- Modify: `CLAUDE.md` (Development Environment section)

- [ ] **Step 1: Update Python env path in CLAUDE.md**

In the `## Development Environment` section, update the Python env line from:
```
- **Python env:** `python` = base conda py3.13 (wrong for this project). Always use `/opt/conda/envs/nerfstudio/bin/python` (py3.11).
```
to:
```
- **Python env:** `python` in base shell may be py3.13 (wrong for this project). Always use `/opt/venv/reconstruction/bin/python` (py3.11), or activate the venv: `source /opt/venv/reconstruction/bin/activate`.
```

In the `## Testing` section, update the test run command from:
```
- Run: `/opt/conda/envs/nerfstudio/bin/python -m pytest tests/`
```
to:
```
- Run: `/opt/venv/reconstruction/bin/python -m pytest tests/`
```

In `## Development Commands`, update:
```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/             # test
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py --help      # eval
```
to:
```bash
/opt/venv/reconstruction/bin/python -m pytest tests/               # test
/opt/venv/reconstruction/bin/python evals/eval_gt.py --help        # eval
```

- [ ] **Step 2: Commit**

```bash
cd /workspace/collab-splats/.worktrees/uv-migration
git add CLAUDE.md
git commit -m "docs(claude): update Python env paths for uv venv"
```

---

### Task 8: Validation checklist

This is an infrastructure change — validation requires a Docker build. Run these commands to validate. If Docker is unavailable in the current environment, record these as manual verification steps.

- [ ] **Step 1: Verify Dockerfile structure is correct**

```bash
grep -n "^FROM\|^RUN\|^ENV\|^COPY\|^ARG" /workspace/collab-splats/.worktrees/uv-migration/Dockerfile
```

Expected output (in order):
```
1:  # syntax comment
6:  ARG UBUNTU_VERSION
7:  ARG NVIDIA_CUDA_VERSION
8:  ARG PYTHON_VERSION
9:  ARG CUDA_ARCHITECTURES
10: ARG TORCH_ARCH_LIST
...
FROM nvidia/cuda:...-devel-... AS builder
FROM colmap/colmap:... AS colmap-source
FROM nvidia/cuda:...-runtime-... AS runtime
```

No `FROM continuumio/miniconda3` should appear.

- [ ] **Step 2: Verify no conda references remain in migrated files**

```bash
grep -rn "conda" \
    /workspace/collab-splats/.worktrees/uv-migration/Dockerfile \
    /workspace/collab-splats/.worktrees/uv-migration/setup.sh \
    /workspace/collab-splats/.worktrees/uv-migration/setup/feedforward.sh \
    /workspace/collab-splats/.worktrees/uv-migration/setup/hloc.sh \
    /workspace/collab-splats/.worktrees/uv-migration/setup/vggt_slam.sh
```

Expected: no matches.

- [ ] **Step 3: Verify no /opt/conda references remain in migrated files**

```bash
grep -rn "/opt/conda" \
    /workspace/collab-splats/.worktrees/uv-migration/Dockerfile \
    /workspace/collab-splats/.worktrees/uv-migration/setup.sh \
    /workspace/collab-splats/.worktrees/uv-migration/setup/feedforward.sh \
    /workspace/collab-splats/.worktrees/uv-migration/setup/hloc.sh \
    /workspace/collab-splats/.worktrees/uv-migration/setup/vggt_slam.sh
```

Expected: no matches.

- [ ] **Step 4: Full docker build (run from outside container / CI)**

```bash
cd /workspace/collab-splats/.worktrees/uv-migration
docker pull nvidia/cuda:12.1.1-devel-ubuntu22.04
docker build --platform=linux/amd64 --progress=plain -t collab-env:cu121-uv .
```

Expected: build completes, smoke test line printed:
```
[Runtime] torch=2.5.1+cu121, cuda=12.1
[Runtime] env verified
```

- [ ] **Step 5: Post-build setup.sh smoke test**

```bash
docker run --rm --gpus all collab-env:cu121-uv bash -c "bash /workspace/setup.sh && echo SETUP_OK"
```

Expected: ends with `SETUP_OK`.

- [ ] **Step 6: Commit validation note if Docker unavailable**

If Docker build cannot be run in this environment, create a note:

```bash
cd /workspace/collab-splats/.worktrees/uv-migration
git commit --allow-empty -m "chore: uv migration complete — docker build validation pending"
```
