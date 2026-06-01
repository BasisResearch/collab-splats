# syntax=docker/dockerfile:1
# CUDA 12.1 + torch 2.5.1 + Python 3.11 + gsplat-rade
# Build: docker pull nvidia/cuda:12.1.1-devel-ubuntu22.04 && docker build --platform=linux/amd64 --progress=plain -t collab-env:cu121 .
# After build: bash setup.sh  (installs nerfstudio + bae + collab-splats)

ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.1.1
ARG PYTHON_VERSION=3.11
ARG CUDA_ARCHITECTURES="90;89;86;80;75;70"
ARG TORCH_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

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
    LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH} \
    CPATH=/usr/local/cuda/include:${CPATH} \
    CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    TORCH_CUDA_ARCH_LIST=${TORCH_ARCH_LIST}

# torch 2.5.1 + cu121
RUN pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Verify torch reachable
RUN python -c 'import torch; print(f"[Builder] torch={torch.__version__}, cuda={torch.version.cuda}")'

# Build the full env at image-build time: copy the repo and run the single-source setup.
# uv sync installs all deps incl. cuda-toolkit (nvcc) then compiles bae + gsplat.
# nvcc cross-compiles to TORCH_CUDA_ARCH_LIST; no GPU needed during build.
WORKDIR /workspace/collab-splats
COPY . /workspace/collab-splats
RUN bash setup.sh

# rclone
RUN curl https://rclone.org/install.sh | bash

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
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
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
    } >> /root/.bashrc

WORKDIR /workspace

CMD bash -c "\
mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && \
service ssh start && \
sleep infinity"
