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
# Stage 1: Builder — conda + Python 3.11 + torch + gsplat-rade
##################################################

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder
ARG PYTHON_VERSION
ARG TORCH_ARCH_LIST

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl ca-certificates gnupg build-essential git \
        unzip xz-utils cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p /opt/conda \
 && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:${PATH}

RUN conda config --set always_yes true \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# reconstruction env with Python 3.11
RUN conda create -n reconstruction python=${PYTHON_VERSION} -y && conda clean -afy

# CUDA 12.1 toolkit inside the conda env (matches torch cu121 ABI; used by gsplat-rade compile)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate reconstruction && \
    conda install -c 'nvidia/label/cuda-12.1.0' cuda-toolkit -y && conda clean -afy"

ENV CUDA_HOME=/opt/conda/envs/reconstruction \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    PATH=/opt/conda/envs/reconstruction/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    LIBRARY_PATH=/opt/conda/envs/reconstruction/lib:/usr/local/cuda/lib64:${LIBRARY_PATH} \
    CPATH=/opt/conda/envs/reconstruction/include:/usr/local/cuda/include:${CPATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_ARCH_LIST}

# torch 2.5.1 + cu121
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate reconstruction && \
    pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121"

# Verify torch reachable
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate reconstruction && \
    python -c 'import torch; print(f\"[Builder] torch={torch.__version__}, cuda={torch.version.cuda}\")'"

# gsplat-rade fork (compiles CUDA kernels — slow step)
# --no-build-isolation: setup.py imports torch at top-level to query CUDA ABI; env already has torch
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate reconstruction && \
    pip install --no-cache-dir --no-build-isolation \
        setuptools wheel ninja && \
    pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/brian-xu/gsplat-rade.git"

# rclone
RUN curl https://rclone.org/install.sh | bash

##################################################
# Pre-built sources for runtime stage
##################################################

FROM continuumio/miniconda3:latest AS conda-source
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

# Conda base + reconstruction env (torch + gsplat-rade compiled in builder)
COPY --from=conda-source /opt/conda/ /opt/conda
COPY --from=builder /opt/conda/envs/reconstruction/ /opt/conda/envs/reconstruction/

# Colmap binary
COPY --from=colmap-source /usr/local/bin/colmap /usr/local/bin/
COPY --from=colmap-source /usr/local/lib/libcolmap* /usr/local/lib/

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda \
    PATH=/opt/conda/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/conda/envs/reconstruction/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    TORCH_HOME=/workspace/models \
    HF_HOME=/workspace/models

# Smoke test — verifies torch + env importable after copy across stages
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate reconstruction && \
    python -c 'import torch; print(f\"[Runtime] torch={torch.__version__}, cuda={torch.version.cuda}\")' && \
    echo '[Runtime] env verified'"

# SSH
RUN echo "PermitRootLogin yes"        >> /etc/ssh/sshd_config && \
    echo "PermitTTY yes"              >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no"  >> /etc/ssh/sshd_config

# Bashrc: activate reconstruction env in interactive sessions
# Note: conda env lib excluded from LD_LIBRARY_PATH here to avoid libtinfo.so.6 warning;
# conda activate sets it correctly at shell init time.
RUN { \
    echo 'export PATH="/opt/conda/bin:$PATH"'; \
    echo 'export TORCH_HOME="/workspace/models"'; \
    echo 'export HF_HOME="/workspace/models"'; \
    echo 'export CUDA_HOME=/usr/local/cuda'; \
    echo 'export CUDA_ROOT=/usr/local/cuda'; \
    echo 'export PATH="/usr/local/cuda/bin:${PATH}"'; \
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"'; \
    echo 'export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc'; \
    echo 'source /opt/conda/etc/profile.d/conda.sh'; \
    echo 'conda activate reconstruction'; \
    echo 'export PATH="/opt/conda/envs/reconstruction/bin:${PATH}"'; \
    } >> /root/.bashrc

WORKDIR /workspace

CMD bash -c "\
mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && \
service ssh start && \
sleep infinity"
