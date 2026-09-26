# syntax=docker/dockerfile:1
# CUDA 12.1 + torch 2.5.1 + Python 3.11; the env is whatever setup.sh builds.
# Build (from the collab-splats checkout, with collab-data cloned beside it):
#   docker build --platform=linux/amd64 --progress=plain \
#     --build-context collab-data=../collab-data \
#     --build-arg MAX_JOBS=4 -t collab-splats:release .
# - collab-data is private and locked as a path dep at /workspace/collab-data
# - MAX_JOBS: Docker memory GB / 8 (one cicc peaks ~7.3 GB), ceiling 6
#
# Apple Silicon: every nvcc call runs under amd64 emulation
# - Docker Desktop > General: enable "Use Rosetta for x86_64/amd64 emulation"; QEMU is far slower
# - Docker Desktop > Resources: raise memory, then set MAX_JOBS from it
# - first build still takes hours; the CUDA layer is then cached until pyproject.toml, uv.lock,
#   setup.sh or collab-data change
#
# GPU support: TORCH_ARCH_LIST="8.0+PTX" (fused-ssim: CUDA_ARCHITECTURES="80;90", set in setup.sh)
# - native: Ampere + Ada, sm_80/86/89 (A100, A40, A6000, A10, L4, L40, RTX 3090, RTX 4090)
# - JIT on first launch: Hopper, sm_90 (H100, H200); cached per container via CUDA_CACHE_MAXSIZE
# - NOT supported: Volta/Turing (V100, T4, RTX 2080); add "7.0;7.5" to TORCH_ARCH_LIST
# - NOT supported: Blackwell (B200, RTX 5090); torch 2.5.1+cu121 itself ships no kernels for it

ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.1.1
ARG PYTHON_VERSION=3.11
ARG TORCH_ARCH_LIST="8.0+PTX"

##################################################
# Stage 1: Builder — runs setup.sh (uv sync + AOT CUDA extensions)
##################################################

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder
ARG PYTHON_VERSION
ARG TORCH_ARCH_LIST
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl ca-certificates gnupg build-essential git \
        unzip xz-utils cmake ninja-build \
        libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

# uv manages the Python install and the venv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:${PATH}

RUN uv python install ${PYTHON_VERSION} \
 && uv venv /opt/venv/reconstruction --python ${PYTHON_VERSION}

# System toolkit from the devel image; nvcc cross-compiles to TORCH_CUDA_ARCH_LIST, no GPU needed
# - CPATH is the toolkit's own include; never add the CCCL overlay here (setup.sh explains)
ENV CUDA_HOME=/usr/local/cuda \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    PATH=/opt/venv/reconstruction/bin:/root/.local/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH} \
    CPATH=/usr/local/cuda/include:${CPATH} \
    CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    TORCH_CUDA_ARCH_LIST=${TORCH_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS}

# collab-data: only what its setup.py installs (never .git, notebooks or the config-local symlink)
COPY --from=collab-data setup.py setup.cfg README.rst /workspace/collab-data/
COPY --from=collab-data collab_data /workspace/collab-data/collab_data

# Single source of truth: setup.sh syncs the lock, compiles bae/gsplat/fused-ssim/nvdiffrast AOT,
# and clones the pinned third_party sources (VDA, LoGeR)
# - pass 1 sees only the lock: the slow CUDA compiles stay cached across source edits
# - pass 2 installs the project itself and the non-lock extras
WORKDIR /workspace/collab-splats
COPY pyproject.toml uv.lock LICENSE setup.sh /workspace/collab-splats/
RUN SETUP_DEPS_ONLY=1 bash setup.sh
COPY . /workspace/collab-splats
RUN bash setup.sh

##################################################
# Stage 2: Runtime — no nvcc; every CUDA extension was built AOT above
##################################################

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
        libc6 libgcc-s1 libgl1 libgl1-mesa-glx \
        libx11-6 libxext6 libsm6 libice6 \
        libhdf5-dev xvfb \
        build-essential ffmpeg libsuitesparse-dev \
        wget curl unzip xz-utils git vim htop tmux less \
        openssh-server gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://rclone.org/install.sh | bash

# uv — manages the venv post-build (setup.sh reruns, uv pip into the venv)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:${PATH}

# venv + the uv-managed interpreter it points at
COPY --from=builder /opt/venv/reconstruction/ /opt/venv/reconstruction/
COPY --from=builder /root/.local/share/uv/ /root/.local/share/uv/

# collab-splats is an editable install: its path must match the builder's
COPY --from=builder /workspace/collab-splats /workspace/collab-splats

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda \
    PATH=/opt/venv/reconstruction/bin:/root/.local/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_HOME=/workspace/models \
    HF_HOME=/workspace/models \
    CUDA_CACHE_MAXSIZE=4294967296

# Smoke test: torch imports and the AOT extensions survived the stage copy
# - no GPU during build, so the creator chain is checked at run time (--gpus), not here
# - a missing .so would JIT-compile on first use, and this stage has no nvcc
# - cv2/open3d/pycolmap import here so a missing system lib fails the build, not the first run
RUN python - <<'EOF'
import glob, sysconfig
import cv2, open3d, pycolmap, torch

assert pycolmap.has_cuda, "pycolmap is the CPU wheel; expected pycolmap-cuda12"

site = sysconfig.get_paths()["purelib"]
for pattern in ("gsplat/csrc*.so", "_nvdiffrast_c*.so", "bae/sparse/*.so"):
    assert glob.glob(f"{site}/{pattern}"), f"missing AOT build: {pattern}"
print(f"[Runtime] torch={torch.__version__} cuda={torch.version.cuda}; AOT extensions present; pycolmap={pycolmap.__version__}")
EOF

# SSH
RUN echo "PermitRootLogin yes"        >> /etc/ssh/sshd_config && \
    echo "PermitTTY yes"              >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no"  >> /etc/ssh/sshd_config

# Bashrc: activate the venv in interactive sessions
RUN { \
    echo 'export TORCH_HOME="/workspace/models"'; \
    echo 'export HF_HOME="/workspace/models"'; \
    echo 'export CUDA_HOME=/usr/local/cuda'; \
    echo 'export CUDA_ROOT=/usr/local/cuda'; \
    echo 'export PATH="/usr/local/cuda/bin:${PATH}"'; \
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"'; \
    echo 'source /opt/venv/reconstruction/bin/activate'; \
    } >> /root/.bashrc

WORKDIR /workspace

CMD bash -c "\
mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && \
service ssh start && \
sleep infinity"
