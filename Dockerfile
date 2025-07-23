# Docker multi-stage build
# syntax=docker/dockerfile:1
# Build with the following commands:
# docker build --platform=linux/amd64 --progress=plain -t tommybotch/collab-splats .

ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0
ARG CUDA_ARCHITECTURES="90;89;86;80;75;70;61"
ARG NERFSTUDIO_VERSION=""
ARG NUMBER_OF_CORES=8

##################################################
#           Builder stage (for compilation)      #
##################################################

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder
ARG CUDA_ARCHITECTURES
ARG NUMBER_OF_CORES

# Use faster apt mirror
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirror.math.princeton.edu/pub/ubuntu/|g' /etc/apt/sources.list

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        curl \
        unzip \
        build-essential \
        g++-11 \
        gcc-11 \
        cmake \
        ninja-build \
        git \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libboost-test-dev \
        libsuitesparse-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libatlas-base-dev \
        libsuitesparse-dev \
        libhdf5-dev \  
        && rm -rf /var/lib/apt/lists/*

# Install conda in builder stage
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Copy environment files
COPY ./env.yml /tmp/env.yml
COPY ./requirements.txt /tmp/requirements.txt
COPY ./ /opt/collab-splats

## Building dependencies from source (found here: https://github.com/cvg/pixel-perfect-sfm/issues/41)

# Eigen
RUN git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git /opt/eigen && \
    cd /opt/eigen && \
    mkdir build && cd build && \
    cmake .. && \
    make install

# Ceres Solver - Build with LTO disabled
RUN git clone https://ceres-solver.googlesource.com/ceres-solver /opt/ceres-solver && \
    cd /opt/ceres-solver && \
    git checkout 2.1.0rc2 && \
    mkdir build && cd build && \
    cmake .. \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
        -DCMAKE_CXX_FLAGS="-fno-lto -O2" \
        -DCMAKE_C_FLAGS="-fno-lto -O2" \
        -G Ninja && \
    ninja -j${NUMBER_OF_CORES} && \
    ninja install

# COLMAP (with CUDA and Ninja)
RUN git clone https://github.com/colmap/colmap.git /opt/colmap && \
    cd /opt/colmap && \
    git checkout 3.8 && \
    mkdir build && cd build && \
    cmake .. \
        -DCUDA_ENABLED=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
        -DCMAKE_CXX_FLAGS="-fno-lto -O2" \
        -DCMAKE_C_FLAGS="-fno-lto -O2" \
        -G Ninja && \
    ninja -j${NUMBER_OF_CORES} && \
    ninja install

# Set CUDA architectures
# 61 = GTX 10xx -- e.g. 1080Ti (Pascal SM61)
# 70 = V100 (Volta SM70)
# 75 = RTX 20xx -- e.g. 2080Ti (Turing SM75)
# 80 = A100 (Ampere SM80)
# 86 = RTX 30xx + A40 -- e.g. 3090 (Ampere SM86)
# 89 = H100, GH100 -- (Hopper SM89)
# 90 = RTX 5090, H200 -- Ada Lovelace SM90
# Set CUDA environment variables
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}

# Accept Anaconda Terms of Service non-interactively
RUN conda config --set always_yes true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Build everything in conda environment --> last step is to install buildtools
RUN /bin/bash -c \
    "source /opt/conda/etc/profile.d/conda.sh && \
    conda env create -n nerfstudio -f /tmp/env.yml && \
    conda activate nerfstudio && \
    \
    # Set compiler versions and disable LTO globally
    export CC=/usr/bin/gcc-11 && \
    export CXX=/usr/bin/g++-11 && \
    export CUDA_HOME=/opt/conda/envs/nerfstudio && \
    export PATH=\${CUDA_HOME}/bin:\${PATH} && \
    export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH} && \
    \
    # Comprehensive LTO and optimization flags
    export CXXFLAGS='-fno-lto -O2 -fPIC' && \
    export CFLAGS='-fno-lto -O2 -fPIC' && \
    export LDFLAGS='-fno-lto' && \
    export CMAKE_CXX_FLAGS='-fno-lto -O2 -fPIC' && \
    export CMAKE_C_FLAGS='-fno-lto -O2 -fPIC' && \
    export CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF && \
    \
    # Control parallelization to avoid build issues
    export MAKEFLAGS='-j2' && \
    export CMAKE_BUILD_PARALLEL_LEVEL=2 && \
    export MAX_JOBS=2 && \
    \
    # Install torch and other dependencies
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    conda install -c 'nvidia/label/cuda-11.8.0' cuda-toolkit -y && \
    conda install -c conda-forge setuptools==69.5.1 'numpy<2.0.0' && \
    \
    # Install hloc and pixel-perfect-sfm
    git clone https://github.com/cvg/pixel-perfect-sfm --recursive /opt/pixel-perfect-sfm && \
    cd /opt/pixel-perfect-sfm && \
    pip install -r requirements.txt && \
    \
    # Now install hloc
    git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git /opt/hloc && \
    cd /opt/hloc && \
    git checkout v1.4 && \
    pip install -e . && \
    \
    # Install pixsfm with explicit compiler flags
    cd /opt/pixel-perfect-sfm && \
    CMAKE_ARGS='-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DCMAKE_CXX_FLAGS=\"-fno-lto -O2 -fPIC\" -DCMAKE_C_FLAGS=\"-fno-lto -O2 -fPIC\"' pip install -e . --no-cache-dir -v && \
    \
    # Bump pycolmap to 0.6.0
    pip install pycolmap==0.6.0 && \ 
    \
    # Install gsplat-rade with proper flags
    pip install -v ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch && \
    export TORCH_CUDA_ARCH_LIST=\"\$(echo \"${CUDA_ARCHITECTURES}\" | tr ';' '\n' | awk '\$0 > 70 {print substr(\$0,1,1)\".\"substr(\$0,2)}' | tr '\n' ' ' | sed 's/ \$//')\" && \
    CMAKE_ARGS='-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DCMAKE_CXX_FLAGS=\"-fno-lto -O2 -fPIC\" -DCMAKE_C_FLAGS=\"-fno-lto -O2 -fPIC\"' pip install git+https://github.com/brian-xu/gsplat-rade.git && \
    pip install nerfstudio && \
    \
    # Bump the conda version back down --> nerfstudio upgrades for some reason in previous step
    conda install -c conda-forge 'numpy<2.0.0' && \ 
    conda install -c conda-forge cmake>3.5 ninja gmp cgal ipykernel && \
    pip install -r /tmp/requirements.txt && \
    \
    cd /opt/collab-splats && \
    pip install -e ."

##################################################
#           Get pre-built components             #
##################################################

# Get conda from official image
FROM continuumio/miniconda3:latest as conda-source

# Get nerfstudio components
# FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.5 as nerfstudio

##################################################
#           Runtime stage                        #
##################################################

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-filesystem1.74.0 \
        libboost-program-options1.74.0 \
        libc6 \
        libceres2 \
        libfreeimage3 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v5 \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5 \
        libgl1-mesa-glx \
        libhdf5-dev \
        xvfb \
        python3.10 \
        python3.10-dev \
        python-is-python3 \
        python3-pip \
        build-essential \
        ffmpeg \
        wget \
        curl \
        unzip \
        git \
        vim \
        htop \
        tmux \
        openssh-server \
        && rm -rf /var/lib/apt/lists/*

# Copy conda installation from conda-source
COPY --from=conda-source /opt/conda/ /opt/conda/

# Copy compiled conda environment from builder
COPY --from=builder /opt/conda/envs/nerfstudio/ /opt/conda/envs/nerfstudio/

# Copy collab-splats project source
COPY --from=builder /opt/collab-splats /opt/collab-splats

# Copy COLMAP (compiled from source)
COPY --from=builder /usr/local/bin/colmap /usr/local/bin/
COPY --from=builder /usr/local/lib/libcolmap* /usr/local/lib/
COPY --from=builder /usr/local/include/colmap /usr/local/include/colmap

# Copy Ceres (compiled from source)
COPY --from=builder /usr/local/lib/libceres* /usr/local/lib/
COPY --from=builder /usr/local/include/ceres /usr/local/include/ceres

# Copy Eigen (header-only)
COPY --from=builder /usr/local/include/eigen3 /usr/local/include/eigen3

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Set environment variables
ENV PATH="/opt/conda/bin:$PATH"
ENV TORCH_HOME="/workspace/models"
ENV HF_HOME="/workspace/models"

# SSH configuration
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PermitTTY yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

# Add environment setup to bashrc
RUN echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc && \
    echo 'export TORCH_HOME="/workspace/models"' >> ~/.bashrc && \
    echo 'export HF_HOME="/workspace/models"' >> ~/.bashrc && \
    echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc && \
    echo 'export CUDA_ROOT=/usr/local/cuda' >> ~/.bashrc && \
    echo 'export PATH="/usr/local/cuda/bin:${PATH}"' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"' >> ~/.bashrc && \
    echo 'export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc' >> ~/.bashrc && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate nerfstudio' >> ~/.bashrc

WORKDIR /workspace

CMD bash -c "\
apt update && \
mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && \
service ssh start && \
sleep infinity"