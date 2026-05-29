# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0
ARG CUDA_ARCHITECTURES="90;89;86;80;75;70;61"
ARG NERFSTUDIO_VERSION=""

##################################################
#           Builder stage (for compilation)      #
##################################################

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder
ARG CUDA_ARCHITECTURES

# Install build dependencies including rclone
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
        && rm -rf /var/lib/apt/lists/*

# Install rclone
RUN curl https://rclone.org/install.sh | bash

# Install conda in builder stage
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Copy environment files
COPY ./env.yml /tmp/env.yml
COPY ./requirements.txt /tmp/requirements.txt
# COPY ./ /opt/collab-splats

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


# Create conda environment
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda env create -n nerfstudio -f /tmp/env.yml"

# Set up build environment variables (cached separately)
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11
ENV CUDA_HOME=/opt/conda/envs/nerfstudio
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install torch and cuda toolkit (cached separately)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate nerfstudio && \
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    conda install -c 'nvidia/label/cuda-11.8.0' cuda-toolkit -y && \
    pip install 'kornia>=0.6.11'"

# Install and setup hloc (cached separately)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate nerfstudio && \
    git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git /opt/hloc && \
    cd /opt/hloc && \
    git checkout v1.4 && \
    git submodule update --init --recursive && \
    pip install -e . --no-cache-dir"

# Install pycolmap for hloc interface (cached separately)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate nerfstudio && \
    pip install --no-cache-dir pycolmap==0.4.0"

# Fix numpy version after hloc (cached separately)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate nerfstudio && \
    conda install -c conda-forge setuptools==69.5.1 'numpy<2.0.0'"

# Install tiny-cuda-nn (cached separately - this is slow)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate nerfstudio && \
    export TORCH_CUDA_ARCH_LIST=\"\$(echo \"${CUDA_ARCHITECTURES}\" | tr ';' '\n' | awk '\$0 > 70 {print substr(\$0,1,1)\".\"substr(\$0,2)}' | tr '\n' ' ' | sed 's/ \$//')\" && \
    pip install -v ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# Install gsplat-rade (cached separately - this is the slowest)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate nerfstudio && \
    export TORCH_CUDA_ARCH_LIST=\"\$(echo \"${CUDA_ARCHITECTURES}\" | tr ';' '\n' | awk '\$0 > 70 {print substr(\$0,1,1)\".\"substr(\$0,2)}' | tr '\n' ' ' | sed 's/ \$//')\" && \
    pip install git+https://github.com/brian-xu/gsplat-rade.git"

# Install nerfstudio from GitHub (this layer will rebuild when you change it)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate nerfstudio && \
    git clone https://github.com/nerfstudio-project/nerfstudio.git /opt/nerfstudio && \
    cd /opt/nerfstudio && \
    pip install . --no-cache-dir"

# Final package installations and numpy fix (cached separately)
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate nerfstudio && \
    conda install -c conda-forge 'numpy<2.0.0' && \
    conda install -c conda-forge cmake>3.5 ninja gmp cgal ipykernel && \
    pip install -r /tmp/requirements.txt"

# Copy example data from GCS using rclone
# Expects the API key to be provided at build time as a build context file
COPY api-key.json /tmp/api-key.json
RUN mkdir -p /opt/data && \
    rclone config create collab-data "google cloud storage" service_account_file=/tmp/api-key.json && \
    rclone copy collab-data:fieldwork_processed/2024_02_06-session_0001/SplatsSD/C0043.MP4 /opt/data/ && \
    rm -f /tmp/api-key.json && \
    rm -rf ~/.config/rclone
# # Build everything in conda environment --> last step is to install buildtools
# RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
#     conda env create -n nerfstudio -f /tmp/env.yml && \
#     conda activate nerfstudio && \

#     # Hack to install our version of rade_gs atm
#     export CC=/usr/bin/gcc-11 && \
#     export CXX=/usr/bin/g++-11 && \
#     export CUDA_HOME=/opt/conda/envs/nerfstudio && \
#     export PATH=\${CUDA_HOME}/bin:\${PATH} && \
#     export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH} && \

#     # Install torch and cuda toolkit
#     pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
#     conda install -c 'nvidia/label/cuda-11.8.0' cuda-toolkit -y && \
#     pip install 'kornia>=0.6.11' && \
  
#     # Install hloc
#     git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git /opt/hloc && \
#     cd /opt/hloc && \
#     git checkout v1.4 && \
#     git submodule update --init --recursive && \
#     pip install -e . --no-cache-dir && \
#     cd ~ && \

#     # Bump down for hloc interface
#     pip install --no-cache-dir pycolmap==0.4.0 && \ 

#     # Now bump back down to numpy 1.26.4
#     conda install -c conda-forge setuptools==69.5.1 'numpy<2.0.0' && \

#     # Install tiny-cuda-nn
#     pip install -v ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch && \
#     export TORCH_CUDA_ARCH_LIST=\"\$(echo \"${CUDA_ARCHITECTURES}\" | tr ';' '\n' | awk '\$0 > 70 {print substr(\$0,1,1)\".\"substr(\$0,2)}' | tr '\n' ' ' | sed 's/ \$//')\" && \
    
#     # Install gsplat-rade
#     pip install git+https://github.com/brian-xu/gsplat-rade.git && \

#     # Changing to clone from github (newer features useful)
#     git clone https://github.com/nerfstudio-project/nerfstudio.git /opt/nerfstudio && \
#     cd /opt/nerfstudio && \
#     pip install -e . && \

#     # pip install nerfstudio && \

#     # Bump the conda version back down --> nerfstudio upgrades for some reason in previous step
#     conda install -c conda-forge 'numpy<2.0.0' && \ 
#     conda install -c conda-forge cmake>3.5 ninja gmp cgal ipykernel && \
#     pip install -r /tmp/requirements.txt"

#     # cd /opt/collab-splats && \
#     # pip install -e ."

##################################################
#           Get pre-built components             #
##################################################

# Get conda from official image
FROM continuumio/miniconda3:latest as conda-source

# Get nerfstudio components
FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.5 as nerfstudio

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
        less \
        && rm -rf /var/lib/apt/lists/*

# Install rclone for runtime use (without keys)
RUN curl https://rclone.org/install.sh | bash

# Copy conda installation from conda-source
COPY --from=conda-source /opt/conda/ /opt/conda

# Copy compiled conda environment from builder
COPY --from=builder /opt/conda/envs/nerfstudio/ /opt/conda/envs/nerfstudio/

# Copy rade_gs from builder for during development --> otherwise we need to run pip install . (instead of -e)
# COPY --from=builder /opt/collab-splats /opt/collab-splats
COPY --from=builder /opt/hloc /opt/hloc

# Copy nerfstudio from builder
COPY --from=builder /opt/nerfstudio /opt/nerfstudio

# Copy example data from builder
COPY --from=builder /opt/data /opt/data

# Copy colmap from nerfstudio
COPY --from=nerfstudio /usr/local/bin/colmap /usr/local/bin/
COPY --from=nerfstudio /usr/local/lib/libcolmap* /usr/local/lib/

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