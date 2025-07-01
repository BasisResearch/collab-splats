# rade_gs
RaDe-GS implemented for nerfstudio

STILL TESTING

## Installation

Follow the [NerfStudio instllation instructions](https://docs.nerf.studio/quickstart/installation.html) to install a conda environment. For convenience, here are the commands I've used to successfully build a nerfstudio environment.

**Note:** This requires cuda developer tools -- specifically nvcc

Create an isolated conda environment (I've successfully built with python3.10)

```bash
# Set our system
export UBUNTU_VERSION=22.04
export NVIDIA_CUDA_VERSION=11.8.0

# You can remove some of these and fit them to your system needs 
export CUDA_ARCHITECTURES="90;89;86;80;75;70;61" 

conda create --name nerfstudio -y python=3.10
conda activate nerfstudio
```

Next install torch and torchvision built for cuda11.8 -- this specifically has to be run via pip for tinycuda-nn to detect the packages.

```bash
# Install torch and torchvision (from specified URL)
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install cuda developer tools 
conda install -c 'nvidia/label/cuda-11.8.0' cuda-toolkit -y

# Downgrade setuptools to avoid tinycuda-nn error --> also need a numpy 1.X.X version
conda install -c conda-forge setuptools==69.5.1 'numpy<2.0.0'
```

Now is where pain begins... tinycuda-nn is the big snag point of installation

```bash
# Note which CUDA architectures to build for
export TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}

pip install -v ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Lastly install gsplat-rade and nerfstudio -- this gsplat version **is required** to run this code, as it contains the CUDA kernel for calculating depth and normal maps. 

```bash
# Install specific gsplat version
pip install git+https://github.com/brian-xu/gsplat-rade.git

# Install nerfstudio
pip install nerfstudio
```

### Optional additions

There are a number of "nice-to-haves" for additional functionality. Currently working on building out tools for feature-splatting with RaDe as the base model. 

```bash

```

here are the commands I run to install nerfstudio on two machines.


Needs to run the following additional extensions

```

conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal

```ls
