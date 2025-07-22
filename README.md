# collab-splats

Extension tools for nerfstudio enabling depth/normal derivation and meshing (among other functions) for gaussian splatting.

## Installation

### Docker

We provide a docker image setup for running nerfstudio with collab-splats (along with other abilities!) at ```tommybotch/collab-environment:latest```

### Conda
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

Now is where pain begins... tinycuda-nn is the big snag point of installation -- it will also take the most amount of time.

```bash
# Note which CUDA architectures to build for
export TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}

pip install -v ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Install gsplat-rade and nerfstudio -- this gsplat version **is required** to run this code, as it contains the CUDA kernel for calculating depth and normal maps. 

```bash
# Install specific gsplat version
pip install git+https://github.com/brian-xu/gsplat-rade.git

# Install nerfstudio
pip install nerfstudio

# Bump the numpy version back down (nerfstudio upgrades for some reason)
conda install -c conda-forge 'numpy<2.0.0'
```

Lastly, install collab-splats -- currently doing direct clone and egg installation due to private repository

```bash
## If public repository could do -- pip install git+https://github.com/BasisResearch/collab-splats
git clone https://github.com/BasisResearch/collab-splats/
cd collab-splats
pip install -e .
```

## Usage

collab-splats is built to integrate different gaussian splatting codebases that enable depth and normal map creation. Specifically, it implements the depth-normal consistency loss

Two models are currently offered:
- **rade-gs:** the baseline extension model that enables depth and normal map creation within the rasterization process. This is built on top of [gsplat-rade](https://github.com/brian-xu/gsplat-rade) and is heavily inspired by the [scaffold-gs-nerfstudio](https://github.com/brian-xu/scaffold-gs-nerfstudio) implementation.
- **rade-features:**  extends rade-gs to enable splatting of ANN feature spaces. This draws inspiration from the original [feature-splatting-ns](https://github.com/vuer-ai/feature-splatting) implementation but contains additional functionality.

Within the class ```Splatter``` we provide the ability to preprocess, train, and visualize splatting models within nerfstudio. We also enable meshing as a post-processing strategy for all splatting outputs.

For examples of these different functionalities, please navigate to the ```examples/``` directory.
