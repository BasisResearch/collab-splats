# collab-splats

```collab-splats``` provides a flexible platform to derive 3D environment models with integrated semantic information from images and videos. Environment models can be initialized from traditional or modern, feedforward approaches (e.g., COLMAP vs. VGGT respectively). Derived camera poses can then be improved through iterative refinement methods (i.e., bundle adjustment, loop closure). New camera poses can be estimated with keypoint matching. Semantic structure can be lifted into the 3D environment through gaussian splatting or a lightweight compression module. 

## Usage


   ---
   Bundle Adjustment                                                                                                                            ↓

 Purpose: Minimize reprojection error across all frames to tighten camera poses after a feedforward pass. Feedforward models produce plausible but geometrically inconsistent poses; BA enforces multi-view consistency.                                                                    ↓

   What we provide: BundleAdjustment wraps any feedforward creator transparently. It:
   1. Runs the base creator to get initial poses.                                                                                               ↓
   2. Extracts 2D–3D tracks via VGGSfM (extract_tracks_vggsfm).
   3. Runs Levenberg-Marquardt BA (run_bundle_adjustment) to refine intrinsics + extrinsics jointly.                                            ↓
                                                                                                                                                ↓
   Configurable via BundleAdjustmentConfig (learning rate, iterations, convergence thresholds). The wrapper is composable: BundleAdjustment(LoopClosure(VGGTXCreator())) runs LC then BA.                                                                               ↓

   ---
   Loop Closure
                                                                                                                                                ↓
   Purpose: Correct long-range drift accumulation in feedforward reconstructions. Poses are estimated locally and errors compound over long     ↓sequences; loop closure detects when the camera revisits a scene region and enforces global consistency.

   What we provide: LoopClosure wraps any feedforward creator and:                                                                              ↓
   1. Splits the sequence into overlapping submaps.
   2. Computes DINO-SALAD global descriptors per frame for place recognition.
   3. Retrieves candidate loop-closure pairs above a similarity threshold.                                                                      ↓
   4. Runs Sim3 pose graph optimization to propagate corrections across submaps.
   5. Merges windowed outputs into a single unified PointcloudResult.
                                                                                                                                                ↓
   Configurable via LoopClosureConfig (submap size, retrieval threshold, optimization steps, Sim3 scale handling).
                                                                                                                                                ↓
   ---
   Implementation Steps                                                                                                                         ↓

   1. Open README.md
   2. Fill Preprocessing section (after the ### Preprocessing header)                                                                           ↓
   3. Fill Pointclouds section (after ### Pointclouds header)
   4. Fill Refinement intro paragraph
   5. Fill Bundle Adjustment subsection                                                                                                         ↓
   6. Fill Loop Closure subsection

   Verification                                                                                                                                 ↓

   Read the resulting README and confirm:                                                                                                       ↓
   - Each section has prose (no empty body)
   - Technical terms match actual class/method names in codebase
   - No placeholder text remains


### Preprocessing

Extract representative keyframes from a video before reconstruction. Feeding every frame is wasteful and introduces redundancy. Within our repository we provide two frame-selection strategies:
   - FPS (uniform) — samples at a fixed frame rate; fast, no GPU, works well for smooth camera motion
   - Optical flow — Lucas-Kanade-based motion/coverage scorer that favors frames with high disparity and spatial coverage; better for variable-speed capture. Tunable via min_disparity, motion_weight, coverage_weight, max_frames.

Both paths output a set of images that are written to an image directory, used for pointcloud creation.

### Pointclouds

Estimate sparse 3D pointclouds and per-frame camera poses (intrinsics + extrinsics) from a set of images. Our package provides an interface to two families of pointcloud estimators:
   - Structure-from-Motion (SfM): uses feature-matching with incremental reconstruction to define a scene. Slower but geometrically consistent.
   - Feedforward: single forward pass through a pre-trained model 

   What we provide: Two families of creators with a unified interface (reconstruct(image_dir, output_dir) → PointcloudResult):                  ↓

   - Feedforward — single forward pass through a learned model; no iterative refinement, fast.
     - VGGT-X (VGGTXCreator) — joint pose + depth from facebook/VGGT-1B; handles up to 256-frame chunks.                                        ↓
     - MapAnything (MapAnythingCreator) — multiview confidence-weighted depth + pose from facebook/map-anything.
   - Traditional SfM — feature matching + incremental reconstruction; slower but geometrically consistent.
     - COLMAP (ColmapCreator) — SIFT features, exhaustive matching.                                                                             ↓
     - HLoc (HlocCreator) — SuperPoint + SuperGlue learned features; stronger for textureless scenes.

   PointcloudResult carries: world-frame XYZ points, RGB colors, optional per-point confidence, camera poses (M, 4, 4), intrinsics (M, 3, 3), an↓a raw COLMAP reconstruction.
                                                                                                                                                ↓
   Factory: make_creator(name, use_ba=False, use_lc=False, **kwargs) wires up creators with optional refinement wrappers.
                                                                                                                                                ↓

### Refinement

Camera pose estimation is notoriously difficult particularly for modern methods. Feedforward approaches do not enforce geometric consistency and suffer from long-range drift. To address this, we provide a few options that enable the refinement of camera poses. 

#### Bundle Adjustment

Bundle adjustment seeks to minimize reprojection error across all frames.

 Minimize reprojection error across all frames to tighten camera poses after a feedforward pass. Feedforward models produce plausible but geometrically inconsistent poses; BA enforces multi-view consistency.                                                                    ↓

   What we provide: BundleAdjustment wraps any feedforward creator transparently. It:
   1. Runs the base creator to get initial poses.                                                                                               ↓
   2. Extracts 2D–3D tracks via VGGSfM (extract_tracks_vggsfm).
   3. Runs Levenberg-Marquardt BA (run_bundle_adjustment) to refine intrinsics + extrinsics jointly.                                            ↓
                                                                                                                                                ↓
   Configurable via BundleAdjustmentConfig (learning rate, iterations, convergence thresholds). The wrapper is composable: BundleAdjustment(LoopClosure(VGGTXCreator())) runs LC then BA.                                                                               ↓


#### Loop Closure




### Semantics

### Localization


### Meshing

**Documentation:** https://basisresearch.github.io/collab-splats/ — tutorials and API reference. Enable via GitHub repo → Settings → Pages → `gh-pages` branch after first merge to `main`.

## Installation

### Docker

We provide a docker image setup for running nerfstudio with collab-splats (along with other abilities!) at ```tommybotch/collab-splats:latest```

Once the docker image is loaded, please clone and install the repository as follows

```bash
## If public repository could do -- pip install git+https://github.com/BasisResearch/collab-splats
git clone https://github.com/BasisResearch/collab-splats/
cd collab-splats

# This performs pip install -e .
bash setup.sh
```

For use of gcloud data interfaces, please also install collab-data

```bash
pip install git+https://github.com/BasisResearch/collab-data.git
```

#### Building the docker image

The Docker image includes an example video file (`C0043.MP4`) downloaded from Google Cloud Storage during the build process. Follow these steps to build the image:

**Prerequisites:**
1. Obtain a Google Cloud Storage service account key with access to the `collab-data` bucket
2. Save the key as a JSON file

**Build Steps:**

1. **Place the service account key file:**
   ```bash
   # Copy your GCS service account key to the build directory
   cp /path/to/your/service-account-key.json ./api-key.json
   ```
   **Important:** The key file MUST be named exactly `api-key.json` and placed in the same directory as the Dockerfile.

2. **Build the Docker image:**
   ```bash
   docker build --platform=linux/amd64 -t collab-splats:latest .
   ```

3. **Clean up the key file after build:**
   ```bash
   rm ./api-key.json
   ```

**What happens during build:**
- The Docker build process installs rclone
- Uses your service account key to configure GCS access
- Downloads `fieldwork_processed/2024_02_06-session_0001/SplatsSD/C0043.MP4` to `/opt/data/` in the image
- Securely removes all credentials from the final image
- The final image contains rclone (without any stored credentials) and the example video file

**Security Notes:**
- The service account key is only used during build time
- No credentials are stored in the final Docker image
- The key file and rclone configuration are completely removed after the download completes

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
```

Install hloc toolbox for SFM options.

```bash
# Install hloc
git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git /opt/hloc
cd /opt/hloc
git checkout v1.4
git submodule update --init --recursive
pip install -e . --no-cache-dir
cd ~

# Bump down for hloc interface
pip install --no-cache-dir pycolmap==0.4.0 
```

Downgrade setuptools to avoid tinycuda-nn error --> also need a numpy 1.X.X version

```bash
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

# Install nerfstudio from github (newer features available that are useful)
git clone https://github.com/nerfstudio-project/nerfstudio.git /opt/nerfstudio
cd /opt/nerfstudio
pip install . --no-cache-dir

# Bump the numpy version back down (nerfstudio upgrades for some reason)
conda install -c conda-forge 'numpy<2.0.0'
conda install -c conda-forge 'cmake>3.5' ninja gmp cgal ipykernel
pip install -r /tmp/requirements.txt"
```

Lastly, install collab-splats -- currently doing direct clone and egg installation due to private repository. For full functionality, you can optionally install collab-data

```bash
## If public repository could do -- pip install git+https://github.com/BasisResearch/collab-splats
git clone https://github.com/BasisResearch/collab-splats/
cd collab-splats

# Runs pip install -e .
bash setup.sh

# Optional install of collab-data
pip install git+https://github.com/BasisResearch/collab-data.git
```

## Usage

collab-splats is built to integrate different gaussian splatting codebases that enable depth and normal map creation. Specifically, it implements the depth-normal consistency loss

Two models are currently offered:
- **rade-gs:** the baseline extension model that enables depth and normal map creation within the rasterization process. This is built on top of [gsplat-rade](https://github.com/brian-xu/gsplat-rade) and is heavily inspired by the [scaffold-gs-nerfstudio](https://github.com/brian-xu/scaffold-gs-nerfstudio) implementation.
- **rade-features:**  extends rade-gs to enable splatting of ANN feature spaces. This draws inspiration from the original [feature-splatting-ns](https://github.com/vuer-ai/feature-splatting) implementation but contains additional functionality.

Within the class ```Splatter``` we provide the ability to preprocess, train, and visualize splatting models within nerfstudio. We also enable meshing as a post-processing strategy for all splatting outputs.

For examples of these different functionalities, please navigate to the ```examples/``` directory.

The Docker image contains an example splat video at `/opt/data/C0043.MP4`.

## Problems

Things aren't showing up in plots? Check [VSCode forwarding settings](https://github.com/pyvista/pyvista/issues/5296#issuecomment-1971079419)