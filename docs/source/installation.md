# Installation

Install with `bash setup.sh` (see the [README](https://github.com/BasisResearch/collab-splats#install)). `setup.sh` is safe to re-run; it only builds what is missing. On RunPod, the `/workspace` volume hides the image's own checkout, so clone into `/workspace` and run `setup.sh` there. For a lighter install, sync extras individually:

```sh
uv sync                                         # core deps only
uv sync --extra semantics                       # + features and segmentation (no CUDA build)
uv sync --extra reconstruction --extra feedforward --extra remote   # + full pipeline and CLI
uv sync --extra instantsfm                      # + InstantSfM backend (needs libsuitesparse-dev)
uv sync --extra hloc                            # + hloc backend
uv sync --extra gpu                             # + CUDA headers / libs torch does not ship
uv sync --extra dashboard                       # + dashboard
uv sync --extra docs                            # + Sphinx docs build
uv sync --extra dev                             # + lint / test tooling
uv sync --all-extras                            # everything (what setup.sh does)
```

Every `uv sync` needs the two path sources in the lock to exist first; `setup.sh` creates both:

- `collab-data` at `/workspace/collab-data` (see [Private dependency](#private-dependency-collab-data)).
- the hloc clone in `third_party/hloc`: run `bash setup/hloc.sh`.

The `loger` backend is a source clone, not a package: run `bash setup/loger.sh`.

> **Use `setup.sh`, not a bare `uv sync`.**
> - The CUDA extensions build from source and need `nvcc` + `build-essential`; `setup.sh` points the build at your CUDA toolkit and warns if `nvcc` is missing.
> - `uv pip install` from outside this repo skips our dependency overrides, so it does not match `uv.lock`.

All later commands assume the venv is active (`source /opt/venv/reconstruction/bin/activate`), or prefix them with `uv run`.

## Semantics only

Other projects can install just the semantics stack. It needs no CUDA toolkit and skips `bae`, `pycolmap`, `open3d`, `gsplat` and `collab-data`:

```sh
uv pip install 'collab_splats[semantics] @ git+https://github.com/BasisResearch/collab-splats@<commit>'
```

uv still reads our package sources, so on Linux this pulls torch from the CUDA 12.1 index. To use your own torch (e.g. CPU-only), add `--no-sources` and pick torch yourself:

```sh
uv pip install --no-sources --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  'torch==2.5.1+cpu' 'torchvision==0.20.1+cpu' \
  'collab_splats[semantics] @ git+https://github.com/BasisResearch/collab-splats@<commit>'
```

In another uv project, set `no-sources = true` under its `[tool.uv]` instead.

This covers the `talk2dino`, `dinov2` and `ocr_lens` features and the `insid3` and `skywater` segmenters. `maskclip` and `mobilesamv2` need the `reconstruction` extra, and `sam3` has its own install. `ocr_lens` also needs the WordNet corpus (`setup.sh` downloads it): `python -m nltk.downloader wordnet`.

## Private dependency (collab-data)

`collab-data` is a private BasisResearch repo that must sit at `/workspace/collab-data`. `setup.sh` clones it there if missing (needs GitHub credentials).

The dashboard and `reconstruct remote` read and write scenes through an rclone remote named `collab-data` (Google Cloud Storage). Install `rclone` and `jq`:

```sh
brew install rclone jq                                                  # macOS
curl -fsSLo rclone.zip https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip \
  && unzip -q rclone.zip && sudo install -m 0755 rclone-v1.75.2-linux-amd64/rclone /usr/local/bin/ \
  && sudo apt-get install -y jq                                         # Linux (apt's rclone is too old)
```

The Docker image already has both.

Save the GCS service-account key to `collab-data/config-local/collab-data.json`, then run the setup script from the collab-data repo:

```sh
./scripts/setup_local_rclone.sh
```

## System requirements

You provide these; `setup.sh` never installs system packages.

- **NVIDIA GPU + driver** for the feedforward models, splats and mesh texturing. Other stages fall back to CPU, but slowly and untested.
- **build-essential** (gcc/g++) + a CUDA toolkit to compile the extensions.
- **libsuitesparse-dev** for `scikit-sparse` (the `instantsfm` extra).
- Optional: `ffmpeg` + `exiftool` for the curated-video scripts; `rclone` + `jq` for GCloud scenes.

