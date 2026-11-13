# `third_party/`

Source clones that are not pip-installable or not in the lock.
Contents are gitignored — only this README is tracked.

## Populating

```bash
bash setup.sh          # uv sync (all extras) + Video-Depth-Anything clone + setup/loger.sh
bash setup/loger.sh    # LoGeR only
bash setup/hloc.sh     # hloc clone @ c13273b (also called by setup.sh; --prefetch caches weights)
```

- scripts are idempotent: an existing clone is not re-cloned
- VDA, LoGeR and hloc clones are re-pinned to their commit on every run
- `hloc/` is a uv path source: it must exist before `uv lock` / `uv sync` (run `setup/hloc.sh` first)

## Policy

- setup scripts own the clone + pin lifecycle; never patch a clone by hand
- deps with no package metadata use a `sys.path` insert at import time
  (see `collab_splats/pointcloud/depth.py`, `collab_splats/pointcloud/feedforward/loger.py`)
- prefer a `[tool.uv.sources]` git pin in `pyproject.toml` over a clone here
- new entries: add a row below and wire it into a setup script

## Current entries

| Path | Upstream | Populated by | Used by |
|---|---|---|---|
| `LoGeR/` | `Junyi42/LoGeR` @ `7685b7a` | `setup/loger.sh` (called by `setup.sh`) | `loger` feedforward backend (`collab_splats/pointcloud/feedforward/loger.py`); `sys.path` insert inside `_load_model`. **No LICENSE file upstream.** |
| `Video-Depth-Anything/` | `DepthAnything/Video-Depth-Anything` @ `4f5ae23` (source only — no weights) | `setup.sh` (VDA block) | Metric video depth for the InstantSfM backend (`collab_splats/pointcloud/depth.py::estimate_depth`); `sys.path` insert of the clone root. vitl checkpoint fetched from HF `depth-anything/Metric-Video-Depth-Anything-Large` on first use. Code Apache-2.0; **vitl weights CC-BY-NC-4.0**. |
| `hloc/` | `cvg/Hierarchical-Localization` @ `c13273b` (recursive: SuperGlue, d2net, r2d2, deep-image-retrieval) | `setup/hloc.sh` (called by `setup.sh` before the sync) | `hloc` sfm backend (`collab_splats/pointcloud/sfm/hloc.py`); editable uv path source, extra `hloc`. **SuperPoint/SuperGlue weights: Magic Leap non-commercial.** |

## Not here — installed from `pyproject.toml`

Git pins in `[tool.uv.sources]`, installed by `uv sync` in `setup.sh`:

| package | upstream | pin |
|---|---|---|
| `vggt` (VGGT-X) | `Linketic/VGGT-X` | `26d1b956` |
| `vggt-omega` | `facebookresearch/vggt-omega` | `39a0cb8` |
| `mapanything` | `facebookresearch/map-anything` | unpinned |
| `lightglue` | `cvg/LightGlue` | unpinned |
| `salad` (DINO-SALAD) | `Dominic101/salad` | unpinned |
| `bae` | `pypose/bae` | `0.2.4` |

- xfeat (and the other local matchers) come from PyPI `vismatch==1.3.1`, not a clone
- InstantSfM is `uv pip install`ed by `setup.sh` from a git URL (`--no-deps`), not cloned here
