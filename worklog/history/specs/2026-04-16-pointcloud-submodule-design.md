# Pointcloud Submodule Design

**Date:** 2026-04-16
**Branch:** `refactor/semantics-pointcloud` (from `tlb-improve-mesh`)
**Prereq:** Agent 1 semantics refactor complete — `collab_splats.semantics.extractors.BaseExtractor` exists

## Problem

Pointcloud creation split across three files, no shared interface:
- `collab_splats/utils/pointcloud.py` — geometric ops (Open3D)
- `stage/feedforward.py` — `Reconstructor` wrapping MapAnything
- `stage/mapanything_utils.py` — MapAnything + COLMAP export

`Splatter` hardcodes MapAnything. Downstream reads `transforms.json` directly — if NS replaced, all callers break.

## Goals

1. Single `BasePointcloudCreator` interface — swap backends without touching `Splatter`
2. `PointcloudResult` carries camera poses in **cam2world OpenGL** — no `transforms.json` dependency
3. `MapAnythingCreator` accepts `BaseExtractor` from Agent 1 semantics module
4. Thin NS SfM wrapper now — cheap insurance against future NS replacement
5. VGGT-X: out of scope, slots in later via same `_colmap_recon_to_result()` helper

## Non-Goals

- Migrating `stage/` (wrap only — remove in follow-up)
- VGGT-X backend (follow-up)
- Changes to training pipeline

## Architecture

```
collab_splats/pointcloud/
    __init__.py       # get_creator(name) registry
    base.py           # PointcloudResult + BasePointcloudCreator + _colmap_recon_to_result()
    utils.py          # clean_pcd, remove_far_points, density_filter (moved from utils/)
    sfm.py            # NerfstudioSfmCreator
    feedforward.py    # MapAnythingCreator
```

`collab_splats/utils/pointcloud.py` → thin re-export shim, removed in follow-up.

## Data Contract

```python
@dataclass
class PointcloudResult:
    points: np.ndarray                    # (N, 3) float32, world XYZ
    colors: np.ndarray                    # (N, 3) uint8, RGB
    confidence: np.ndarray | None         # (N,) float32 — MapAnything only
    camera_poses: np.ndarray | None       # (M, 4, 4) float32, cam2world OpenGL
    camera_intrinsics: np.ndarray | None  # (M, 3, 3) float32, K per image
    colmap_reconstruction: Any | None     # pycolmap.Reconstruction — NS interop only
```

`PointcloudResult` is a plain dataclass of numpy arrays. Open3D types stay inside `utils.py` only.

## Coordinate Convention

All backends output **cam2world OpenGL** in `camera_poses`.

`_colmap_recon_to_result()` in `base.py` normalizes COLMAP output:
```
COLMAP w2c (OpenCV) --[inv]--> c2w (OpenCV) --[c2w[:,1:3] *= -1]--> c2w (OpenGL)
```
Mirrors `nerfstudio/data/dataparsers/colmap_dataparser.py:167-169`.
Splatfacto reads result directly — no further conversion needed.

## Backends

### NerfstudioSfmCreator
- `use_hloc=True`: wraps `nerfstudio.process_data.hloc_utils.run_hloc()`
- `use_hloc=False`: pycolmap extract + match_exhaustive + incremental_mapping
- Both → `_colmap_recon_to_result(recon)`

### MapAnythingCreator
- Wraps `stage.feedforward.Reconstructor`
- `extractor: BaseExtractor | None` — injected from semantics module, None = no features
- Points/colors/confidence from `feedforward_to_pointcloud()`; poses from `build_colmap_reconstruction()` → `_colmap_recon_to_result()`
- Requires updating `stage/feedforward.py:feedforward_to_pointcloud()` to accept `extractor: BaseExtractor | None`

## Error Handling

Both raise `RuntimeError("reconstruction failed — <reason>")`. No silent fallbacks.

## Integration Points

| File | Change |
|------|--------|
| `collab_splats/utils/pointcloud.py` | Re-export shim |
| `stage/feedforward.py` | Add `extractor: BaseExtractor | None` to `feedforward_to_pointcloud()` |
| `collab_splats/wrapper/splatter.py` | Add `pointcloud_method: str = "sfm"`, dispatch via registry |

## Follow-up (out of scope)

- `VGGTCreator` wrapping `nerfstudio.process_data.vggt_utils.run_vggt()` — same `_colmap_recon_to_result()`
- Remove `stage/` once `MapAnythingCreator` stable
- Remove `utils/pointcloud.py` shim
