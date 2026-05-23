# Pointcloud Submodule — Feedforward Integration & Architecture Refactor

**Date:** 2026-04-20
**Branch:** `refactor/core-modules` → new branch from this
**Supersedes:** `worklog/history/specs/2026-04-16-pointcloud-submodule-design.md` (Phase 1 — implemented)
**Prereqs:** Phase 1 complete — `collab_splats/pointcloud/{base,sfm,feedforward}.py` exist with `MapAnythingCreator` stub

---

## Problem

Phase 1 established the skeleton but left several issues unresolved:

1. **Two feedforward methods not integrated:** VGGT-X (`stage/vggt_utils.py` in nerfstudio fork) and MapAnything (`stage/mapanything_utils.py`) both work but live outside the submodule with no shared interface.
2. **`NerfstudioSfmCreator` bundles two independent backends** (pycolmap SIFT + hloc) behind a `use_hloc` flag — should be separate named creators.
3. **Nerfstudio coupling in SfM input path** — hloc is a standalone package; the nerfstudio wrapper is unnecessary.
4. **Output path bug:** `ColmapCreator` (current `_run_pycolmap`) writes to `output_dir/sparse/0/` instead of the required `output_dir/colmap/sparse/0/`.
5. **Coordinate convention incomplete:** `_colmap_recon_to_result()` applies OpenCV→OpenGL camera flip but omits the COLMAP world → nerfstudio world reorientation. `camera_poses` in `PointcloudResult` is in COLMAP world frame, not nerfstudio world frame — silently diverges from `transforms.json`.
6. **No disk output from feedforward path:** `MapAnythingCreator.create()` builds `PointcloudResult` in-memory but writes no COLMAP binary or `transforms.json`. `Splatter.load_cameras()` requires `transforms.json` on disk.
7. **`stage/feedforward.py` is a reimplementation of `Splatter`** — redundant, should be eliminated.
8. **`vggt_utils.py` buried in nerfstudio fork** — should live in `stage/` alongside `mapanything_utils.py`.
9. **`PointcloudResult` field annotation** `colmap_reconstruction: Any | None  # NS interop only` is wrong — nerfstudio reads from disk, not this field. Real use case: pycolmap API access (bundle adjustment etc).

---

## Goals

1. Two working feedforward creators: `MapAnythingCreator` + `VGGTXCreator`
2. Two clean SfM creators: `ColmapCreator` + `HlocCreator` (no nerfstudio dep)
3. All four creators share a unified disk output contract
4. `PointcloudResult` carries correct coordinate metadata and `world_transform`
5. Eliminate `stage/feedforward.py`; copy `vggt_utils.py` to `stage/` (original kept in nerfstudio fork)

---

## Non-Goals

- Bundle adjustment (shelved — pycolmap API available via `colmap_reconstruction` field when ready)
- Loop closure
- Dense reconstruction / MVS
- Changes to training pipeline (`ns-train` still used as-is)
- `to_frame()` coordinate conversion utility (future)

---

## Architecture

```
BasePointcloudCreator (base.py)
  reconstruct(image_dir, output_dir) → PointcloudResult  [abstract]
  _write_transforms(sparse_dir, output_dir)               [shared util]

  ├── ColmapCreator (sfm.py)
  │     pycolmap direct: extract_features → match_exhaustive → incremental_mapping
  │     writes colmap/sparse/0/ then calls _write_transforms
  │
  ├── HlocCreator (sfm.py)
  │     hloc direct: extract_features → match_features → reconstruction.main()
  │     no nerfstudio dependency
  │     writes colmap/sparse/0/ then calls _write_transforms
  │
  └── BaseFeedforwardCreator (feedforward.py)
        reconstruct():
          sparse_dir = output_dir / "colmap" / "sparse" / "0"
          recon = self._run_inference(image_dir, output_dir)   [abstract]
          self._write_transforms(sparse_dir, output_dir)       [shared, once]
          return _colmap_recon_to_result(recon)
        _run_inference() → pycolmap.Reconstruction             [abstract]

        ├── MapAnythingCreator
        │     _run_inference(image_dir, output_dir):
        │       sparse_dir = output_dir / "colmap" / "sparse" / "0"
        │       mapanything steps 1–4 → recon (pycolmap.Reconstruction)
        │       sparse_dir.mkdir(parents=True, exist_ok=True)
        │       recon.write_binary(str(sparse_dir))
        │       return recon
        │
        └── VGGTXCreator
              _run_inference(image_dir, output_dir):
                run_vggt(image_dir, colmap_dir=output_dir/"colmap")
                → reads back pycolmap.Reconstruction from disk
                  (sparse_dir = output_dir/"colmap"/"sparse"/"0")
```

**Registry:**
```python
_REGISTRY = {
    "colmap":      ColmapCreator,
    "hloc":        HlocCreator,
    "mapanything": MapAnythingCreator,
    "vggtx":       VGGTXCreator,
}
```

Registry keys match nerfstudio's `sfm_tool` naming convention where applicable (`"colmap"`, `"hloc"`).

---

## Unified Disk Output Contract

Every `reconstruct(image_dir, output_dir)` call produces:

```
{output_dir}/
  colmap/
    sparse/
      0/
        cameras.bin
        images.bin
        points3D.bin
  transforms.json       ← written by _write_transforms() via colmap_to_json()
  sparse_pc.ply         ← written by colmap_to_json() as a side effect
```

`{output_dir}` is the data root passed to `ns-train --data {output_dir}`. No separate `colmap_dir` parameter — it is always derived as `output_dir / "colmap"`.

`_write_transforms(sparse_dir, output_dir)` is the shared base utility:
```python
def _write_transforms(self, sparse_dir: Path, output_dir: Path) -> None:
    from nerfstudio.process_data.colmap_utils import colmap_to_json
    colmap_to_json(recon_dir=sparse_dir, output_dir=output_dir)
```
Called exactly once per `reconstruct()`, by each subclass after binary files exist. No double-write.

---

## Data Contract

```python
class CoordinateFrame(str, Enum):
    COLMAP     = "colmap"      # w2c, OpenCV axes, world -Y up
    NERFSTUDIO = "nerfstudio"  # c2w, OpenGL axes, world +Z up

@dataclass
class PointcloudResult:
    points: np.ndarray                    # (N, 3) float32, world XYZ
    colors: np.ndarray                    # (N, 3) uint8, RGB
    confidence: np.ndarray | None         # (N,) float32 — feedforward only
    camera_poses: np.ndarray | None       # (M, 4, 4) float32
    camera_intrinsics: np.ndarray | None  # (M, 3, 3) float32, K per image
    colmap_reconstruction: Any | None     # pycolmap.Reconstruction — BA + pycolmap API
    frame: CoordinateFrame = CoordinateFrame.NERFSTUDIO
    world_transform: np.ndarray | None = None
    # (3, 4) applied_transform: COLMAP world → nerfstudio world.
    # Matches transforms.json["applied_transform"].
    # Invert to recover COLMAP world convention.
    # None when keep_original_world_coordinate=True was used.
```

---

## Coordinate Convention

**Two distinct transforms** applied in `_colmap_recon_to_result()`:

**Transform A — camera axes:** OpenCV → OpenGL
```python
c2w = np.linalg.inv(w2c)
c2w[:3, 1:3] *= -1   # flip Y and Z columns
```

**Transform B — world reorientation:** COLMAP gravity guess (-Y) → nerfstudio (+Z up)
```python
c2w = c2w[np.array([0, 2, 1, 3]), :]   # swap rows 1 ↔ 2
c2w[2, :] *= -1                          # negate new row 2
```

`world_transform` stores B as a `(3, 4)` matrix — matches `transforms.json["applied_transform"]`. This ensures `PointcloudResult.camera_poses` is in the same frame as `transforms.json` frame poses, which is what `Splatter.load_cameras()` reads.

**Phase 1 bug:** `_colmap_recon_to_result()` applied only A. The fix applies both A and B, populates `world_transform` from the same matrix used by `colmap_to_json()`.

**The nerfstudio transform separation** (important for visualization):
- `transforms.json` = preprocessing poses (nerfstudio world, gravity-aligned)
- `dataparser_transforms.json` = written by `ns-train`, additional normalization (auto-scale + auto-orient into unit sphere)
- `Splatter.load_cameras()` composes both — this is correct and intentional

---

## Bugs Fixed

| Location | Bug | Fix |
|---|---|---|
| `sfm.py:50` | `_run_pycolmap` writes to `output_dir/sparse/0/` | Write to `output_dir/colmap/sparse/0/` |
| `base.py` | `_colmap_recon_to_result()` omits world reorientation (transform B) | Apply both A and B; populate `world_transform` |
| `base.py` | `colmap_reconstruction` annotated `# NS interop only` | Correct annotation: pycolmap API / BA |
| `feedforward.py` | `MapAnythingCreator.create()` writes no files to disk | `_run_inference()` writes binary; base writes `transforms.json` |

---

## Stage File Changes

| File | Action |
|---|---|
| `stage/feedforward.py` | **Delete** — but absorb `feedforward_to_pointcloud()` + `prepare_outputs_for_export()` logic first |
| `stage/mapanything_utils.py` | Keep — used by `MapAnythingCreator._run_inference()` |
| `stage/vggt_utils.py` | **Add** — copied from `nerfstudio/process_data/vggt_utils.py` in fork (original kept) |
| `stage/preproc_utils.py` | Keep — image loading utilities |
| `stage/optical_flow.py` | **Delete** — confirmed absorbed into `collab_splats/utils/frame_sampling.py` |

`BaseFeedforwardCreator` imports `stage/` via `sys.path` insert at repo root (same pattern as current `MapAnythingCreator`). Frame sampling uses `collab_splats.utils.frame_sampling`, not `stage/optical_flow.py`.

---

## VGGTXCreator Notes

`run_vggt()` signature: `run_vggt(image_dir, colmap_dir, ..., use_global_alignment=True)`.

`use_global_alignment=True` calls `_run_global_alignment()` — the cross-camera alignment step that was unresolved in `tlb-improve-mesh`. Default `VGGTXCreator` param: `use_global_alignment: bool = False` until alignment is validated. Exposed as a creator param so callers can opt in.

`run_vggt()` writes to `colmap_dir/sparse/0/` internally. `VGGTXCreator._run_inference()` calls with `colmap_dir = output_dir / "colmap"` — no confusion about nesting level.

---

## MapAnythingCreator Notes

`run_mapanything_pipeline()` (the stage convenience function) includes step 6 (`convert_to_nerfstudio_format()` → writes `transforms.json`). **Do not call `run_mapanything_pipeline()`** — call individual steps 1–4 to avoid double-write. `BaseFeedforwardCreator.reconstruct()` calls `_write_transforms()` exactly once.

**Inference params on `MapAnythingCreator`** (from notebook `FeedforwardMeshing.ipynb`):
- `confidence_percentile: float = 35.0` — passed to `run_mapanything_inference()`; keeps points above this confidence percentile (keep top 65%)
- `use_multiview_confidence: bool = True` — multi-view depth consistency filter
- `minibatch_size: int = 1` — memory-efficient inference; increase if VRAM allows

`conf_threshold=1.5` and `subsample_factor=1` from Phase 1 were artifacts of `Reconstructor.feedforward_to_pointcloud()` — not used in actual notebook workflows. Drop them.

**`stage/feedforward.py` absorption required before deletion:** `feedforward_to_pointcloud()` contains the logic that builds `(points, colors, confidence)` arrays from raw depth map outputs. This is NOT in `mapanything_utils.py`. Absorb this logic into `MapAnythingCreator._run_inference()` (or a helper in `mapanything_utils.py`) before deleting the file. `prepare_outputs_for_export()` (returns extrinsics dict for `build_colmap_reconstruction()`) must also be absorbed.

---

## HlocCreator Notes

Calls hloc directly — no nerfstudio dependency:
```python
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction
```
`sfm_dir = output_dir / "colmap" / "sparse" / "0"` passed to `reconstruction.main()`. hloc writes there directly. Then calls `_write_transforms()`.

This removes the nerfstudio dep for the SfM input path. nerfstudio remains only for model training (`ns-train`).

---

## Files Modified

| File | Change |
|---|---|
| `collab_splats/pointcloud/base.py` | Add `CoordinateFrame`; update `PointcloudResult` (frame, world_transform); fix `_colmap_recon_to_result()` (transform B + world_transform); add `_write_transforms()` |
| `collab_splats/pointcloud/sfm.py` | Split `NerfstudioSfmCreator` → `ColmapCreator` + `HlocCreator`; fix output path; hloc calls directly |
| `collab_splats/pointcloud/feedforward.py` | Add `BaseFeedforwardCreator`; refactor `MapAnythingCreator` → `_run_inference()`; add `VGGTXCreator` |
| `collab_splats/pointcloud/__init__.py` | Update registry: `"colmap"`, `"hloc"`, `"mapanything"`, `"vggtx"` |
| `stage/vggt_utils.py` | New file (copied from nerfstudio fork — original kept) |
| `stage/feedforward.py` | Deleted |
| `stage/optical_flow.py` | Deleted |
| `tests/pointcloud/` | Update tests for new class names + registry keys; add `VGGTXCreator` smoke test |

---

## Testing

- `test_base.py` — `PointcloudResult` fields including `frame`, `world_transform`; `CoordinateFrame` enum
- `test_sfm_creator.py` — `ColmapCreator` + `HlocCreator` (separate); output path assertions
- `test_mapanything_creator.py` — `_run_inference()` interface; disk output contract
- `test_registry.py` — all four keys resolve
- New: `test_vggtx_creator.py` — defaults, missing dir raises, `use_global_alignment` param

All GPU-dependent tests marked `@pytest.mark.gpu`.

---

## Open Questions

1. Does `run_mapanything_pipeline()` write COLMAP binary to `output_dir/colmap/sparse/0/` specifically? **Must verify before implementation** — if not, adapter needed in `MapAnythingCreator._run_inference()`.
2. `colmap_to_json()` remains a nerfstudio import in `_write_transforms()`. Acceptable for now (we use nerfstudio for training). Future: write own implementation.
