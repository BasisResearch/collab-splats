# Scene Viewer — generic live 3D visualization (design)

**Date:** 2026-07-19
**Status:** approved
**Scope note:** the loop-closure wiring is *deferred* — the LC module is about to change.
This spec builds only the generic viewer; the LC hook design is recorded below so it
survives the refactor.

## Goal

A browser-viewable live 3D scene that long-running processes (loop closure, BA,
localization) can push point clouds, camera frusta, and lines into while they run.
Immediate motivation: watching submaps land in the global scene during loop closure,
with a balanced point budget per submap.

## Decisions (from brainstorm)

- **Viewer backend:** viser web server. Pure-pip, no GL/Xvfb (websocket only) — works
  from a tmux run on a headless container, viewed at `http://<ip>:<port>`.
  rerun-sdk is excluded from this project's install; pyvista/Panel needs a server
  thread + Xvfb inside the producing process — worse fit.
- **One class, no adapter.** LC-specific glue is a few lines at the call sites
  (VGGT-SLAM's solver→viewer pattern, cf. `MIT-SPARK/VGGT-SLAM vggt_slam/viewer.py`).
  An adapter class would wrap three call sites and add no value.
- **Balance semantics:** fixed per-submap cap (`max_points` per named node), not a
  total budget (would force re-touching old submaps) and not voxel size (uniform
  density but unequal counts).
- **Placement during a run:** raw stitched poses live (drift visible), one full
  re-upload with corrected poses after PGO (scene "snaps"). No incremental PGO.

## Components

### 1. `collab_splats/viewer.py` — `Viewer`

Generic viser wrapper. Domain-neutral: arrays in, named scene nodes out. No imports
from `geometry/` or `pointcloud/`.

```python
class Viewer:
    def __init__(self, port: int = 8080): ...
    def add_points(self, name, points, colors, point_size=0.01): ...   # (N,3), (N,3) uint8
    def add_frusta(self, name, poses, intrinsics, images=None): ...    # (K,4,4) world-to-cam, (K,3,3)
    def add_lines(self, name, segments, color): ...                    # (M,2,3)
```

- **Upsert by name:** adding a node with an existing name replaces it (viser native
  behavior). This is how a caller refreshes a submap after correction — re-upload,
  no `set_node_pose` API needed.
- **Frusta:** one frustum per frame under `{name}/frame_{i}`, images optional
  (thumbnails downscaled before upload). Poses are world-to-cam (repo convention,
  `assert_world_to_cam`); Viewer inverts internally for viser's cam-to-world frames.
- **GUI:** two checkboxes — *Show cameras* (toggle all frusta), *Color by node*
  (deterministic palette per named points node; makes submap boundaries and balance
  visible vs RGB).
- Handles stored per name so toggles and replacement work; plain variable names.
- **Not included (YAGNI):** walkthrough animation, OBB drawing, `set_node_pose`,
  remove-node API.

### 2. `subsample_points()` — `collab_splats/pointcloud/utils.py`

```python
def subsample_points(points, colors=None, conf=None, max_points=50_000, conf_percentile=20.0):
    """Confidence-filter then randomly cap a point set to max_points."""
```

- Drops points below the `conf_percentile`-th percentile of `conf` (when given),
  then random-samples down to `max_points` (when over).
- Returns `(points, colors)` index-aligned.
- **Why a new function:** the existing `voxel_downsample` is voxel-*size*-based —
  output count varies with scene extent, so it cannot deliver an equal per-submap
  budget. No count-capped downsampler exists in the repo.
- Lives beside `voxel_downsample`/`clean_pointcloud`; generic (no Submap import).

### 3. Dependency

`viser` added to `pyproject.toml` main deps (light, pure-python). `viewer.py` imports
it at top of module per the hard-imports rule; nothing else imports `viewer.py`, so
non-viz code paths pay no import cost.

## Deferred: loop-closure wiring (record for post-refactor)

Three guarded hook sites in `_run_lc_loop` (or its successor), each
`if self.viz is not None:` and wrapped in try/except-log — a viz failure must never
kill an hours-long LC run:

1. **After `submaps.append(submap)`:** re-base the submap's local `world_points` to
   the global raw-stitched frame (per-frame rigid re-basing, same first-writer-wins
   math as `_assemble_precorrection_extrinsics` — reuse it), then
   `subsample_points(...)` and `viz.add_points(f"submap_{i}", ...)` +
   `viz.add_frusta(f"submap_{i}/cams", ...)`.
2. **On accepted loop:** `viz.add_lines` between query/detected camera centers.
3. **After PGO:** re-upload every submap using `corrected_extrinsics` — same
   re-basing function, corrected pose source.

Config knobs land on `LoopClosureConfig` then: `viz_max_points_per_submap` (50k),
conf percentile. `LoopClosure` gains optional `viz: Viewer | None = None`.

## Testing

Flat test functions (`tests/pointcloud/test_utils_subsample.py`, `tests/test_viewer.py`):

- `subsample_points`: cap respected, conf filter drops low-conf points, no-op when
  under budget, colors stay aligned, `colors=None`/`conf=None` paths.
- `Viewer` smoke: instantiate on an ephemeral port, `add_points`/`add_frusta`/
  `add_lines` with tiny arrays, upsert-by-name replaces. Viser needs no display, so
  this runs in CI.
- LC hook tests come with the deferred wiring (duck-typed stub viz recording calls).

## Non-goals

Incremental PGO, dashboard integration, mesh preview, walkthrough animation,
file-snapshot export.
