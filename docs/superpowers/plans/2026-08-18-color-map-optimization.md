# Mesh Color Map Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opt-in rigid Zhou–Koltun color map optimization (`mesh.color_map_iterations`, default 0 = off) that recolors mesh.ply in place after TSDF fusion, sharpening vertex colors by refining a private copy of the camera poses for photo-consistency.

**Architecture:** New `optimize_color_map(mesh_path, depths, rgbs, c2w, intrinsics, iterations, depth_trunc)` in `collab_splats/mesh/utils.py`, wrapping `o3d.pipelines.color_map.run_rigid_optimizer` (open3d 0.19.0, verified in venv). `pointcloud_to_mesh` gains `color_map_iterations: int = 0` and calls it AFTER `mesher.create()` returns — fusion and clean_repair both happen inside `create()`, so the optimizer colors final geometry. RGBD images are built from the SAME arrays the fusion consumed (model-res and native paths both work by construction). Poses are report-only — nothing writes back to COLMAP or the zarr. Wiring: `_run_tsdf_mesh` + `Reconstructor.mesh()` + one base.yaml key.

**Tech Stack:** Open3D `pipelines.color_map` (rigid only), numpy. Zero new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-18-color-map-optimization-design.md`

**Environment:** always `/opt/venv/reconstruction/bin/python` (base `python` may be py3.13). Repo: `/workspace/collab-splats`, branch `refactor/cu121-uv-migration` (single-branch strategy — commit directly, no feature branch).

**CRITICAL — base.yaml has uncommitted USER edits (conf_percentile: 20, native_resolution: true, fps 2.0, deleted loger block, mesh voxel/sdf/trunc/clean_repair changes). NEVER `git add configs/base.yaml`. Task 3 commits ONLY the new key via a synthetic blob (`git hash-object -w` + `git update-index --cacheinfo`). Same caution for pyproject.toml, collab_splats/remote/rerun.py, docs/examples/run_pipeline_remote.py, tests/examples/test_run_pipeline_remote.py, data/tutorial/README.md — never stage them.**

**File structure:**
- Modify: `collab_splats/mesh/utils.py` — new `optimize_color_map` + hook in `pointcloud_to_mesh`
- Modify: `collab_splats/wrapper/reconstructor.py` — `_run_tsdf_mesh` param + `mesh()` plumbing
- Modify: `configs/base.yaml` — `mesh.color_map_iterations: 0` (synthetic-blob staged)
- Test: `tests/mesh/test_utils.py` — 3 new tests
- Modify: `tests/wrapper/test_reconstructor.py:1397` — extend `test_base_yaml_mesh_has_fidelity_keys`

---

### Task 1: `optimize_color_map` in mesh/utils.py

**Files:**
- Modify: `collab_splats/mesh/utils.py`
- Test: `tests/mesh/test_utils.py`

Context: `mesh/utils.py` holds `_feedforward_to_tsdf_inputs` (adapter producing `depths (N,H,W) float32`, `rgbs (N,H,W,3) uint8 OR float [0,1]`, `c2w (N,4,4)`, `intrinsics (N,3,3)`) and `pointcloud_to_mesh`. `Open3DTSDFFusion.create` (mesh/tsdf.py:38) writes `output_dir/mesh.ply`, optionally runs `clean_repair_mesh` on the saved file, returns `MeshResult(mesh_path=...)`. The new function loads that PLY, optimizes, overwrites it — same in-place contract as `clean_repair_mesh`.

- [ ] **Step 1: Write the failing test**

Append to `tests/mesh/test_utils.py` (add imports at top of file if missing: `import open3d as o3d`, `from collab_splats.mesh.tsdf import Open3DTSDFFusion`, `from collab_splats.mesh.utils import optimize_color_map`):

```python
def test_optimize_color_map_runs_and_recolors(tmp_path):
    """Rigid optimizer runs on a tiny synthetic scene and leaves a valid colored mesh in place."""
    # Constant-depth plane seen by 3 slightly-translated cameras; left half bright so the
    # optimizer has an image gradient to work with
    n, h, w = 3, 32, 32
    depths = np.full((n, h, w), 1.0, np.float32)
    rgbs = np.zeros((n, h, w, 3), np.uint8)
    rgbs[:, :, : w // 2] = 200
    K = np.array([[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]], np.float32)
    intrinsics = np.tile(K, (n, 1, 1))
    c2w = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    c2w[:, 0, 3] = np.linspace(-0.02, 0.02, n)

    fusion = Open3DTSDFFusion(
        output_dir=tmp_path, voxel_size=0.05, sdf_trunc=0.15, depth_trunc=5.0
    )
    result = fusion.create(depths, rgbs, c2w, intrinsics)

    optimize_color_map(
        result.mesh_path, depths, rgbs, c2w, intrinsics, iterations=5, depth_trunc=5.0
    )

    mesh = o3d.io.read_triangle_mesh(str(result.mesh_path))
    assert len(mesh.vertices) > 0
    assert len(mesh.vertex_colors) == len(mesh.vertices)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py::test_optimize_color_map_runs_and_recolors -v`
Expected: FAIL — `ImportError: cannot import name 'optimize_color_map'`

- [ ] **Step 3: Implement `optimize_color_map`**

In `collab_splats/mesh/utils.py`: add `import open3d as o3d` to the top-level imports if not already present (the mesh package hard-requires open3d via tsdf.py, so a top-level import is correct per code style — no lazy import needed). Then add, after `_feedforward_to_tsdf_inputs` and before `pointcloud_to_mesh`:

```python
def optimize_color_map(
    mesh_path: Path,
    depths: np.ndarray,
    rgbs: np.ndarray,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    iterations: int,
    depth_trunc: float,
) -> None:
    """Rigid Zhou-Koltun color map optimization — recolors mesh_path in place.

    Refines a private copy of the camera poses for photo-consistency and reassigns vertex
    colors from the refined poses. Poses are report-only: nothing is written back to COLMAP
    or the zarr — the overwritten mesh file is the only output.

    Args:
        mesh_path:   PLY to load, recolor, and overwrite (same in-place contract as
                     clean_repair_mesh).
        depths:      (N, H, W) float32 — the SAME array the TSDF fusion consumed.
        rgbs:        (N, H, W, 3) uint8, or float32 in [0, 1] (converted at this boundary).
        c2w:         (N, 4, 4) float32 cam-to-world OpenCV.
        intrinsics:  (N, 3, 3) float32.
        iterations:  rigid optimizer iteration count (upstream default is 300).
        depth_trunc: visibility cutoff — must match the fusion's depth_trunc; the option's
                     2.5 default assumes metric depth and ours is non-metric.
    """
    # Float [0,1] RGB (model-res path) -> uint8 at the boundary; native path is already uint8
    if rgbs.dtype != np.uint8:
        rgbs = (np.clip(rgbs, 0.0, 1.0) * 255.0).astype(np.uint8)

    # RGBD list + camera trajectory from the same arrays the fusion consumed — resolution
    # consistency with the mesh is guaranteed by construction
    height, width = depths.shape[1:3]
    rgbd_images = []
    cam_params = []
    for i in range(depths.shape[0]):
        color = o3d.geometry.Image(np.ascontiguousarray(rgbs[i]))
        depth = o3d.geometry.Image(np.ascontiguousarray(depths[i].astype(np.float32)))
        rgbd_images.append(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
        )
        K = intrinsics[i]
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        )
        cam.extrinsic = np.linalg.inv(c2w[i])  # optimizer wants world-to-camera
        cam_params.append(cam)
    trajectory = o3d.camera.PinholeCameraTrajectory()
    trajectory.parameters = cam_params

    # Run the rigid optimizer and overwrite the mesh; the refined trajectory is discarded
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    option = o3d.pipelines.color_map.RigidOptimizerOption(
        maximum_iteration=int(iterations),
        maximum_allowable_depth=float(depth_trunc),
    )
    logger.info("Color map optimization: %d frames, %d iterations", depths.shape[0], iterations)
    mesh, _ = o3d.pipelines.color_map.run_rigid_optimizer(mesh, rgbd_images, trajectory, option)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
```

Note: if `run_rigid_optimizer` in the installed 0.19.0 returns only a mesh (not a tuple), adjust the unpack accordingly — probe with a one-liner if the test errors on unpacking. Everything else stays.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py::test_optimize_color_map_runs_and_recolors -v`
Expected: PASS (open3d may print optimizer progress to stdout — fine)

- [ ] **Step 5: Run the whole mesh test file**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/ -v`
Expected: all pass (no regressions)

- [ ] **Step 6: Commit**

```bash
git add tests/mesh/test_utils.py collab_splats/mesh/utils.py
git commit -m "feat(mesh): optimize_color_map — rigid Zhou-Koltun recolor of mesh.ply in place

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `pointcloud_to_mesh` hook

**Files:**
- Modify: `collab_splats/mesh/utils.py:614-657` (`pointcloud_to_mesh`)
- Test: `tests/mesh/test_utils.py`

Context: `pointcloud_to_mesh` currently ends with:

```python
    depths, rgbs, c2w, intrinsics = _feedforward_to_tsdf_inputs(
        result,
        conf_percentile=conf_percentile,
        frame_store=frame_store,
        native_intrinsics=native_intrinsics,
    )
    mesher = get_mesh_creator(method, Path(output_dir), **mesher_kwargs)
    return mesher.create(depths, rgbs, c2w, intrinsics)
```

`tests/mesh/test_utils.py` has a `_tiny_ff_result(with_confidence=True)` fixture (2 frames, 8x8 model res, 16x16 original res).

- [ ] **Step 1: Write the failing tests**

Append to `tests/mesh/test_utils.py` (add `import pytest` at top if missing):

```python
def test_pointcloud_to_mesh_default_skips_color_map(tmp_path, monkeypatch):
    """color_map_iterations=0 (default) never touches the optimizer — shipping path unchanged."""
    called = []
    monkeypatch.setattr(
        "collab_splats.mesh.utils.optimize_color_map",
        lambda *a, **k: called.append(1),
    )
    result = _tiny_ff_result()
    pointcloud_to_mesh(
        result, tmp_path, voxel_size=0.05, sdf_trunc=0.15, depth_trunc=10.0
    )
    assert called == []


def test_pointcloud_to_mesh_color_map_requires_tsdf(tmp_path):
    """A non-TSDF method with color_map_iterations>0 fails loudly before any work."""
    result = _tiny_ff_result()
    with pytest.raises(ValueError, match="color_map_iterations"):
        pointcloud_to_mesh(
            result, tmp_path, method="depth_normal_poisson", color_map_iterations=10
        )


def test_pointcloud_to_mesh_color_map_called_with_fusion_arrays(tmp_path, monkeypatch):
    """color_map_iterations>0 calls the optimizer with the mesh path and the fusion's arrays."""
    calls = {}

    def fake_optimize(mesh_path, depths, rgbs, c2w, intrinsics, iterations, depth_trunc):
        calls.update(
            mesh_path=mesh_path,
            n_frames=depths.shape[0],
            iterations=iterations,
            depth_trunc=depth_trunc,
        )

    monkeypatch.setattr("collab_splats.mesh.utils.optimize_color_map", fake_optimize)
    result = _tiny_ff_result()
    mesh_result = pointcloud_to_mesh(
        result,
        tmp_path,
        voxel_size=0.05,
        sdf_trunc=0.15,
        depth_trunc=10.0,
        color_map_iterations=7,
    )
    assert calls["mesh_path"] == mesh_result.mesh_path
    assert calls["n_frames"] == result.depth.shape[0]
    assert calls["iterations"] == 7
    assert calls["depth_trunc"] == 10.0
```

If `pointcloud_to_mesh` is not yet imported in the test file, add it to the existing `from collab_splats.mesh.utils import ...` line.

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py -v -k color_map`
Expected: `test_optimize_color_map_runs_and_recolors` PASS (Task 1); the two new `pointcloud_to_mesh` param tests FAIL with `TypeError: pointcloud_to_mesh() got an unexpected keyword argument 'color_map_iterations'`; the default-skip test PASSES trivially (nothing calls the optimizer yet) — that's expected, it exists to lock the behavior in.

- [ ] **Step 3: Implement the hook**

In `pointcloud_to_mesh`: add `color_map_iterations: int = 0` to the signature (after `native_intrinsics`, before `**mesher_kwargs`), add to the docstring Args:

```
        color_map_iterations: Rigid color-map optimization iterations run on mesh.ply after
                           fusion + clean_repair (0 = off). Only "open3d_tsdf" supports it.
```

and replace the function tail with:

```python
    from collab_splats.mesh import get_mesh_creator

    # Loud failure before any work — only the TSDF path has the depth_trunc + mesh.ply
    # contract the optimizer needs
    if color_map_iterations and method != "open3d_tsdf":
        raise ValueError(
            f"color_map_iterations requires method='open3d_tsdf', got {method!r}"
        )

    depths, rgbs, c2w, intrinsics = _feedforward_to_tsdf_inputs(
        result,
        conf_percentile=conf_percentile,
        frame_store=frame_store,
        native_intrinsics=native_intrinsics,
    )
    mesher = get_mesh_creator(method, Path(output_dir), **mesher_kwargs)
    mesh_result = mesher.create(depths, rgbs, c2w, intrinsics)

    # Color-map optimization AFTER create(): fusion and clean_repair both run inside it, so
    # the optimizer colors the final geometry instead of speckle about to be deleted
    if color_map_iterations:
        optimize_color_map(
            mesh_result.mesh_path,
            depths,
            rgbs,
            c2w,
            intrinsics,
            iterations=color_map_iterations,
            depth_trunc=mesher.depth_trunc,
        )
    return mesh_result
```

Also update the `Raises:` docstring line to mention the new ValueError.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/mesh/test_utils.py collab_splats/mesh/utils.py
git commit -m "feat(mesh): color_map_iterations hook in pointcloud_to_mesh (0 = off, TSDF-only)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Reconstructor wiring + base.yaml key

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:397-457` (`_run_tsdf_mesh`) and `:954-964` (`mesh()` call site)
- Modify: `configs/base.yaml` (mesh block — SYNTHETIC-BLOB STAGING ONLY, see below)
- Test: `tests/wrapper/test_reconstructor.py:1397` (`test_base_yaml_mesh_has_fidelity_keys`)

**WARNING — configs/base.yaml carries uncommitted user edits. Do NOT `git add configs/base.yaml`. Follow Step 4 exactly.**

- [ ] **Step 1: Extend the failing config test**

In `tests/wrapper/test_reconstructor.py`, extend `test_base_yaml_mesh_has_fidelity_keys`:

```python
def test_base_yaml_mesh_has_fidelity_keys():
    """New mesh keys exist and default OFF — shipping output stays byte-identical."""
    cfg = yaml.safe_load((Path(__file__).parents[2] / "configs" / "base.yaml").read_text())
    assert cfg["mesh"]["conf_percentile"] is None
    assert cfg["mesh"]["native_resolution"] is False
    assert cfg["mesh"]["color_map_iterations"] == 0
```

NOTE: the first two asserts may already FAIL in the working tree because the user has locally enabled these features (`conf_percentile: 20`, `native_resolution: true`) — that is a known pre-existing failure class, NOT yours to fix. Your new assert must pass once Step 3 adds the key. Judge this test on the committed base.yaml: `git stash push configs/base.yaml` is FORBIDDEN (user edits) — instead verify the committed version passes via `git show HEAD:configs/base.yaml` in Step 5.

- [ ] **Step 2: Wire `_run_tsdf_mesh` and `mesh()`**

In `_run_tsdf_mesh` signature (reconstructor.py:397), add `color_map_iterations: int = 0,` after `native_resolution: bool = False,`. Forward it in the `pointcloud_to_mesh(...)` call (after `native_intrinsics=native_intrinsics,`):

```python
        color_map_iterations=color_map_iterations,
```

At the `mesh()` call site (~line 954-964), after `native_resolution=mesh_cfg["native_resolution"],` add:

```python
        color_map_iterations=mesh_cfg["color_map_iterations"],
```

- [ ] **Step 3: Add the key to the WORKING-TREE base.yaml**

Edit `configs/base.yaml` (Edit tool, working tree — this keeps the user's local runs working with the strict `mesh_cfg["color_map_iterations"]` access): after the `native_resolution:` line in the mesh block, add:

```yaml
  color_map_iterations: 0  # rigid color-map optimization iterations after fusion (0 = off)
```

- [ ] **Step 4: Stage ONLY the new key via synthetic blob**

The committed change must be HEAD's base.yaml + the one new line — none of the user's edits:

```bash
cd /workspace/collab-splats
BLOB=$(git show HEAD:configs/base.yaml \
  | sed '/native_resolution:/a\  color_map_iterations: 0  # rigid color-map optimization iterations after fusion (0 = off)' \
  | git hash-object -w --stdin)
git update-index --cacheinfo 100644,$BLOB,configs/base.yaml
git diff --cached configs/base.yaml   # MUST show exactly one added line
```

Verify the cached diff is exactly the one `color_map_iterations` line. If it shows any user edit (fps, voxel_size, loger, clean_repair...), STOP — reset with `git restore --staged configs/base.yaml` and redo.

- [ ] **Step 5: Verify the committed-config test logic**

```bash
git show :configs/base.yaml | /opt/venv/reconstruction/bin/python -c "
import sys, yaml
cfg = yaml.safe_load(sys.stdin.read())
assert cfg['mesh']['color_map_iterations'] == 0
assert cfg['mesh']['conf_percentile'] is None
assert cfg['mesh']['native_resolution'] is False
print('STAGED CONFIG OK')
"
```

Expected: `STAGED CONFIG OK`

- [ ] **Step 6: Run the wrapper tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k fidelity`
Expected: the new `color_map_iterations` assert passes; the `conf_percentile`/`native_resolution` asserts may fail against the user's working-tree values — pre-existing, report but do not fix. Also run `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/ -q` — all pass.

- [ ] **Step 7: Commit (base.yaml already staged from Step 4)**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "feat(mesh): wire color_map_iterations through Reconstructor.mesh + base.yaml (default 0)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git show --stat HEAD   # confirm exactly 3 files
git diff HEAD~1 HEAD -- configs/base.yaml   # confirm one added line only
```

---

### Task 4: Measured verification on the reference scene

**Files:**
- Modify: `/tmp/claude-0/-workspace-collab-splats/a5e0ab39-0895-4524-9574-6fbd037e8da0/scratchpad/task5_fuse.py` (add `--color-map-iterations` flag)
- No repo changes. All runs in the scratchpad (`SP` below). Reference scene: `/workspace/outputs/2026_07_15-Goprosplat-GH010229`, backend `vggt_omega`, 300 frames.

**Memory discipline: 46.6 GB cgroup cap — run each fuse/optimize sequentially in background (`run_in_background` Bash or tmux), never in parallel. Native fuse alone peaked 16.8 GB; the RGBD list adds ~4.4 GB.**

- [ ] **Step 1: Extend the driver**

`SP/task5_fuse.py` fuses via the repo's mesh path and prints a `TASK5_STATS` JSON line plus a `CODE <path>` provenance line. Add an argument `--color-map-iterations` (int, default 0) and forward it into the existing `pointcloud_to_mesh`/`_run_tsdf_mesh` call the driver makes. Time the whole run as the driver already does; the optimizer's share = (run with N iters) − (fuse-only run).

- [ ] **Step 2: Regression — defaults byte-identical**

```bash
SP=/tmp/claude-0/-workspace-collab-splats/a5e0ab39-0895-4524-9574-6fbd037e8da0/scratchpad
cd /tmp && PYTHONPATH=/workspace/collab-splats /opt/venv/reconstruction/bin/python \
  $SP/task5_fuse.py --out $SP/run_cmo_reg
cmp $SP/run_cmo_reg/mesh.ply $SP/run_defaults_head/mesh.ply && echo CMO_REGRESSION_BYTE_IDENTICAL
```

Expected: `CMO_REGRESSION_BYTE_IDENTICAL` (the hook is dead code at 0 iterations). Check the driver's `CODE` line prints the repo path.

- [ ] **Step 3: Baseline native+p20 fuse (no optimization)**

```bash
cd /tmp && PYTHONPATH=/workspace/collab-splats /usr/bin/time -v /opt/venv/reconstruction/bin/python \
  $SP/task5_fuse.py --out $SP/run_native_p20 --native --conf-percentile 20 \
  > $SP/cmo_base.log 2>&1
```

Record wall-clock + `Maximum resident set size` from the log.

- [ ] **Step 4: Probe 10 iterations, extrapolate runtime**

```bash
cd /tmp && PYTHONPATH=/workspace/collab-splats /usr/bin/time -v /opt/venv/reconstruction/bin/python \
  $SP/task5_fuse.py --out $SP/run_native_p20_cmo10 --native --conf-percentile 20 \
  --color-map-iterations 10 > $SP/cmo_10.log 2>&1
```

Compute per-iteration cost = (this run − Step 3 run) / 10. If 100 iterations extrapolates beyond ~2 hours, cap the sweep at what fits (e.g. {30} only) and REPORT the cap — no silent truncation.

- [ ] **Step 5: Sweep {30, 100} iterations (as runtime allows)**

Same command with `--color-map-iterations 30` → `run_native_p20_cmo30`, then `100` → `run_native_p20_cmo100`. Sequential, backgrounded, each logged with `/usr/bin/time -v`.

- [ ] **Step 6: Renders for user eyeball**

`SP/task5_render.py` renders camera-0 view (OffscreenRenderer, defaultUnlit, 1280x720 — works headless despite "XDG_RUNTIME_DIR not set" stderr):

```bash
for run in run_native_p20 run_native_p20_cmo10 run_native_p20_cmo30 run_native_p20_cmo100; do
  [ -f $SP/$run/mesh.ply ] && PYTHONPATH=/workspace/collab-splats /opt/venv/reconstruction/bin/python \
    $SP/task5_render.py $SP/$run/mesh.ply $SP/render_$run.png
done
```

- [ ] **Step 7: Record measured results in the plan**

Fill a results table in this plan file (iterations, wall-clock delta, peak RSS, mesh bytes, render path) and commit:

```bash
git add -f docs/superpowers/plans/2026-08-18-color-map-optimization.md
git commit -m "docs(plans): color map optimization — measured results on Goprosplat reference

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Measured results (fill in):**

| Run | Iterations | Wall-clock | Peak RSS | mesh.ply bytes | Render |
|-----|-----------|-----------|----------|----------------|--------|
| regression (defaults) | 0 | — | — | byte-identical? | — |
| native+p20 | 0 | | | | render_run_native_p20.png |
| native+p20 | 10 | | | | |
| native+p20 | 30 | | | | |
| native+p20 | 100 | | | | |

---

## Self-review notes

- Spec coverage: config key (T3), `optimize_color_map` (T1), hook placement + ValueError (T2), wiring (T3), regression + A/B verification (T4), all four spec tests present (T1 synthetic, T2 default-off + ValueError, T3 base.yaml key). Poses report-only: the refined trajectory is discarded in T1's implementation (`mesh, _ = ...`).
- Type consistency: `optimize_color_map(mesh_path, depths, rgbs, c2w, intrinsics, iterations, depth_trunc)` used identically in T1 impl, T2 hook, T2 mock.
- Known uncertainty flagged inline: `run_rigid_optimizer` return arity (T1 Step 3 note).
