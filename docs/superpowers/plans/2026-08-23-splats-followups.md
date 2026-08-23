# Splats Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the items the splats-module plan left owed: `mesh.source: splats` wired into `Reconstructor.mesh()`, `evals/scripts/eval_splats.py`, a measured 3DGS-vs-2DGS report on `data/tutorial/`, the `03_splats` / `06_mesh` tutorials, and the doc updates. Everything folds into the single squash commit `34b034bb` on `refactor/cu121-uv-migration`.

**Architecture:** `splats.zarr` already holds `(rgb, depth, normal, alpha, c2w, K)` per view in the COLMAP frame, so meshing from splats is a second input adapter (`_splats_to_tsdf_inputs`) feeding the same TSDF mesher; `alpha` plays the role confidence plays on the feedforward path. The eval script trains both primitives straight from a `feedforward.zarr` (no COLMAP needed — the zarr carries poses, K, points, colours, depth) and reads the trainer's own quality report; GT depth error is optional via the `eval_multiview_conf` helpers.

**Tech Stack:** gsplat d2f5c0f, Open3D TSDF (`collab_splats/mesh`), zarr v3, jupyter nbconvert for executed tutorials.

**Worktree:** `/workspace/collab-splats-splats`, branch `feat/splats-followups` (from `34b034bb`). `PY=/opt/venv/reconstruction/bin/python`, run with `PYTHONPATH=/workspace/collab-splats-splats` (editable install points at the main checkout). `data/outputs` is gitignored; symlink `data/outputs -> /workspace/collab-splats/data/outputs` in the worktree so `tutorial_config.py` resolves (that dir holds the tutorial's `frames.zarr` + `feedforward.zarr`, 30 frames at 688×384).

**Rules (from CLAUDE.md + user):** never index/slice inside a call's argument list — unpack to named locals first; no `.get(key, default)` on config dicts (`test_no_inline_defaults_in_source`); imports at top except gsplat/CUDA-only modules; block comments; `"""` on their own lines; flat test functions; do NOT run `black` on existing files; **do not commit** — the controller commits.

---

### Task 1: `mesh.source: splats` — adapter, config, Reconstructor wiring

**Files:**
- Modify: `collab_splats/mesh/utils.py` — add `_splats_to_tsdf_inputs`, split the tail of `pointcloud_to_mesh` into `mesh_from_tsdf_inputs`
- Modify: `collab_splats/wrapper/reconstructor.py` — `_run_tsdf_mesh` gains `source`; `Reconstructor.mesh()` reads `mesh.source`
- Modify: `configs/base.yaml` — `mesh.source: feedforward`
- Modify: `configs/README.md` — key row + the "Splats train from…" section sentence about `mesh.source`
- Test: `tests/mesh/test_splats_adapter.py`, `tests/wrapper/test_splats_stage.py` (append)

- [ ] **Step 1: Failing tests**

`tests/mesh/test_splats_adapter.py`:
```python
"""
_splats_to_tsdf_inputs: splats.zarr -> (depths, rgbs, c2w, K) with alpha as the confidence gate.
"""

import numpy as np
import pytest
import zarr

from collab_splats.mesh.utils import _splats_to_tsdf_inputs


def _write_splats_zarr(path, n_views=3, height=4, width=5):
    store = zarr.open_group(path, mode="w")
    rgb = np.full((n_views, height, width, 3), 7, np.uint8)
    depth = np.ones((n_views, height, width), np.float32)
    alpha = np.linspace(0.0, 1.0, n_views * height * width, dtype=np.float32).reshape(n_views, height, width)
    c2w = np.tile(np.eye(4, dtype=np.float32), (n_views, 1, 1))
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (n_views, 1, 1))
    for name, array in (("rgb", rgb), ("depth", depth), ("alpha", alpha), ("c2w", c2w), ("K", intrinsics)):
        store.create_array(name, data=array)
    store.attrs["primitive"] = "3dgs"
    return alpha


def test_adapter_shapes_and_types(tmp_path):
    _write_splats_zarr(tmp_path / "splats.zarr")
    depths, rgbs, c2w, intrinsics = _splats_to_tsdf_inputs(tmp_path / "splats.zarr", conf_percentile=None)
    assert depths.shape == (3, 4, 5) and depths.dtype == np.float32
    assert rgbs.shape == (3, 4, 5, 3) and rgbs.dtype == np.uint8
    assert c2w.shape == (3, 4, 4) and intrinsics.shape == (3, 3, 3)
    assert depths.min() == 1.0  # no percentile -> nothing dropped except alpha == 0


def test_adapter_alpha_percentile_zeroes_depth(tmp_path):
    alpha = _write_splats_zarr(tmp_path / "splats.zarr")
    depths, _, _, _ = _splats_to_tsdf_inputs(tmp_path / "splats.zarr", conf_percentile=50)
    dropped = depths == 0.0
    assert 0.45 < dropped.mean() < 0.55  # global percentile over all views
    assert np.all(alpha[dropped] <= np.percentile(alpha, 50))


def test_adapter_always_drops_zero_alpha(tmp_path):
    _write_splats_zarr(tmp_path / "splats.zarr")
    depths, _, _, _ = _splats_to_tsdf_inputs(tmp_path / "splats.zarr", conf_percentile=None)
    assert depths.flat[0] == 0.0  # alpha == 0 at the first pixel -> no observation


def test_adapter_missing_zarr_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="splats.zarr"):
        _splats_to_tsdf_inputs(tmp_path / "splats.zarr", conf_percentile=None)
```

Append to `tests/wrapper/test_splats_stage.py` (reuse its `_stub_reconstructor`; give the stub `"mesh": {...}` the full key set `voxel_size, sdf_trunc, depth_trunc, clean_repair, conf_percentile, native_resolution, color_map_iterations, source`):
```python
def test_mesh_source_splats_without_zarr_raises(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "splats"
    with pytest.raises(ValueError, match="mesh.source: splats"):
        recon.mesh()


def test_mesh_source_splats_fuses_from_splats_zarr(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "splats"
    recon.config["mesh"]["native_resolution"] = True  # ignored on this path, must not raise
    splats_dir = recon.backend_dir / "splats"
    splats_dir.mkdir(parents=True)
    _write_minimal_splats_zarr(splats_dir / "splats.zarr", n_views=3)
    with patch("collab_splats.wrapper.reconstructor.mesh_from_tsdf_inputs") as fuse:
        fuse.return_value = SimpleNamespace(mesh_path=recon.backend_dir / "mesh.ply")
        out = recon.mesh()
    depths, rgbs, c2w, intrinsics = fuse.call_args.args[:4]
    assert depths.shape[0] == 3 and out == recon.backend_dir / "mesh.ply"


def test_mesh_source_unknown_raises(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "nerf"
    with pytest.raises(ValueError, match="mesh.source"):
        recon.mesh()
```
(`_write_minimal_splats_zarr` = same writer as the adapter test; duplicate it locally, do not import across test modules.)  Adjust the patch target to wherever `mesh_from_tsdf_inputs` is imported in `reconstructor.py` (lazy import inside `_run_tsdf_mesh` → patch `collab_splats.mesh.utils.mesh_from_tsdf_inputs` instead).

- [ ] **Step 2: Run → FAIL** (`ImportError: _splats_to_tsdf_inputs`, `KeyError: 'source'`).

- [ ] **Step 3: `mesh/utils.py`**

Add after `_feedforward_to_tsdf_inputs`:
```python
def _splats_to_tsdf_inputs(
    splats_zarr: Path, conf_percentile: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack splats.zarr (rendered views) into (depths, rgbs, c2w, intrinsics) for TSDF fusion.

    - `alpha` is the confidence: pixels with alpha == 0 never reach Open3D, and
      `conf_percentile` drops the lowest-alpha percentile globally (same rule as the
      feedforward path's confidence gate).
    - Poses are the zarr's `c2w` — what was actually rendered, including pose-opt deltas.
    - Already native to the training frames, so there is no upsampling path.
    """
```
Body: `if not splats_zarr.exists(): raise FileNotFoundError(f"{splats_zarr} — run the splats stage first")`; open read-only; `depths = np.ascontiguousarray(store["depth"][:], np.float32)`; `alpha = store["alpha"][:]`; `keep = alpha > 0`; `if conf_percentile is not None: keep &= confidence_mask(alpha, conf_percentile)`; `depths[~keep] = 0.0`; log the dropped fraction like the feedforward path; return `(depths, np.ascontiguousarray(store["rgb"][:]), store["c2w"][:].astype(np.float32), store["K"][:].astype(np.float32))`.

Split `pointcloud_to_mesh`: everything from `mesher = get_mesh_creator(...)` to the return moves into
```python
def mesh_from_tsdf_inputs(
    depths, rgbs, c2w, intrinsics, output_dir: Path, method: str = "open3d_tsdf",
    color_map_iterations: int = 0, **mesher_kwargs,
) -> MeshResult:
    """
    Fuse pre-built (depths, rgbs, c2w, intrinsics) with any registered mesher; optional colour-map pass.
    """
```
and `pointcloud_to_mesh` becomes adapter + `return mesh_from_tsdf_inputs(...)`. Keep the `color_map_iterations` / non-TSDF guard inside `mesh_from_tsdf_inputs` (it is the one place both callers pass through).

- [ ] **Step 4: `reconstructor.py`**

`_run_tsdf_mesh(..., source: str = "feedforward", splats_zarr: Path | None = None)`. At the top:
```python
    # Splats source: rendered depth/RGB/alpha + the poses actually rendered; no native path
    if source == "splats":
        from collab_splats.mesh.utils import _splats_to_tsdf_inputs, mesh_from_tsdf_inputs

        if native_resolution:
            logger.info("mesh.native_resolution ignored: splats renders are already at frame resolution")
        depths, rgbs, c2w, intrinsics = _splats_to_tsdf_inputs(splats_zarr, conf_percentile=conf_percentile)
        n_views = depths.shape[0]
        n_poses = result.extrinsics.shape[0]
        if n_views != n_poses:
            raise ValueError(f"Frame-count mismatch: COLMAP has {n_poses} images but {splats_zarr} has {n_views}")
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_result = mesh_from_tsdf_inputs(
            depths, rgbs, c2w, intrinsics, output_dir, method="open3d_tsdf",
            voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc,
            clean_repair=clean_repair, color_map_iterations=color_map_iterations,
        )
        return mesh_result.mesh_path
```
`Reconstructor.mesh()`: read `source = mesh_cfg["source"]`; `if source not in ("feedforward", "splats"): raise ValueError(...)`; for `splats`, `splats_zarr = self.backend_dir / "splats" / "splats.zarr"`, `if not splats_zarr.exists(): raise ValueError("mesh.source: splats needs <path> — run the splats stage first (it is never auto-run)")`; the `feedforward.zarr` existence check stays only on the feedforward branch; pass `source=`/`splats_zarr=` through. `_STAGE_DEPS["mesh"]` unchanged. Update the `mesh()` docstring.

- [ ] **Step 5: Config + README**

`configs/base.yaml` `mesh:` block, first key after `enabled`: `source: feedforward     # feedforward (model depth) | splats (rendered depth; alpha = confidence)`.
`configs/README.md`: key row `| `mesh.source` | str | `feedforward` | `feedforward` fuses feedforward.zarr depth; `splats` fuses splats.zarr renders (needs the splats stage; `native_resolution` ignored) |`; in the "Splats train from the published COLMAP + frames.zarr" section replace the "fusing the splat renders (`mesh.source: splats`) is a follow-on" sentence with "`mesh.source: splats` fuses the renders instead (alpha as confidence, poses as rendered)". Also touch the matching sentence in `docs/superpowers/specs/2026-08-22-splats-module-design.md` (the "Deferred to a follow-on plan" bullet → "Built 2026-08-23 (this plan)").

- [ ] **Step 6: Run** `tests/mesh tests/wrapper/test_splats_stage.py tests/wrapper/test_reconstructor.py -q -p no:randomly` → PASS. Report DONE.

---

### Task 2: `evals/scripts/eval_splats.py`

**Files:** Create `evals/scripts/eval_splats.py`; Test: `tests/evals/test_eval_splats.py` (create `tests/evals/__init__.py` if absent — check first).

Purpose: one CLI that trains each primitive from a `feedforward.zarr` and tabulates the trainer's own quality report, optionally scoring rendered depth against 7-Scenes GT. CLI/tmux only.

- [ ] **Step 1: Failing tests** (CPU-only, no training)

```python
"""
eval_splats: feedforward.zarr -> trainer inputs, and the summary row built from a quality report.
"""

import json

import numpy as np
import zarr

from evals.scripts.eval_splats import inputs_from_feedforward_zarr, summarise_run


def _write_ff_zarr(path, n=2, h=4, w=6, scale=255.0):
    store = zarr.open_group(path, mode="w")
    images = np.random.rand(n, 3, h, w).astype(np.float32) * scale
    for name, array in (
        ("images", images), ("depth", np.ones((n, h, w), np.float32)),
        ("confidence", np.ones((n, h, w), np.float32)),
        ("extrinsics", np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))),
        ("intrinsics", np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))),
        ("points", np.zeros((120, 3), np.float32)), ("colors", np.zeros((120, 3), np.uint8)),
    ):
        store.create_array(name, data=array)
    store.attrs["image_paths"] = [f"frame_{i:06d}.jpg" for i in range(n)]


def test_inputs_uint8_hwc_for_both_image_scales(tmp_path):
    for scale in (1.0, 255.0):
        _write_ff_zarr(tmp_path / f"ff_{scale}.zarr", scale=scale)
        inputs = inputs_from_feedforward_zarr(tmp_path / f"ff_{scale}.zarr")
        assert inputs.images.shape == (2, 4, 6, 3) and inputs.images.dtype == np.uint8
        assert inputs.images.max() > 1  # [0,1] stores are rescaled, [0,255] stores are not squashed
        assert inputs.world_to_cam.shape == (2, 4, 4) and inputs.depth_targets.shape == (2, 4, 6)


def test_summarise_run_reads_report(tmp_path):
    report = {"summary": {"psnr": 30.0, "ssim": 0.9, "n_gaussians": 10, "seconds": 12.0,
                          "final_losses": {"depth": 0.1}, "config": {"primitive": "3dgs", "max_steps": 5}},
              "per_frame": [{"image_id": 0, "psnr": 30.0, "ssim": 0.9}]}
    (tmp_path / "splats_quality_report.json").write_text(json.dumps(report))
    row = summarise_run(tmp_path)
    assert row == {"primitive": "3dgs", "max_steps": 5, "psnr": 30.0, "ssim": 0.9, "n_gaussians": 10,
                   "seconds": 12.0, "ms_per_step": 2400.0, "final_losses": {"depth": 0.1}}
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Script**

Module docstring in the style of `eval_multiview_conf.py` (purpose, metric, "CLI/tmux only", usage). Pieces:
- `@dataclass SplatInputs(images, world_to_cam, intrinsics, points, colors, depth_targets)`.
- `inputs_from_feedforward_zarr(path) -> SplatInputs`: `FeedforwardResult.load_zarr(path, load_images=True, load_world_points=False)`; images `(N,3,H,W)` float → HWC; if `images.max() <= 1.0` multiply by 255 (MapAnything `[0,1]` vs VGGT `[0,255]` — cite the known drift); `.round().clip(0,255).astype(np.uint8)`; `world_to_cam = extrinsics` (4×4 w2c); depth targets = `depth` masked by `confidence_mask(confidence, conf_percentile)` when `--conf-percentile` given (default 20, matches base.yaml `mesh.conf_percentile`).
- `summarise_run(out_dir) -> dict` exactly as the test row (`ms_per_step = 1000 * seconds / max_steps`).
- Optional GT depth: `--seq <7scenes dir>` → reuse `load_7scenes_depth` + `median_align` + `retained_error` imported from `evals.scripts.eval_multiview_conf`, comparing `splats.zarr["depth"]` (resized nearest to GT res) where `alpha > 0.5` against GT; add `depth_vs_gt` to the row.
- `main()`: `--zarr` (required), `--out` (dir, required), `--primitives` (nargs, default `3dgs 2dgs`), `--max-steps` (default 30000), `--conf-percentile`, `--seq`, `--pose-opt` flag. Per primitive: `SplatsConfig(primitive=..., max_steps=..., pose_opt=...)` (losses = the dataclass default for that primitive), `train(...)` into `out/<primitive>/`, `summarise_run`, log a one-line row. Write `out/summary.json` `{"zarr": ..., "rows": [...]}` and print a markdown table (`primitive | steps | psnr | ssim | gaussians | seconds | ms/step`).

- [ ] **Step 4: Run tests → PASS.** `$PY evals/scripts/eval_splats.py --help` prints. Report DONE.

---

### Task 3: Measured report (GPU — run AFTER Task 1 + 2 land; nothing else on the GPU)

**Files:** Create `docs/superpowers/specs/2026-08-23-splats-measured-report.md`. Results under `evals/results/splats/tutorial/` (gitignored).

- [ ] **Step 1:** `PYTHONPATH=/workspace/collab-splats-splats $PY evals/scripts/eval_splats.py --zarr data/outputs/feedforward.zarr --out evals/results/splats/tutorial --max-steps 30000 2>&1 | tee evals/results/splats/tutorial/log.txt` (run in tmux `splats-eval`; poll). Record per-primitive: psnr, ssim, n_gaussians, seconds, ms/step, final_losses, peak GPU memory (`nvidia-smi` sampled in the log every 500 steps is fine, or `torch.cuda.max_memory_allocated` if the trainer logs it — otherwise note "not measured").
- [ ] **Step 2:** normal sanity for 3DGS `extra_signals` normals: from `evals/results/splats/tutorial/3dgs/splats.zarr`, fraction of pixels with `alpha > 0.5` whose `normal` has norm in `[0.9, 1.1]`, and mean angle to `depth_to_normal`-style finite-difference normals of the rendered depth (reuse `collab_splats.splats.rendering.depth_to_normal` if it exists, else `np.gradient`). Same for 2DGS. This answers "are the 3DGS extra_signals normals usable".
- [ ] **Step 3:** mesh comparison on the same scene — `mesh_from_tsdf_inputs` over (a) `_feedforward_to_tsdf_inputs(ff, conf_percentile=20)` and (b) `_splats_to_tsdf_inputs(3dgs splats.zarr, 20)` and (c) 2dgs, with base.yaml voxel/sdf/depth_trunc and `clean_repair=True`, `color_map_iterations=0`: vertex count, component count (`open3d cluster_connected_triangles`), wall time. Save the three PLYs under `evals/results/splats/tutorial/mesh_*.ply`.
- [ ] **Step 4:** write the report: setup (scene, 30 frames, 688×384, commit), three tables (train, normals, mesh), a 4–6 line verdict (which primitive by default? is 2DGS worth 2× time? are 3DGS normals usable for `normal_consistency`?), traps hit, and "Not measured" (7-Scenes GT depth error, pose_opt effect). Numbers must come from the run — no placeholders. Report DONE with the verdict lines.

---

### Task 4: Tutorials `03_splats/train_splats.ipynb` + `06_mesh/splats_mesh.ipynb` (GPU — after Task 3)

**Files:** Create both notebooks; modify `docs/source/tutorials/index.rst` (toctree entries "03 · Splats" between 02 and 04, "06 · Mesh" between 05 and 07); `README.md` tutorial rows 03/06 (currently "owed") → real links.

Follow `02_pointcloud/feedforward_mesh.ipynb` for cell style (`%run ../tutorial_config.py`, `notebook_utils.py` helpers, headless static backend). Use `MAX_STEPS = 3000` so execution stays under ~3 min; say so in the intro and point at the measured report for 30k numbers.

`03_splats/train_splats.ipynb` sections: §1 inputs (load `RECON` via `FeedforwardResult.load_zarr(load_images=True)`, reuse `inputs_from_feedforward_zarr` from `evals.scripts.eval_splats`), §2 `SplatsConfig` + `train()` into `OUTPUT_DIR / "splats"`, §3 outputs (`splats.ply`, `ckpt.pt`, `splats.zarr`, report — show the summary dict and a per-frame PSNR bar), §4 render gallery (rgb / depth / normal / alpha for 3 views via matplotlib), §5 2DGS in two lines (config diff only; not executed — explain why: distortion loss + DefaultStrategy).

`06_mesh/splats_mesh.ipynb`: §1 feedforward-source mesh via `pointcloud_to_mesh(ff, ..., conf_percentile=20)`, §2 splats-source mesh via `_splats_to_tsdf_inputs` + `mesh_from_tsdf_inputs`, §3 side-by-side stats (verts, components) + two PyVista static renders, §4 how the same thing runs in the pipeline (`mesh.source: splats`, `--stages splats mesh`), §5 **semantic mesh query** — the old `Splatter.query_mesh` (nerfstudio `mesh_decoder.pt`) is gone; rebuild it from surviving parts exactly as `05_lifting/semantic_lifting.ipynb` does for points: MaskCLIP extractor (`BaseFeatureExtractor.get("maskclip")`) → `FeatureAutoencoder` → `lift_features(compressed, out)` per-point features (re-use `05_lifting` cells; if `OUTPUT_DIR` already has the nb05 artefacts load them instead of recomputing), then `features2vertex(mesh.vertices, ff.points, point_features, k=5, sdf_trunc=...)` to put them on the splats-source mesh, `score_queries(vertex_feats, positive=[...], negative=[...], temperature=0.05)` for a text query, paint the mesh (R = score), `mesh_clustering(mesh, scores, similarity_threshold=0.8, ...)` to pick the largest cluster, save `query_mesh.ply` beside it. Explain each step in one markdown line; this IS the replacement for the deleted `06_mesh/create_mesh.ipynb` §5–§6.

- [ ] Execute both with `jupyter nbconvert --to notebook --execute --inplace` (`PYTHONPATH` set, `cwd` = the notebook dir). Strip nothing — executed outputs are the deliverable (as the other tutorials). Report DONE with cell counts and wall time.

---

### Task 5: Docs + gates

- `CLAUDE.md` splats-module paragraph: replace "`mesh.source: splats` deferred to a follow-on" with the built behaviour; replace the owed list with what is still owed (Docker rebuild — no docker in this container; 7-Scenes GT depth error; pose_opt effect; PAGaS); add one sentence with the measured headline numbers from Task 3.
- `docs/superpowers/specs/2026-08-22-splats-module-design.md` "Owed measurements" → tick what Task 3 measured with the numbers.
- `docs/known-test-failures.md`: nothing new unless the gates below add something.
- Gates: `$PY -m collab_splats.dashboard --smoke` → `SMOKE PASS`; `$PY -m pytest tests/mesh tests/wrapper tests/splats tests/evals -q -p no:randomly`; `graphify update .` in the main checkout after the squash.

---

### Squash (controller)

In the worktree: `git add -A` (force-add `docs/superpowers` + the two notebooks), one commit. In the main checkout: `git stash push` the concurrent edits, `git reset --soft 09fa30f2` + `git commit -C 34b034bb` over the followups tree (i.e. build one commit whose tree == followups tip, message = 34b034bb's message + a "Follow-ups (2026-08-23)" paragraph), `git stash pop`, resolve `configs/base.yaml` (+ keep the regenerated `uv.lock`).
