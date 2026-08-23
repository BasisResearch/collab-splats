# Splats Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `collab_splats/nerfstudio/` + `wrapper/splatter.py` (gsplat-rade fork + nerfstudio fork) with a small `collab_splats/splats/` package that trains 3DGS/2DGS splats on upstream gsplat from an existing pointcloud stage, wired in as leaf stage `splats`.

**Architecture:** Five focused files — `cameras.py` (vendored pose refinement), `losses.py` (registry of small loss functions + one scheduled loop), `rendering.py` (one `render_view` for both primitives), `trainer.py` (config with all tunables, Gaussian init, strategy, training loop), `outputs.py` (ply / ckpt / rendered zarr / quality report). `train()` takes plain arrays; `Reconstructor.splats()` assembles them from `PointcloudResult` + `FrameStore` + `feedforward.zarr`. Training happens in the COLMAP world frame (no normalisation). Densification: `MCMCStrategy` for 3DGS (fixed Gaussian budget, no gradient heuristics), `DefaultStrategy` for 2DGS (the only pairing upstream ships).

**Tech Stack:** gsplat upstream pinned at commit `d2f5c0f` (git source, `no-build-isolation`), torch, zarr v3, numpy, sklearn (kNN), pytest. Python: `/opt/venv/reconstruction/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-22-splats-module-design.md`

---

## Binding conventions (read before any task)

- **Work in a worktree.** Task 0 creates `../collab-splats-splats` on branch `feat/splats-module` off `refactor/cu121-uv-migration`. Every later command runs from that worktree. The main checkout has foreign unstaged edits (`pyproject.toml`, `uv.lock`, `configs/base.yaml`, …) from a concurrent session — the worktree sidesteps them.
- `export PY=/opt/venv/reconstruction/bin/python` and use `$PY` everywhere. The venv is editable-installed against the MAIN checkout, so in the worktree always `export PYTHONPATH=$PWD` and confirm `$PY -c "import collab_splats; print(collab_splats.__file__)"` prints the worktree path.
- Commit with `git commit --only <files>` (new files must be `git add <path>`-ed first — `--only` rejects untracked pathspecs); check `ls .git/sequencer 2>/dev/null` is empty first. Plans/specs need `git add -f docs/superpowers/...`.
- Never run repo-wide `black .`. Format only touched files: `$PY -m black <paths> && $PY -m isort <paths>`. Black 120 / isort 88 — write long imports parenthesized.
- **Readability rules (user-mandated for this module):**
  - Every logical block gets a one-line comment saying what it does. Blocks are separated by a blank line.
  - Docstrings open with `"""` on its own line, text on the next line, closing `"""` on its own line.
  - Names say what things are: `cam_to_world` not `c2w`, `n_views` not `N`, `write_splat_outputs` not `_save`.
  - **Unpack before you call.** Never index/slice/compute inside a call's argument list. `rendered_depth = render["depth"][has_target]` on its own line, then `depth_l1_loss(rendered_depth, target_depth, scene_scale)`.
  - Tunables live on `SplatsConfig` with defaults, not as module constants. Module constants are only for things that are math or fixed by gsplat (`SH_DC_NORMALISER`, `PRIMITIVES`).
  - `########` dividers between major sections.
- CUDA tests: `pytest.mark.skipif(not torch.cuda.is_available(), ...)`. Do not run CUDA tests while a heavy tmux job is running (46.6 GB cgroup cap).
- Vendored code cites repo + commit + file + lines at the site.

### Spec deviations recorded here (Task 9 writes them back into the spec)

1. **Depth-loss targets.** Spec says COLMAP `points3D` tracks. Feedforward reconstructions carry NO `Point2D` tracks (`build_pycolmap_reconstruction`, `collab_splats/pointcloud/feedforward/base.py:747`, adds points with empty tracks) — that path would be inert. Targets instead come from `feedforward.zarr` `depth` masked by `confidence_mask(confidence, mesh.conf_percentile)` — the depth the `mesh` stage already trusts — resized nearest to frame resolution inside the trainer. `train()` receives an optional `(n_views, h, w)` float32 array, `0 = no target`.
2. **`mesh.source: splats` is deferred** to a follow-on plan. This plan ends at the `splats` stage producing `splats.zarr`; `mesh` keeps reading `feedforward.zarr`.
3. **Densification strategy.** 3DGS uses `MCMCStrategy` (upstream `mcmc` preset: `cap_max`, `opacity_reg=0.01`, `scale_reg=0.01`), not `DefaultStrategy`. 2DGS keeps `DefaultStrategy` (upstream `simple_trainer_2dgs.py` ships no MCMC pairing). No `strategy` config key — primitive decides.

---

## File structure

**Create**
- `collab_splats/splats/__init__.py` — `GSPLAT_COMMIT`, exports `SplatsConfig`, `train`.
- `collab_splats/splats/cameras.py` — vendored `CameraOptModule`, `rotation_6d_to_matrix`.
- `collab_splats/splats/losses.py` — loss registry (`depth_loss`, `normal_consistency_loss`, `distortion_loss`, `opacity_reg_loss`, `scale_reg_loss`), `compute_losses`.
- `collab_splats/splats/rendering.py` — `gaussian_normals_in_camera_frame`, `render_view`.
- `collab_splats/splats/trainer.py` — `SplatsConfig`, `compute_scene_scale`, `init_gaussians_from_points`, `make_strategy`, `make_pose_refiner`, `prepare_training_target`, `train`.
- `collab_splats/splats/outputs.py` — `render_all_views`, `write_splat_outputs`.
- `tests/splats/__init__.py`, `tests/splats/synthetic.py`, `tests/splats/test_cameras.py`, `test_losses.py`, `test_rendering.py`, `test_trainer.py`, `test_outputs.py`
- `tests/wrapper/test_splats_stage.py`

**Modify**
- `pyproject.toml`, `setup.sh`, `Dockerfile`, `README.md` — gsplat pin + nerfstudio removal.
- `tests/test_cu121_migration.py` — module list + gsplat assertions.
- `collab_splats/__init__.py`, `collab_splats/wrapper/__init__.py`, `collab_splats/wrapper/config.py` — strip Splatter.
- `collab_splats/wrapper/reconstructor.py` — drop `nerfstudio` method; add `splats` stage.
- `configs/base.yaml` — drop `nerfstudio:`; add `splats:`.
- `configs/README.md`, `CLAUDE.md`, `collab_splats/mesh/tsdf.py` comments, `docs/source/{conf.py,index.rst,getting_started.md,api/wrapper.rst,tutorials/index.rst}`, `docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb` (one markdown sentence).

**Delete**
- `collab_splats/nerfstudio/`, `collab_splats/wrapper/splatter.py`, `tests/nerfstudio_methods/`, `tests/test_models.py`, `tests/wrapper/test_splatter_mesh.py`, `tests/wrapper/test_splatter_query.py`, `docs/source/tutorials/03_splats/`, `docs/source/tutorials/06_mesh/`.

---

### Task 0: Worktree

- [x] **Step 1: Create the worktree**

```bash
cd /workspace/collab-splats
git worktree add ../collab-splats-splats -b feat/splats-module refactor/cu121-uv-migration
cd ../collab-splats-splats
export PY=/opt/venv/reconstruction/bin/python PYTHONPATH=$PWD
$PY -c "import collab_splats; print(collab_splats.__file__)"
```
Expected: prints `/workspace/collab-splats-splats/collab_splats/__init__.py`. If it prints the main checkout, `PYTHONPATH` is not exported — fix before continuing.

- [x] **Step 2: Baseline**

Run: `$PY -m pytest tests/wrapper/test_verify_stage.py -q`
Expected: PASS.

---

### Task 1: Swap gsplat to upstream, drop nerfstudio dependency (incl. Dockerfile)

**Files:**
- Modify: `pyproject.toml` (lines 4, 74–76, 93–95, 179–181, 210–211, 258, 262), `setup.sh:12,55,60,63`, `Dockerfile:2,4,13,19,50`, `README.md:30,45,52`, `tests/test_cu121_migration.py:122-133,174-237`

The Dockerfile only runs `setup.sh` / `uv sync`, so the lock file drives what gets installed; its edits here are comment/wording only. A full image rebuild is verification owed after this plan.

- [x] **Step 1: Write the failing dependency test**

In `tests/test_cu121_migration.py` replace `test_gsplat_rade_fork` and `test_gsplat_not_overwritten` (~174–190) with:

```python
def test_gsplat_upstream_pinned():
    """
    gsplat is upstream nerfstudio-project/gsplat @ d2f5c0f: gsplat.losses, 2DGS, MCMC and extra_signals exist.
    """
    import inspect

    import gsplat
    from gsplat.losses import depth_l1_loss, normal_cosine_loss, opacity_reg_loss, scale_reg_loss, ssim_loss  # noqa: F401
    from gsplat.strategy import MCMCStrategy  # noqa: F401

    assert hasattr(gsplat, "rasterization_2dgs")
    assert not hasattr(gsplat, "rasterization_2dgs_inria_wrapper"), "gsplat-rade fork is still installed"
    assert "extra_signals" in inspect.signature(gsplat.rasterization).parameters
```

Delete `test_nerfstudio_installed_local` (~212–237). In the module list (~122–133) replace every `"collab_splats.nerfstudio.*"` and `"collab_splats.wrapper.splatter"` entry with:

```python
        "collab_splats.splats",
        "collab_splats.splats.cameras",
        "collab_splats.splats.losses",
        "collab_splats.splats.rendering",
        "collab_splats.splats.trainer",
        "collab_splats.splats.outputs",
```

- [x] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_cu121_migration.py::test_gsplat_upstream_pinned -v`
Expected: FAIL — `ImportError` on `gsplat.losses`.

- [x] **Step 3: Edit pyproject.toml**

Line 258, replace `gsplat = { git = "https://github.com/brian-xu/gsplat-rade.git" }` with:
```toml
# Upstream gsplat, main @ 2026-07-09 (version string 1.5.3; the v1.5.3 TAG lacks gsplat.losses and
# extra_signals). Newest rev that builds on torch 2.5.1+cu121: 4561ac4 (2026-06-01) needs CCCL
# `cuda::ceil_div` (CUDA >= 12.3) and 31f5b3d (2026-06-08) needs `c10d::wait_tensor` (torch >= 2.7).
# Headers still need CCCL >= 2.2 for <cuda/std/optional> — see setup.sh. Bump deliberately.
gsplat = { git = "https://github.com/nerfstudio-project/gsplat.git", rev = "d2f5c0f" }
```
Line 262: delete the `nerfstudio = { git = ... }` source. Lines 93–95: delete the nerfstudio comment + `"nerfstudio",`. Lines 74–76: comment → `# gsplat: upstream, built from source against the cu121 toolchain (see [tool.uv.sources])`. Lines 179–181: delete `[project.entry-points."nerfstudio.method_configs"]`. Lines 210–211: delete the `tests/nerfstudio_methods` pytest comment. Line 4: description → `"Feedforward reconstruction, Gaussian-splat training, meshing and localization pipeline"`. Keep `no-build-isolation-package = ["bae", "gsplat"]`.

- [x] **Step 4: Reword shell/docker/readme**

`setup.sh` 12, 55, 60, 63: `gsplat-rade` → `gsplat`; keep the `import gsplat` smoke; drop any `rasterization_2dgs_inria_wrapper` probe.
`Dockerfile` 2, 4, 13, 19, 50: `gsplat-rade` → `gsplat`, drop "installs nerfstudio" wording.
`README.md` 30, 45, 52: `gsplat-rade` → `gsplat`; delete the nerfstudio install sentence.

- [x] **Step 5: Re-lock and sync**

Run: `uv lock && uv sync`
Expected: gsplat builds from source (minutes). Then `grep -c 'gsplat-rade\|name = "nerfstudio"' uv.lock` → `0`.

- [x] **Step 6: Run test**

Run: `$PY -m pytest tests/test_cu121_migration.py -v -k gsplat` → PASS. (The module-import test fails on `collab_splats.splats` until Task 3 — expected.)

- [x] **Step 7: Commit**

```bash
git commit --only pyproject.toml uv.lock setup.sh Dockerfile README.md tests/test_cu121_migration.py \
  -m "build(deps): pin upstream gsplat d2f5c0f, drop gsplat-rade fork and nerfstudio"
```

---

### Task 2: Retire nerfstudio, Splatter, and their tutorials

**Files:**
- Delete: `collab_splats/nerfstudio/`, `collab_splats/wrapper/splatter.py`, `tests/nerfstudio_methods/`, `tests/test_models.py`, `tests/wrapper/test_splatter_mesh.py`, `tests/wrapper/test_splatter_query.py`, `docs/source/tutorials/03_splats/`, `docs/source/tutorials/06_mesh/`
- Modify: `collab_splats/__init__.py`, `collab_splats/wrapper/__init__.py:10-16`, `collab_splats/wrapper/config.py:1`, `collab_splats/wrapper/reconstructor.py:54,715-716,887-…`, `configs/base.yaml:111-114`, `configs/README.md:324`, `CLAUDE.md:91-92`, `collab_splats/mesh/tsdf.py:22,119`, `docs/source/conf.py:40`, `docs/source/index.rst:4`, `docs/source/getting_started.md:13,26-50`, `docs/source/api/wrapper.rst:6`, `docs/source/tutorials/index.rst:42-45`, `docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb:16`

`query_mesh` lives only in `splatter.py` (no other callers) — it goes with it.

**Found during implementation:** `BasePointcloudCreator._write_transforms` imported `nerfstudio.process_data.colmap_utils.colmap_to_json` — every creator raised `ModuleNotFoundError` once nerfstudio left the env. Its output never survived (`Reconstructor._write_transforms_json` replaced `frames`, `_write_ply` replaced the ply; `ply_file_path`/`applied_transform` only served splatfacto), so the method and its three call sites were deleted and the Reconstructor now writes `transforms.json` outright (`camera_model` + `frames`, no merge). Commit `95c3fbcf`.

- [x] **Step 1: Write the failing test**

Append to `tests/wrapper/test_reconstructor.py`:

```python
def test_nerfstudio_method_rejected():
    """
    pointcloud.method=nerfstudio is gone — validation only knows feedforward and sfm.
    """
    from collab_splats.wrapper.reconstructor import _VALID_METHODS

    assert _VALID_METHODS == {"feedforward", "sfm"}
    assert not hasattr(Reconstructor, "_run_nerfstudio")
```

- [x] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/wrapper/test_reconstructor.py::test_nerfstudio_method_rejected -v` → FAIL on the set comparison.

- [x] **Step 3: Delete files**

```bash
git rm -r collab_splats/nerfstudio tests/nerfstudio_methods docs/source/tutorials/03_splats docs/source/tutorials/06_mesh
git rm collab_splats/wrapper/splatter.py tests/test_models.py tests/wrapper/test_splatter_mesh.py tests/wrapper/test_splatter_query.py
```

- [x] **Step 4: Strip code references**

`collab_splats/__init__.py` → whole file:
```python
"""
collab-splats: feedforward reconstruction, splat training, meshing and localization.
"""

__version__ = "0.0.1"
```
`collab_splats/wrapper/__init__.py` 10–16: delete the guarded `Splatter`/`SplatterConfig` import and the two `__all__` entries.
`collab_splats/wrapper/config.py:1`: docstring → `Configuration loading utilities for Reconstructor workflows.`
`collab_splats/wrapper/reconstructor.py`: line 54 `_VALID_METHODS = {"feedforward", "sfm"}`; delete the `if method == "nerfstudio": result = self._run_nerfstudio()` branch (715–716); delete the whole `_run_nerfstudio` method (~887 to the next `def`). Then `grep -n nerfstudio collab_splats/wrapper/reconstructor.py` must print nothing.
`configs/base.yaml` 111–114: delete comment + `nerfstudio:` block.
`configs/README.md:324`: `pointcloud.method` allowed values → `feedforward | sfm`.
`CLAUDE.md` 91–92: delete the `nerfstudio/` tree line; `wrapper/` line → `# stage orchestration: Reconstructor (config-driven pipeline), batch drivers`.
`collab_splats/mesh/tsdf.py` 22, 119: drop the nerfstudio mention (`Accepts rendered depth + RGB frames as numpy arrays.`).

- [x] **Step 5: Strip docs references**

`docs/source/conf.py:40`: remove `nerfstudio` from the autodoc mock list.
`docs/source/index.rst:4`: → `Feedforward reconstruction, Gaussian splats, semantics, and mesh export — built on gsplat.`
`docs/source/api/wrapper.rst:6`: delete the `.. automodule:: collab_splats.wrapper.splatter` block (directive + option lines).
`docs/source/tutorials/index.rst`: delete the `03_splats/*` and `06_mesh/create_mesh` toctree entries (lines 42–45).
`docs/source/getting_started.md`: line 13 → `This installs the package in development mode along with gsplat and all CUDA dependencies.`; replace the "Gaussian Splatting with depth and normals" quick-start (lines 26–50) with:

````markdown
### Reconstruct a video

```bash
python docs/examples/run_pipeline.py data/tutorial/<video.mp4> --config configs/base.yaml
```

Stages run in order: frame sampling → feedforward pointcloud → (optional) splats, mesh,
semantics, localization. Enable Gaussian-splat training with `splats.enabled: true` in the
config; outputs land in `<output>/<backend>/splats/`.
````
`docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb:16`: change the markdown sentence to `"Gaussian-splat training lives in the `splats` stage (`collab_splats/splats/`)."` (edit the JSON string in place; no cell execution).

- [x] **Step 6: Verify no dangling references**

Run: `grep -rn 'splatter\|nerfstudio\|ns-train' --include='*.py' --include='*.yaml' --include='*.toml' --include='*.rst' --include='*.md' collab_splats configs tests docs/source pyproject.toml | grep -v 'display_name\|kernelspec\|/opt/conda/envs'`
Expected: no output.

- [x] **Step 7: Run tests**

Run: `$PY -m pytest tests/wrapper/test_reconstructor.py tests/test_cu121_migration.py -v -x -k "not splats"` → PASS.

- [x] **Step 8: Commit**

```bash
git status --short
git commit --only collab_splats/nerfstudio tests/nerfstudio_methods docs/source/tutorials/03_splats docs/source/tutorials/06_mesh \
  collab_splats/wrapper/splatter.py tests/test_models.py tests/wrapper/test_splatter_mesh.py tests/wrapper/test_splatter_query.py \
  collab_splats/__init__.py collab_splats/wrapper/__init__.py collab_splats/wrapper/config.py collab_splats/wrapper/reconstructor.py \
  configs/base.yaml configs/README.md CLAUDE.md collab_splats/mesh/tsdf.py docs/source/conf.py docs/source/index.rst \
  docs/source/getting_started.md docs/source/api/wrapper.rst docs/source/tutorials/index.rst \
  docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb tests/wrapper/test_reconstructor.py \
  -m "refactor(splats): retire nerfstudio package, Splatter, and their tutorials"
```

---

### Task 3: `splats/cameras.py` — vendored pose refinement

**Files:** Create `collab_splats/splats/__init__.py`, `collab_splats/splats/cameras.py`, `tests/splats/__init__.py` (empty), `tests/splats/test_cameras.py`

Body diffed against `/tmp/gsplat-main/examples/utils.py` lines 27–63 and 132–153: identical logic, only names/comments ours.

- [x] **Step 1: Write the failing test**

```python
"""
CameraOptModule: zero-init is identity; random-init perturbs; 6D rotations are proper rotations.
"""

import torch

from collab_splats.splats.cameras import CameraOptModule, rotation_6d_to_matrix


def test_zero_init_leaves_poses_unchanged():
    refiner = CameraOptModule(3)
    refiner.zero_init()
    cam_to_world = torch.eye(4).expand(3, 4, 4).clone()
    cam_to_world[:, :3, 3] = torch.arange(3).float()[:, None]
    camera_ids = torch.arange(3)
    refined = refiner(cam_to_world, camera_ids)
    assert torch.allclose(refined, cam_to_world)


def test_random_init_changes_poses():
    refiner = CameraOptModule(2)
    refiner.random_init(std=0.1)
    cam_to_world = torch.eye(4).expand(2, 4, 4).clone()
    camera_ids = torch.arange(2)
    refined = refiner(cam_to_world, camera_ids)
    assert not torch.allclose(refined, cam_to_world)


def test_rotation_6d_gives_proper_rotations():
    rotations = rotation_6d_to_matrix(torch.randn(5, 6))
    identity = torch.eye(3).expand(5, 3, 3)
    gram = rotations @ rotations.transpose(-1, -2)
    determinants = torch.linalg.det(rotations)
    assert torch.allclose(gram, identity, atol=1e-5)
    assert torch.allclose(determinants, torch.ones(5), atol=1e-5)
```

- [x] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/splats/test_cameras.py -v` → `ModuleNotFoundError: No module named 'collab_splats.splats'`.

- [x] **Step 3: Write the module**

`collab_splats/splats/__init__.py`:
```python
"""
Gaussian-splat training on upstream gsplat from an existing pointcloud stage.
"""

# The gsplat commit pinned in pyproject.toml; recorded in every splats.zarr for provenance
GSPLAT_COMMIT = "d2f5c0f"
```

`collab_splats/splats/cameras.py`:
```python
"""
Per-camera pose refinement for splat training.

Vendored from nerfstudio-project/gsplat @ d2f5c0f, examples/utils.py:
``CameraOptModule`` lines 27-63, ``rotation_6d_to_matrix`` lines 132-153.
``examples/`` is not shipped in the gsplat wheel, so the two pieces we need are copied verbatim.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """
    Gram-Schmidt 6D rotation representation (Zhou et al. 2019) -> (..., 3, 3) rotation matrices.
    """
    # First basis vector: normalised first triple
    first_triple = rotation_6d[..., :3]
    second_triple = rotation_6d[..., 3:]
    basis_x = F.normalize(first_triple, dim=-1)

    # Second basis vector: second triple with its projection onto basis_x removed
    projection = (basis_x * second_triple).sum(-1, keepdim=True) * basis_x
    basis_y = F.normalize(second_triple - projection, dim=-1)

    # Third basis vector completes the right-handed frame
    basis_z = torch.cross(basis_x, basis_y, dim=-1)
    return torch.stack((basis_x, basis_y, basis_z), dim=-2)


class CameraOptModule(torch.nn.Module):
    """
    Learned per-camera SE(3) delta, applied on the right of camera-to-world.
    """

    def __init__(self, n_cameras: int):
        super().__init__()

        # One 9-vector per camera: translation delta (3) + rotation delta in 6D form (6)
        self.embeds = torch.nn.Embedding(n_cameras, 9)

        # Identity rotation in 6D form; the learned rotation delta is added to it
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, cam_to_world: Tensor, camera_ids: Tensor) -> Tensor:
        """
        Apply the learned deltas: cam_to_world (..., 4, 4), camera_ids (...) -> refined (..., 4, 4).
        """
        assert cam_to_world.shape[:-2] == camera_ids.shape
        batch_shape = cam_to_world.shape[:-2]

        # Split each camera's 9-vector into translation and rotation deltas
        pose_deltas = self.embeds(camera_ids)
        translation_delta = pose_deltas[..., :3]
        rotation_delta = pose_deltas[..., 3:]
        identity_6d = self.identity.expand(*batch_shape, -1)
        rotation = rotation_6d_to_matrix(rotation_delta + identity_6d)

        # Build the 4x4 delta transform and compose it onto the input pose
        delta_transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        delta_transform[..., :3, :3] = rotation
        delta_transform[..., :3, 3] = translation_delta
        return torch.matmul(cam_to_world, delta_transform)
```

- [x] **Step 4: Run test**

Run: `$PY -m pytest tests/splats/test_cameras.py -v` → 3 PASSED.

- [x] **Step 5: Commit**

```bash
git commit --only collab_splats/splats/__init__.py collab_splats/splats/cameras.py tests/splats/__init__.py tests/splats/test_cameras.py \
  -m "feat(splats): vendor gsplat CameraOptModule for pose optimization"
```

---

### Task 4: `splats/losses.py` — loss registry + scheduled loop

**Files:** Create `collab_splats/splats/losses.py`, `tests/splats/test_losses.py`

Contract: `compute_losses(step, render, target, gaussians, loss_schedule, scene_scale) -> (total, {name: value})`. Photometric `0.8·L1 + 0.2·(1−SSIM)` always on. `loss_schedule` is the yaml `losses:` mapping `{name: {weight, start}}`; a loss contributes iff `weight > 0`, `step >= start`, and its function returns a value (returns `None` when its input is absent). Every optional loss has the same signature `(render, target, gaussians, scene_scale) -> Tensor | None`, so adding one (PAGaS later) is one function + one registry entry.

`render` keys: `rgb`, `alpha`, `depth` `(1,H,W,C)`, `normal`, `depth_normal` `(1,H,W,3)`, and `distortion` `(1,H,W,1)` **only for 2dgs** (key absent otherwise). `target` keys: `rgb` `(1,H,W,3)` in `[0,1]`, `depth` `(1,H,W,1)` or `None` (0 = no target). `gaussians` is the `ParameterDict` (raw `opacities` logits, raw `scales` logs — what `gsplat.losses` regularisers expect).

- [x] **Step 1: Write the failing tests**

```python
"""
compute_losses: photometric always on; optional losses gated by weight > 0, start step, and input presence.
"""

import pytest
import torch

from collab_splats.splats.losses import OPTIONAL_LOSSES, compute_losses


def _render(height=16, width=16, with_distortion=True):
    gen = torch.Generator().manual_seed(0)
    normal = torch.nn.functional.normalize(torch.randn(1, height, width, 3, generator=gen), dim=-1)
    noisy_normal = torch.nn.functional.normalize(normal + 0.1 * torch.randn_like(normal), dim=-1)
    render = {
        "rgb": torch.rand(1, height, width, 3, generator=gen),
        "alpha": torch.rand(1, height, width, 1, generator=gen),
        "depth": torch.rand(1, height, width, 1, generator=gen) + 0.5,
        "normal": normal,
        "depth_normal": noisy_normal,
    }
    if with_distortion:
        render["distortion"] = torch.rand(1, height, width, 1, generator=gen)
    return render


def _target(height=16, width=16, with_depth=True):
    gen = torch.Generator().manual_seed(1)
    depth = torch.rand(1, height, width, 1, generator=gen) + 0.5
    depth[:, :4] = 0.0  # rows without a target
    rgb = torch.rand(1, height, width, 3, generator=gen)
    return {"rgb": rgb, "depth": depth if with_depth else None}


def _gaussians(n_points=50):
    opacities = torch.nn.Parameter(torch.zeros(n_points))
    scales = torch.nn.Parameter(torch.zeros(n_points, 3))
    return torch.nn.ParameterDict({"opacities": opacities, "scales": scales})


def test_registry_names():
    assert set(OPTIONAL_LOSSES) == {"depth", "normal_consistency", "distortion", "opacity_reg", "scale_reg"}


def test_photometric_only_when_no_optional_losses():
    total, values = compute_losses(0, _render(), _target(), _gaussians(), {}, 1.0)
    expected = 0.8 * values["l1"] + 0.2 * values["ssim"]
    assert set(values) == {"l1", "ssim"}
    assert total.item() == pytest.approx(expected, rel=1e-5)


def test_loss_waits_for_its_start_step():
    schedule = {"depth": {"weight": 1.0, "start": 100}}
    _, before = compute_losses(99, _render(), _target(), _gaussians(), schedule, 1.0)
    _, after = compute_losses(100, _render(), _target(), _gaussians(), schedule, 1.0)
    assert "depth" not in before and "depth" in after


def test_zero_weight_skips_loss():
    schedule = {"depth": {"weight": 0.0}}
    _, values = compute_losses(0, _render(), _target(), _gaussians(), schedule, 1.0)
    assert "depth" not in values


def test_depth_loss_ignores_zero_targets_and_scales_with_scene_scale():
    schedule = {"depth": {"weight": 1.0}}
    render, target = _render(), _target()
    _, at_scale_1 = compute_losses(0, render, target, _gaussians(), schedule, 1.0)
    _, at_scale_2 = compute_losses(0, render, target, _gaussians(), schedule, 2.0)
    assert at_scale_2["depth"] == pytest.approx(2 * at_scale_1["depth"], rel=1e-5)

    # Perfect prediction on targeted pixels -> zero loss, whatever the untargeted rows hold
    render["depth"] = target["depth"].clone()
    render["depth"][:, :4] = 123.0
    _, perfect = compute_losses(0, render, target, _gaussians(), schedule, 1.0)
    assert perfect["depth"] == pytest.approx(0.0, abs=1e-6)


def test_depth_loss_skipped_without_targets():
    schedule = {"depth": {"weight": 1.0}}
    target = _target(with_depth=False)
    _, values = compute_losses(0, _render(), target, _gaussians(), schedule, 1.0)
    assert "depth" not in values


def test_normal_consistency_is_zero_for_identical_normals():
    render = _render()
    render["depth_normal"] = render["normal"].clone()
    render["alpha"] = torch.ones_like(render["alpha"])
    schedule = {"normal_consistency": {"weight": 1.0}}
    _, values = compute_losses(0, render, _target(), _gaussians(), schedule, 1.0)
    assert values["normal_consistency"] == pytest.approx(0.0, abs=1e-5)


def test_distortion_skipped_when_render_has_no_map():
    schedule = {"distortion": {"weight": 1.0}}
    render_without_map = _render(with_distortion=False)
    _, with_map = compute_losses(0, _render(), _target(), _gaussians(), schedule, 1.0)
    _, without_map = compute_losses(0, render_without_map, _target(), _gaussians(), schedule, 1.0)
    assert "distortion" in with_map and "distortion" not in without_map


def test_regularisers_read_raw_gaussian_params():
    schedule = {"opacity_reg": {"weight": 1.0}, "scale_reg": {"weight": 1.0}}
    _, values = compute_losses(0, _render(), _target(), _gaussians(), schedule, 1.0)
    assert values["opacity_reg"] > 0 and values["scale_reg"] > 0


def test_total_is_weighted_sum():
    schedule = {"depth": {"weight": 0.3}, "normal_consistency": {"weight": 0.7}}
    total, values = compute_losses(0, _render(), _target(), _gaussians(), schedule, 1.0)
    expected = 0.8 * values["l1"] + 0.2 * values["ssim"] + 0.3 * values["depth"] + 0.7 * values["normal_consistency"]
    assert total.item() == pytest.approx(expected, rel=1e-5)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/splats/test_losses.py -v` → `ModuleNotFoundError`.

- [x] **Step 3: Write the module**

```python
"""
Loss registry and the scheduled weighted sum over it.

Photometric (0.8 L1 + 0.2 (1 - SSIM)) is always on. Each optional loss is one small function
with the same signature; ``compute_losses`` loops over the yaml schedule ``name: {weight, start}``
and adds a loss iff weight > 0, step >= start, and the function returns a value.
"""

import torch
from gsplat import losses as gsplat_losses
from torch import Tensor

########################################
# Optional losses — each returns None when its input is absent
########################################


def depth_loss(render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float) -> Tensor | None:
    """
    Disparity L1 against the depth target on the pixels that have one (0 = no target).
    """
    target_depth = target["depth"]
    if target_depth is None:
        return None

    # Only pixels with a target contribute
    has_target = target_depth > 0
    rendered_depth = render["depth"][has_target]
    target_depth = target_depth[has_target]
    return gsplat_losses.depth_l1_loss(rendered_depth, target_depth, scene_scale)


def normal_consistency_loss(render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float) -> Tensor:
    """
    Cosine distance between rendered normals and normals finite-differenced from rendered depth.
    """
    # Depth normal is weighted by (detached) alpha so empty pixels do not pull
    rendered_normal = render["normal"]
    alpha = render["alpha"].detach()
    depth_normal = render["depth_normal"] * alpha
    cosine_distance = gsplat_losses.normal_cosine_loss(rendered_normal, depth_normal)
    return cosine_distance.mean()


def distortion_loss(render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float) -> Tensor | None:
    """
    Mean of the 2DGS rasterizer's per-pixel distortion map; None when the primitive has none.
    """
    distortion_map = render.get("distortion")
    if distortion_map is None:
        return None
    return distortion_map.mean()


def opacity_reg_loss(render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float) -> Tensor:
    """
    Opacity regulariser from gsplat (MCMC); expects raw logit opacities.
    """
    opacities = gaussians["opacities"]
    return gsplat_losses.opacity_reg_loss(opacities)


def scale_reg_loss(render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float) -> Tensor:
    """
    Scale regulariser from gsplat (MCMC); expects raw log scales.
    """
    log_scales = gaussians["scales"]
    return gsplat_losses.scale_reg_loss(log_scales)


# Name in the yaml `losses:` block -> function. Also the allow-list for config validation.
OPTIONAL_LOSSES = {
    "depth": depth_loss,
    "normal_consistency": normal_consistency_loss,
    "distortion": distortion_loss,
    "opacity_reg": opacity_reg_loss,
    "scale_reg": scale_reg_loss,
}

########################################
# Weighted sum
########################################


def compute_losses(
    step: int,
    render: dict,
    target: dict,
    gaussians: torch.nn.ParameterDict,
    loss_schedule: dict[str, dict],
    scene_scale: float,
) -> tuple[Tensor, dict[str, float]]:
    """
    Weighted sum of the losses active at `step`. Returns (total, {name: value}).
    """
    # Photometric: L1 + SSIM between rendered and target RGB (ssim_loss wants NCHW)
    rendered_rgb = render["rgb"]
    target_rgb = target["rgb"]
    rendered_nchw = rendered_rgb.permute(0, 3, 1, 2)
    target_nchw = target_rgb.permute(0, 3, 1, 2)
    l1 = gsplat_losses.l1_loss(rendered_rgb, target_rgb).mean()
    ssim = gsplat_losses.ssim_loss(rendered_nchw, target_nchw)
    total = 0.8 * l1 + 0.2 * ssim
    values = {"l1": l1.item(), "ssim": ssim.item()}

    # Optional losses: skip when not started, zero-weighted, or the loss has no input this step
    for name, spec in loss_schedule.items():
        weight = spec["weight"]
        start = spec.get("start", 0)
        if weight <= 0 or step < start:
            continue
        loss_fn = OPTIONAL_LOSSES[name]
        value = loss_fn(render, target, gaussians, scene_scale)
        if value is None:
            continue
        total = total + weight * value
        values[name] = value.item()

    return total, values
```

- [x] **Step 4: Run tests**

Run: `$PY -m pytest tests/splats/test_losses.py -v` → 10 PASSED (CPU).

- [x] **Step 5: Commit**

```bash
git commit --only collab_splats/splats/losses.py tests/splats/test_losses.py \
  -m "feat(splats): loss registry + compute_losses scheduled loop"
```

---

### Task 5: `splats/rendering.py` — one render for both primitives

**Files:** Create `collab_splats/splats/rendering.py`, `tests/splats/synthetic.py`, `tests/splats/test_rendering.py`

- [x] **Step 1: Write the shared synthetic scene helper**

`tests/splats/synthetic.py`:
```python
"""
Synthetic splat-training scene: random coloured points in a box, cameras on a ring, analytic depth.
"""

import numpy as np

FOCAL = 60.0


def make_scene(n_views=8, height=64, width=64, n_points=200):
    """
    Returns (images uint8 (n,H,W,3), world_to_cam (n,4,4), intrinsics (n,3,3), points, colors, depths (n,H,W)).
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 0.5, (n_points, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (n_points, 3)).astype(np.uint8)
    intrinsics = np.array([[FOCAL, 0, width / 2], [0, FOCAL, height / 2], [0, 0, 1]], dtype=np.float32)

    images, depths, world_to_cam = [], [], []
    for view in range(n_views):
        # Camera on a ring of radius 3, looking at the origin (OpenCV: +z forward)
        angle = 2 * np.pi * view / n_views
        position = np.array([3 * np.cos(angle), 0.3, 3 * np.sin(angle)], dtype=np.float32)
        forward = -position / np.linalg.norm(position)
        right = np.cross([0, 1, 0], forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation_c2w = np.stack([right, down, forward], axis=1)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation_c2w.T
        pose[:3, 3] = -rotation_c2w.T @ position
        world_to_cam.append(pose)

        # Paint each point as a 3x3 dot, far to near so nearer points overwrite
        image = np.zeros((height, width, 3), np.uint8)
        depth = np.zeros((height, width), np.float32)
        points_cam = (pose[:3, :3] @ points.T + pose[:3, 3:]).T
        far_to_near = np.argsort(-points_cam[:, 2])
        for idx in far_to_near:
            z = points_cam[idx, 2]
            if z <= 0:
                continue
            pixel = intrinsics[:2, :2] @ (points_cam[idx, :2] / z) + intrinsics[:2, 2]
            u, v = pixel.round().astype(int)
            if 1 <= u < width - 1 and 1 <= v < height - 1:
                image[v - 1 : v + 2, u - 1 : u + 2] = colors[idx]
                depth[v - 1 : v + 2, u - 1 : u + 2] = z
        images.append(image)
        depths.append(depth)

    intrinsics_per_view = np.stack([intrinsics] * n_views)
    return np.stack(images), np.stack(world_to_cam), intrinsics_per_view, points, colors, np.stack(depths)
```

- [x] **Step 2: Write the failing render test**

`tests/splats/test_rendering.py`:
```python
"""
render_view: output shape contract for both primitives; 3DGS normals face the camera.
"""

import pytest
import torch

from collab_splats.splats.rendering import gaussian_normals_in_camera_frame, render_view

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


def _gaussians(n_points=200, device="cuda"):
    gen = torch.Generator().manual_seed(0)
    gaussians = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.rand(n_points, 3, generator=gen) - 0.5),
            "scales": torch.nn.Parameter(torch.full((n_points, 3), -3.0)),
            "quats": torch.nn.Parameter(torch.rand(n_points, 4, generator=gen)),
            "opacities": torch.nn.Parameter(torch.zeros(n_points)),
            "sh0": torch.nn.Parameter(torch.rand(n_points, 1, 3, generator=gen)),
            "shN": torch.nn.Parameter(torch.zeros(n_points, 15, 3)),
        }
    )
    return gaussians.to(device)


def _camera(device="cuda"):
    cam_to_world = torch.eye(4, device=device)[None]
    cam_to_world[0, 2, 3] = -4.0
    intrinsics = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device)[None]
    return cam_to_world, intrinsics


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_render_view_shapes(primitive):
    cam_to_world, intrinsics = _camera()
    render, info = render_view(primitive, _gaussians(), cam_to_world, intrinsics, 64, 64, sh_degree=0, absgrad=False)
    assert render["rgb"].shape == (1, 64, 64, 3)
    assert render["alpha"].shape == (1, 64, 64, 1)
    assert render["depth"].shape == (1, 64, 64, 1)
    assert render["normal"].shape == (1, 64, 64, 3)
    assert render["depth_normal"].shape == (1, 64, 64, 3)
    assert ("distortion" in render) == (primitive == "2dgs")
    expected_gradient_key = "means2d" if primitive == "3dgs" else "gradient_2dgs"
    assert expected_gradient_key in info


@cuda
def test_gaussian_normals_face_the_camera():
    gaussians = _gaussians()
    cam_to_world, _ = _camera()
    world_to_cam = torch.linalg.inv(cam_to_world)[0]
    scales = torch.exp(gaussians["scales"])
    normals = gaussian_normals_in_camera_frame(gaussians["quats"], scales, gaussians["means"], world_to_cam)

    rotation_w2c = world_to_cam[:3, :3]
    translation_w2c = world_to_cam[:3, 3]
    means_cam = gaussians["means"] @ rotation_w2c.T + translation_w2c
    normal_lengths = normals.norm(dim=-1)
    facing = (normals * means_cam).sum(-1)
    assert normals.shape == (200, 3)
    assert torch.allclose(normal_lengths, torch.ones(200, device="cuda"), atol=1e-5)
    assert (facing <= 1e-6).all()
```

- [x] **Step 3: Run test to verify it fails**

Run: `$PY -m pytest tests/splats/test_rendering.py -v` → `ModuleNotFoundError`.

- [x] **Step 4: Write the module**

```python
"""
One render call for both splat primitives.

3DGS goes through ``gsplat.rasterization`` (fast kernel, antialiased); per-Gaussian normals are
rendered as an extra signal and the depth normal is finite-differenced from rendered depth.
2DGS goes through ``gsplat.rasterization_2dgs`` which returns both normals natively plus a
distortion map. Normals are camera-space for both so the consistency loss compares like with like.
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization, rasterization_2dgs
from gsplat.utils import depth_to_normal, normalized_quat_to_rotmat
from torch import Tensor


def gaussian_normals_in_camera_frame(quats: Tensor, scales: Tensor, means: Tensor, world_to_cam: Tensor) -> Tensor:
    """
    Per-Gaussian normal (shortest scale axis), rotated into the camera frame and flipped to face it. (N, 3).
    """
    # Shortest axis of each Gaussian is its normal direction in world space
    unit_quats = F.normalize(quats, dim=-1)
    rotations = normalized_quat_to_rotmat(unit_quats)
    shortest_axis = scales.argmin(dim=-1)
    gaussian_index = torch.arange(len(rotations), device=rotations.device)
    normals_world = rotations[gaussian_index, :, shortest_axis]

    # Rotate normals and positions into the camera frame
    rotation_w2c = world_to_cam[:3, :3]
    translation_w2c = world_to_cam[:3, 3]
    normals_cam = normals_world @ rotation_w2c.T
    means_cam = means @ rotation_w2c.T + translation_w2c

    # A normal pointing away from the camera (positive dot with the view ray) is flipped
    faces_away = (normals_cam * means_cam).sum(-1, keepdim=True) > 0
    return torch.where(faces_away, -normals_cam, normals_cam)


def render_view(
    primitive: str,
    gaussians: torch.nn.ParameterDict,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    sh_degree: int,
    absgrad: bool,
) -> tuple[dict[str, Tensor], dict]:
    """
    Render one camera. Returns ({rgb, alpha, depth, normal, depth_normal[, distortion]}, strategy info).
    """
    # Activate the raw parameters: log-scales -> scales, logit-opacities -> opacities, SH bands concatenated
    means = gaussians["means"]
    quats = gaussians["quats"]
    scales = torch.exp(gaussians["scales"])
    opacities = torch.sigmoid(gaussians["opacities"])
    sh_coeffs = torch.cat([gaussians["sh0"], gaussians["shN"]], dim=1)
    world_to_cam = torch.linalg.inv(cam_to_world)
    shared_kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_coeffs,
        viewmats=world_to_cam,
        Ks=intrinsics,
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
        absgrad=absgrad,
        render_mode="RGB+ED",
    )

    # 2DGS: the rasterizer returns rgb+depth, alpha, both normals, and the distortion map
    if primitive == "2dgs":
        rgb_depth, alpha, normal, depth_normal, distortion, _median_depth, info = rasterization_2dgs(
            **shared_kwargs, distloss=True
        )
        rgb = rgb_depth[..., :3]
        depth = rgb_depth[..., 3:4]
        render = {
            "rgb": rgb,
            "alpha": alpha,
            "depth": depth,
            "normal": normal,
            "depth_normal": depth_normal,
            "distortion": distortion,
        }
        return render, info

    # 3DGS: normals ride along as an extra per-Gaussian signal; depth normal is finite-differenced
    # in camera space (identity pose) so it lives in the same frame as the rendered normals
    normals_cam = gaussian_normals_in_camera_frame(quats, scales, means, world_to_cam[0])
    rgb_depth, alpha, info = rasterization(**shared_kwargs, rasterize_mode="antialiased", extra_signals=normals_cam)
    rgb = rgb_depth[..., :3]
    depth = rgb_depth[..., 3:4]
    rendered_normals = info["render_extra_signals"]
    identity_pose = torch.eye(4, device=cam_to_world.device)[None]
    render = {
        "rgb": rgb,
        "alpha": alpha,
        "depth": depth,
        "normal": F.normalize(rendered_normals, dim=-1),
        "depth_normal": depth_to_normal(depth, identity_pose, intrinsics),
    }
    return render, info
```

- [x] **Step 5: Run tests**

Run: `$PY -m pytest tests/splats/test_rendering.py -v` → 3 PASSED. If `render_extra_signals` is missing from `info`, re-check the pin (`extra_signals` must be in `inspect.signature(gsplat.rasterization).parameters`).

- [x] **Step 6: Commit**

```bash
git commit --only collab_splats/splats/rendering.py tests/splats/synthetic.py tests/splats/test_rendering.py \
  -m "feat(splats): render_view — one call for 3dgs (extra-signal normals) and 2dgs"
```

---

### Task 6: `splats/trainer.py` — config, Gaussian init, strategy, training loop

**Files:** Create `collab_splats/splats/trainer.py`, `tests/splats/test_trainer.py`; modify `collab_splats/splats/__init__.py`; stub `collab_splats/splats/outputs.py`

`train()` calls `write_splat_outputs` from Task 7, so this task stubs it; the end-to-end test lands in Task 7.

- [x] **Step 1: Write the failing tests**

`tests/splats/test_trainer.py`:
```python
"""
SplatsConfig validation, scene scale, Gaussian init, strategy choice and training-target preparation.
"""

import numpy as np
import pytest
import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from collab_splats.splats.trainer import (
    SplatsConfig,
    compute_scene_scale,
    init_gaussians_from_points,
    make_strategy,
    prepare_training_target,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


def test_config_from_dict_keeps_given_values_and_defaults():
    block = {"enabled": True, "primitive": "2dgs", "max_steps": 10, "losses": {"depth": {"weight": 0.1}}}
    cfg = SplatsConfig.from_dict(block)
    assert (cfg.primitive, cfg.max_steps, cfg.pose_opt, cfg.sh_degree) == ("2dgs", 10, False, 3)
    assert cfg.losses == {"depth": {"weight": 0.1}}


def test_default_losses_match_primitive():
    losses_3dgs = SplatsConfig(primitive="3dgs").losses
    losses_2dgs = SplatsConfig(primitive="2dgs").losses
    assert {"opacity_reg", "scale_reg"} <= set(losses_3dgs)
    assert "distortion" in losses_2dgs and "opacity_reg" not in losses_2dgs


@pytest.mark.parametrize(
    "bad",
    [
        {"primitive": "4dgs"},
        {"losses": {"tv": {"weight": 1.0}}},
        {"losses": {"depth": {"weight": 1.0, "stop": 5}}},
        {"losses": {"depth": {"start": 5}}},
        {"primitive": "3dgs", "losses": {"distortion": {"weight": 0.1}}},
        {"unknown_key": 1},
    ],
)
def test_config_rejects_invalid(bad):
    with pytest.raises(ValueError):
        SplatsConfig.from_dict(bad)


def test_scene_scale_is_max_camera_spread_times_margin():
    cam_to_world = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    cam_to_world[:, 0, 3] = [-1.0, 0.0, 1.0]
    scene_scale = compute_scene_scale(torch.from_numpy(cam_to_world))
    assert scene_scale == pytest.approx(1.1)


def test_strategy_follows_primitive():
    mcmc = make_strategy(SplatsConfig(primitive="3dgs", cap_max=1234))
    default = make_strategy(SplatsConfig(primitive="2dgs"))
    assert isinstance(mcmc, MCMCStrategy) and mcmc.cap_max == 1234
    assert isinstance(default, DefaultStrategy) and default.key_for_gradient == "gradient_2dgs"


@cuda
def test_init_gaussians_shapes_and_optimizers():
    points = np.random.default_rng(0).uniform(-1, 1, (200, 3)).astype(np.float32)
    colors = np.full((200, 3), 128, np.uint8)
    cfg = SplatsConfig()
    gaussians, optimizers = init_gaussians_from_points(cfg, points, colors, scene_scale=1.0, device="cuda")
    assert gaussians["means"].shape == (200, 3) and gaussians["sh0"].shape == (200, 1, 3)
    assert gaussians["shN"].shape == (200, 15, 3) and gaussians["opacities"].shape == (200,)

    opacities = torch.sigmoid(gaussians["opacities"])
    expected_opacity = torch.full((200,), cfg.init_opacity, device="cuda")
    assert torch.allclose(opacities, expected_opacity)
    assert set(optimizers) == {"means", "scales", "quats", "opacities", "sh0", "shN"}
    means_lr = optimizers["means"].param_groups[0]["lr"]
    assert means_lr == pytest.approx(cfg.means_lr)


def test_prepare_training_target_scales_rgb_and_resizes_depth():
    image = np.full((8, 8, 3), 255, np.uint8)
    depth = np.array([[1.0, 0.0], [2.0, 3.0]], np.float32)
    target = prepare_training_target(image, depth, device="cpu")
    assert target["rgb"].shape == (1, 8, 8, 3) and target["rgb"].max() == 1.0
    assert target["depth"].shape == (1, 8, 8, 1)
    assert target["depth"][0, 0, 0, 0] == 1.0 and target["depth"][0, 0, 7, 0] == 0.0 and target["depth"][0, 7, 7, 0] == 3.0

    no_depth = prepare_training_target(image, None, device="cpu")
    assert no_depth["depth"] is None
```

- [x] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/splats/test_trainer.py -v` → `ModuleNotFoundError`.

- [x] **Step 3: Write the module**

```python
"""
Trainer for 3DGS / 2DGS splats on upstream gsplat.

Training runs in the COLMAP world frame (no normalisation) so poses, depth and the ply line up
with every other stage artifact. ``scene_scale`` — 1.1 x the largest camera distance from the
camera centroid, as in gsplat's simple_trainer — only scales the means learning rate, the
densification thresholds and the depth loss.

Densification: ``MCMCStrategy`` for 3DGS (fixed budget ``cap_max``, dead Gaussians relocated,
no gradient heuristics — upstream's ``mcmc`` preset), ``DefaultStrategy`` for 2DGS (the only
pairing upstream ships; prunes opacity < 0.005 and oversized Gaussians, resets opacity every 3000).
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR

from collab_splats.splats.cameras import CameraOptModule
from collab_splats.splats.losses import OPTIONAL_LOSSES, compute_losses
from collab_splats.splats.outputs import write_splat_outputs
from collab_splats.splats.rendering import render_view
from collab_splats.utils.progress import progress

logger = logging.getLogger(__name__)

PRIMITIVES = ("3dgs", "2dgs")
SH_DC_NORMALISER = 0.28209479177387814  # rgb -> SH degree-0 coefficient (1 / (2 sqrt(pi)))

########################################
# Config — every tunable, with gsplat simple_trainer defaults
########################################


def _default_losses(primitive: str) -> dict[str, dict]:
    """
    Default loss schedule per primitive: MCMC regularisers for 3dgs, distortion for 2dgs.
    """
    losses = {"depth": {"weight": 0.01}, "normal_consistency": {"weight": 0.05, "start": 7000}}
    if primitive == "3dgs":
        losses["opacity_reg"] = {"weight": 0.01}
        losses["scale_reg"] = {"weight": 0.01}
    else:
        losses["distortion"] = {"weight": 100.0, "start": 3000}
    return losses


@dataclass
class SplatsConfig:
    """
    Trainer knobs; mirrors the ``splats:`` yaml block minus ``enabled``. Defaults follow gsplat's examples.
    """

    primitive: str = "3dgs"
    max_steps: int = 30000
    pose_opt: bool = False
    losses: dict[str, dict] | None = None  # None -> _default_losses(primitive)

    # Appearance
    sh_degree: int = 3
    sh_degree_interval: int = 1000  # one more SH band unlocked every this many steps
    init_opacity: float = 0.1

    # Learning rates (means_lr and pose_lr are multiplied by scene_scale; means/pose decay 0.01x over the run)
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 1.25e-4
    pose_lr: float = 1e-5

    # Densification
    cap_max: int = 1_000_000  # 3dgs (MCMC): Gaussian budget
    grow_grad2d: float = 8e-4  # 2dgs (Default): 2D-gradient threshold to split/duplicate

    log_every: int = 500

    def __post_init__(self):
        if self.losses is None:
            self.losses = _default_losses(self.primitive)

    @classmethod
    def from_dict(cls, block: dict) -> "SplatsConfig":
        """
        Build from the yaml block; rejects unknown keys, unknown/ill-formed losses, and distortion with 3dgs.
        """
        # Unknown top-level keys are almost always typos — refuse rather than silently default
        allowed_keys = {"enabled", *cls.__dataclass_fields__}
        unknown_keys = set(block) - allowed_keys
        if unknown_keys:
            raise ValueError(f"splats: unknown keys {sorted(unknown_keys)}; allowed {sorted(allowed_keys)}")
        fields = {key: value for key, value in block.items() if key != "enabled"}
        cfg = cls(**fields)

        # Primitive and loss entries must be ones the trainer knows, each with a weight
        if cfg.primitive not in PRIMITIVES:
            raise ValueError(f"splats.primitive must be one of {PRIMITIVES}, got '{cfg.primitive}'")
        for name, spec in cfg.losses.items():
            if name not in OPTIONAL_LOSSES:
                raise ValueError(f"splats.losses: unknown loss '{name}'; allowed {sorted(OPTIONAL_LOSSES)}")
            unknown_spec_keys = set(spec) - {"weight", "start"}
            if unknown_spec_keys or "weight" not in spec:
                raise ValueError(f"splats.losses.{name}: expected {{weight[, start]}}, got {sorted(spec)}")

        # The distortion map only exists for 2DGS
        distortion_spec = cfg.losses.get("distortion", {})
        distortion_weight = distortion_spec.get("weight", 0.0)
        if cfg.primitive == "3dgs" and distortion_weight > 0:
            raise ValueError("splats.losses.distortion is 2dgs-only; set its weight to 0 or use primitive: 2dgs")
        return cfg


########################################
# Setup helpers
########################################


def compute_scene_scale(cam_to_world: Tensor) -> float:
    """
    1.1 x the largest camera distance from the camera centroid (gsplat's scene-extent proxy).
    """
    positions = cam_to_world[:, :3, 3]
    centroid = positions.mean(0)
    spread = (positions - centroid).norm(dim=-1).max()
    return float(spread) * 1.1


def init_gaussians_from_points(
    cfg: SplatsConfig, points: np.ndarray, colors: np.ndarray, scene_scale: float, device: str
) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Optimizer]]:
    """
    One Gaussian per seed point (scale from kNN spacing, colour as SH DC) plus one Adam per parameter.

    Port of create_splats_with_optimizers, gsplat @ d2f5c0f examples/simple_trainer.py.
    """
    n_points = len(points)

    # Initial scale: mean distance to the 3 nearest neighbours, stored as log-scale
    neighbour_dists, _ = NearestNeighbors(n_neighbors=4).fit(points).kneighbors(points)
    mean_spacing = np.sqrt((neighbour_dists[:, 1:] ** 2).mean(-1))
    spacing = torch.from_numpy(mean_spacing).float()
    log_scales = torch.log(spacing).unsqueeze(-1).repeat(1, 3)

    # Colour: RGB goes into the degree-0 SH band, higher bands start at zero
    rgb = torch.from_numpy(colors).float() / 255.0
    n_sh_coeffs = (cfg.sh_degree + 1) ** 2
    sh_coeffs = torch.zeros(n_points, n_sh_coeffs, 3)
    sh_coeffs[:, 0, :] = (rgb - 0.5) / SH_DC_NORMALISER

    # Raw parameters: random orientation, logit-opacity so sigmoid gives init_opacity
    initial_opacities = torch.logit(torch.full((n_points,), cfg.init_opacity))
    means = torch.from_numpy(points).float()
    gaussians = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(log_scales),
            "quats": torch.nn.Parameter(torch.rand(n_points, 4)),
            "opacities": torch.nn.Parameter(initial_opacities),
            "sh0": torch.nn.Parameter(sh_coeffs[:, :1, :]),
            "shN": torch.nn.Parameter(sh_coeffs[:, 1:, :]),
        }
    ).to(device)

    # One Adam per parameter so the densification strategy can grow/prune optimizer state per tensor
    learning_rates = {
        "means": cfg.means_lr * scene_scale,
        "scales": cfg.scales_lr,
        "quats": cfg.quats_lr,
        "opacities": cfg.opacities_lr,
        "sh0": cfg.sh0_lr,
        "shN": cfg.shN_lr,
    }
    optimizers = {}
    for name, lr in learning_rates.items():
        param_group = {"params": gaussians[name], "lr": lr, "name": name}
        optimizers[name] = torch.optim.Adam([param_group], eps=1e-15)
    return gaussians, optimizers


def make_strategy(cfg: SplatsConfig) -> MCMCStrategy | DefaultStrategy:
    """
    MCMC for 3dgs (budgeted, no gradient heuristics); Default with the 2D-gradient key for 2dgs.
    """
    if cfg.primitive == "3dgs":
        return MCMCStrategy(cap_max=cfg.cap_max, verbose=True)
    return DefaultStrategy(absgrad=False, grow_grad2d=cfg.grow_grad2d, key_for_gradient="gradient_2dgs", verbose=True)


def make_pose_refiner(
    cfg: SplatsConfig, n_views: int, scene_scale: float, lr_gamma: float, device: str
) -> tuple[CameraOptModule, torch.optim.Optimizer, ExponentialLR]:
    """
    Zero-initialised CameraOptModule with its Adam optimizer and exponential lr decay.
    """
    refiner = CameraOptModule(n_views).to(device)
    refiner.zero_init()
    pose_lr = cfg.pose_lr * scene_scale
    optimizer = torch.optim.Adam(refiner.parameters(), lr=pose_lr, weight_decay=1e-6)
    scheduler = ExponentialLR(optimizer, gamma=lr_gamma)
    return refiner, optimizer, scheduler


def prepare_training_target(image: np.ndarray, depth_target: np.ndarray | None, device: str) -> dict:
    """
    One view's targets as tensors: rgb (1, H, W, 3) in [0, 1]; depth (1, H, W, 1) resized nearest, or None.
    """
    rgb = torch.from_numpy(image).to(device).float()[None] / 255.0
    if depth_target is None:
        return {"rgb": rgb, "depth": None}

    # Depth targets may be at model resolution; nearest resize keeps zeros (no target) as zeros
    height, width = image.shape[:2]
    depth_nchw = torch.from_numpy(depth_target).to(device)[None, None]
    depth_nchw = F.interpolate(depth_nchw, size=(height, width), mode="nearest")
    depth_nhwc = depth_nchw.permute(0, 2, 3, 1)
    return {"rgb": rgb, "depth": depth_nhwc}


########################################
# Training
########################################


def train(
    cfg: SplatsConfig,
    images: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsics: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    out_dir: Path,
    depth_targets: np.ndarray | None = None,
) -> None:
    """
    Train splats and write splats.ply / ckpt.pt / splats.zarr / splats_quality_report.json to out_dir.

    Args:
        images: (n_views, H, W, 3) uint8 frames.
        world_to_cam: (n_views, 4, 4) COLMAP-convention poses; intrinsics: (n_views, 3, 3) at frame resolution.
        points, colors: (P, 3) float32 / uint8 seed points in the same world frame.
        depth_targets: optional (n_views, h, w) float32 depth at any resolution, 0 = no target.
    """
    device = "cuda"
    n_views, height, width = images.shape[:3]
    out_dir = Path(out_dir)

    # Refuse inputs that cannot train: too few seed points, or mismatched per-view arrays
    n_points = len(points)
    if n_points < 100:
        raise ValueError(f"splats: need >= 100 seed points, got {n_points}")
    n_poses = len(world_to_cam)
    n_intrinsics = len(intrinsics)
    n_depth = None if depth_targets is None else len(depth_targets)
    per_view_counts = {n_views, n_poses, n_intrinsics}
    if n_depth is not None:
        per_view_counts.add(n_depth)
    if len(per_view_counts) != 1:
        raise ValueError(
            f"splats: frames mismatch — images {n_views}, world_to_cam {n_poses}, "
            f"intrinsics {n_intrinsics}, depth_targets {n_depth}"
        )

    # Cameras on the GPU; frames stay uint8 on the CPU and move one view at a time
    cam_to_world_np = np.linalg.inv(world_to_cam)
    cam_to_world = torch.from_numpy(cam_to_world_np).float().to(device)
    intrinsics_gpu = torch.from_numpy(intrinsics).float().to(device)
    scene_scale = compute_scene_scale(cam_to_world)

    # Gaussians, densification strategy, and the lr decay on the means (0.01x over the run)
    gaussians, optimizers = init_gaussians_from_points(cfg, points, colors, scene_scale, device)
    strategy = make_strategy(cfg)
    strategy.check_sanity(gaussians, optimizers)
    if isinstance(strategy, MCMCStrategy):
        strategy_state = strategy.initialize_state()
    else:
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    lr_gamma = 0.01 ** (1.0 / cfg.max_steps)
    means_scheduler = ExponentialLR(optimizers["means"], gamma=lr_gamma)
    schedulers = [means_scheduler]

    # Optional joint pose refinement
    pose_refiner, pose_optimizer = None, None
    if cfg.pose_opt:
        pose_refiner, pose_optimizer, pose_scheduler = make_pose_refiner(cfg, n_views, scene_scale, lr_gamma, device)
        schedulers.append(pose_scheduler)

    start_time = time.perf_counter()
    loss_values: dict[str, float] = {}
    use_pre_backward_hook = isinstance(strategy, DefaultStrategy)
    for step in progress(range(cfg.max_steps), desc=f"splats[{cfg.primitive}]"):
        # Pick one random view and its (possibly refined) camera
        view = int(torch.randint(n_views, (1,)))
        view_image = images[view]
        view_depth_target = None if depth_targets is None else depth_targets[view]
        target = prepare_training_target(view_image, view_depth_target, device)
        view_cam_to_world = cam_to_world[view : view + 1]
        view_intrinsics = intrinsics_gpu[view : view + 1]
        if pose_refiner is not None:
            camera_id = torch.tensor([view], device=device)
            view_cam_to_world = pose_refiner(view_cam_to_world, camera_id)

        # Render with the SH bands unlocked so far, over a random background so transparency cannot hide
        sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        absgrad = use_pre_backward_hook and strategy.absgrad
        render, info = render_view(
            cfg.primitive, gaussians, view_cam_to_world, view_intrinsics, width, height, sh_degree, absgrad
        )
        background = torch.rand(1, 3, device=device)
        transparency = 1.0 - render["alpha"]
        render["rgb"] = render["rgb"] + background * transparency

        # Loss + backward; DefaultStrategy needs a hook before backward to retain 2D-means gradients
        if use_pre_backward_hook:
            strategy.step_pre_backward(gaussians, optimizers, strategy_state, step, info)
        loss, loss_values = compute_losses(step, render, target, gaussians, cfg.losses, scene_scale)
        loss.backward()

        # Densify / prune / relocate — the two strategies take different extra arguments
        if isinstance(strategy, MCMCStrategy):
            means_lr_now = means_scheduler.get_last_lr()[0]
            strategy.step_post_backward(gaussians, optimizers, strategy_state, step, info, lr=means_lr_now)
        else:
            strategy.step_post_backward(gaussians, optimizers, strategy_state, step, info, packed=False)

        # Optimizer steps for Gaussians (and poses), then lr decay
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if pose_optimizer is not None:
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()

        if step % cfg.log_every == 0:
            n_gaussians = len(gaussians["means"])
            rounded = {name: round(value, 4) for name, value in loss_values.items()}
            logger.info("splats step %d loss %.4f gaussians %d %s", step, loss.item(), n_gaussians, rounded)

    train_seconds = time.perf_counter() - start_time
    write_splat_outputs(
        cfg, gaussians, pose_refiner, images, cam_to_world, intrinsics_gpu, out_dir, train_seconds, loss_values
    )
```

Replace `collab_splats/splats/__init__.py` with:
```python
"""
Gaussian-splat training on upstream gsplat from an existing pointcloud stage.
"""

# The gsplat commit pinned in pyproject.toml; recorded in every splats.zarr for provenance
GSPLAT_COMMIT = "d2f5c0f"

from .trainer import SplatsConfig, train  # noqa: E402

__all__ = ["GSPLAT_COMMIT", "SplatsConfig", "train"]
```

Stub `collab_splats/splats/outputs.py` (Task 7 replaces it):
```python
"""
Splat-stage artifacts (stub; implemented in the next task).
"""


def write_splat_outputs(*args, **kwargs):
    raise NotImplementedError
```

- [x] **Step 4: Run tests**

Run: `$PY -m pytest tests/splats/test_trainer.py -v` → all PASSED.

- [x] **Step 5: Format and commit**

```bash
$PY -m black collab_splats/splats tests/splats && $PY -m isort collab_splats/splats tests/splats
git commit --only collab_splats/splats/trainer.py collab_splats/splats/__init__.py collab_splats/splats/outputs.py tests/splats/test_trainer.py \
  -m "feat(splats): SplatsConfig, Gaussian init, MCMC/Default strategy, training loop"
```

---

### Task 7: `splats/outputs.py` — ply, ckpt, rendered zarr, quality report; end-to-end test

**Files:** Replace `collab_splats/splats/outputs.py`; create `tests/splats/test_outputs.py`

- [x] **Step 1: Write the failing end-to-end tests**

`tests/splats/test_outputs.py`:
```python
"""
End-to-end: train() on the synthetic scene writes every artifact with the documented shapes/attrs.
"""

import json

import numpy as np
import pytest
import torch
import zarr

from collab_splats.splats import GSPLAT_COMMIT
from collab_splats.splats.trainer import SplatsConfig, train
from tests.splats.synthetic import make_scene

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_train_writes_all_outputs(tmp_path, primitive):
    images, world_to_cam, intrinsics, points, colors, depths = make_scene()
    losses = {"depth": {"weight": 0.1}, "normal_consistency": {"weight": 0.05, "start": 10}}
    if primitive == "2dgs":
        losses["distortion"] = {"weight": 0.01, "start": 10}
    cfg = SplatsConfig(primitive=primitive, max_steps=50, cap_max=500, losses=losses)

    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, depth_targets=depths)

    # Files
    for name in ("splats.ply", "ckpt.pt", "splats.zarr", "splats_quality_report.json"):
        assert (tmp_path / name).exists(), name

    # Report
    report = json.loads((tmp_path / "splats_quality_report.json").read_text())
    summary = report["summary"]
    first_frame = report["per_frame"][0]
    assert summary["n_gaussians"] > 0 and np.isfinite(summary["psnr"]) and summary["config"]["max_steps"] == 50
    assert {"depth", "normal_consistency"} <= set(summary["final_losses"])
    assert len(report["per_frame"]) == 8 and {"image_id", "psnr", "ssim"} <= set(first_frame)

    # Rendered zarr
    store = zarr.open_group(tmp_path / "splats.zarr", mode="r")
    assert store["rgb"].shape == (8, 64, 64, 3) and store["rgb"].dtype == np.uint8
    assert store["depth"].shape == (8, 64, 64) and store["normal"].shape == (8, 64, 64, 3)
    assert store["alpha"].shape == (8, 64, 64) and store["c2w"].shape == (8, 4, 4) and store["K"].shape == (8, 3, 3)
    assert store.attrs["primitive"] == primitive and store.attrs["gsplat_commit"] == GSPLAT_COMMIT
    assert list(store.attrs["image_ids"]) == list(range(8)) and store.attrs["pose_opt"] is False

    # Checkpoint
    ckpt = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    assert set(ckpt) == {"splats", "pose_adjust", "config"}
    assert ckpt["pose_adjust"] is None and "means" in ckpt["splats"] and ckpt["config"]["primitive"] == primitive


@cuda
def test_pose_opt_refines_poses_slightly(tmp_path):
    images, world_to_cam, intrinsics, points, colors, _ = make_scene()
    cfg = SplatsConfig(max_steps=30, pose_opt=True, cap_max=500, losses={})
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path)

    store = zarr.open_group(tmp_path / "splats.zarr", mode="r")
    cam_to_world_in = np.linalg.inv(world_to_cam)
    cam_to_world_out = store["c2w"][:]
    assert not np.allclose(cam_to_world_out, cam_to_world_in, atol=1e-7)  # moved
    assert np.allclose(cam_to_world_out, cam_to_world_in, atol=1e-2)  # but only a little
    ckpt = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["pose_adjust"] is not None


@cuda
def test_train_rejects_bad_inputs(tmp_path):
    images, world_to_cam, intrinsics, points, colors, _ = make_scene()
    cfg = SplatsConfig(max_steps=1)
    few_points, few_colors = points[:50], colors[:50]
    with pytest.raises(ValueError, match="seed points"):
        train(cfg, images, world_to_cam, intrinsics, few_points, few_colors, tmp_path)
    fewer_images = images[:4]
    with pytest.raises(ValueError, match="frames mismatch"):
        train(cfg, fewer_images, world_to_cam, intrinsics, points, colors, tmp_path)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/splats/test_outputs.py -v` → `NotImplementedError` from the stub (the bad-inputs test passes already — fine).

- [x] **Step 3: Write the module**

```python
"""
Splat-stage artifacts: splats.ply, ckpt.pt, splats.zarr (every view re-rendered) and the quality report.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from gsplat.exporter import export_splats
from gsplat.losses import ssim_loss
from torch import Tensor

from collab_splats.splats import GSPLAT_COMMIT
from collab_splats.splats.cameras import CameraOptModule
from collab_splats.splats.rendering import render_view
from collab_splats.utils.progress import progress

logger = logging.getLogger(__name__)


def render_all_views(
    cfg,
    gaussians: torch.nn.ParameterDict,
    pose_refiner: CameraOptModule | None,
    images: np.ndarray,
    cam_to_world: Tensor,
    intrinsics: Tensor,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """
    Re-render every training view at full SH over black. Returns (arrays for the zarr, per-frame psnr/ssim).
    """
    n_views, height, width = images.shape[:3]
    device = cam_to_world.device
    arrays = {
        "rgb": np.zeros((n_views, height, width, 3), np.uint8),
        "depth": np.zeros((n_views, height, width), np.float32),
        "normal": np.zeros((n_views, height, width, 3), np.float32),
        "alpha": np.zeros((n_views, height, width), np.float32),
        "c2w": cam_to_world.cpu().numpy().copy(),
        "K": intrinsics.cpu().numpy(),
    }
    per_frame = []

    with torch.no_grad():
        for view in progress(range(n_views), desc="splats render"):
            # Refined pose if poses were optimised; the zarr stores what was actually rendered
            view_cam_to_world = cam_to_world[view : view + 1]
            view_intrinsics = intrinsics[view : view + 1]
            if pose_refiner is not None:
                camera_id = torch.tensor([view], device=device)
                view_cam_to_world = pose_refiner(view_cam_to_world, camera_id)
                arrays["c2w"][view] = view_cam_to_world[0].cpu().numpy()

            # Render and score against the training frame
            render, _ = render_view(
                cfg.primitive, gaussians, view_cam_to_world, view_intrinsics, width, height, cfg.sh_degree, absgrad=False
            )
            rendered_rgb = render["rgb"].clamp(0, 1)
            target_rgb = torch.from_numpy(images[view]).to(device).float()[None] / 255.0
            rendered_nchw = rendered_rgb.permute(0, 3, 1, 2)
            target_nchw = target_rgb.permute(0, 3, 1, 2)
            mse = F.mse_loss(rendered_rgb, target_rgb).item()
            ssim_distance = ssim_loss(rendered_nchw, target_nchw).item()
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            per_frame.append({"image_id": view, "psnr": psnr, "ssim": 1.0 - ssim_distance})

            # Stash the render
            rgb_uint8 = (rendered_rgb[0] * 255).round().byte()
            arrays["rgb"][view] = rgb_uint8.cpu().numpy()
            arrays["depth"][view] = render["depth"][0, ..., 0].cpu().numpy()
            arrays["normal"][view] = render["normal"][0].cpu().numpy()
            arrays["alpha"][view] = render["alpha"][0, ..., 0].cpu().numpy()

    return arrays, per_frame


def write_splat_outputs(
    cfg,
    gaussians: torch.nn.ParameterDict,
    pose_refiner: CameraOptModule | None,
    images: np.ndarray,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    out_dir: Path,
    train_seconds: float,
    final_losses: dict[str, float],
) -> None:
    """
    Write splats.ply, ckpt.pt, splats.zarr and splats_quality_report.json to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(cfg)

    # splats.ply: standard 3DGS ply (raw log-scales / logit-opacities, as every viewer expects)
    ply_path = str(out_dir / "splats.ply")
    export_splats(
        means=gaussians["means"],
        scales=gaussians["scales"],
        quats=gaussians["quats"],
        opacities=gaussians["opacities"],
        sh0=gaussians["sh0"],
        shN=gaussians["shN"],
        format="ply",
        save_to=ply_path,
    )

    # ckpt.pt: raw parameters + pose deltas + config — everything needed to re-render or continue
    splats_cpu = {name: param.detach().cpu() for name, param in gaussians.items()}
    pose_adjust = None if pose_refiner is None else pose_refiner.state_dict()
    checkpoint = {"splats": splats_cpu, "pose_adjust": pose_adjust, "config": config_dict}
    torch.save(checkpoint, out_dir / "ckpt.pt")

    # splats.zarr: one chunk per view so downstream stages can read frames independently
    arrays, per_frame = render_all_views(cfg, gaussians, pose_refiner, images, cam_to_world, intrinsics)
    store = zarr.open_group(out_dir / "splats.zarr", mode="w")
    for name, array in arrays.items():
        per_view_chunks = (1, *array.shape[1:])
        store.create_array(name, data=array, chunks=per_view_chunks)
    image_ids = list(range(len(images)))
    store.attrs.update(
        image_ids=image_ids,
        primitive=cfg.primitive,
        pose_opt=cfg.pose_opt,
        gsplat_commit=GSPLAT_COMMIT,
        config=config_dict,
    )

    # splats_quality_report.json: same shape as the other stage reports (summary + per_frame)
    mean_psnr = float(np.mean([frame["psnr"] for frame in per_frame]))
    mean_ssim = float(np.mean([frame["ssim"] for frame in per_frame]))
    n_gaussians = int(len(gaussians["means"]))
    report = {
        "summary": {
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "n_gaussians": n_gaussians,
            "seconds": round(train_seconds, 1),
            "final_losses": final_losses,
            "config": config_dict,
        },
        "per_frame": per_frame,
    }
    (out_dir / "splats_quality_report.json").write_text(json.dumps(report, indent=2))
    logger.info(
        "splats: %d gaussians, psnr %.2f, ssim %.3f, %.0fs -> %s", n_gaussians, mean_psnr, mean_ssim, train_seconds, out_dir
    )
```

- [x] **Step 4: Run the whole splats suite**

Run: `$PY -m pytest tests/splats/ -v` → all PASSED (≈1–2 min GPU).

- [x] **Step 5: Format and commit**

```bash
$PY -m black collab_splats/splats tests/splats && $PY -m isort collab_splats/splats tests/splats
git commit --only collab_splats/splats/outputs.py tests/splats/test_outputs.py \
  -m "feat(splats): write_splat_outputs — ply, ckpt, rendered zarr, quality report"
```

---

### Task 8: `Reconstructor.splats()` leaf stage + config

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` — `_STAGE_ORDER`/`_STAGE_DEPS` (55–78), `_stage_output_exists` (~1283), `run_pipeline` (~1320), new method after `verify()`
- Modify: `configs/base.yaml` — add `splats:` block after `mesh:`
- Test: `tests/wrapper/test_splats_stage.py`

- [x] **Step 1: Write the failing tests**

```python
"""
Splats-stage wiring: leaf registration, base.yaml default, and the arrays handed to train().
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.wrapper.reconstructor import _STAGE_DEPS, _STAGE_ORDER, LEAF_STAGES, Reconstructor

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_splats_is_a_leaf_stage():
    assert "splats" in _STAGE_ORDER
    assert _STAGE_ORDER.index("splats") < _STAGE_ORDER.index("mesh")
    assert _STAGE_DEPS["splats"] == ["pointcloud"]
    assert "splats" in LEAF_STAGES


def test_base_yaml_defaults():
    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())["splats"]
    assert cfg["enabled"] is False and cfg["primitive"] == "3dgs" and cfg["cap_max"] == 1_000_000
    assert set(cfg["losses"]) == {"depth", "normal_consistency", "opacity_reg", "scale_reg"}


def _stub_reconstructor(tmp_path, n_views=3, height=8, width=8):
    """
    Reconstructor with config + frames.zarr + a fake PointcloudResult; image_paths reversed to prove index lookup.
    """
    recon = Reconstructor.__new__(Reconstructor)
    recon.config = {
        "output_path": str(tmp_path),
        "pointcloud": {"backend": "vggtx"},
        "mesh": {"conf_percentile": 20},
        "splats": {"enabled": True, "max_steps": 1, "losses": {"depth": {"weight": 0.1}}},
    }
    recon._stage_output_exists = lambda stage: False

    frames = np.stack([np.full((height, width, 3), view * 10, np.uint8) for view in range(n_views)])
    records = [{"frame_idx": view} for view in range(n_views)]
    FrameStore.create(recon.frames_zarr, frames, records, provenance={"video_path": "v"})
    image_paths = [Path(f"frame_{view:06d}.jpg") for view in reversed(range(n_views))]
    recon._resolve_result = lambda: SimpleNamespace(
        image_paths=image_paths,
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n_views, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n_views, 1, 1)),
        points=np.zeros((200, 3), np.float32),
        colors=np.zeros((200, 3), np.uint8),
    )
    return recon


def test_splats_stage_assembles_arrays_in_image_path_order(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    feedforward = SimpleNamespace(depth=np.ones((3, 4, 4), np.float32), confidence=torch.ones(3, 4, 4))
    with patch("collab_splats.splats.trainer.train") as train, patch(
        "collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=feedforward
    ):
        out = recon.splats()

    assert out == recon.backend_dir / "splats" / "splats.zarr"
    cfg, images, world_to_cam, intrinsics, points, colors, out_dir = train.call_args.args
    depth_targets = train.call_args.kwargs["depth_targets"]
    assert cfg.max_steps == 1
    assert images[0, 0, 0, 0] == 20 and images[2, 0, 0, 0] == 0  # follows image_paths (reversed), not store order
    assert world_to_cam.shape == (3, 4, 4) and intrinsics.shape == (3, 3, 3) and points.shape == (200, 3)
    assert out_dir == recon.backend_dir / "splats"
    assert depth_targets.shape == (3, 4, 4)


def test_splats_stage_skips_depth_targets_when_depth_loss_off(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.config["splats"]["losses"] = {}
    with patch("collab_splats.splats.trainer.train") as train:
        recon.splats()
    assert train.call_args.kwargs["depth_targets"] is None


def test_splats_stage_skips_when_output_exists(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon._stage_output_exists = lambda stage: stage == "splats"
    with patch("collab_splats.splats.trainer.train") as train:
        recon.splats()
    train.assert_not_called()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/wrapper/test_splats_stage.py -v` → FAIL (`'splats' in _STAGE_ORDER`, `KeyError: 'splats'`).

- [x] **Step 3: Register the stage**

`_STAGE_ORDER` → `["preproc", "pointcloud", "refine", "semantics", "splats", "mesh", "localize", "verify", "reconstruction_quality_report"]`.

`_STAGE_DEPS` — add in the neighbours' comment style:
```python
    # splats: trains on COLMAP poses/points + frames.zarr; leaf — nothing reads it yet
    "splats": ["pointcloud"],
```

`_stage_output_exists` — before the `mesh` check:
```python
        if stage == "splats":
            return (self.backend_dir / "splats" / "splats.zarr").exists()
```

`run_pipeline` — after the `semantics` append:
```python
            if self.config["splats"]["enabled"]:
                stages.append("splats")
```
dispatch loop, after the `semantics` branch:
```python
            elif stage == "splats":
                self.splats(overwrite=overwrite)
```
and add `"splats"` to the `stages:` docstring list.

New method after `verify()`:

```python
    def splats(self, overwrite: bool = False) -> Path:
        """
        Train Gaussian splats from the pointcloud stage. Returns path to splats/splats.zarr.
        """
        out_dir = self.backend_dir / "splats"
        splats_zarr = out_dir / "splats.zarr"
        if not overwrite and self._stage_output_exists("splats"):
            logger.info("Splats exist at %s, skipping", out_dir)
            return splats_zarr

        result = self._resolve_result()
        if result is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        from collab_splats.pointcloud.feedforward.base import FeedforwardResult
        from collab_splats.pointcloud.utils import confidence_mask
        from collab_splats.preproc.frame_store import FrameStore
        from collab_splats.splats.trainer import SplatsConfig, train

        cfg = SplatsConfig.from_dict(self.config["splats"])

        # Frames in COLMAP image order, looked up in frames.zarr by the frame index in each name
        store = FrameStore.open(self.frames_zarr)
        frame_indices = [FrameStore.frame_idx_from_path(path) for path in result.image_paths]
        images = np.stack([store.image_by_frame_idx(frame_idx) for frame_idx in frame_indices])
        n_images = len(images)

        # Depth targets: feedforward depth masked like the mesh stage masks it (0 = no target)
        depth_targets = None
        depth_spec = cfg.losses.get("depth", {})
        depth_weight = depth_spec.get("weight", 0.0)
        if depth_weight > 0:
            feedforward_zarr = self.backend_dir / "feedforward.zarr"
            feedforward = FeedforwardResult.load_zarr(feedforward_zarr, load_images=False, load_world_points=False)
            n_depth = None if feedforward.depth is None else len(feedforward.depth)
            if n_depth != n_images:
                raise ValueError(f"splats depth loss needs feedforward depth for all {n_images} frames; got {n_depth}")
            depth_targets = np.ascontiguousarray(feedforward.depth, dtype=np.float32)
            conf_percentile = self.config["mesh"]["conf_percentile"]
            if conf_percentile is not None and feedforward.confidence is not None:
                confidence = feedforward.confidence.cpu().numpy()
                keep = confidence_mask(confidence, conf_percentile)
                depth_targets = np.where(keep, depth_targets, 0.0).astype(np.float32)

        train(
            cfg, images, result.extrinsics, result.intrinsics, result.points, result.colors, out_dir,
            depth_targets=depth_targets,
        )
        logger.info("Splats saved to %s", out_dir)
        return splats_zarr
```
(`np` is imported at the top of `reconstructor.py` — verify with `grep -n '^import numpy' collab_splats/wrapper/reconstructor.py`.)

- [x] **Step 4: Add the config block**

`configs/base.yaml`, after the `mesh:` block (every key mirrors a `SplatsConfig` field; values are the dataclass defaults, listed so the knobs are visible):
```yaml
# Gaussian splats trained from the pointcloud stage (leaf stage `splats`, upstream gsplat).
# Photometric (0.8 L1 + 0.2 SSIM) is always on; each loss below is active iff weight > 0 and
# step >= start. Depth targets = feedforward depth masked by mesh.conf_percentile.
# 3dgs densifies with MCMC (fixed budget cap_max; needs opacity_reg/scale_reg), 2dgs with
# gsplat's DefaultStrategy (grow_grad2d) and gets the distortion loss instead.
splats:
  enabled: false
  primitive: 3dgs            # 3dgs (fast kernel, antialiased) | 2dgs (surface-aligned)
  max_steps: 30000
  pose_opt: false            # refine camera poses jointly (CameraOptModule)
  sh_degree: 3
  sh_degree_interval: 1000
  init_opacity: 0.1
  means_lr: 1.6e-4           # x scene_scale, decays 0.01x over the run
  scales_lr: 5.0e-3
  quats_lr: 1.0e-3
  opacities_lr: 5.0e-2
  sh0_lr: 2.5e-3
  shN_lr: 1.25e-4
  pose_lr: 1.0e-5            # x scene_scale
  cap_max: 1000000           # 3dgs only: Gaussian budget
  grow_grad2d: 8.0e-4        # 2dgs only
  log_every: 500
  losses:
    depth: {weight: 0.01}
    normal_consistency: {weight: 0.05, start: 7000}
    opacity_reg: {weight: 0.01}
    scale_reg: {weight: 0.01}
    # 2dgs: replace the two regularisers with  distortion: {weight: 100.0, start: 3000}
```

- [x] **Step 5: Run tests**

Run: `$PY -m pytest tests/wrapper/test_splats_stage.py tests/wrapper/test_verify_stage.py tests/wrapper/test_reconstructor.py -v` → PASS. If `test_reconstructor.py` enumerates `_STAGE_ORDER` literally, add `"splats"` between `semantics` and `mesh` there.

- [x] **Step 6: Commit**

```bash
git commit --only collab_splats/wrapper/reconstructor.py configs/base.yaml tests/wrapper/test_splats_stage.py tests/wrapper/test_reconstructor.py \
  -m "feat(wrapper): splats leaf stage — Reconstructor.splats() + splats: config block"
```

---

### Task 9: Docs, spec deviations, gates

**Files:** `configs/README.md` (key table ~313+, layout table ~425, section ~461), `CLAUDE.md`, `docs/superpowers/specs/2026-08-22-splats-module-design.md`

- [x] **Step 1: configs/README.md**

Key table: one row per `splats.*` key from the yaml block above, in the neighbours' column format, plus `splats.losses.<name>.{weight,start}`.
Layout table: rows for `<backend>/splats/splats.ply`, `ckpt.pt`, `splats.zarr`, `splats_quality_report.json`.
Replace the "#### `ns-train --data` does not work on a published scene, by design" section with:

```markdown
#### Splats train from the published COLMAP + frames.zarr

`--stages splats` pulls a processed scene and trains directly on `colmap/` poses + points and
`frames.zarr` — no image directory, no transforms.json round-trip. Every `splats/` artifact is in
the COLMAP world frame; nothing is normalised. `mesh` still fuses `feedforward.zarr`; fusing the
splat renders (`mesh.source: splats`) is a follow-on.
```
Add `splats` to the "Re-running one stage" leaf list.

- [x] **Step 2: CLAUDE.md**

Arch tree, under `collab_splats/`:
```
  splats/                  # Gaussian-splat training on upstream gsplat: cameras, losses, rendering, trainer, outputs
```
Add a "Recently completed (2026-08-22): **splats-module**" paragraph (≤6 lines): nerfstudio/gsplat-rade retired; `splats` leaf stage; MCMC for 3dgs / Default for 2dgs; depth targets from feedforward depth (spec deviation); `mesh.source: splats` deferred; 3DGS `extra_signals` normals unmeasured (owed); PAGaS seam = `render_view` + one `OPTIONAL_LOSSES` entry.

- [x] **Step 3: Spec deviations**

In the spec: (a) replace the depth-target sentence(s) (grep `points3D` / `tracks`) with Deviation 1; (b) mark the `mesh.source` subsection "Deferred to a follow-on plan (2026-08-22)"; (c) replace the strategy paragraph with Deviation 3; (d) update the layout to the 5 files listed here and the config block to the full key list.

- [x] **Step 4: Gates**

```bash
$PY -m collab_splats.dashboard --smoke                       # must print SMOKE PASS
$PY -m pytest tests/ -x -q -p no:randomly                      # full suite; no tmux eval concurrently
grep -rn 'gsplat-rade\|nerfstudio\|Splatter' --include='*.py' --include='*.md' --include='*.yaml' --include='*.toml' --include='*.rst' . \
  | grep -v 'docs/superpowers\|graphify-out\|docs/_build\|update_kernelspecs\|evals/datasets.py\|known-test-failures\|display_name\|/opt/conda/envs'
graphify update .
```
Expected: `SMOKE PASS`; suite green except `docs/known-test-failures.md` entries; the grep prints nothing.

- [x] **Step 5: Commit, then merge back**

```bash
git add -f docs/superpowers/specs/2026-08-22-splats-module-design.md
git commit --only configs/README.md CLAUDE.md docs/superpowers/specs/2026-08-22-splats-module-design.md graphify-out \
  -m "docs(splats): config keys, processed-scene layout, spec deviations"
```
Merging `feat/splats-module` into `refactor/cu121-uv-migration` is a human step (the main checkout carries a concurrent session's uncommitted edits to `pyproject.toml` / `uv.lock` / `configs/base.yaml`, which will conflict with Tasks 1 and 8).

---

## Owed after this plan (not in scope)

- `mesh.source: splats` (TSDF over `splats.zarr`, alpha as confidence) — separate plan.
- New tutorial `docs/source/tutorials/03_splats/train_splats.ipynb` on `data/tutorial/` + a semantic mesh-query tutorial replacing the deleted `06_mesh` one.
- Measured report on `data/tutorial/`: 3DGS-MCMC vs 2DGS-Default (psnr/ssim/n_gaussians/time), and whether the 3DGS `extra_signals` normals are usable.
- Docker image rebuild against the new lock.
- `evals/scripts/eval_splats.py`, PAGaS (gsplat `bd64a47` + CUDA patch, seam = `render_view`), feature distillation, dense mono priors.
