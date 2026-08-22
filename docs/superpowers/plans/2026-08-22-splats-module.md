# Splats Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `collab_splats/nerfstudio/` + `wrapper/splatter.py` (gsplat-rade fork + nerfstudio fork) with a ~300-LOC `collab_splats/splats/` package that trains 3DGS/2DGS splats on upstream gsplat from an existing pointcloud stage, and lets `mesh` fuse the rendered depth instead of feedforward depth.

**Architecture:** Three files — `trainer.py` (config + one loop + save), `losses.py` (scheduled weighted sum over `gsplat.losses`), `cameras.py` (vendored `CameraOptModule`). `train()` takes plain arrays (images/poses/K/points/colors); `Reconstructor.splats()` does the ~20-line assembly from `PointcloudResult` + `FrameStore` + `feedforward.zarr`. Outputs under `<backend>/splats/`: `splats.ply`, `ckpt.pt`, `splats.zarr` (rendered rgb/depth/normal/alpha + poses), `splats_quality_report.json`. New leaf stage `splats` (`_STAGE_DEPS == ["pointcloud"]`); `mesh.source: splats` reads `splats.zarr` through the existing `get_mesh_creator` path.

**Tech Stack:** gsplat upstream pinned at commit `90d7b4b` (git source, `no-build-isolation`), torch, zarr v3, pycolmap (already present), pytest. Python: `/opt/venv/reconstruction/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-22-splats-module-design.md`

---

## Binding conventions (read before any task)

- Run everything with `/opt/venv/reconstruction/bin/python` (alias below as `$PY`). `export PY=/opt/venv/reconstruction/bin/python`.
- Commit with `git commit --only <files>` (shared index race, see memory). Check `ls .git/sequencer 2>/dev/null` is empty before committing. Plans/specs need `git add -f docs/superpowers/...`.
- Never run repo-wide `black .`. Format only the files you touched: `$PY -m black <file> && $PY -m isort <file>`. Black line length 120, isort wraps at 88 — write long imports parenthesized.
- Code style: imports at top (heavy deps inline only inside `Reconstructor` stage methods, matching `mesh()`/`verify()`), `logging` not print, block comments, `########` dividers, one-line docstrings, flat test functions.
- CUDA tests: mark with `pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")`. Don't run CUDA tests while another heavy job is in tmux (46.6 GB cgroup cap).
- Vendored code cites repo + commit + file + lines at the site (feedback memory `attribute-ported-code`).
- **Working tree is shared with a concurrent session.** At plan-writing time `pyproject.toml`, `uv.lock`, `configs/base.yaml`, `collab_splats/remote/rerun.py`, `docs/examples/run_pipeline_remote.py`, `tests/examples/test_run_pipeline_remote.py`, `data/tutorial/README.md` already had foreign unstaged edits. Before Task 1 run `git status --short`; if those are still dirty, `git diff pyproject.toml configs/base.yaml` and keep the foreign hunks intact — commit only your own hunks (`git add -p` then `git commit --only` is NOT enough; use `git add -p <file>` + `git commit` without `--only` for those specific files, or wait for the other session to land).

### Spec deviation recorded here (update spec in Task 9)

The spec says depth-loss targets come from COLMAP `points3D` tracks. **Feedforward reconstructions carry no `Point2D` tracks** — `build_pycolmap_reconstruction` (`collab_splats/pointcloud/feedforward/base.py:747`) adds points with empty tracks, so that path would be inert for the primary use case. Depth targets instead come from `feedforward.zarr` `depth` (model-res, dense) masked by `confidence_mask(confidence, mesh.conf_percentile)` — the same depth + mask the `mesh` stage already trusts — resized nearest to frame resolution inside the trainer. `train()` takes them as an optional `(N, h, w)` float32 array with `0 = no target`, so the trainer stays decoupled from zarr and the synthetic test can pass analytic depth.

---

## File structure

**Create**
- `collab_splats/splats/__init__.py` — exports `SplatsConfig`, `train`.
- `collab_splats/splats/cameras.py` — vendored `CameraOptModule`, `rotation_6d_to_matrix`.
- `collab_splats/splats/losses.py` — `compute_losses(step, out, batch, losses, scene_scale)`.
- `collab_splats/splats/trainer.py` — `SplatsConfig`, `_create_splats_with_optimizers`, `_gaussian_normals`, `_render`, `train`, `_save`.
- `tests/splats/__init__.py`, `tests/splats/test_cameras.py`, `tests/splats/test_losses.py`, `tests/splats/test_trainer.py`
- `tests/wrapper/test_splats_stage.py`

**Modify**
- `pyproject.toml` — gsplat source → upstream `90d7b4b`; drop nerfstudio dep/source/entry points; description.
- `setup.sh`, `Dockerfile`, `README.md` — gsplat-rade/nerfstudio wording.
- `tests/test_cu121_migration.py` — module list + gsplat assertions.
- `collab_splats/__init__.py`, `collab_splats/wrapper/__init__.py`, `collab_splats/wrapper/config.py` — strip Splatter exports.
- `collab_splats/wrapper/reconstructor.py` — drop `nerfstudio` method; add `splats` stage; `mesh.source`.
- `configs/base.yaml` — drop `nerfstudio:`; add `splats:` + `mesh.source`.
- `configs/README.md`, `CLAUDE.md`, `collab_splats/mesh/tsdf.py` comments, `docs/source/conf.py`, `docs/source/getting_started.md`.

**Delete**
- `collab_splats/nerfstudio/`, `collab_splats/wrapper/splatter.py`, `tests/nerfstudio_methods/`, `tests/test_models.py`, `tests/wrapper/test_splatter_mesh.py`, `tests/wrapper/test_splatter_query.py`.

---

### Task 1: Swap gsplat to upstream, drop nerfstudio dependency

**Files:**
- Modify: `pyproject.toml` (lines 4, 74–76, 93–95, 179–181, 210–211, 258, 262)
- Modify: `setup.sh:12,55,60,63`, `Dockerfile:2,4,13,19,50`, `README.md:30,45,52`
- Modify: `tests/test_cu121_migration.py:122-133,174-237`

- [ ] **Step 1: Write the failing dependency test**

Replace `test_gsplat_rade_fork` (line ~174) and `test_gsplat_not_overwritten` (line ~179) in `tests/test_cu121_migration.py` with:

```python
def test_gsplat_upstream_pinned():
    """gsplat is upstream nerfstudio-project/gsplat at >= 90d7b4b: has gsplat.losses + 2DGS + extra_signals."""
    import inspect

    import gsplat
    from gsplat.losses import depth_l1_loss, normal_cosine_loss, ssim_loss  # noqa: F401

    assert hasattr(gsplat, "rasterization_2dgs")
    assert not hasattr(gsplat, "rasterization_2dgs_inria_wrapper"), "gsplat-rade fork is still installed"
    assert "extra_signals" in inspect.signature(gsplat.rasterization).parameters
```

Delete `test_nerfstudio_installed_local` (lines ~212–237) entirely.

In the module list (lines ~122–133) replace every `"collab_splats.nerfstudio.*"` and `"collab_splats.wrapper.splatter"` entry with:

```python
        "collab_splats.splats",
        "collab_splats.splats.trainer",
        "collab_splats.splats.losses",
        "collab_splats.splats.cameras",
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_cu121_migration.py::test_gsplat_upstream_pinned -v`
Expected: FAIL — `ImportError: cannot import name 'depth_l1_loss' from 'gsplat.losses'` (or `No module named 'gsplat.losses'`).

- [ ] **Step 3: Edit pyproject.toml**

Line 258, replace:
```toml
gsplat = { git = "https://github.com/brian-xu/gsplat-rade.git" }
```
with:
```toml
# Upstream gsplat, main @ 2026-08-20 (reports version 1.6.0 but no tag exists; v1.5.3 lacks
# gsplat.losses + the fast 3DGS kernel). Bump the rev deliberately — extra_signals/2DGS APIs move.
gsplat = { git = "https://github.com/nerfstudio-project/gsplat.git", rev = "90d7b4b" }
```

Line 262: delete the `nerfstudio = { git = ... }` source line.
Lines 93–95: delete the nerfstudio comment + `"nerfstudio",` dependency entry.
Lines 74–76: reword comment to `# gsplat: upstream, built from source against the cu121 toolchain (see [tool.uv.sources])`.
Lines 179–181: delete the `[project.entry-points."nerfstudio.method_configs"]` table.
Lines 210–211: delete the pytest comment referencing `tests/nerfstudio_methods`.
Line 4: description → `"Feedforward reconstruction, Gaussian-splat training, meshing and localization pipeline"`.
Keep line 247–248 `no-build-isolation-package = ["bae", "gsplat"]`.

- [ ] **Step 4: Reword shell/docker/readme references**

`setup.sh` lines 12, 55, 60, 63: replace `gsplat-rade` → `gsplat (upstream)`; keep the import smoke (`python -c "import gsplat"`), drop any `rasterization_2dgs_inria_wrapper` mention if present.
`Dockerfile` lines 2, 4, 13, 19, 50: replace `gsplat-rade`/`nerfstudio` wording with `gsplat`; delete any `ns-*` install line.
`README.md` lines 30, 45, 52: `gsplat-rade` → `gsplat` and drop the nerfstudio install sentence.

- [ ] **Step 5: Re-lock and sync**

Run: `cd /workspace/collab-splats && uv lock && uv sync`
Expected: lock resolves; gsplat builds from source (several minutes, needs CUDA toolchain). `uv.lock` must no longer contain `gsplat-rade` or `nerfstudio`: `grep -c 'gsplat-rade\|name = "nerfstudio"' uv.lock` → `0`.

- [ ] **Step 6: Run test to verify it passes**

Run: `$PY -m pytest tests/test_cu121_migration.py -v -k "gsplat or import"`
Expected: `test_gsplat_upstream_pinned PASSED`. The module-import test will FAIL on `collab_splats.splats` until Task 3 — that is expected; note it and move on.

- [ ] **Step 7: Commit**

```bash
git commit --only pyproject.toml uv.lock setup.sh Dockerfile README.md tests/test_cu121_migration.py \
  -m "build(deps): pin upstream gsplat 90d7b4b, drop gsplat-rade fork and nerfstudio"
```

---

### Task 2: Retire the nerfstudio package and Splatter

**Files:**
- Delete: `collab_splats/nerfstudio/`, `collab_splats/wrapper/splatter.py`, `tests/nerfstudio_methods/`, `tests/test_models.py`, `tests/wrapper/test_splatter_mesh.py`, `tests/wrapper/test_splatter_query.py`
- Modify: `collab_splats/__init__.py`, `collab_splats/wrapper/__init__.py:10-16`, `collab_splats/wrapper/config.py:1`, `collab_splats/wrapper/reconstructor.py:54,715-716,887-…`, `configs/base.yaml:111-114`, `configs/README.md:324`, `CLAUDE.md:91-92`, `collab_splats/mesh/tsdf.py:22,119`, `docs/source/conf.py:40`, `docs/source/getting_started.md:13`

- [ ] **Step 1: Write the failing test**

Append to `tests/wrapper/test_reconstructor.py`:

```python
def test_nerfstudio_method_rejected():
    """pointcloud.method=nerfstudio is gone — Reconstructor refuses it at validation."""
    from collab_splats.wrapper.reconstructor import _VALID_METHODS

    assert _VALID_METHODS == {"feedforward", "sfm"}
    assert not hasattr(Reconstructor, "_run_nerfstudio")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/wrapper/test_reconstructor.py::test_nerfstudio_method_rejected -v`
Expected: FAIL — `assert {'feedforward', 'sfm', 'nerfstudio'} == {'feedforward', 'sfm'}`.

- [ ] **Step 3: Delete files**

```bash
git rm -r collab_splats/nerfstudio tests/nerfstudio_methods
git rm collab_splats/wrapper/splatter.py tests/test_models.py \
  tests/wrapper/test_splatter_mesh.py tests/wrapper/test_splatter_query.py
```

- [ ] **Step 4: Strip exports and the nerfstudio method**

`collab_splats/__init__.py` — replace whole file with:

```python
"""collab-splats: feedforward reconstruction, splat training, meshing and localization."""

__version__ = "0.0.1"
```

`collab_splats/wrapper/__init__.py` lines 10–16: delete the guarded `from .splatter import Splatter, SplatterConfig` block and remove the two names from `__all__`.

`collab_splats/wrapper/config.py:1`: docstring → `"""Configuration loading utilities for Reconstructor workflows."""`.

`collab_splats/wrapper/reconstructor.py`:
- line 54: `_VALID_METHODS = {"feedforward", "sfm"}`
- lines 715–716: delete the `if method == "nerfstudio": result = self._run_nerfstudio()` branch (keep the `else`/raise that follows intact — it now catches any non-feedforward/sfm).
- delete the whole `_run_nerfstudio` method starting at line ~887 (through the end of its body, before the next `def`).
- grep the file for any remaining `nerfstudio` string and remove it: `grep -n nerfstudio collab_splats/wrapper/reconstructor.py` → must print nothing.

`configs/base.yaml` lines 111–114: delete the comment + `nerfstudio:` block.
`configs/README.md:324`: the `pointcloud.method` row → allowed values `feedforward | sfm`.
`CLAUDE.md` lines 91–92: delete the `nerfstudio/` tree line; `wrapper/` line → `# stage orchestration: Reconstructor (config-driven pipeline), batch drivers`.
`collab_splats/mesh/tsdf.py:22,119`: rewrite comments without the nerfstudio mention (e.g. `Accepts rendered depth + RGB frames as numpy arrays.`).
`docs/source/conf.py:40`: drop `nerfstudio` from the autodoc mock list.
`docs/source/getting_started.md:13`: drop the nerfstudio sentence.

- [ ] **Step 5: Verify no dangling references**

Run: `grep -rn 'splatter\|nerfstudio' --include='*.py' --include='*.yaml' --include='*.toml' collab_splats configs tests pyproject.toml`
Expected: no output. (`scripts/update_kernelspecs.py` and `evals/datasets.py:124` only name the conda env — leave them.)

- [ ] **Step 6: Run tests**

Run: `$PY -m pytest tests/wrapper/test_reconstructor.py tests/test_cu121_migration.py -v -x -k "not splats"`
Expected: all PASS (the `collab_splats.splats` import entries are excluded by `-k`).

- [ ] **Step 7: Commit**

```bash
git commit --only collab_splats/nerfstudio tests/nerfstudio_methods collab_splats/wrapper/splatter.py \
  tests/test_models.py tests/wrapper/test_splatter_mesh.py tests/wrapper/test_splatter_query.py \
  collab_splats/__init__.py collab_splats/wrapper/__init__.py collab_splats/wrapper/config.py \
  collab_splats/wrapper/reconstructor.py configs/base.yaml configs/README.md CLAUDE.md \
  collab_splats/mesh/tsdf.py docs/source/conf.py docs/source/getting_started.md tests/wrapper/test_reconstructor.py \
  -m "refactor(splats): retire nerfstudio package, Splatter, and the nerfstudio pointcloud method"
```
(Deleted paths are legal `--only` arguments. Verify `git status --short` shows nothing unexpected first.)

---

### Task 3: `splats/cameras.py` — vendored pose refinement

**Files:**
- Create: `collab_splats/splats/__init__.py`, `collab_splats/splats/cameras.py`
- Test: `tests/splats/__init__.py` (empty), `tests/splats/test_cameras.py`

- [ ] **Step 1: Write the failing test**

```python
"""CameraOptModule: zero-init is identity; random-init perturbs; 6D rotation is orthonormal."""

import torch

from collab_splats.splats.cameras import CameraOptModule, rotation_6d_to_matrix


def test_zero_init_is_identity():
    m = CameraOptModule(3)
    m.zero_init()
    c2w = torch.eye(4).expand(3, 4, 4).clone()
    c2w[:, :3, 3] = torch.arange(3).float()[:, None]
    out = m(c2w, torch.arange(3))
    assert torch.allclose(out, c2w)


def test_random_init_changes_pose():
    m = CameraOptModule(2)
    m.random_init(std=0.1)
    c2w = torch.eye(4).expand(2, 4, 4).clone()
    out = m(c2w, torch.arange(2))
    assert not torch.allclose(out, c2w)


def test_rotation_6d_is_orthonormal():
    R = rotation_6d_to_matrix(torch.randn(5, 6))
    eye = torch.eye(3).expand(5, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(5), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/splats/test_cameras.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.splats'`.

- [ ] **Step 3: Write the module**

`collab_splats/splats/__init__.py`:

```python
"""Gaussian-splat training on upstream gsplat from an existing pointcloud stage."""

from .trainer import SplatsConfig, train

__all__ = ["SplatsConfig", "train"]
```
(The trainer import will fail until Task 5 — write `cameras.py` first and temporarily run the cameras test with `PYTHONPATH` import of the submodule only; or, simpler, create `trainer.py` in Task 5 and for now make `__init__.py` just the docstring. **Do the latter**: write only the docstring line now, add the imports in Task 5.)

`collab_splats/splats/cameras.py`:

```python
"""Per-camera pose refinement for splat training.

Vendored from nerfstudio-project/gsplat @ 90d7b4b, examples/utils.py:
``CameraOptModule`` lines 27-63, ``rotation_6d_to_matrix`` lines 132-153.
``examples/`` is not shipped in the gsplat wheel, so the two pieces we need are copied verbatim.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """Gram-Schmidt 6D rotation representation (Zhou et al. 2019) -> (..., 3, 3) rotation matrices."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class CameraOptModule(torch.nn.Module):
    """Learned per-camera SE(3) delta applied on the right of camera-to-world."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3) + delta rotations (6), one embedding row per camera
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Apply the learned deltas: camtoworlds (..., 4, 4), embed_ids (...) -> (..., 4, 4)."""
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(drot + self.identity.expand(*batch_shape, -1))
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/splats/test_cameras.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/splats/__init__.py collab_splats/splats/cameras.py tests/splats/__init__.py tests/splats/test_cameras.py \
  -m "feat(splats): vendor gsplat CameraOptModule for pose optimization"
```

---

### Task 4: `splats/losses.py` — scheduled weighted losses

**Files:**
- Create: `collab_splats/splats/losses.py`
- Test: `tests/splats/test_losses.py`

Contract (from spec): `compute_losses(step, out, batch, losses, scene_scale) -> (total: Tensor, logs: dict[str, float])`. Photometric `0.8·L1 + 0.2·(1−SSIM)` always on. Optional loss `name` is active iff `name in losses and losses[name]["weight"] > 0 and step >= losses[name].get("start", 0)`. `out` keys: `rgb`, `alpha`, `depth` (all `(1,H,W,C)`), `normal`, `depth_normal` `(1,H,W,3)`, `distort` `(1,H,W,1)` or `None`. `batch` keys: `rgb` `(1,H,W,3)` in `[0,1]`, `depth` `(1,H,W,1)` or `None` (0 = no target).

- [ ] **Step 1: Write the failing tests**

```python
"""compute_losses: photometric always on, optional losses gated by weight>0 and start step."""

import pytest
import torch

from collab_splats.splats.losses import compute_losses


def _out(H=16, W=16, distort=True):
    g = torch.Generator().manual_seed(0)
    n = torch.nn.functional.normalize(torch.randn(1, H, W, 3, generator=g), dim=-1)
    return {
        "rgb": torch.rand(1, H, W, 3, generator=g),
        "alpha": torch.rand(1, H, W, 1, generator=g),
        "depth": torch.rand(1, H, W, 1, generator=g) + 0.5,
        "normal": n,
        "depth_normal": torch.nn.functional.normalize(n + 0.1 * torch.randn(1, H, W, 3, generator=g), dim=-1),
        "distort": torch.rand(1, H, W, 1, generator=g) if distort else None,
    }


def _batch(H=16, W=16, depth=True):
    g = torch.Generator().manual_seed(1)
    d = torch.rand(1, H, W, 1, generator=g) + 0.5
    d[:, :4] = 0.0  # no-target rows
    return {"rgb": torch.rand(1, H, W, 3, generator=g), "depth": d if depth else None}


def test_photometric_only_when_no_optional_losses():
    total, logs = compute_losses(0, _out(), _batch(), {}, 1.0)
    assert set(logs) == {"l1", "ssim"}
    assert torch.isfinite(total)
    assert total.item() == pytest.approx(0.8 * logs["l1"] + 0.2 * logs["ssim"], rel=1e-5)


def test_depth_loss_gated_by_start_and_weight():
    losses = {"depth": {"weight": 0.5, "start": 100}}
    _, before = compute_losses(99, _out(), _batch(), losses, 1.0)
    _, after = compute_losses(100, _out(), _batch(), losses, 1.0)
    assert "depth" not in before and "depth" in after
    _, zero = compute_losses(100, _out(), _batch(), {"depth": {"weight": 0.0}}, 1.0)
    assert "depth" not in zero


def test_depth_loss_ignores_zero_targets_and_scales_with_scene_scale():
    losses = {"depth": {"weight": 1.0}}
    out, batch = _out(), _batch()
    _, a = compute_losses(0, out, batch, losses, 1.0)
    _, b = compute_losses(0, out, batch, losses, 2.0)
    assert b["depth"] == pytest.approx(2 * a["depth"], rel=1e-5)
    # Perfect prediction on the valid pixels -> zero loss regardless of the zero rows
    out["depth"] = batch["depth"].clone()
    out["depth"][:, :4] = 123.0
    _, c = compute_losses(0, out, batch, losses, 1.0)
    assert c["depth"] == pytest.approx(0.0, abs=1e-6)


def test_depth_loss_skipped_without_targets():
    _, logs = compute_losses(0, _out(), _batch(depth=False), {"depth": {"weight": 1.0}}, 1.0)
    assert "depth" not in logs


def test_normal_consistency_is_zero_for_identical_normals():
    out = _out()
    out["depth_normal"] = out["normal"].clone()
    out["alpha"] = torch.ones_like(out["alpha"])
    _, logs = compute_losses(0, out, _batch(), {"normal_consistency": {"weight": 1.0}}, 1.0)
    assert logs["normal_consistency"] == pytest.approx(0.0, abs=1e-5)


def test_distortion_requires_render_output():
    losses = {"distortion": {"weight": 1.0}}
    _, logs = compute_losses(0, _out(), _batch(), losses, 1.0)
    assert "distortion" in logs
    with pytest.raises(ValueError, match="distortion"):
        compute_losses(0, _out(distort=False), _batch(), losses, 1.0)


def test_total_is_weighted_sum():
    losses = {"depth": {"weight": 0.3}, "normal_consistency": {"weight": 0.7}}
    total, logs = compute_losses(0, _out(), _batch(), losses, 1.0)
    expected = 0.8 * logs["l1"] + 0.2 * logs["ssim"] + 0.3 * logs["depth"] + 0.7 * logs["normal_consistency"]
    assert total.item() == pytest.approx(expected, rel=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/splats/test_losses.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.splats.losses'`.

- [ ] **Step 3: Write the module**

```python
"""Scheduled, weighted loss sum over gsplat's loss functions.

Photometric (0.8 L1 + 0.2 (1 - SSIM)) is always on. Each optional loss is a yaml entry
``{weight, start}`` and is active iff weight > 0 and step >= start.
"""

import torch
from gsplat.losses import depth_l1_loss, l1_loss, normal_cosine_loss, ssim_loss
from torch import Tensor

OPTIONAL_LOSSES = ("depth", "normal_consistency", "distortion")


def _active(name: str, losses: dict[str, dict], step: int) -> bool:
    """True iff `name` is configured with weight > 0 and its start step has been reached."""
    spec = losses.get(name)
    return spec is not None and spec.get("weight", 0.0) > 0 and step >= spec.get("start", 0)


def compute_losses(
    step: int,
    out: dict[str, Tensor | None],
    batch: dict[str, Tensor | None],
    losses: dict[str, dict],
    scene_scale: float,
) -> tuple[Tensor, dict[str, float]]:
    """Weighted sum of the losses active at `step`; returns (total, {name: value})."""
    # Photometric: L1 + SSIM on (1, H, W, 3); ssim_loss wants NCHW
    rgb, gt = out["rgb"], batch["rgb"]
    l1 = l1_loss(rgb, gt).mean()
    ssim = ssim_loss(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))
    total = 0.8 * l1 + 0.2 * ssim
    logs = {"l1": l1.item(), "ssim": ssim.item()}

    # Depth: disparity L1 on pixels with a target (0 = no target)
    if _active("depth", losses, step) and batch.get("depth") is not None:
        valid = batch["depth"] > 0
        if valid.any():
            d = depth_l1_loss(out["depth"][valid], batch["depth"][valid], scene_scale)
            total = total + losses["depth"]["weight"] * d
            logs["depth"] = d.item()

    # Normal consistency: rendered normals vs normals finite-differenced from rendered depth,
    # the depth normal weighted by (detached) alpha so empty pixels do not pull
    if _active("normal_consistency", losses, step):
        n = normal_cosine_loss(out["normal"], out["depth_normal"] * out["alpha"].detach()).mean()
        total = total + losses["normal_consistency"]["weight"] * n
        logs["normal_consistency"] = n.item()

    # Distortion (2DGS only): the rasterizer's per-pixel distortion map
    if _active("distortion", losses, step):
        if out.get("distort") is None:
            raise ValueError("distortion loss configured but the renderer produced no distortion map (3dgs?)")
        dist = out["distort"].mean()
        total = total + losses["distortion"]["weight"] * dist
        logs["distortion"] = dist.item()

    return total, logs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/splats/test_losses.py -v`
Expected: 7 PASSED (CPU; `ssim_loss` falls back to the torch implementation when fused-ssim is absent — a warning is fine).

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/splats/losses.py tests/splats/test_losses.py \
  -m "feat(splats): compute_losses — scheduled weighted sum over gsplat losses"
```

---

### Task 5: `splats/trainer.py` part 1 — config, splat init, render

**Files:**
- Create: `collab_splats/splats/trainer.py`
- Modify: `collab_splats/splats/__init__.py`
- Test: `tests/splats/test_trainer.py`

- [ ] **Step 1: Write the failing tests (config + render)**

```python
"""SplatsConfig validation and _render shape contract (both primitives)."""

import numpy as np
import pytest
import torch

from collab_splats.splats.trainer import SplatsConfig, _create_splats_with_optimizers, _render

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


def test_config_defaults_and_from_dict():
    cfg = SplatsConfig.from_dict({"enabled": True, "primitive": "2dgs", "max_steps": 10,
                                  "losses": {"depth": {"weight": 0.1}}})
    assert cfg.primitive == "2dgs" and cfg.max_steps == 10 and cfg.pose_opt is False
    assert cfg.losses == {"depth": {"weight": 0.1}}


@pytest.mark.parametrize("bad", [
    {"primitive": "4dgs"},
    {"losses": {"tv": {"weight": 1.0}}},
    {"losses": {"depth": {"weight": 1.0, "stop": 5}}},
    {"primitive": "3dgs", "losses": {"distortion": {"weight": 0.1}}},
    {"unknown_key": 1},
])
def test_config_rejects(bad):
    with pytest.raises(ValueError):
        SplatsConfig.from_dict(bad)


def _scene(n_pts=200, device="cuda"):
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, (n_pts, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (n_pts, 3)).astype(np.uint8)
    return _create_splats_with_optimizers(points, colors, scene_scale=1.0, device=device)


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_render_shapes(primitive):
    splats, _ = _scene()
    c2w = torch.eye(4, device="cuda")[None]
    c2w[0, 2, 3] = -4.0
    K = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device="cuda")[None]
    out, info = _render(primitive, splats, c2w, K, 64, 64, sh_degree=0, absgrad=False)
    assert out["rgb"].shape == (1, 64, 64, 3)
    assert out["alpha"].shape == (1, 64, 64, 1)
    assert out["depth"].shape == (1, 64, 64, 1)
    assert out["normal"].shape == (1, 64, 64, 3)
    assert out["depth_normal"].shape == (1, 64, 64, 3)
    assert (out["distort"] is None) == (primitive == "3dgs")
    key = "means2d" if primitive == "3dgs" else "gradient_2dgs"
    assert key in info
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/splats/test_trainer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.splats.trainer'`.

- [ ] **Step 3: Write trainer.py (part 1)**

```python
"""One trainer for 3DGS / 2DGS splats on upstream gsplat.

Trains in the COLMAP world frame (no normalisation) so the output poses, depth and ply line up
with every other stage artifact. ``scene_scale`` (1.1 x max camera distance from the camera
centroid, as in gsplat's simple_trainer) only scales the means learning rate, the densification
thresholds and the depth loss.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from gsplat import rasterization, rasterization_2dgs
from gsplat.exporter import export_splats
from gsplat.losses import ssim_loss
from gsplat.strategy import DefaultStrategy
from gsplat.utils import depth_to_normal, normalized_quat_to_rotmat
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR

from collab_splats.splats.cameras import CameraOptModule
from collab_splats.splats.losses import OPTIONAL_LOSSES, compute_losses
from collab_splats.utils.progress import progress

logger = logging.getLogger(__name__)

GSPLAT_COMMIT = "90d7b4b"

########################################
# Constants (gsplat examples/simple_trainer.py defaults — not config knobs)
########################################

PRIMITIVES = ("3dgs", "2dgs")
SH_DEGREE = 3
SH_DEGREE_INTERVAL = 1000
INIT_OPACITY = 0.1
INIT_SCALE = 1.0
LOG_EVERY = 500
# Per-primitive densification: 3DGS uses absgrad + antialiased (examples/simple_trainer.py);
# 2DGS uses the 2D gradient key and classic rasterization (examples/simple_trainer_2dgs.py)
_STRATEGY = {
    "3dgs": dict(absgrad=True, grow_grad2d=8e-4, key_for_gradient="means2d"),
    "2dgs": dict(absgrad=False, key_for_gradient="gradient_2dgs"),
}

########################################
# Config
########################################


@dataclass
class SplatsConfig:
    """Trainer knobs; mirrors the `splats:` yaml block (minus `enabled`)."""

    primitive: str = "3dgs"
    max_steps: int = 30000
    pose_opt: bool = False
    losses: dict[str, dict] = field(
        default_factory=lambda: {
            "depth": {"weight": 0.01},
            "normal_consistency": {"weight": 0.05, "start": 7000},
            "distortion": {"weight": 0.0, "start": 3000},
        }
    )

    @classmethod
    def from_dict(cls, d: dict) -> "SplatsConfig":
        """Build from the yaml block, rejecting unknown keys, unknown losses and 3dgs+distortion."""
        known = {"enabled", *cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"splats: unknown keys {sorted(unknown)}; allowed {sorted(known)}")
        cfg = cls(**{k: v for k, v in d.items() if k != "enabled"})
        if cfg.primitive not in PRIMITIVES:
            raise ValueError(f"splats.primitive must be one of {PRIMITIVES}, got '{cfg.primitive}'")
        for name, spec in cfg.losses.items():
            if name not in OPTIONAL_LOSSES:
                raise ValueError(f"splats.losses: unknown loss '{name}'; allowed {OPTIONAL_LOSSES}")
            bad = set(spec) - {"weight", "start"}
            if bad:
                raise ValueError(f"splats.losses.{name}: unknown keys {sorted(bad)}; allowed ['weight', 'start']")
        if cfg.primitive == "3dgs" and cfg.losses.get("distortion", {}).get("weight", 0.0) > 0:
            raise ValueError("splats.losses.distortion is 2dgs-only; set weight 0 or primitive: 2dgs")
        return cfg


########################################
# Splat initialisation (examples/simple_trainer.py create_splats_with_optimizers @ 90d7b4b)
########################################


def _knn_dist2(points: np.ndarray, k: int = 4) -> np.ndarray:
    """Mean squared distance to the k-1 nearest neighbours of each point (sklearn, CPU)."""
    nn = NearestNeighbors(n_neighbors=k).fit(points)
    dists, _ = nn.kneighbors(points)
    return (dists[:, 1:] ** 2).mean(-1)


def _create_splats_with_optimizers(
    points: np.ndarray, colors: np.ndarray, scene_scale: float, device: str
) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Optimizer]]:
    """Gaussians seeded on the sparse points (scale from kNN spacing, colour as SH DC) + one Adam per param."""
    pts = torch.from_numpy(points).float().to(device)
    rgb = torch.from_numpy(colors).float().to(device) / 255.0
    n = len(pts)
    scales = torch.log(torch.from_numpy(np.sqrt(_knn_dist2(points))).float().to(device) * INIT_SCALE)
    scales = scales.unsqueeze(-1).repeat(1, 3)
    sh = torch.zeros(n, (SH_DEGREE + 1) ** 2, 3, device=device)
    sh[:, 0, :] = (rgb - 0.5) / 0.28209479177387814  # rgb_to_sh
    params = [
        ("means", torch.nn.Parameter(pts), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(torch.rand(n, 4, device=device)), 1e-3),
        ("opacities", torch.nn.Parameter(torch.logit(torch.full((n,), INIT_OPACITY, device=device))), 5e-2),
        ("sh0", torch.nn.Parameter(sh[:, :1, :]), 2.5e-3),
        ("shN", torch.nn.Parameter(sh[:, 1:, :]), 2.5e-3 / 20),
    ]
    splats = torch.nn.ParameterDict({k: v for k, v, _ in params}).to(device)
    optimizers = {
        k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15) for k, _, lr in params
    }
    return splats, optimizers


########################################
# Rendering
########################################


def _gaussian_normals(quats: Tensor, scales: Tensor, means: Tensor, viewmat: Tensor) -> Tensor:
    """Per-Gaussian camera-space normal: shortest scale axis, flipped to face the camera. (N, 3)."""
    R = normalized_quat_to_rotmat(F.normalize(quats, dim=-1))  # (N, 3, 3)
    axis = scales.argmin(dim=-1)
    n_world = R[torch.arange(len(R), device=R.device), :, axis]  # column = rotated axis
    R_cw, t_cw = viewmat[:3, :3], viewmat[:3, 3]
    n_cam = n_world @ R_cw.T
    p_cam = means @ R_cw.T + t_cw
    flip = (n_cam * p_cam).sum(-1, keepdim=True) > 0
    return torch.where(flip, -n_cam, n_cam)


def _render(
    primitive: str,
    splats: torch.nn.ParameterDict,
    c2w: Tensor,
    K: Tensor,
    width: int,
    height: int,
    sh_degree: int,
    absgrad: bool,
) -> tuple[dict[str, Tensor | None], dict]:
    """Render one camera. Returns ({rgb, alpha, depth, normal, depth_normal, distort}, strategy info).

    Normals are camera-space for both primitives; `distort` is None for 3dgs.
    """
    means, quats = splats["means"], splats["quats"]
    scales, opacities = torch.exp(splats["scales"]), torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    viewmat = torch.linalg.inv(c2w)
    common = dict(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors, viewmats=viewmat,
        Ks=K, width=width, height=height, sh_degree=sh_degree, packed=False, absgrad=absgrad,
        render_mode="RGB+ED",
    )
    if primitive == "2dgs":
        rgbd, alpha, normal, depth_normal, distort, _median, info = rasterization_2dgs(**common, distloss=True)
        out = dict(rgb=rgbd[..., :3], alpha=alpha, depth=rgbd[..., 3:4], normal=normal,
                   depth_normal=depth_normal, distort=distort)
    else:
        # Normals rendered as an extra per-Gaussian signal; depth normal finite-differenced
        # in camera space (identity c2w) so both sides of the consistency loss share a frame
        normals_cam = _gaussian_normals(quats, scales, means, viewmat[0])
        rgbd, alpha, info = rasterization(**common, rasterize_mode="antialiased", extra_signals=normals_cam)
        depth = rgbd[..., 3:4]
        eye = torch.eye(4, device=c2w.device)[None]
        out = dict(rgb=rgbd[..., :3], alpha=alpha, depth=depth,
                   normal=F.normalize(info["render_extra_signals"], dim=-1),
                   depth_normal=depth_to_normal(depth, eye, K), distort=None)
    return out, info
```

Then set `collab_splats/splats/__init__.py` to:

```python
"""Gaussian-splat training on upstream gsplat from an existing pointcloud stage."""

from .trainer import SplatsConfig, train

__all__ = ["SplatsConfig", "train"]
```

(`train` is added in Task 6 — add a stub `def train(*a, **k): raise NotImplementedError` at the bottom of `trainer.py` for now so the import resolves; Task 6 replaces it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/splats/test_trainer.py -v`
Expected: `test_config_defaults_and_from_dict`, 5× `test_config_rejects`, 2× `test_render_shapes` PASSED. If `test_render_shapes[3dgs]` fails on `render_extra_signals` missing, confirm with `$PY -c "import inspect,gsplat;print('extra_signals' in inspect.signature(gsplat.rasterization).parameters)"` — must be `True` (Task 1 pin).

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/splats/trainer.py collab_splats/splats/__init__.py tests/splats/test_trainer.py \
  -m "feat(splats): SplatsConfig, splat init and unified 3dgs/2dgs _render"
```

---

### Task 6: `splats/trainer.py` part 2 — train loop + save

**Files:**
- Modify: `collab_splats/splats/trainer.py` (replace the `train` stub)
- Test: `tests/splats/test_trainer.py`

- [ ] **Step 1: Write the failing end-to-end test**

Append to `tests/splats/test_trainer.py`:

```python
from collab_splats.splats.trainer import train


def _synthetic_scene(n_frames=8, H=64, W=64, n_pts=200):
    """Random coloured points in a unit box, cameras on a ring looking at the origin, analytic depth."""
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 0.5, (n_pts, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (n_pts, 3)).astype(np.uint8)
    K = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]], dtype=np.float32)
    w2c, images, depths = [], [], []
    for i in range(n_frames):
        a = 2 * np.pi * i / n_frames
        cam = np.array([3 * np.cos(a), 0.3, 3 * np.sin(a)], dtype=np.float32)
        z = -cam / np.linalg.norm(cam)                      # look at origin (OpenCV +z forward)
        x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=1)                     # c2w rotation (columns = camera axes)
        T = np.eye(4, dtype=np.float32); T[:3, :3] = R.T; T[:3, 3] = -R.T @ cam
        w2c.append(T)
        # Image: splat each point as a 3x3 dot; depth: z of the nearest point per pixel
        img = np.zeros((H, W, 3), np.uint8); dep = np.zeros((H, W), np.float32)
        pc = (T[:3, :3] @ points.T + T[:3, 3:]).T
        order = np.argsort(-pc[:, 2])                       # far to near so near overwrites
        for j in order:
            if pc[j, 2] <= 0: continue
            u, v = (K[:2, :2] @ (pc[j, :2] / pc[j, 2]) + K[:2, 2]).round().astype(int)
            if 1 <= u < W - 1 and 1 <= v < H - 1:
                img[v - 1:v + 2, u - 1:u + 2] = colors[j]; dep[v - 1:v + 2, u - 1:u + 2] = pc[j, 2]
        images.append(img); depths.append(dep)
    return (np.stack(images), np.stack(w2c), np.stack([K] * n_frames), points, colors, np.stack(depths))


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_train_end_to_end(tmp_path, primitive):
    images, w2c, K, points, colors, depths = _synthetic_scene()
    losses = {"depth": {"weight": 0.1}, "normal_consistency": {"weight": 0.05, "start": 10}}
    if primitive == "2dgs":
        losses["distortion"] = {"weight": 0.01, "start": 10}
    cfg = SplatsConfig(primitive=primitive, max_steps=50, losses=losses)
    report = train(cfg, images, w2c, K, points, colors, tmp_path, depth_targets=depths)

    assert (tmp_path / "splats.ply").exists()
    assert (tmp_path / "ckpt.pt").exists()
    assert (tmp_path / "splats_quality_report.json").exists()
    s = report["summary"]
    assert s["steps"] == 50 and s["n_gaussians"] > 0 and np.isfinite(s["psnr"])
    assert len(report["per_frame"]) == 8
    assert {"depth", "normal_consistency"} <= set(s["final_losses"])

    import zarr
    g = zarr.open_group(tmp_path / "splats.zarr", mode="r")
    assert g["rgb"].shape == (8, 64, 64, 3) and g["rgb"].dtype == np.uint8
    assert g["depth"].shape == (8, 64, 64) and g["normal"].shape == (8, 64, 64, 3)
    assert g["alpha"].shape == (8, 64, 64) and g["c2w"].shape == (8, 4, 4) and g["K"].shape == (8, 3, 3)
    assert g.attrs["primitive"] == primitive and g.attrs["gsplat_commit"] == "90d7b4b"
    assert list(g.attrs["image_ids"]) == list(range(8))

    ckpt = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["step"] == 50 and ckpt["pose_adjust"] is None and "means" in ckpt["splats"]


@cuda
def test_train_pose_opt_moves_poses(tmp_path):
    images, w2c, K, points, colors, _ = _synthetic_scene()
    cfg = SplatsConfig(max_steps=30, pose_opt=True, losses={})
    train(cfg, images, w2c, K, points, colors, tmp_path)
    import zarr
    g = zarr.open_group(tmp_path / "splats.zarr", mode="r")
    c2w_in = np.linalg.inv(w2c)
    assert not np.allclose(g["c2w"][:], c2w_in, atol=1e-7)  # refined
    assert np.allclose(g["c2w"][:], c2w_in, atol=1e-2)      # but only slightly
    ckpt = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["pose_adjust"] is not None


@cuda
def test_train_rejects_bad_inputs(tmp_path):
    images, w2c, K, points, colors, _ = _synthetic_scene()
    with pytest.raises(ValueError, match="points"):
        train(SplatsConfig(max_steps=1), images, w2c, K, points[:50], colors[:50], tmp_path)
    with pytest.raises(ValueError, match="frames"):
        train(SplatsConfig(max_steps=1), images[:4], w2c, K, points, colors, tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/splats/test_trainer.py -v -k train`
Expected: FAIL — `NotImplementedError` from the stub.

- [ ] **Step 3: Replace the stub with train + _save**

Append to `collab_splats/splats/trainer.py` (delete the stub first):

```python
########################################
# Training
########################################


def train(
    cfg: SplatsConfig,
    images: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    out_dir: Path,
    depth_targets: np.ndarray | None = None,
) -> dict:
    """Train splats and write splats.ply / ckpt.pt / splats.zarr / splats_quality_report.json to out_dir.

    Args:
        images: (N, H, W, 3) uint8 frames.
        w2c: (N, 4, 4) world-to-camera (COLMAP convention), K: (N, 3, 3) at frame resolution.
        points, colors: (P, 3) float32 / uint8 sparse seed points in the same world frame.
        depth_targets: optional (N, h, w) float32 depth (any resolution, 0 = no target).

    Returns the quality report dict.
    """
    device = "cuda"
    N, H, W = images.shape[:3]
    if len(points) < 100:
        raise ValueError(f"splats: need >= 100 seed points, got {len(points)}")
    if not (N == len(w2c) == len(K)) or (depth_targets is not None and len(depth_targets) != N):
        raise ValueError(f"splats: frames mismatch — images {N}, w2c {len(w2c)}, K {len(K)}, "
                         f"depth_targets {None if depth_targets is None else len(depth_targets)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scene tensors: poses/K on GPU, frames stay uint8 on CPU and move one at a time
    c2w_all = torch.from_numpy(np.linalg.inv(w2c)).float().to(device)
    Ks = torch.from_numpy(K).float().to(device)
    cam_pos = c2w_all[:, :3, 3]
    scene_scale = float((cam_pos - cam_pos.mean(0)).norm(dim=-1).max()) * 1.1

    # Splats, densification strategy, schedulers, optional pose refinement
    splats, optimizers = _create_splats_with_optimizers(points, colors, scene_scale, device)
    strategy = DefaultStrategy(verbose=True, **_STRATEGY[cfg.primitive])
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    gamma = 0.01 ** (1.0 / cfg.max_steps)
    schedulers = [ExponentialLR(optimizers["means"], gamma=gamma)]
    pose_adjust, pose_opt = None, None
    if cfg.pose_opt:
        pose_adjust = CameraOptModule(N).to(device)
        pose_adjust.zero_init()
        pose_opt = torch.optim.Adam(pose_adjust.parameters(), lr=1e-5 * scene_scale, weight_decay=1e-6)
        schedulers.append(ExponentialLR(pose_opt, gamma=gamma))

    def batch_for(i: int) -> dict[str, Tensor | None]:
        rgb = torch.from_numpy(images[i]).to(device).float()[None] / 255.0
        depth = None
        if depth_targets is not None:
            d = torch.from_numpy(depth_targets[i]).to(device)[None, None]
            depth = F.interpolate(d, size=(H, W), mode="nearest").permute(0, 2, 3, 1)
        return {"rgb": rgb, "depth": depth}

    # Main loop: one random frame per step
    t0 = time.perf_counter()
    logs: dict[str, float] = {}
    for step in progress(range(cfg.max_steps), desc=f"splats[{cfg.primitive}]"):
        i = int(torch.randint(N, (1,)))
        batch = batch_for(i)
        c2w = c2w_all[i : i + 1]
        if pose_adjust is not None:
            c2w = pose_adjust(c2w, torch.tensor([i], device=device))
        sh_degree = min(step // SH_DEGREE_INTERVAL, SH_DEGREE)
        out, info = _render(cfg.primitive, splats, c2w, Ks[i : i + 1], W, H, sh_degree, strategy.absgrad)
        # Random background so transparent regions cannot hide behind a fixed colour
        bkgd = torch.rand(1, 3, device=device)
        out["rgb"] = out["rgb"] + bkgd * (1.0 - out["alpha"])

        strategy.step_pre_backward(splats, optimizers, state, step, info)
        loss, logs = compute_losses(step, out, batch, cfg.losses, scene_scale)
        loss.backward()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if pose_opt is not None:
            pose_opt.step()
            pose_opt.zero_grad(set_to_none=True)
        for sched in schedulers:
            sched.step()
        if step % LOG_EVERY == 0:
            logger.info("splats step %d loss %.4f gaussians %d %s", step, loss.item(), len(splats["means"]),
                        {k: round(v, 4) for k, v in logs.items()})

    seconds = time.perf_counter() - t0
    return _save(cfg, splats, pose_adjust, images, c2w_all, Ks, out_dir, seconds, logs)


########################################
# Save: ply + ckpt + rendered zarr + quality report
########################################


def _save(cfg, splats, pose_adjust, images, c2w_all, Ks, out_dir, seconds, final_losses) -> dict:
    """Write splats.ply, ckpt.pt, splats.zarr (full-res renders of every frame) and the quality report."""
    N, H, W = images.shape[:3]
    device = c2w_all.device
    export_splats(means=splats["means"], scales=splats["scales"], quats=splats["quats"],
                  opacities=splats["opacities"], sh0=splats["sh0"], shN=splats["shN"],
                  format="ply", save_to=str(out_dir / "splats.ply"))
    torch.save({"splats": {k: v.detach().cpu() for k, v in splats.items()},
                "pose_adjust": None if pose_adjust is None else pose_adjust.state_dict(),
                "config": asdict(cfg), "step": cfg.max_steps}, out_dir / "ckpt.pt")

    # Render every frame once at full SH degree over a black background; score psnr/ssim as we go
    rgb_all = np.zeros((N, H, W, 3), np.uint8)
    depth_all = np.zeros((N, H, W), np.float32)
    normal_all = np.zeros((N, H, W, 3), np.float32)
    alpha_all = np.zeros((N, H, W), np.float32)
    c2w_out = c2w_all.clone()
    per_frame = []
    with torch.no_grad():
        for i in progress(range(N), desc="splats render"):
            c2w = c2w_all[i : i + 1]
            if pose_adjust is not None:
                c2w = pose_adjust(c2w, torch.tensor([i], device=device))
                c2w_out[i] = c2w[0]
            out, _ = _render(cfg.primitive, splats, c2w, Ks[i : i + 1], W, H, SH_DEGREE, absgrad=False)
            gt = torch.from_numpy(images[i]).to(device).float()[None] / 255.0
            rgb = out["rgb"].clamp(0, 1)
            mse = F.mse_loss(rgb, gt).item()
            per_frame.append({"image_id": i, "psnr": 10 * np.log10(1.0 / max(mse, 1e-12)),
                              "ssim": 1.0 - ssim_loss(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).item()})
            rgb_all[i] = (rgb[0] * 255).round().byte().cpu().numpy()
            depth_all[i] = out["depth"][0, ..., 0].cpu().numpy()
            normal_all[i] = out["normal"][0].cpu().numpy()
            alpha_all[i] = out["alpha"][0, ..., 0].cpu().numpy()

    g = zarr.open_group(out_dir / "splats.zarr", mode="w")
    for name, arr in (("rgb", rgb_all), ("depth", depth_all), ("normal", normal_all), ("alpha", alpha_all),
                      ("c2w", c2w_out.cpu().numpy()), ("K", Ks.cpu().numpy())):
        g.create_array(name, data=arr, chunks=(1, *arr.shape[1:]))
    g.attrs.update(image_ids=list(range(N)), primitive=cfg.primitive, pose_opt=cfg.pose_opt,
                   gsplat_commit=GSPLAT_COMMIT, config=asdict(cfg))

    report = {
        "summary": {
            "psnr": float(np.mean([f["psnr"] for f in per_frame])),
            "ssim": float(np.mean([f["ssim"] for f in per_frame])),
            "n_gaussians": int(len(splats["means"])),
            "steps": cfg.max_steps,
            "seconds": round(seconds, 1),
            "final_losses": final_losses,
            "config": asdict(cfg),
        },
        "per_frame": per_frame,
    }
    (out_dir / "splats_quality_report.json").write_text(json.dumps(report, indent=2))
    logger.info("splats: %d gaussians, psnr %.2f, ssim %.3f, %.0fs -> %s",
                report["summary"]["n_gaussians"], report["summary"]["psnr"], report["summary"]["ssim"],
                seconds, out_dir)
    return report
```


- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/splats/ -v`
Expected: all PASS (≈1–2 min on GPU). Known pitfalls: `strategy.absgrad` is a dataclass field on `DefaultStrategy` — if `AttributeError`, use `_STRATEGY[cfg.primitive]["absgrad"]`. If `export_splats` complains about `shN` being empty for `sh_degree=0`, it is not — we always build `(SH_DEGREE+1)**2 = 16` coefficients.

- [ ] **Step 5: Format and commit**

```bash
$PY -m black collab_splats/splats tests/splats && $PY -m isort collab_splats/splats tests/splats
git commit --only collab_splats/splats/trainer.py tests/splats/test_trainer.py \
  -m "feat(splats): train loop (DefaultStrategy, sh schedule, pose opt) + ply/ckpt/zarr/report save"
```

---

### Task 7: `Reconstructor.splats()` leaf stage + config

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` — `_STAGE_ORDER`/`_STAGE_DEPS` (55–78), `_stage_output_exists` (~1283), `run_pipeline` (~1320: enabled-stage list + dispatch), new method after `verify`
- Modify: `configs/base.yaml` — add `splats:` block after `mesh:`
- Test: `tests/wrapper/test_splats_stage.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Splats-stage wiring: leaf registration, config default, array assembly handed to train()."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from collab_splats.wrapper.reconstructor import _STAGE_DEPS, _STAGE_ORDER, LEAF_STAGES, Reconstructor

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_splats_is_a_leaf_stage():
    assert "splats" in _STAGE_ORDER
    assert _STAGE_ORDER.index("splats") < _STAGE_ORDER.index("mesh")
    assert _STAGE_DEPS["splats"] == ["pointcloud"]
    assert "splats" in LEAF_STAGES


def test_base_yaml_defaults():
    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    assert cfg["splats"]["enabled"] is False
    assert cfg["splats"]["primitive"] == "3dgs"
    assert set(cfg["splats"]["losses"]) == {"depth", "normal_consistency", "distortion"}
    assert cfg["splats"]["losses"]["distortion"]["weight"] == 0.0
    assert cfg["mesh"]["source"] == "feedforward"


def _stub_reconstructor(tmp_path, n=3, H=8, W=8):
    r = Reconstructor.__new__(Reconstructor)
    r.config = {
        "output_path": str(tmp_path),
        "pointcloud": {"backend": "vggtx"},
        "mesh": {"conf_percentile": 20},
        "splats": {"enabled": True, "max_steps": 1, "losses": {"depth": {"weight": 0.1}}},
    }
    r._stage_output_exists = lambda stage: False
    # frames.zarr with n frames named frame_000000.. ; image_paths in reversed order to prove lookup-by-idx
    from collab_splats.preproc.frame_store import FrameStore
    images = np.stack([np.full((H, W, 3), i * 10, np.uint8) for i in range(n)])
    FrameStore.create(r.frames_zarr, images, [{"frame_idx": i} for i in range(n)], provenance={"video_path": "v"})
    paths = [Path(f"frame_{i:06d}.jpg") for i in reversed(range(n))]
    r._resolve_result = lambda: SimpleNamespace(
        image_paths=paths,
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        points=np.zeros((200, 3), np.float32),
        colors=np.zeros((200, 3), np.uint8),
    )
    return r, images


def test_splats_stage_assembles_arrays_and_calls_train(tmp_path):
    r, images = _stub_reconstructor(tmp_path)
    ff = SimpleNamespace(depth=np.ones((3, 4, 4), np.float32), confidence=None)
    with patch("collab_splats.splats.trainer.train") as train, patch(
        "collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=ff
    ):
        train.return_value = {"summary": {}}
        (r.backend_dir / "feedforward.zarr").mkdir(parents=True)
        out = r.splats()
    assert out == r.backend_dir / "splats" / "splats.zarr"
    args, kwargs = train.call_args
    cfg, imgs, w2c, K, pts, cols, out_dir = args
    assert cfg.max_steps == 1
    # image_paths were reversed -> stacked images must follow image_paths order, not store order
    assert imgs[0, 0, 0, 0] == 20 and imgs[2, 0, 0, 0] == 0
    assert w2c.shape == (3, 4, 4) and K.shape == (3, 3, 3) and pts.shape == (200, 3)
    assert out_dir == r.backend_dir / "splats"
    assert kwargs["depth_targets"].shape == (3, 4, 4)


def test_splats_stage_skips_depth_targets_when_loss_off(tmp_path):
    r, _ = _stub_reconstructor(tmp_path)
    r.config["splats"]["losses"] = {}
    with patch("collab_splats.splats.trainer.train", return_value={}) as train:
        r.splats()
    assert train.call_args.kwargs["depth_targets"] is None


def test_splats_stage_refuses_existing_without_overwrite(tmp_path):
    r, _ = _stub_reconstructor(tmp_path)
    r._stage_output_exists = lambda stage: stage == "splats"
    with patch("collab_splats.splats.trainer.train") as train:
        r.splats()
    train.assert_not_called()
```

(`FrameStore.create(path, frames, records, *, provenance)` — every record needs `frame_idx`; same call shape as `tests/preproc/test_frame_store.py:17`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/wrapper/test_splats_stage.py -v`
Expected: FAIL — `assert 'splats' in _STAGE_ORDER`; `KeyError: 'splats'` on base.yaml.

- [ ] **Step 3: Register the stage**

`collab_splats/wrapper/reconstructor.py`:

`_STAGE_ORDER` → `["preproc", "pointcloud", "refine", "semantics", "splats", "mesh", "localize", "verify", "reconstruction_quality_report"]`.

`_STAGE_DEPS` — add, with a comment in the same style as the neighbours:
```python
    # splats: trains from COLMAP poses/points + frames.zarr; mesh.source: splats reads its zarr
    "splats": ["pointcloud"],
```

`_stage_output_exists` — add before the `mesh` check:
```python
        if stage == "splats":
            return (self.backend_dir / "splats" / "splats.zarr").exists()
```

`run_pipeline` enabled-stage list — after the `semantics` append:
```python
            if self.config["splats"]["enabled"]:
                stages.append("splats")
```
Dispatch loop — after the `semantics` branch:
```python
            elif stage == "splats":
                self.splats(overwrite=overwrite)
```
Also extend the `stages:` docstring list with `"splats"`.

New method, placed after `verify()`:

```python
    def splats(self, overwrite: bool = False) -> Path:
        """Train Gaussian splats from the pointcloud stage. Returns path to splats/splats.zarr."""
        out_dir = self.backend_dir / "splats"
        if not overwrite and self._stage_output_exists("splats"):
            logger.info("Splats exist at %s, skipping", out_dir)
            return out_dir / "splats.zarr"

        result = self._resolve_result()
        if result is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        from collab_splats.pointcloud.feedforward.base import FeedforwardResult
        from collab_splats.pointcloud.utils import confidence_mask
        from collab_splats.preproc.frame_store import FrameStore
        from collab_splats.splats.trainer import SplatsConfig, train

        cfg = SplatsConfig.from_dict(self.config["splats"])

        # Frames in image_paths order (COLMAP image order), looked up by frame index
        store = FrameStore.open(self.frames_zarr)
        images = np.stack([store.image_by_frame_idx(FrameStore.frame_idx_from_path(p)) for p in result.image_paths])

        # Depth targets: feedforward depth masked like the mesh stage (see plan deviation note)
        depth_targets = None
        if cfg.losses.get("depth", {}).get("weight", 0.0) > 0:
            ff = FeedforwardResult.load_zarr(
                self.backend_dir / "feedforward.zarr", load_images=False, load_world_points=False
            )
            if ff.depth is None or len(ff.depth) != len(images):
                raise ValueError(
                    f"splats depth loss needs feedforward depth for all {len(images)} frames; "
                    f"feedforward.zarr has {None if ff.depth is None else len(ff.depth)}"
                )
            depth_targets = np.ascontiguousarray(ff.depth, dtype=np.float32)
            pct = self.config["mesh"]["conf_percentile"]
            if pct is not None and ff.confidence is not None:
                conf = ff.confidence.cpu().numpy() if hasattr(ff.confidence, "cpu") else np.asarray(ff.confidence)
                depth_targets = np.where(confidence_mask(conf, pct), depth_targets, 0.0).astype(np.float32)

        train(cfg, images, result.extrinsics, result.intrinsics, result.points, result.colors, out_dir,
              depth_targets=depth_targets)
        logger.info("Splats saved to %s", out_dir)
        return out_dir / "splats.zarr"
```

(`np` is already imported at the top of `reconstructor.py` — verify with `grep -n '^import numpy' collab_splats/wrapper/reconstructor.py`.)

- [ ] **Step 4: Add the config block**

`configs/base.yaml` — insert after the `mesh:` block (keep the file's comment style):

```yaml
# Gaussian splats trained from the pointcloud stage (leaf stage `splats`, upstream gsplat).
# Losses: photometric (0.8 L1 + 0.2 SSIM) is always on; each entry below is active iff
# weight > 0 and step >= start. Depth targets = feedforward depth masked by mesh.conf_percentile.
splats:
  enabled: false
  primitive: 3dgs            # 3dgs (fast kernel, antialiased) | 2dgs (surface-aligned)
  max_steps: 30000
  pose_opt: false            # refine camera poses jointly (CameraOptModule)
  losses:
    depth: {weight: 0.01}
    normal_consistency: {weight: 0.05, start: 7000}
    distortion: {weight: 0.0, start: 3000}   # 2dgs only; >0 with 3dgs is a ValueError
```

And in the `mesh:` block add, directly under `enabled:`:
```yaml
  source: feedforward        # feedforward | splats — which depth maps TSDF fuses
```

- [ ] **Step 5: Run tests**

Run: `$PY -m pytest tests/wrapper/test_splats_stage.py tests/wrapper/test_verify_stage.py tests/wrapper/test_reconstructor.py -v`
Expected: all PASS. If `test_reconstructor.py` has a test enumerating `_STAGE_ORDER` literally, update its expected list to include `"splats"` between `semantics` and `mesh`.

- [ ] **Step 6: Commit**

```bash
git commit --only collab_splats/wrapper/reconstructor.py configs/base.yaml tests/wrapper/test_splats_stage.py tests/wrapper/test_reconstructor.py \
  -m "feat(wrapper): splats leaf stage — Reconstructor.splats() + splats: config block"
```

---

### Task 8: `mesh.source: splats`

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` — `mesh()` (~1078–1116) + new module-level `_run_splats_mesh` next to `_run_tsdf_mesh`
- Test: `tests/wrapper/test_splats_stage.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/wrapper/test_splats_stage.py`:

```python
import zarr

from collab_splats.wrapper.reconstructor import _run_splats_mesh


def _write_splats_zarr(path, n=2, H=8, W=8):
    g = zarr.open_group(path, mode="w")
    rng = np.random.default_rng(0)
    g.create_array("rgb", data=rng.integers(0, 255, (n, H, W, 3)).astype(np.uint8))
    g.create_array("depth", data=np.full((n, H, W), 1.0, np.float32))
    g.create_array("normal", data=np.zeros((n, H, W, 3), np.float32))
    alpha = np.ones((n, H, W), np.float32); alpha[:, :2] = 0.1
    g.create_array("alpha", data=alpha)
    g.create_array("c2w", data=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)))
    g.create_array("K", data=np.tile(np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]], np.float32), (n, 1, 1)))


def test_run_splats_mesh_masks_alpha_and_calls_mesher(tmp_path):
    _write_splats_zarr(tmp_path / "splats.zarr")
    with patch("collab_splats.mesh.get_mesh_creator") as gmc, patch(
        "collab_splats.mesh.utils.optimize_color_map"
    ) as ocm:
        mesher = gmc.return_value
        mesher.depth_trunc = 2.0
        mesher.create.return_value = SimpleNamespace(mesh_path=tmp_path / "mesh.ply")
        out = _run_splats_mesh(tmp_path / "splats.zarr", tmp_path, voxel_size=0.01, sdf_trunc=0.04,
                               depth_trunc=2.0, clean_repair=False, conf_percentile=20, color_map_iterations=3)
    assert out == tmp_path / "mesh.ply"
    depths = mesher.create.call_args.args[0]
    assert (depths[:, :2] == 0).all() and (depths[:, 2:] == 1.0).all()   # low-alpha rows masked
    assert mesher.create.call_args.args[1].dtype == np.uint8
    ocm.assert_called_once()


def test_run_splats_mesh_requires_zarr(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="splats.zarr"):
        _run_splats_mesh(tmp_path / "splats.zarr", tmp_path, voxel_size=0.01, sdf_trunc=0.04,
                         depth_trunc=2.0, clean_repair=False, conf_percentile=None, color_map_iterations=0)


def test_mesh_dispatches_on_source(tmp_path):
    r, _ = _stub_reconstructor(tmp_path)
    r.config["mesh"].update(enabled=True, source="splats", voxel_size=0.01, sdf_trunc=0.04, depth_trunc=2.0,
                            clean_repair=False, native_resolution=True, color_map_iterations=0)
    with patch("collab_splats.wrapper.reconstructor._run_splats_mesh", return_value=tmp_path / "m.ply") as rsm:
        out = r.mesh()
    rsm.assert_called_once()
    assert rsm.call_args.args[0] == r.backend_dir / "splats" / "splats.zarr"
    assert out == tmp_path / "m.ply"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/wrapper/test_splats_stage.py -v -k mesh`
Expected: FAIL — `ImportError: cannot import name '_run_splats_mesh'`.

- [ ] **Step 3: Implement**

Module-level function in `reconstructor.py`, directly after `_run_tsdf_mesh`:

```python
def _run_splats_mesh(
    splats_zarr: Path,
    output_dir: Path,
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
    clean_repair: bool,
    conf_percentile: float | None,
    color_map_iterations: int,
) -> Path:
    """TSDF-fuse the splat renders in splats.zarr (rendered depth/rgb/poses); alpha plays the confidence role."""
    import zarr

    from collab_splats.mesh import get_mesh_creator
    from collab_splats.mesh.utils import optimize_color_map
    from collab_splats.pointcloud.utils import confidence_mask

    if not splats_zarr.exists():
        raise ValueError(f"mesh.source: splats needs {splats_zarr}; run the splats stage first")
    g = zarr.open_group(splats_zarr, mode="r")
    depths, rgbs = g["depth"][:].astype(np.float32), g["rgb"][:]
    c2w, K, alpha = g["c2w"][:], g["K"][:], g["alpha"][:]

    # Mask low-alpha pixels exactly as the feedforward path masks low-confidence ones
    if conf_percentile is not None:
        depths = np.where(confidence_mask(alpha, conf_percentile), depths, 0.0).astype(np.float32)

    mesher = get_mesh_creator(
        "open3d_tsdf", Path(output_dir), voxel_size=voxel_size, sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc, clean_repair=clean_repair,
    )
    mesh_path = mesher.create(depths, rgbs, c2w, K).mesh_path
    if color_map_iterations > 0:
        optimize_color_map(mesh_path, depths, rgbs, c2w, K, iterations=color_map_iterations,
                           depth_trunc=mesher.depth_trunc)
    return mesh_path
```

(`optimize_color_map(mesh_path, depths, rgbs, c2w, intrinsics, iterations, depth_trunc)` — `collab_splats/mesh/utils.py:615`; the call above matches.)

In `Reconstructor.mesh()`, after the skip check and before `result = result or self._resolve_result()`:

```python
        mesh_cfg = self.config["mesh"]
        if mesh_cfg["source"] == "splats":
            if mesh_cfg["native_resolution"]:
                logger.info("mesh.source: splats renders at frame resolution already; native_resolution ignored")
            out = _run_splats_mesh(
                self.backend_dir / "splats" / "splats.zarr", self.backend_dir,
                voxel_size=mesh_cfg["voxel_size"], sdf_trunc=mesh_cfg["sdf_trunc"],
                depth_trunc=mesh_cfg["depth_trunc"], clean_repair=mesh_cfg["clean_repair"],
                conf_percentile=mesh_cfg["conf_percentile"], color_map_iterations=mesh_cfg["color_map_iterations"],
            )
            logger.info("Mesh saved to %s", out)
            return out
        if mesh_cfg["source"] != "feedforward":
            raise ValueError(f"mesh.source must be 'feedforward' or 'splats', got '{mesh_cfg['source']}'")
```
and delete the later duplicate `mesh_cfg = self.config["mesh"]` line.

Also: in `run_pipeline`'s dependency validation, nothing changes — `mesh` still depends on `pointcloud`; a `source: splats` run without the splats zarr fails loud inside `mesh()`.

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/wrapper/test_splats_stage.py tests/wrapper/test_reconstructor.py -v`
Expected: all PASS. `test_reconstructor.py` stubs that build a `mesh` config dict without `source` will `KeyError` — add `"source": "feedforward"` to those fixtures.

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_splats_stage.py tests/wrapper/test_reconstructor.py \
  -m "feat(mesh): mesh.source: splats — TSDF-fuse rendered splat depth from splats.zarr"
```

---

### Task 9: Docs, spec deviation, gates

**Files:**
- Modify: `configs/README.md` (key table ~313+, layout table ~425, section ~461), `CLAUDE.md` (arch tree + In-Flight/Recently completed), `docs/superpowers/specs/2026-08-22-splats-module-design.md`

- [ ] **Step 1: configs/README.md**

Key table: add rows for `splats.enabled`, `splats.primitive`, `splats.max_steps`, `splats.pose_opt`, `splats.losses.<name>.{weight,start}`, `mesh.source` (one line each, same column format as neighbours).

Processed-scene layout table: add `<backend>/splats/splats.ply`, `ckpt.pt`, `splats.zarr`, `splats_quality_report.json` rows.

Replace section "#### `ns-train --data` does not work on a published scene, by design" with:

```markdown
#### Splats train from the published COLMAP + frames.zarr

`--stages splats` pulls a processed scene and trains directly on `colmap/` poses + points and
`frames.zarr` — no image directory, no transforms.json round-trip. `mesh.source: splats` then
fuses `splats/splats.zarr` (rendered depth, alpha as confidence) instead of `feedforward.zarr`.
Every `splats/` artifact is in the COLMAP world frame; nothing is normalised.
```

Add "Re-running one stage" list: `splats` is a leaf stage (`--stages splats`, `--stages mesh` with `source: splats`).

- [ ] **Step 2: CLAUDE.md**

Arch tree: add under `collab_splats/`:
```
  splats/                  # Gaussian-splat training on upstream gsplat (trainer, losses, vendored cameras)
```
Add a "Recently completed (2026-08-22): **splats-module**" paragraph (≤6 lines): nerfstudio/gsplat-rade retired, `splats` leaf stage, `mesh.source`, depth targets from feedforward depth (spec deviation), 3DGS normals via `extra_signals` unmeasured (owed), PAGaS follow-on via `_render()`.

- [ ] **Step 3: Record the deviation in the spec**

In `docs/superpowers/specs/2026-08-22-splats-module-design.md`, find the depth-loss target sentence (grep `points3D` / `tracks`) and replace it with: *Depth targets come from `feedforward.zarr` depth masked by `confidence_mask(conf, mesh.conf_percentile)`, resized nearest to frame resolution — feedforward COLMAP models carry no Point2D tracks (`build_pycolmap_reconstruction` adds points with empty tracks), so the track path would be inert. `train()` receives them as an optional `(N, h, w)` array (0 = no target).*

- [ ] **Step 4: Gates**

```bash
$PY -m collab_splats.dashboard --smoke          # must print SMOKE PASS
$PY -m pytest tests/ -x -q -p no:randomly         # full suite (CUDA tests run; no tmux eval concurrently)
grep -rn 'gsplat-rade\|nerfstudio' --include='*.py' --include='*.md' --include='*.yaml' --include='*.toml' . \
  | grep -v 'docs/superpowers\|graphify-out\|update_kernelspecs\|evals/datasets.py\|known-test-failures'
graphify update .
```
Expected: `SMOKE PASS`; suite green except entries already in `docs/known-test-failures.md`; the grep prints nothing.

- [ ] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-08-22-splats-module-design.md
git commit --only configs/README.md CLAUDE.md docs/superpowers/specs/2026-08-22-splats-module-design.md graphify-out \
  -m "docs(splats): config keys, processed-scene layout, spec deviation (depth targets from feedforward depth)"
```

---

## Owed after this plan (not in scope)

- Measured report: 3DGS `extra_signals` normals + `depth_to_normal` consistency vs 2DGS on `data/tutorial/` (psnr/ssim + mesh vertex count/components).
- `evals/scripts/eval_splats.py`, PAGaS (gsplat `bd64a47` + CUDA patch, seam = `_render`), feature distillation, dense mono priors.
- `cfg.losses` default includes `distortion: weight 0.0` for discoverability only — `SplatsConfig()` is valid with 3dgs because the check is `weight > 0`.
