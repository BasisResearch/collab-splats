# Scaffold-GS on gsplat — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Scaffold-GS as a `splats.representation: scaffold` axis — anchors + MLP-decoded neural Gaussians — feeding the existing gsplat `rasterization` (3dgs) and `rasterization_2dgs` (2dgs) paths.

**Architecture:** One new file `collab_splats/splats/scaffold.py` holds the config, three MLP heads, the `AnchorField` (parameters + init + per-view `decode`), and `AnchorStrategy` (gsplat `Strategy` subclass doing anchor growing/pruning). `rendering.py` splits "produce Gaussians for this view" from "rasterize" so both representations share one rasterize path. `trainer.py` branches on `representation` for init, optimizers and strategy; `outputs.py` bakes a viewer-loadable ply from the view-dependent decode.

**Tech Stack:** PyTorch 2.x, gsplat @ `d2f5c0f` (`rasterization`, `rasterization_2dgs`, `strategy.base.Strategy`, `strategy.ops`), zarr 3.1.5, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-scaffold-gs-gsplat-design.md`

**Python:** every command below uses `/opt/venv/reconstruction/bin/python` (py3.11). `python` in the base shell is the wrong interpreter.

**Licensing:** Scaffold-GS is Inria-licensed (non-commercial) and GS-SR has no licence at all. Reimplement from the paper; **copy no source**. Every ported concept gets a comment citing `repo + commit + file + line`, as elsewhere in this codebase.

---

### Task 1: Split the render seam (pure refactor, no behavior change)

`render_view` currently activates raw parameters and rasterizes in one function. Split it so a scaffold decode can feed the same rasterizer. Behavior must be byte-identical for vanilla.

**Files:**
- Modify: `collab_splats/splats/rendering.py:44-160`
- Test: `tests/splats/test_rendering.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/splats/test_rendering.py`:

```python
@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_activated_dict_renders_identically_to_parameter_dict(primitive):
    """render_gaussians on a pre-activated dict == render_view on the raw ParameterDict."""
    from collab_splats.splats.rendering import activate_vanilla, render_gaussians

    cam_to_world, intrinsics = _camera()
    gaussians = _gaussians()
    reference, _ = render_view(primitive, gaussians, cam_to_world, intrinsics, 64, 64, sh_degree=0, absgrad=False)
    decoded = activate_vanilla(gaussians)
    actual, _ = render_gaussians(primitive, decoded, cam_to_world, intrinsics, 64, 64, sh_degree=0, absgrad=False)
    for key, expected in reference.items():
        assert torch.equal(actual[key], expected), key
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_rendering.py::test_activated_dict_renders_identically_to_parameter_dict -v`
Expected: FAIL with `ImportError: cannot import name 'activate_vanilla'`

- [ ] **Step 3: Implement the split**

In `collab_splats/splats/rendering.py`, add above `render_view`:

```python
def activate_vanilla(gaussians: torch.nn.ParameterDict) -> dict[str, Tensor]:
    """
    Activate raw vanilla parameters into the tensors the rasterizer takes.

    - log-scales -> scales, logit-opacities -> opacities, SH bands concatenated.
    - `colors` here are SH coefficients: the caller passes `sh_degree` to the rasterizer.
    """
    return {
        "means": gaussians["means"],
        "quats": gaussians["quats"],
        "scales": torch.exp(gaussians["scales"]),
        "opacities": torch.sigmoid(gaussians["opacities"]),
        "colors": torch.cat([gaussians["sh0"], gaussians["shN"]], dim=1),
    }
```

Rename the body of `render_view` to `render_gaussians`, taking `decoded: dict[str, Tensor]` in place of
`gaussians: torch.nn.ParameterDict`, and replace the activation block at the top with:

```python
    means = decoded["means"]
    quats = decoded["quats"]
    scales = decoded["scales"]
    opacities = decoded["opacities"]
    colors = decoded["colors"]
    world_to_cam = torch.linalg.inv(cam_to_world)
    shared_kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=world_to_cam,
        Ks=intrinsics,
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
        absgrad=absgrad,
        render_mode="RGB+ED",
    )
```

Then make `render_view` a thin wrapper that keeps the existing signature and call sites working:

```python
def render_view(
    primitive: str,
    gaussians: torch.nn.ParameterDict,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    sh_degree: int,
    absgrad: bool,
    render_normals: bool = True,
) -> tuple[dict[str, Tensor], dict]:
    """
    Render one camera from raw vanilla parameters (activate, then rasterize).
    """
    return render_gaussians(
        primitive,
        activate_vanilla(gaussians),
        cam_to_world,
        intrinsics,
        width,
        height,
        sh_degree,
        absgrad,
        render_normals=render_normals,
    )
```

`sh_degree=None` must reach the rasterizer when `colors` are post-activation RGB — scaffold uses that in
Task 6, so `render_gaussians` passes `sh_degree` straight through with no clamping of its own.

- [ ] **Step 4: Run the whole rendering suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_rendering.py -v`
Expected: PASS, all tests including the pre-existing shape/normal ones.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/splats/rendering.py tests/splats/test_rendering.py
git commit --only collab_splats/splats/rendering.py tests/splats/test_rendering.py -m "refactor(splats): split gaussian activation from rasterization"
```

---

### Task 2: ScaffoldConfig and the representation axis

**Files:**
- Create: `collab_splats/splats/scaffold.py`
- Modify: `collab_splats/splats/trainer.py:74-190` (`SplatsConfig` fields + `from_dict` validation)
- Test: `tests/splats/test_scaffold.py` (create), `tests/splats/test_trainer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/splats/test_scaffold.py`:

```python
"""
Scaffold-GS anchors: config, MLP heads, decode, and anchor densification.
"""

import numpy as np
import pytest
import torch

from collab_splats.splats.scaffold import ScaffoldConfig
from collab_splats.splats.trainer import SplatsConfig

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


def test_representation_defaults_to_vanilla():
    cfg = SplatsConfig()
    assert cfg.representation == "vanilla"
    assert cfg.scaffold is None


def test_scaffold_block_parses_into_scaffold_config():
    cfg = SplatsConfig.from_dict({"representation": "scaffold", "scaffold": {"n_offsets": 5, "feat_dim": 16}})
    assert isinstance(cfg.scaffold_config, ScaffoldConfig)
    assert cfg.scaffold_config.n_offsets == 5
    assert cfg.scaffold_config.feat_dim == 16


def test_unknown_representation_is_rejected():
    with pytest.raises(ValueError, match="representation"):
        SplatsConfig.from_dict({"representation": "octree"})


def test_unknown_scaffold_key_is_rejected():
    with pytest.raises(ValueError, match="n_offset"):
        SplatsConfig.from_dict({"representation": "scaffold", "scaffold": {"n_offset": 5}})


def test_scaffold_block_without_scaffold_representation_is_rejected():
    with pytest.raises(ValueError, match="representation: scaffold"):
        SplatsConfig.from_dict({"scaffold": {"n_offsets": 5}})


def test_sh_degree_is_rejected_under_scaffold():
    with pytest.raises(ValueError, match="sh_degree"):
        SplatsConfig.from_dict({"representation": "scaffold", "sh_degree": 3})
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'collab_splats.splats.scaffold'`

- [ ] **Step 3: Write `scaffold.py`'s config section**

Create `collab_splats/splats/scaffold.py`:

```python
"""
Scaffold-GS anchors on gsplat: anchor parameters, MLP decode heads, and anchor densification.

Reimplemented from the Scaffold-GS paper (Lu et al., CVPR 2024). The reference implementation
(city-super/Scaffold-GS) is under the Inria/MPII Gaussian-Splatting licence and is NOT vendored —
each ported concept cites its upstream site for provenance only.

Anchors carry a feature vector; three MLP heads decode `n_offsets` neural Gaussians per anchor per
view, which then go through the same gsplat rasterizer the vanilla representation uses. Densification
operates on anchors (voxel growing / opacity pruning), not on the decoded Gaussians.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import _update_param_with_optimizer
from torch import Tensor

logger = logging.getLogger(__name__)

########################################
# Config
########################################


@dataclass
class ScaffoldConfig:
    """
    Anchor-representation knobs; mirrors the ``splats.scaffold:`` yaml block.
    """

    n_offsets: int = 10
    feat_dim: int = 32

    # Anchor voxel size = median kNN spacing of the seed points x voxel_multiplier, so it is
    # scale-free under normalize_scene either way. voxel_size overrides it in world units.
    voxel_multiplier: float = 1.0
    voxel_size: float | None = None

    # Densification window and thresholds. grad_threshold is Scaffold's published value and is
    # directly comparable because AnchorStrategy renormalizes gradients the way gsplat's
    # DefaultStrategy does (strategy/default.py:243-249).
    update_from: int = 1500
    update_until: int = 15000
    refine_every: int = 100
    grad_threshold: float = 2e-4
    min_opacity: float = 0.005
    update_depth: int = 3  # coarse-to-fine growing levels
    update_hierarchy_factor: int = 4

    # Scaffold's own per-image appearance embedding, concatenated into the color MLP input.
    # 0 disables it. Independent of splats.appearance_opt (our per-image affine module):
    # both ship, neither retires the other, and the 2x2 is measured.
    appearance_dim: int = 0

    # Learning rates (anchor / offset lrs are multiplied by scene_scale like means_lr)
    anchor_lr: float = 1.6e-4
    offset_lr: float = 1e-2
    anchor_feat_lr: float = 7.5e-3
    scaling_lr: float = 7e-3
    rotation_lr: float = 2e-3
    opacities_lr: float = 2e-2
    mlp_lr: float = 2e-3

    @classmethod
    def from_dict(cls, block: dict) -> "ScaffoldConfig":
        """
        Build from the yaml block; rejects unknown keys (they are almost always typos).
        """
        unknown = set(block) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"splats.scaffold: unknown keys {sorted(unknown)}; allowed {sorted(cls.__dataclass_fields__)}"
            )
        cfg = cls(**block)
        if cfg.n_offsets < 1:
            raise ValueError(f"splats.scaffold.n_offsets must be >= 1, got {cfg.n_offsets}")
        if cfg.voxel_size is not None and cfg.voxel_size <= 0:
            raise ValueError(f"splats.scaffold.voxel_size must be > 0 when set, got {cfg.voxel_size}")
        return cfg
```

- [ ] **Step 4: Wire the representation axis into `SplatsConfig`**

In `collab_splats/splats/trainer.py`, import at the top (`from collab_splats.splats.scaffold import ScaffoldConfig`)
and add next to `PRIMITIVES`:

```python
REPRESENTATIONS = ("vanilla", "scaffold")
```

Add the fields to `SplatsConfig` (next to `primitive`):

```python
    representation: str = "vanilla"  # vanilla | scaffold
    scaffold: dict | None = None  # scaffold block; only with representation: scaffold
```

Add a non-field attribute set in `__post_init__` so the parsed config travels with the dataclass:

```python
        # Parsed scaffold block. Kept off the dataclass fields so asdict(cfg) — which lands in
        # ckpt.pt and the zarr attrs — stays plain yaml-shaped data.
        self.scaffold_config = None if self.scaffold is None else ScaffoldConfig.from_dict(self.scaffold)
```

In `from_dict`, after the primitive check:

```python
        # Representation and its block: a scaffold block without the representation is a silent no-op
        if cfg.representation not in REPRESENTATIONS:
            raise ValueError(f"splats.representation must be one of {REPRESENTATIONS}, got '{cfg.representation}'")
        if cfg.scaffold is not None and cfg.representation != "scaffold":
            raise ValueError("splats.scaffold requires representation: scaffold")

        # Scaffold decodes RGB from an MLP, so the SH schedule has nothing to act on
        if cfg.representation == "scaffold" and ("sh_degree" in block or "sh_degree_interval" in block):
            raise ValueError(
                "splats.sh_degree / sh_degree_interval are vanilla-only; scaffold decodes RGB from mlp_color"
            )
```

- [ ] **Step 5: Run the tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py tests/splats/test_trainer.py -v`
Expected: PASS (all six new tests, plus the existing trainer config tests unchanged).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/splats/scaffold.py collab_splats/splats/trainer.py tests/splats/test_scaffold.py
git commit --only collab_splats/splats/scaffold.py collab_splats/splats/trainer.py tests/splats/test_scaffold.py \
  -m "feat(splats): scaffold representation axis and ScaffoldConfig"
```

---

### Task 3: MLP decode heads

**Files:**
- Modify: `collab_splats/splats/scaffold.py` (new `######## MLP heads` section)
- Test: `tests/splats/test_scaffold.py`

- [ ] **Step 1: Write the failing test**

```python
def test_mlp_heads_emit_per_offset_outputs():
    from collab_splats.splats.scaffold import ScaffoldMLPs

    cfg = ScaffoldConfig(n_offsets=4, feat_dim=8)
    mlps = ScaffoldMLPs(cfg)
    features = torch.zeros(6, cfg.feat_dim + 4)  # feat + view dir (3) + view distance (1)
    opacity, cov, color = mlps(features, camera_id=None)
    assert opacity.shape == (6, 4)
    assert cov.shape == (6, 4 * 7)
    assert color.shape == (6, 4 * 3)
    assert opacity.min() >= -1.0 and opacity.max() <= 1.0  # tanh
    assert color.min() >= 0.0 and color.max() <= 1.0  # sigmoid


def test_appearance_embedding_changes_color_only_when_enabled():
    from collab_splats.splats.scaffold import ScaffoldMLPs

    features = torch.zeros(3, 8 + 4)
    camera_id = torch.zeros(3, dtype=torch.long)

    off = ScaffoldMLPs(ScaffoldConfig(n_offsets=2, feat_dim=8, appearance_dim=0), n_views=5)
    assert off.embedding_appearance is None
    off(features, camera_id)  # camera_id is accepted and ignored

    on = ScaffoldMLPs(ScaffoldConfig(n_offsets=2, feat_dim=8, appearance_dim=6), n_views=5)
    assert on.embedding_appearance is not None
    assert on.embedding_appearance.weight.shape == (5, 6)
    with pytest.raises(ValueError, match="camera_id"):
        on(features, camera_id=None)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -k mlp -v`
Expected: FAIL with `ImportError: cannot import name 'ScaffoldMLPs'`

- [ ] **Step 3: Implement the heads**

Append to `collab_splats/splats/scaffold.py`:

```python
########################################
# MLP heads
########################################

VIEW_DIM = 4  # unit view direction (3) + view distance (1)


class ScaffoldMLPs(torch.nn.Module):
    """
    The three Scaffold-GS decode heads: opacity, covariance, color.

    - Input is [anchor_feat, view_dir, view_dist] per visible anchor; every head emits one row of
      ``n_offsets`` outputs per anchor.
    - opacity ends in tanh (its sign is the offset visibility mask), color in sigmoid (RGB),
      covariance is raw (3 log-ish scale factors + 4 quaternion components per offset).
    - ``appearance_dim > 0`` adds Scaffold's per-image embedding to the color head input only.

    Head shapes follow city-super/Scaffold-GS scene/gaussian_model.py (MLP definitions); weights and
    code are not copied.
    """

    def __init__(self, cfg: ScaffoldConfig, n_views: int = 0):
        super().__init__()
        self.n_offsets = cfg.n_offsets
        width = cfg.feat_dim
        base_dim = cfg.feat_dim + VIEW_DIM

        self.mlp_opacity = torch.nn.Sequential(
            torch.nn.Linear(base_dim, width),
            torch.nn.ReLU(True),
            torch.nn.Linear(width, cfg.n_offsets),
            torch.nn.Tanh(),
        )
        self.mlp_cov = torch.nn.Sequential(
            torch.nn.Linear(base_dim, width),
            torch.nn.ReLU(True),
            torch.nn.Linear(width, 7 * cfg.n_offsets),
        )

        # Scaffold's appearance embedding rides the color head only
        self.embedding_appearance = None
        color_dim = base_dim
        if cfg.appearance_dim > 0:
            if n_views < 1:
                raise ValueError("scaffold.appearance_dim > 0 needs n_views >= 1 to size the embedding")
            self.embedding_appearance = torch.nn.Embedding(n_views, cfg.appearance_dim)
            torch.nn.init.zeros_(self.embedding_appearance.weight)
            color_dim += cfg.appearance_dim
        self.mlp_color = torch.nn.Sequential(
            torch.nn.Linear(color_dim, width),
            torch.nn.ReLU(True),
            torch.nn.Linear(width, 3 * cfg.n_offsets),
            torch.nn.Sigmoid(),
        )

    def forward(self, features: Tensor, camera_id: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Decode (opacity, covariance, color) for every visible anchor. Shapes [A, K], [A, 7K], [A, 3K].
        """
        opacity = self.mlp_opacity(features)
        cov = self.mlp_cov(features)

        # The embedding is per-image, so the color head needs to know which view is being rendered
        color_input = features
        if self.embedding_appearance is not None:
            if camera_id is None:
                raise ValueError("scaffold.appearance_dim > 0 requires camera_id at decode time")
            embedding = self.embedding_appearance(camera_id[:1]).expand(len(features), -1)
            color_input = torch.cat([features, embedding], dim=-1)
        color = self.mlp_color(color_input)
        return opacity, cov, color
```

- [ ] **Step 4: Run the tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -k mlp -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/splats/scaffold.py tests/splats/test_scaffold.py
git commit --only collab_splats/splats/scaffold.py tests/splats/test_scaffold.py \
  -m "feat(splats): scaffold MLP decode heads"
```

---

### Task 4: AnchorField parameters and voxel init

**Files:**
- Modify: `collab_splats/splats/scaffold.py` (new `######## Anchor field` section)
- Test: `tests/splats/test_scaffold.py`

- [ ] **Step 1: Write the failing tests**

```python
def _seed_points(n=500, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    colors = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
    return points, colors


def test_anchor_init_voxelizes_seed_points():
    from collab_splats.splats.scaffold import AnchorField

    cfg = ScaffoldConfig(n_offsets=4, feat_dim=8)
    points, colors = _seed_points()
    field = AnchorField(cfg, points, colors, scene_scale=1.0, n_views=3, device="cpu")
    n_anchors = len(field.params["anchors"])
    assert 0 < n_anchors <= len(points)
    assert field.params["offsets"].shape == (n_anchors, 4, 3)
    assert field.params["anchor_feat"].shape == (n_anchors, 8)
    assert field.params["scaling"].shape == (n_anchors, 6)
    assert field.params["rotation"].shape == (n_anchors, 4)
    assert field.params["opacities"].shape == (n_anchors, 1)
    assert field.voxel_size > 0


def test_anchor_count_is_invariant_to_scene_scale():
    """voxel_size is derived from kNN spacing, so a 10x bigger copy of a scene gets the same anchors."""
    from collab_splats.splats.scaffold import AnchorField

    cfg = ScaffoldConfig(n_offsets=2, feat_dim=8)
    points, colors = _seed_points()
    small = AnchorField(cfg, points, colors, scene_scale=1.0, n_views=1, device="cpu")
    large = AnchorField(cfg, points * 10.0, colors, scene_scale=10.0, n_views=1, device="cpu")
    assert len(large.params["anchors"]) == len(small.params["anchors"])
    assert large.voxel_size == pytest.approx(small.voxel_size * 10.0, rel=1e-5)


def test_explicit_voxel_size_overrides_the_derived_one():
    from collab_splats.splats.scaffold import AnchorField

    points, colors = _seed_points()
    field = AnchorField(
        ScaffoldConfig(n_offsets=2, feat_dim=8, voxel_size=0.5), points, colors, scene_scale=1.0, n_views=1, device="cpu"
    )
    assert field.voxel_size == 0.5


def test_optimizers_cover_every_anchor_parameter():
    from collab_splats.splats.scaffold import AnchorField

    points, colors = _seed_points()
    field = AnchorField(ScaffoldConfig(n_offsets=2, feat_dim=8), points, colors, 1.0, n_views=1, device="cpu")
    assert set(field.optimizers) == set(field.params)
    for optimizer in field.optimizers.values():
        assert len(optimizer.param_groups) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -k anchor_init -v`
Expected: FAIL with `ImportError: cannot import name 'AnchorField'`

- [ ] **Step 3: Implement the field and its init**

Append to `collab_splats/splats/scaffold.py`:

```python
########################################
# Anchor field
########################################


def voxelize(points: Tensor, voxel_size: float) -> Tensor:
    """
    One representative point per occupied voxel: round to the grid, dedup, return grid centers.
    """
    grid_coords = torch.round(points / voxel_size)
    unique_coords = torch.unique(grid_coords, dim=0)
    return unique_coords * voxel_size


class AnchorField:
    """
    Scaffold-GS anchors: the trainable state plus the per-view decode into neural Gaussians.

    - ``params`` is the ParameterDict the densification strategy grows and prunes; the MLP heads live
      in ``mlps`` and are fixed-size, so they are optimized separately and never handed to the strategy.
    - ``decode`` returns the rasterizer inputs plus ``decode_index``, the (anchor * n_offsets + offset)
      slot each emitted Gaussian came from. Anchor densification is built entirely on that index.
    """

    def __init__(
        self,
        cfg: ScaffoldConfig,
        points: np.ndarray,
        colors: np.ndarray,
        scene_scale: float,
        n_views: int,
        device: str,
    ):
        self.cfg = cfg
        self.device = device

        # Voxel size from the seed points' own spacing: scale-free, so normalize_scene cannot
        # silently change anchor density (Scaffold's absolute 0.001 default assumes a normalized scene)
        points_t = torch.from_numpy(np.asarray(points, dtype=np.float32))
        if cfg.voxel_size is not None:
            self.voxel_size = float(cfg.voxel_size)
        else:
            self.voxel_size = float(median_knn_spacing(points_t) * cfg.voxel_multiplier)

        anchors = voxelize(points_t, self.voxel_size).to(device)
        n_anchors = len(anchors)
        logger.info(
            "scaffold: %d seed points -> %d anchors at voxel_size %.6g", len(points_t), n_anchors, self.voxel_size
        )

        # Offsets start at zero (upstream: new anchors are seeded with zero offsets), scaling holds
        # the offset extent (first 3) and the decoded-Gaussian extent (last 3), both in log space
        log_voxel = math.log(self.voxel_size)
        self.params = torch.nn.ParameterDict(
            {
                "anchors": torch.nn.Parameter(anchors),
                "offsets": torch.nn.Parameter(torch.zeros(n_anchors, cfg.n_offsets, 3, device=device)),
                "anchor_feat": torch.nn.Parameter(torch.zeros(n_anchors, cfg.feat_dim, device=device)),
                "scaling": torch.nn.Parameter(torch.full((n_anchors, 6), log_voxel, device=device)),
                "rotation": torch.nn.Parameter(
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n_anchors, 1)
                ),
                "opacities": torch.nn.Parameter(torch.zeros(n_anchors, 1, device=device)),
            }
        )

        self.mlps = ScaffoldMLPs(cfg, n_views=n_views).to(device)

        # One Adam per anchor tensor so the strategy can grow/prune optimizer state per tensor
        learning_rates = {
            "anchors": cfg.anchor_lr * scene_scale,
            "offsets": cfg.offset_lr * scene_scale,
            "anchor_feat": cfg.anchor_feat_lr,
            "scaling": cfg.scaling_lr,
            "rotation": cfg.rotation_lr,
            "opacities": cfg.opacities_lr,
        }
        self.optimizers = {
            name: torch.optim.Adam([{"params": self.params[name], "lr": lr, "name": name}], eps=1e-15)
            for name, lr in learning_rates.items()
        }
        self.mlp_optimizer = torch.optim.Adam(self.mlps.parameters(), lr=cfg.mlp_lr, eps=1e-15)
```

Add `median_knn_spacing` to the same section, and refactor `trainer.init_gaussians_from_points` to call it
so both representations share one definition:

```python
def median_knn_spacing(points: Tensor, k: int = 3) -> float:
    """
    Median mean-distance to the k nearest neighbors — the seed points' own length scale.
    """
    from sklearn.neighbors import NearestNeighbors

    array = points.detach().cpu().numpy()
    neighbor_dists, _ = NearestNeighbors(n_neighbors=k + 1).fit(array).kneighbors(array)
    return float(np.median(np.sqrt((neighbor_dists[:, 1:] ** 2).mean(-1))))
```

This is the one place in the codebase where a heavy import sits inside a function: `sklearn` is already a
hard dependency and imported at the top of `trainer.py`, so import it at the top of `scaffold.py` too and
delete the inline import — the inline form above is shown only to mark where it goes.

- [ ] **Step 4: Run the tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -k "anchor_init or scene_scale or voxel_size or optimizers" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/splats/scaffold.py tests/splats/test_scaffold.py
git commit --only collab_splats/splats/scaffold.py tests/splats/test_scaffold.py \
  -m "feat(splats): scaffold anchor field with spacing-derived voxel init"
```

---

### Task 5: `AnchorField.decode` — neural Gaussians and `decode_index`

**Files:**
- Modify: `collab_splats/splats/scaffold.py`
- Test: `tests/splats/test_scaffold.py`

- [ ] **Step 1: Write the failing tests**

```python
def _field(n_offsets=4, appearance_dim=0, n_views=3):
    from collab_splats.splats.scaffold import AnchorField

    cfg = ScaffoldConfig(n_offsets=n_offsets, feat_dim=8, appearance_dim=appearance_dim)
    points, colors = _seed_points(n=200)
    return AnchorField(cfg, points, colors, scene_scale=1.0, n_views=n_views, device="cpu")


def _cam(device="cpu"):
    cam_to_world = torch.eye(4, device=device)[None]
    cam_to_world[0, 2, 3] = -4.0
    intrinsics = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device)[None]
    return cam_to_world, intrinsics


def test_decode_returns_rasterizer_inputs_and_index():
    field = _field()
    cam_to_world, intrinsics = _cam()
    decoded, index = field.decode("3dgs", cam_to_world, intrinsics, width=64, height=64, camera_id=None)
    n = len(decoded["means"])
    assert decoded["means"].shape == (n, 3)
    assert decoded["quats"].shape == (n, 4)
    assert decoded["scales"].shape == (n, 3)
    assert decoded["opacities"].shape == (n,)
    assert decoded["colors"].shape == (n, 3)  # post-activation RGB, so sh_degree=None at rasterize
    assert decoded["log_scales"].shape == (n, 3)  # scale_reg reads this instead of a parameter
    assert index.shape == (n,)
    assert index.dtype == torch.int64
    assert int(index.max()) < len(field.params["anchors"]) * field.cfg.n_offsets


def test_decode_index_points_at_the_generating_anchor():
    field = _field(n_offsets=2)
    cam_to_world, intrinsics = _cam()
    decoded, index = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    anchor_ids = index // field.cfg.n_offsets

    # Each decoded mean is its anchor plus a scaled offset, so it must sit within the offset extent
    anchors = field.params["anchors"][anchor_ids]
    offset_extent = torch.exp(field.params["scaling"][anchor_ids][:, :3])
    displacement = (decoded["means"] - anchors).abs()
    assert torch.all(displacement <= offset_extent * 1.001 + 1e-6)


def test_decode_drops_offsets_with_non_positive_opacity():
    field = _field(n_offsets=4)
    cam_to_world, intrinsics = _cam()

    # Force every opacity negative: the tanh head's last bias dominates a zeroed feature input
    with torch.no_grad():
        field.mlps.mlp_opacity[-2].weight.zero_()
        field.mlps.mlp_opacity[-2].bias.fill_(-5.0)
    decoded, index = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    assert len(decoded["means"]) == 0
    assert len(index) == 0


def test_decode_is_differentiable_into_the_mlps():
    field = _field()
    cam_to_world, intrinsics = _cam()
    decoded, _ = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    decoded["colors"].sum().backward()
    assert field.mlps.mlp_color[0].weight.grad is not None
    assert field.params["anchor_feat"].grad is not None


def test_decode_2dgs_zeroes_the_third_scale():
    field = _field()
    cam_to_world, intrinsics = _cam()
    decoded, _ = field.decode("2dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    assert torch.all(decoded["scales"][:, 2] == 0.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -k decode -v`
Expected: FAIL with `AttributeError: 'AnchorField' object has no attribute 'decode'`

- [ ] **Step 3: Implement `decode`**

Add to `AnchorField` in `collab_splats/splats/scaffold.py`:

```python
    def visible_anchors(self, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int) -> Tensor:
        """
        Boolean mask of anchors whose center projects in front of the camera and inside the frame.

        - Deviation from upstream, which reuses the rasterizer's own prefilter pass. A projection test
          is cheaper and needs no extra rasterization; the margin keeps anchors whose Gaussians spill
          into frame from centers just outside it.
        """
        world_to_cam = torch.linalg.inv(cam_to_world)[0]
        anchors_cam = self.params["anchors"] @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
        depth = anchors_cam[:, 2]
        in_front = depth > 1e-3

        # Project with the frame's K; a whole-frame margin keeps off-center anchors that still splat in
        safe_depth = depth.clamp_min(1e-3)
        projected = (anchors_cam[:, :2] / safe_depth[:, None]) @ intrinsics[0, :2, :2].T + intrinsics[0, :2, 2]
        margin_x, margin_y = width * 0.5, height * 0.5
        in_frame = (
            (projected[:, 0] > -margin_x)
            & (projected[:, 0] < width + margin_x)
            & (projected[:, 1] > -margin_y)
            & (projected[:, 1] < height + margin_y)
        )
        return in_front & in_frame

    def decode(
        self,
        primitive: str,
        cam_to_world: Tensor,
        intrinsics: Tensor,
        width: int,
        height: int,
        camera_id: Tensor | None,
    ) -> tuple[dict[str, Tensor], Tensor]:
        """
        Decode this view's neural Gaussians. Returns (rasterizer inputs, decode_index).

        - ``decode_index`` is ``anchor_index * n_offsets + offset_index`` per emitted Gaussian.
        - ``colors`` are post-activation RGB, so the caller rasterizes with ``sh_degree=None``.
        - ``log_scales`` carries the decoded scales in log space for the scale regularizer.

        Follows generate_neural_gaussians in city-super/Scaffold-GS scene/gaussian_model.py
        (reimplemented; no code copied).
        """
        n_offsets = self.cfg.n_offsets
        visible = self.visible_anchors(cam_to_world, intrinsics, width, height)
        anchor_ids = torch.nonzero(visible, as_tuple=False).squeeze(-1)
        anchors = self.params["anchors"][anchor_ids]
        feat = self.params["anchor_feat"][anchor_ids]
        scaling = torch.exp(self.params["scaling"][anchor_ids])
        offsets = self.params["offsets"][anchor_ids]

        # View direction and distance from each anchor to the camera center feed every head
        camera_center = cam_to_world[0, :3, 3]
        to_camera = anchors - camera_center
        view_distance = to_camera.norm(dim=-1, keepdim=True)
        view_direction = to_camera / view_distance.clamp_min(1e-8)
        features = torch.cat([feat, view_direction, view_distance], dim=-1)

        neural_opacity, cov, color = self.mlps(features, camera_id)

        # Offsets with non-positive opacity contribute nothing: dropping them here is what keeps the
        # decoded count far below anchors x n_offsets
        keep = (neural_opacity > 0).reshape(-1)
        slot_index = (anchor_ids[:, None] * n_offsets + torch.arange(n_offsets, device=anchors.device)).reshape(-1)
        decode_index = slot_index[keep]

        # means = anchor + offset scaled by the anchor's offset extent (scaling[:, :3])
        means = (anchors[:, None, :] + offsets * scaling[:, None, :3]).reshape(-1, 3)[keep]

        # scales = the anchor's gaussian extent (scaling[:, 3:6]) modulated per offset
        cov = cov.reshape(-1, 7)[keep]
        scales = scaling[:, 3:6].repeat_interleave(n_offsets, dim=0)[keep] * torch.sigmoid(cov[:, :3])
        quats = F.normalize(cov[:, 3:7], dim=-1)
        opacities = neural_opacity.reshape(-1)[keep]
        colors = color.reshape(-1, 3)[keep]

        # 2DGS reads scales[..., :2]; zero the unused third channel so it can never be misread
        if primitive == "2dgs":
            scales = torch.cat([scales[:, :2], torch.zeros_like(scales[:, :1])], dim=-1)
            log_scales = torch.log(scales[:, :2].clamp_min(1e-12))
            log_scales = torch.cat([log_scales, torch.zeros_like(scales[:, :1])], dim=-1)
        else:
            log_scales = torch.log(scales.clamp_min(1e-12))

        decoded = {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
            "log_scales": log_scales,
        }
        return decoded, decode_index
```

- [ ] **Step 4: Run the tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -k decode -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/splats/scaffold.py tests/splats/test_scaffold.py
git commit --only collab_splats/splats/scaffold.py tests/splats/test_scaffold.py \
  -m "feat(splats): scaffold per-view decode into neural gaussians"
```

---

### Task 6: Render scaffold through the existing rasterizers

**Files:**
- Modify: `collab_splats/splats/rendering.py` (accept `sh_degree=None` for post-activation colors)
- Test: `tests/splats/test_scaffold.py`

- [ ] **Step 1: Write the failing test**

```python
@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_scaffold_decode_renders_through_gsplat(primitive):
    from collab_splats.splats.rendering import render_gaussians

    field = _field()
    field.params = field.params.to("cuda")
    field.mlps = field.mlps.to("cuda")
    cam_to_world, intrinsics = _cam(device="cuda")
    decoded, _ = field.decode(primitive, cam_to_world, intrinsics, 64, 64, camera_id=None)
    render, info = render_gaussians(
        primitive, decoded, cam_to_world, intrinsics, 64, 64, sh_degree=None, absgrad=False
    )
    assert render["rgb"].shape == (1, 64, 64, 3)
    assert render["depth"].shape == (1, 64, 64, 1)
    expected_gradient_key = "means2d" if primitive == "3dgs" else "gradient_2dgs"
    assert expected_gradient_key in info


@cuda
def test_render_gradient_reaches_the_anchor_features():
    from collab_splats.splats.rendering import render_gaussians

    field = _field()
    field.params = field.params.to("cuda")
    field.mlps = field.mlps.to("cuda")
    cam_to_world, intrinsics = _cam(device="cuda")
    decoded, _ = field.decode("3dgs", cam_to_world, intrinsics, 64, 64, camera_id=None)
    render, _ = render_gaussians("3dgs", decoded, cam_to_world, intrinsics, 64, 64, sh_degree=None, absgrad=False)
    render["rgb"].sum().backward()
    assert field.params["anchor_feat"].grad is not None
    assert torch.isfinite(field.params["anchor_feat"].grad).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -k "renders_through_gsplat or gradient_reaches" -v`
Expected: FAIL — `rasterization` raises on the extra `log_scales` key or on SH-shaped color handling.

- [ ] **Step 3: Make `render_gaussians` tolerate decode extras**

In `collab_splats/splats/rendering.py`, inside `render_gaussians`, read only the five rasterizer keys
explicitly (`means`, `quats`, `scales`, `opacities`, `colors`) — `log_scales` and anything else in the
decoded dict is ignored by construction. Add the doc line:

```python
    - `sh_degree=None` with `colors` of shape (N, 3) rasterizes post-activation RGB; that is the
      scaffold path. Vanilla passes SH coefficients and an integer degree.
```

3DGS normals are computed from `quats`/`scales`/`means`, which the decoded dict supplies, so the
extra-signal path needs no change.

- [ ] **Step 4: Run the tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py tests/splats/test_rendering.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/splats/rendering.py tests/splats/test_scaffold.py
git commit --only collab_splats/splats/rendering.py tests/splats/test_scaffold.py \
  -m "feat(splats): rasterize scaffold-decoded gaussians"
```

---

### Task 7: AnchorStrategy — gradient accumulation

**Files:**
- Modify: `collab_splats/splats/scaffold.py` (new `######## Densification` section)
- Test: `tests/splats/test_scaffold.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_gradient_key_follows_the_primitive():
    from collab_splats.splats.scaffold import AnchorStrategy

    assert AnchorStrategy(ScaffoldConfig(), primitive="3dgs").key_for_gradient == "means2d"
    assert AnchorStrategy(ScaffoldConfig(), primitive="2dgs").key_for_gradient == "gradient_2dgs"


def test_accumulation_renormalizes_gradients_like_gsplat():
    """gsplat's DefaultStrategy scales means2d grads to [-1, 1] screen space before thresholding."""
    from collab_splats.splats.scaffold import AnchorStrategy

    cfg = ScaffoldConfig(n_offsets=2, feat_dim=8)
    strategy = AnchorStrategy(cfg, primitive="3dgs")
    state = strategy.initialize_state(n_slots=8)

    means2d = torch.zeros(1, 3, 2)
    means2d.grad = torch.tensor([[[1e-3, 0.0], [0.0, 2e-3], [0.0, 0.0]]])
    info = {"means2d": means2d, "width": 800, "height": 600, "n_cameras": 1}
    decode_index = torch.tensor([0, 5, 7])

    strategy.accumulate(state, info, decode_index, opacities=torch.tensor([0.5, 0.5, 0.5]))
    assert state["grad_accum"][0] == pytest.approx(1e-3 * 400.0)
    assert state["grad_accum"][5] == pytest.approx(2e-3 * 300.0)
    assert state["denom"][0] == 1
    assert state["denom"][1] == 0
    assert state["opacity_accum"][0] == pytest.approx(0.5)


def test_accumulation_is_additive_over_steps():
    from collab_splats.splats.scaffold import AnchorStrategy

    strategy = AnchorStrategy(ScaffoldConfig(n_offsets=2, feat_dim=8), primitive="3dgs")
    state = strategy.initialize_state(n_slots=4)
    means2d = torch.zeros(1, 1, 2)
    means2d.grad = torch.tensor([[[1e-3, 0.0]]])
    info = {"means2d": means2d, "width": 800, "height": 600, "n_cameras": 1}
    for _ in range(3):
        strategy.accumulate(state, info, torch.tensor([2]), opacities=torch.tensor([0.25]))
    assert state["denom"][2] == 3
    assert state["grad_accum"][2] == pytest.approx(3 * 1e-3 * 400.0)
    assert state["opacity_accum"][2] == pytest.approx(0.75)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -k "gradient_key or accumulat" -v`
Expected: FAIL with `ImportError: cannot import name 'AnchorStrategy'`

- [ ] **Step 3: Implement the strategy skeleton and accumulation**

Append to `collab_splats/splats/scaffold.py`:

```python
########################################
# Densification
########################################


class AnchorStrategy(Strategy):
    """
    Scaffold-GS anchor growing and pruning, in gsplat's Strategy shape.

    - Screen-space gradients are accumulated per (anchor, offset) slot through the decode index, then
      averaged by visit count — Scaffold's offset_gradient_accum / offset_denom.
    - Growing adds anchors in unoccupied voxels around high-gradient slots; pruning drops anchors whose
      accumulated decoded opacity stays below min_opacity.
    - Anchor tensors are grown/pruned through gsplat's own optimizer-state surgery, so Adam moments
      follow the parameters. The MLP heads are fixed-size and are never handed to this class.

    Reimplemented from Scaffold-GS (Lu et al., CVPR 2024) — adjust_anchor / anchor_growing /
    training_statis in city-super/Scaffold-GS scene/gaussian_model.py. No code copied.
    """

    def __init__(self, cfg: ScaffoldConfig, primitive: str, voxel_size: float = 0.0, verbose: bool = False):
        self.cfg = cfg
        self.primitive = primitive
        self.voxel_size = voxel_size
        self.verbose = verbose

        # 2DGS backward writes .absgrad on means2d only, never on the gradient_2dgs tensor
        # DefaultStrategy reads, so the key must follow the primitive or accumulation is all zeros
        self.key_for_gradient = "means2d" if primitive == "3dgs" else "gradient_2dgs"

    def initialize_state(self, n_slots: int) -> dict[str, Tensor]:
        """
        Per-slot accumulators; n_slots = n_anchors x n_offsets.
        """
        return {
            "grad_accum": torch.zeros(n_slots),
            "denom": torch.zeros(n_slots),
            "opacity_accum": torch.zeros(n_slots),
        }

    def accumulate(self, state: dict, info: dict, decode_index: Tensor, opacities: Tensor) -> None:
        """
        Scatter this view's screen-space gradient norms and opacities into the per-slot accumulators.

        - Gradients are renormalized to [-1, 1] screen space exactly as gsplat's DefaultStrategy does
          (strategy/default.py:243-249), which is what makes Scaffold's published grad_threshold
          directly usable here.
        """
        gradient_tensor = info[self.key_for_gradient]
        grads = gradient_tensor.grad
        if grads is None:
            return
        grads = grads.detach().clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        grad_norm = grads.reshape(-1, 2).norm(dim=-1)

        device = state["grad_accum"].device
        index = decode_index.to(device)
        state["grad_accum"].index_add_(0, index, grad_norm.to(device))
        state["denom"].index_add_(0, index, torch.ones_like(index, dtype=state["denom"].dtype))
        state["opacity_accum"].index_add_(0, index, opacities.detach().to(device))
```

`initialize_state` allocates on CPU in the tests and on the training device in the trainer — the trainer
calls `.to(device)` on the returned tensors in Task 10.

- [ ] **Step 4: Run the tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -k "gradient_key or accumulat" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/splats/scaffold.py tests/splats/test_scaffold.py
git commit --only collab_splats/splats/scaffold.py tests/splats/test_scaffold.py \
  -m "feat(splats): anchor strategy gradient accumulation"
```

---

### Task 8: AnchorStrategy — anchor growing

**Files:**
- Modify: `collab_splats/splats/scaffold.py`
- Test: `tests/splats/test_scaffold.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_growing_adds_an_anchor_at_a_high_gradient_slot():
    from collab_splats.splats.scaffold import AnchorStrategy

    field = _field(n_offsets=2)
    n_before = len(field.params["anchors"])
    strategy = AnchorStrategy(field.cfg, primitive="3dgs", voxel_size=field.voxel_size)
    state = strategy.initialize_state(n_slots=n_before * 2)

    # One slot far above threshold, displaced a full voxel from its anchor so its cell is empty
    slot = 0
    with torch.no_grad():
        field.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])
    state["grad_accum"][slot] = 1.0
    state["denom"][slot] = 1.0

    strategy.grow(field, state)
    assert len(field.params["anchors"]) == n_before + 1


def test_growing_skips_slots_below_threshold():
    from collab_splats.splats.scaffold import AnchorStrategy

    field = _field(n_offsets=2)
    n_before = len(field.params["anchors"])
    strategy = AnchorStrategy(field.cfg, primitive="3dgs", voxel_size=field.voxel_size)
    state = strategy.initialize_state(n_slots=n_before * 2)
    state["grad_accum"][0] = field.cfg.grad_threshold * 0.5
    state["denom"][0] = 1.0

    strategy.grow(field, state)
    assert len(field.params["anchors"]) == n_before


def test_growing_does_not_duplicate_an_occupied_voxel():
    from collab_splats.splats.scaffold import AnchorStrategy

    field = _field(n_offsets=2)
    n_before = len(field.params["anchors"])
    strategy = AnchorStrategy(field.cfg, primitive="3dgs", voxel_size=field.voxel_size)
    state = strategy.initialize_state(n_slots=n_before * 2)

    # Zero offset: the candidate lands in its own anchor's voxel, which is already occupied
    state["grad_accum"][0] = 1.0
    state["denom"][0] = 1.0
    strategy.grow(field, state)
    assert len(field.params["anchors"]) == n_before


def test_growing_extends_optimizer_state_to_match():
    from collab_splats.splats.scaffold import AnchorStrategy

    field = _field(n_offsets=2)
    strategy = AnchorStrategy(field.cfg, primitive="3dgs", voxel_size=field.voxel_size)
    state = strategy.initialize_state(n_slots=len(field.params["anchors"]) * 2)

    # Take one Adam step so exp_avg exists and must be grown alongside the parameters
    for name, optimizer in field.optimizers.items():
        field.params[name].grad = torch.ones_like(field.params[name])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        field.params["offsets"][0, 0] = torch.tensor([50.0, 50.0, 50.0])
    state["grad_accum"][0] = 1.0
    state["denom"][0] = 1.0
    strategy.grow(field, state)

    n_anchors = len(field.params["anchors"])
    for name, optimizer in field.optimizers.items():
        exp_avg = optimizer.state[field.params[name]]["exp_avg"]
        assert len(exp_avg) == n_anchors, name
    assert len(state["grad_accum"]) == n_anchors * field.cfg.n_offsets
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -k growing -v`
Expected: FAIL with `AttributeError: 'AnchorStrategy' object has no attribute 'grow'`

- [ ] **Step 3: Implement growing**

Add to `AnchorStrategy`:

```python
    def grow(self, field: "AnchorField", state: dict) -> int:
        """
        Add anchors in unoccupied voxels around slots whose mean gradient clears the threshold.

        - Runs ``update_depth`` coarse-to-fine levels: level i raises the threshold by
          ``update_hierarchy_factor ** i`` and coarsens the voxel by the same factor, so a few strong
          slots seed coarse anchors and many weak ones seed fine anchors (Scaffold's anchor_growing).
        - Candidates are the decoded Gaussian positions, deduped against both each other and the
          existing anchor grid; a candidate landing in an occupied voxel is dropped.

        Returns the number of anchors added.
        """
        n_offsets = self.cfg.n_offsets
        denom = state["denom"].clamp_min(1.0)
        mean_grads = state["grad_accum"] / denom
        seen = state["denom"] > 0

        # Candidate positions: every slot's decoded mean, computed without a camera (offsets alone)
        anchors = field.params["anchors"].detach()
        offset_extent = torch.exp(field.params["scaling"].detach()[:, :3])
        candidates_all = (anchors[:, None, :] + field.params["offsets"].detach() * offset_extent[:, None, :]).reshape(
            -1, 3
        )

        added_total = 0
        for level in range(self.cfg.update_depth):
            threshold = self.cfg.grad_threshold * (self.cfg.update_hierarchy_factor**level)
            level_voxel = self.voxel_size * (self.cfg.update_hierarchy_factor**level)
            selected = seen & (mean_grads >= threshold)
            if not bool(selected.any()):
                continue

            # Quantise candidates and the existing anchors onto this level's grid, then keep only
            # candidate cells that no anchor occupies. torch.unique does the candidate-side dedup.
            candidates = candidates_all[selected]
            candidate_cells = torch.round(candidates / level_voxel)
            unique_cells = torch.unique(candidate_cells, dim=0)
            occupied = torch.round(field.params["anchors"].detach() / level_voxel)
            free = ~(unique_cells[:, None, :] == occupied[None, :, :]).all(-1).any(-1)
            new_cells = unique_cells[free]
            if len(new_cells) == 0:
                continue

            new_anchors = new_cells * level_voxel
            self._append_anchors(field, state, new_anchors, level_voxel)
            added_total += len(new_anchors)

            # Later levels must see the anchors this level added, and the accumulators grew with them
            mean_grads = torch.cat([mean_grads, torch.zeros(len(new_anchors) * n_offsets)])
            seen = torch.cat([seen, torch.zeros(len(new_anchors) * n_offsets, dtype=torch.bool)])

        if self.verbose and added_total:
            logger.info("scaffold: grew %d anchors -> %d", added_total, len(field.params["anchors"]))
        return added_total

    def _append_anchors(self, field: "AnchorField", state: dict, new_anchors: Tensor, level_voxel: float) -> None:
        """
        Append new anchors (zero offsets, zero features, level-sized scaling) to params, optimizer state
        and the per-slot accumulators.
        """
        n_new = len(new_anchors)
        device = field.params["anchors"].device
        log_voxel = math.log(level_voxel)
        additions = {
            "anchors": new_anchors.to(device),
            "offsets": torch.zeros(n_new, self.cfg.n_offsets, 3, device=device),
            "anchor_feat": torch.zeros(n_new, self.cfg.feat_dim, device=device),
            "scaling": torch.full((n_new, 6), log_voxel, device=device),
            "rotation": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n_new, 1),
            "opacities": torch.zeros(n_new, 1, device=device),
        }

        # gsplat's helper rebuilds each Parameter and its Adam moments together; new rows start at
        # zero momentum, which is what upstream densification does too
        def param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(torch.cat([param.detach(), additions[name]], dim=0))

        def optimizer_fn(key: str, value: torch.Tensor) -> torch.Tensor:
            return torch.cat([value, torch.zeros((n_new, *value.shape[1:]), device=value.device)], dim=0)

        _update_param_with_optimizer(param_fn, optimizer_fn, field.params, field.optimizers)

        # Accumulators are per slot, so they grow by n_new * n_offsets rows
        n_new_slots = n_new * self.cfg.n_offsets
        for key in ("grad_accum", "denom", "opacity_accum"):
            state[key] = torch.cat([state[key], torch.zeros(n_new_slots, device=state[key].device)])
```

- [ ] **Step 4: Run the tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -k growing -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/splats/scaffold.py tests/splats/test_scaffold.py
git commit --only collab_splats/splats/scaffold.py tests/splats/test_scaffold.py \
  -m "feat(splats): anchor growing with multi-level voxel dedup"
```

---

### Task 9: AnchorStrategy — pruning and the post-backward step

**Files:**
- Modify: `collab_splats/splats/scaffold.py`
- Test: `tests/splats/test_scaffold.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_pruning_removes_persistently_transparent_anchors():
    from collab_splats.splats.scaffold import AnchorStrategy

    field = _field(n_offsets=2)
    n_before = len(field.params["anchors"])
    strategy = AnchorStrategy(field.cfg, primitive="3dgs", voxel_size=field.voxel_size)
    state = strategy.initialize_state(n_slots=n_before * 2)

    # Anchor 0 seen many times at ~zero opacity; anchor 1 seen many times at high opacity
    state["denom"][0:2] = 100.0
    state["opacity_accum"][0:2] = 1e-6
    state["denom"][2:4] = 100.0
    state["opacity_accum"][2:4] = 50.0

    strategy.prune(field, state)
    assert len(field.params["anchors"]) == n_before - 1
    assert len(state["grad_accum"]) == (n_before - 1) * 2


def test_pruning_keeps_anchors_that_were_never_seen():
    from collab_splats.splats.scaffold import AnchorStrategy

    field = _field(n_offsets=2)
    n_before = len(field.params["anchors"])
    strategy = AnchorStrategy(field.cfg, primitive="3dgs", voxel_size=field.voxel_size)
    state = strategy.initialize_state(n_slots=n_before * 2)
    strategy.prune(field, state)
    assert len(field.params["anchors"]) == n_before


def test_pruning_shrinks_optimizer_state_to_match():
    from collab_splats.splats.scaffold import AnchorStrategy

    field = _field(n_offsets=2)
    strategy = AnchorStrategy(field.cfg, primitive="3dgs", voxel_size=field.voxel_size)
    state = strategy.initialize_state(n_slots=len(field.params["anchors"]) * 2)
    for name, optimizer in field.optimizers.items():
        field.params[name].grad = torch.ones_like(field.params[name])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    state["denom"][0:2] = 100.0
    state["opacity_accum"][0:2] = 1e-6
    strategy.prune(field, state)

    n_anchors = len(field.params["anchors"])
    for name, optimizer in field.optimizers.items():
        assert len(optimizer.state[field.params[name]]["exp_avg"]) == n_anchors, name


def test_step_post_backward_refines_only_inside_the_window():
    from collab_splats.splats.scaffold import AnchorStrategy

    cfg = ScaffoldConfig(n_offsets=2, feat_dim=8, update_from=10, update_until=20, refine_every=5)
    field = _field(n_offsets=2)
    field.cfg = cfg
    strategy = AnchorStrategy(cfg, primitive="3dgs", voxel_size=field.voxel_size)
    state = strategy.initialize_state(n_slots=len(field.params["anchors"]) * 2)
    state["denom"][0:2] = 100.0
    state["opacity_accum"][0:2] = 1e-6
    n_before = len(field.params["anchors"])

    strategy.step_post_backward(field, state, step=5)  # before the window
    assert len(field.params["anchors"]) == n_before
    strategy.step_post_backward(field, state, step=12)  # inside, but not a refine step
    assert len(field.params["anchors"]) == n_before
    strategy.step_post_backward(field, state, step=15)  # inside and on the cadence
    assert len(field.params["anchors"]) == n_before - 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -k "prun or post_backward" -v`
Expected: FAIL with `AttributeError: 'AnchorStrategy' object has no attribute 'prune'`

- [ ] **Step 3: Implement pruning and the step hook**

Add to `AnchorStrategy`:

```python
    def prune(self, field: "AnchorField", state: dict) -> int:
        """
        Drop anchors whose mean decoded opacity stayed below min_opacity across the window.

        - Anchors never decoded (denom 0 for all their slots) are kept: no evidence is not evidence
          of transparency, and a frustum-filtered anchor may simply not have been visited yet.

        Returns the number of anchors removed.
        """
        n_offsets = self.cfg.n_offsets
        denom = state["denom"].reshape(-1, n_offsets).sum(dim=1)
        opacity = state["opacity_accum"].reshape(-1, n_offsets).sum(dim=1)
        seen = denom > 0
        mean_opacity = opacity / denom.clamp_min(1.0)
        drop = seen & (mean_opacity < self.cfg.min_opacity)
        if not bool(drop.any()):
            return 0

        keep = ~drop
        keep_slots = keep.repeat_interleave(n_offsets)

        def param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(param.detach()[keep])

        def optimizer_fn(key: str, value: torch.Tensor) -> torch.Tensor:
            return value[keep]

        _update_param_with_optimizer(param_fn, optimizer_fn, field.params, field.optimizers)
        for key in ("grad_accum", "denom", "opacity_accum"):
            state[key] = state[key][keep_slots]

        n_dropped = int(drop.sum())
        if self.verbose:
            logger.info("scaffold: pruned %d anchors -> %d", n_dropped, len(field.params["anchors"]))
        return n_dropped

    def step_post_backward(self, field: "AnchorField", state: dict, step: int) -> None:
        """
        Grow then prune on the refine cadence inside [update_from, update_until]; reset accumulators after.

        - Signature deliberately differs from gsplat's (params, optimizers, state, step, info): the
          anchor field owns both params and optimizers, and gradient accumulation happens in
          ``accumulate`` right after backward, before the optimizer step.
        """
        if step < self.cfg.update_from or step > self.cfg.update_until:
            return
        if step % self.cfg.refine_every != 0:
            return

        self.grow(field, state)
        self.prune(field, state)

        # Statistics are per-window: carrying them across refines would let a long-dead slot's history
        # keep triggering growth (Scaffold resets both accumulators after adjust_anchor)
        for key in ("grad_accum", "denom", "opacity_accum"):
            state[key] = torch.zeros_like(state[key])
```

- [ ] **Step 4: Run the tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_scaffold.py -v`
Expected: PASS (all scaffold tests so far)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/splats/scaffold.py tests/splats/test_scaffold.py
git commit --only collab_splats/splats/scaffold.py tests/splats/test_scaffold.py \
  -m "feat(splats): anchor pruning and refine cadence"
```

---

### Task 10: Trainer wiring

**Files:**
- Modify: `collab_splats/splats/trainer.py` (`train`, `make_strategy`), `collab_splats/splats/losses.py:112-119`
- Test: `tests/splats/test_trainer.py`, `tests/splats/test_losses.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/splats/test_trainer.py`:

```python
@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_train_scaffold_runs_and_writes_outputs(tmp_path, primitive):
    """A short scaffold run produces the same artifact set as a vanilla run."""
    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=4)
    cfg = SplatsConfig.from_dict(
        {
            "representation": "scaffold",
            "primitive": primitive,
            "max_steps": 60,
            "log_every": 10,
            "scaffold": {"n_offsets": 4, "feat_dim": 8, "update_from": 20, "update_until": 50, "refine_every": 10},
        }
    )
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, depth_targets=depths)

    for name in ("splats.ply", "ckpt.pt", "splats.zarr", "splats_quality_report.json"):
        assert (tmp_path / name).exists(), name
    checkpoint = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert "anchors" in checkpoint["splats"]
    assert "mlps" in checkpoint
    assert checkpoint["voxel_size"] > 0


@cuda
def test_scaffold_records_anchor_provenance(tmp_path):
    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=4)
    cfg = SplatsConfig.from_dict(
        {
            "representation": "scaffold",
            "primitive": "3dgs",
            "max_steps": 60,
            "log_every": 10,
            "scaffold": {
                "n_offsets": 4,
                "feat_dim": 8,
                "update_from": 10,
                "update_until": 50,
                "refine_every": 10,
                "min_opacity": 0.5,
            },
        }
    )
    train(cfg, images, world_to_cam, intrinsics, points, colors, tmp_path, depth_targets=depths)

    checkpoint = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    store = zarr.open_group(tmp_path / "splats.zarr", mode="r")
    assert store.attrs["representation"] == "scaffold"
    assert store.attrs["ply_baked"] is True
    assert store.attrs["n_anchors"] == len(checkpoint["splats"]["anchors"])
```

`make_scene` is already imported in `tests/splats/test_trainer.py` (`from tests.splats.synthetic import
make_scene`) and returns six values — `(images, world_to_cam, intrinsics, points, colors, depths)`. Add
`import zarr` to that module's imports; it is not there yet.

Append to `tests/splats/test_losses.py`:

```python
def test_scale_reg_prefers_decoded_log_scales():
    """Under scaffold there is no scales parameter: the regularizer reads the decoded scales."""
    from collab_splats.splats.losses import scale_reg_loss

    render = {"log_scales": torch.full((5, 3), -2.0)}
    gaussians = torch.nn.ParameterDict({"scales": torch.nn.Parameter(torch.zeros(5, 3))})
    decoded_value = scale_reg_loss(render, {}, gaussians, 1.0, {"weight": 1.0})
    parameter_value = scale_reg_loss({}, {}, gaussians, 1.0, {"weight": 1.0})
    assert not torch.isclose(decoded_value, parameter_value)
```

- [ ] **Step 2: Run to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_trainer.py -k scaffold tests/splats/test_losses.py -k scale_reg -v`
Expected: FAIL — `train` has no scaffold branch, `scale_reg_loss` ignores `render`.

- [ ] **Step 3: Point `scale_reg_loss` at the decoded scales**

In `collab_splats/splats/losses.py`, replace `scale_reg_loss`'s body:

```python
def scale_reg_loss(
    render: dict, target: dict, gaussians: torch.nn.ParameterDict, scene_scale: float, spec: dict
) -> Tensor:
    """
    Scale regularizer from gsplat (MCMC); expects raw log scales.

    - Scaffold has no scales parameter — its Gaussians are decoded per view — so the render carries
      ``log_scales`` and that is used when present.
    """
    log_scales = render.get("log_scales")
    if log_scales is None:
        log_scales = gaussians["scales"]
    return gsplat_losses.scale_reg_loss(log_scales)
```

- [ ] **Step 4: Branch the trainer**

In `collab_splats/splats/trainer.py`:

Import `AnchorField` and `AnchorStrategy` from `collab_splats.splats.scaffold`, and `render_gaussians`
from `collab_splats.splats.rendering`.

Replace the init block (`gaussians, optimizers = init_gaussians_from_points(...)` through
`strategy_state = ...`) with:

```python
    # Representation: vanilla Gaussians with a gsplat strategy, or Scaffold anchors with AnchorStrategy
    anchor_field = None
    if cfg.representation == "scaffold":
        anchor_field = AnchorField(cfg.scaffold_config, points, colors, scene_scale, n_views, device)
        gaussians, optimizers = anchor_field.params, anchor_field.optimizers
        strategy = AnchorStrategy(cfg.scaffold_config, cfg.primitive, anchor_field.voxel_size)
        n_slots = len(gaussians["anchors"]) * cfg.scaffold_config.n_offsets
        strategy_state = {k: v.to(device) for k, v in strategy.initialize_state(n_slots).items()}
    else:
        gaussians, optimizers = init_gaussians_from_points(cfg, points, colors, scene_scale, device)
        strategy = make_strategy(cfg, n_views)
        strategy.check_sanity(gaussians, optimizers)
        if isinstance(strategy, MCMCStrategy):
            strategy_state = strategy.initialize_state()
        else:
            strategy_state = strategy.initialize_state(scene_scale=scene_scale)
```

The means lr scheduler keys off a parameter name that differs per representation:

```python
    means_key = "anchors" if cfg.representation == "scaffold" else "means"
    means_optimizer = optimizers[means_key]
```

Inside the loop, replace the render call with a branch:

```python
        # Scaffold decodes this view's Gaussians from the anchors; vanilla activates its parameters
        if anchor_field is not None:
            decoded, decode_index = anchor_field.decode(
                cfg.primitive, view_cam_to_world, view_intrinsics, step_width, step_height, camera_id
            )
            render, info = render_gaussians(
                cfg.primitive,
                decoded,
                view_cam_to_world,
                view_intrinsics,
                step_width,
                step_height,
                sh_degree=None,
                absgrad=False,
                render_normals=render_normals,
            )
            render["log_scales"] = decoded["log_scales"]
        else:
            sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            absgrad = use_pre_backward_hook and strategy.absgrad
            render, info = render_view(
                cfg.primitive,
                gaussians,
                view_cam_to_world,
                view_intrinsics,
                step_width,
                step_height,
                sh_degree,
                absgrad,
                render_normals=render_normals,
            )
```

`use_pre_backward_hook` must not fire for scaffold: change its definition to

```python
    use_pre_backward_hook = isinstance(strategy, DefaultStrategy)
```

which already excludes `AnchorStrategy` — leave it as-is and add a comment saying so. Scaffold instead
needs `means2d` gradients retained, so before `loss.backward()`:

```python
        if anchor_field is not None:
            info[strategy.key_for_gradient].retain_grad()
```

After `loss.backward()` and **before** the optimizer steps, accumulate (the gradient must be read while
it is still on the tensor):

```python
        if anchor_field is not None:
            strategy.accumulate(strategy_state, info, decode_index, decoded["opacities"])
```

Add the MLP optimizer to the step block:

```python
        if anchor_field is not None:
            anchor_field.mlp_optimizer.step()
            anchor_field.mlp_optimizer.zero_grad(set_to_none=True)
```

And branch the post-backward refine:

```python
        if anchor_field is not None:
            strategy.step_post_backward(anchor_field, strategy_state, step)
        elif isinstance(strategy, MCMCStrategy):
            ...
```

Finally, the step log counts anchors instead of Gaussians under scaffold:

```python
        if step % cfg.log_every == 0:
            if anchor_field is not None:
                n_primitives = len(gaussians["anchors"])
                unit = "anchors"
            else:
                n_primitives = len(gaussians["means"])
                unit = "gaussians"
            rounded = {name: round(value, 4) for name, value in loss_values.items()}
            logger.info("splats step %d loss %.4f %s %d %s", step, loss.item(), unit, n_primitives, rounded)
```

`denormalize_outputs` is vanilla-only (it rewrites `means` and `scales`). Under scaffold it must rewrite
`anchors` and `scaling`, so guard it:

```python
    if cfg.normalize_scene:
        if anchor_field is not None:
            denormalize_anchors(gaussians, pose_refiner, cam_to_world, center, scale)
        else:
            denormalize_outputs(gaussians, pose_refiner, cam_to_world, center, scale)
```

with, next to `denormalize_outputs`:

```python
def denormalize_anchors(
    anchors: torch.nn.ParameterDict,
    pose_refiner: CameraOptModule | None,
    cam_to_world: Tensor,
    center: np.ndarray,
    scale: float,
) -> None:
    """
    Undo ``scene_normalization`` in place on the anchors, cameras and pose deltas.

    - anchors: p / scale + center; both halves of the log ``scaling`` shift by -log(scale).
    - offsets are stored in units of the anchor's own extent, so they are scale-free and untouched.
    """
    center_t = torch.as_tensor(center, dtype=torch.float32, device=cam_to_world.device)
    with torch.no_grad():
        anchors["anchors"].data = anchors["anchors"].data / scale + center_t
        anchors["scaling"].data = anchors["scaling"].data - math.log(scale)
        cam_to_world[:, :3, 3] = cam_to_world[:, :3, 3] / scale + center_t
        if pose_refiner is not None:
            pose_refiner.translation.weight /= scale
```

Pass the field through to the writer: `write_splat_outputs(cfg, gaussians, ..., anchor_field=anchor_field)`.

- [ ] **Step 5: Run the tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/ -v`
Expected: the two new trainer tests FAIL on the writer (Task 11); everything else PASSES.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/splats/trainer.py collab_splats/splats/losses.py tests/splats/test_losses.py tests/splats/test_trainer.py
git commit --only collab_splats/splats/trainer.py collab_splats/splats/losses.py tests/splats/test_losses.py tests/splats/test_trainer.py \
  -m "feat(splats): train the scaffold representation"
```

---

### Task 11: Outputs — checkpoint, baked ply, zarr provenance

**Files:**
- Modify: `collab_splats/splats/outputs.py:30-180`
- Test: `tests/splats/test_outputs.py`, `tests/splats/test_trainer.py` (the two from Task 10)

- [ ] **Step 1: Write the failing tests**

Append to `tests/splats/test_outputs.py`:

```python
def _scaffold_field(device="cuda"):
    """AnchorField seeded from the synthetic scene's points, on the given device."""
    from collab_splats.splats.scaffold import AnchorField, ScaffoldConfig

    _, _, _, points, colors, _ = make_scene(n_views=3)
    cfg = ScaffoldConfig(n_offsets=2, feat_dim=8)
    return AnchorField(cfg, points, colors, scene_scale=1.0, n_views=3, device=device)


@cuda
def test_bake_anchor_gaussians_uses_mean_observed_view_direction():
    from collab_splats.splats.outputs import bake_anchor_gaussians

    field = _scaffold_field()
    cam_to_world = torch.eye(4, device="cuda")[None].repeat(3, 1, 1)
    cam_to_world[:, 2, 3] = -4.0
    intrinsics = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device="cuda")[None].repeat(3, 1, 1)

    baked = bake_anchor_gaussians(field, cam_to_world, intrinsics, width=64, height=64)
    n = len(baked["means"])
    assert baked["scales"].shape == (n, 3)  # log scales, as the ply writer expects
    assert baked["quats"].shape == (n, 4)
    assert baked["opacities"].shape == (n,)
    assert baked["sh0"].shape == (n, 1, 3)
    assert baked["shN"].shape == (n, 0, 3)
    assert torch.isfinite(baked["means"]).all()
```

```python
@cuda
def test_scaffold_ply_loads_with_the_expected_field_set():
    """The baked ply is a normal degree-0 3DGS ply: any viewer must be able to read it."""
    from plyfile import PlyData

    images, world_to_cam, intrinsics, points, colors, depths = make_scene(n_views=4)
    cfg = SplatsConfig.from_dict(
        {
            "representation": "scaffold",
            "primitive": "3dgs",
            "max_steps": 30,
            "log_every": 10,
            "scaffold": {"n_offsets": 2, "feat_dim": 8, "update_from": 10, "update_until": 20, "refine_every": 10},
        }
    )
    out_dir = tmp_path_factory.mktemp("scaffold_ply")
    train(cfg, images, world_to_cam, intrinsics, points, colors, out_dir, depth_targets=depths)

    ply = PlyData.read(str(out_dir / "splats.ply"))
    fields = {prop.name for prop in ply["vertex"].properties}
    assert {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"} <= fields
    assert {"scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"} <= fields
    assert len(ply["vertex"]) > 0
```

Take `tmp_path_factory` as the test's argument (the pytest builtin) so the fixture name matches the body.

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_outputs.py -k bake -v`
Expected: FAIL with `ImportError: cannot import name 'bake_anchor_gaussians'`

- [ ] **Step 3: Implement baking and the writer branch**

Add to `collab_splats/splats/outputs.py`:

```python
def bake_anchor_gaussians(
    anchor_field, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int
) -> dict[str, Tensor]:
    """
    Decode each anchor once at its mean observed view direction, for a static viewer-loadable ply.

    - Anchors seen by no training camera fall back to the direction of the nearest camera.
    - Colors are baked into the degree-0 SH band; there are no higher bands to write.
    - Lossy by construction: the trained model is view-dependent. splats.zarr renders are not affected.
    """
    device = anchor_field.params["anchors"].device
    anchors = anchor_field.params["anchors"].detach()

    # Accumulate the unit direction to every camera that can see each anchor
    direction_sum = torch.zeros_like(anchors)
    seen_count = torch.zeros(len(anchors), device=device)
    for view in range(len(cam_to_world)):
        visible = anchor_field.visible_anchors(cam_to_world[view : view + 1], intrinsics[view : view + 1], width, height)
        to_camera = anchors - cam_to_world[view, :3, 3]
        unit = to_camera / to_camera.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction_sum[visible] += unit[visible]
        seen_count[visible] += 1

    # Unseen anchors: use the nearest camera's direction rather than dropping them from the ply
    unseen = seen_count == 0
    if bool(unseen.any()):
        camera_centers = cam_to_world[:, :3, 3]
        nearest = torch.cdist(anchors[unseen], camera_centers).argmin(dim=1)
        to_nearest = anchors[unseen] - camera_centers[nearest]
        direction_sum[unseen] = to_nearest / to_nearest.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        seen_count[unseen] = 1

    mean_direction = direction_sum / seen_count[:, None]
    mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    mean_distance = (anchors - cam_to_world[:, :3, 3].mean(dim=0)).norm(dim=-1, keepdim=True)

    # One decode at that direction, bypassing the frustum filter so every anchor is written
    with torch.no_grad():
        features = torch.cat([anchor_field.params["anchor_feat"].detach(), mean_direction, mean_distance], dim=-1)
        camera_id = torch.zeros(1, dtype=torch.long, device=device)
        neural_opacity, cov, color = anchor_field.mlps(features, camera_id)

        n_offsets = anchor_field.cfg.n_offsets
        scaling = torch.exp(anchor_field.params["scaling"].detach())
        offsets = anchor_field.params["offsets"].detach()
        keep = (neural_opacity > 0).reshape(-1)

        means = (anchors[:, None, :] + offsets * scaling[:, None, :3]).reshape(-1, 3)[keep]
        cov = cov.reshape(-1, 7)[keep]
        scales = scaling[:, 3:6].repeat_interleave(n_offsets, dim=0)[keep] * torch.sigmoid(cov[:, :3])
        quats = torch.nn.functional.normalize(cov[:, 3:7], dim=-1)
        opacities = neural_opacity.reshape(-1)[keep]
        colors = color.reshape(-1, 3)[keep]

    # The ply writer wants the raw forms every viewer expects: log scales, logit opacities, SH DC
    return {
        "means": means,
        "scales": torch.log(scales.clamp_min(1e-12)),
        "quats": quats,
        "opacities": torch.logit(opacities.clamp(1e-4, 1 - 1e-4)),
        "sh0": ((colors - 0.5) / SH_DC_NORMALIZER).unsqueeze(1),
        "shN": torch.zeros(len(means), 0, 3, device=means.device),
    }
```

`SH_DC_NORMALIZER` is already defined in `trainer.py`; import it in `outputs.py` rather than redefining it.

`shN` with zero bands is the degree-0 form `export_splats` expects. If it rejects an empty band dimension,
write one zeroed degree-1 band instead (`torch.zeros(len(means), 3, 3)`) — never fabricate SH content, the
whole point of the bake is that the view-dependent part is gone.

In `write_splat_outputs`, add the `anchor_field=None` keyword argument and branch the ply and checkpoint:

```python
    # splats.ply: standard 3DGS ply. Scaffold has no static Gaussians, so it is baked at each
    # anchor's mean observed view direction — lossy, and marked as such in the zarr provenance.
    ply_path = str(out_dir / "splats.ply")
    ply_baked = anchor_field is not None
    ply_source = (
        bake_anchor_gaussians(anchor_field, cam_to_world, intrinsics, images.shape[2], images.shape[1])
        if ply_baked
        else gaussians
    )
    export_splats(
        means=ply_source["means"],
        scales=ply_source["scales"],
        quats=ply_source["quats"],
        opacities=ply_source["opacities"],
        sh0=ply_source["sh0"],
        shN=ply_source["shN"],
        format="ply",
        save_to=ply_path,
    )
```

```python
    checkpoint = {
        "splats": splats_cpu,
        "pose_adjust": pose_adjust,
        "appearance": appearance_state,
        "config": config_dict,
    }
    # Scaffold's decode heads are half the model: without them the anchors cannot be rendered
    if anchor_field is not None:
        checkpoint["mlps"] = anchor_field.mlps.state_dict()
        checkpoint["voxel_size"] = anchor_field.voxel_size
    torch.save(checkpoint, out_dir / "ckpt.pt")
```

And extend the zarr provenance:

```python
    store.attrs.update(
        image_ids=image_ids,
        primitive=cfg.primitive,
        representation=cfg.representation,
        n_anchors=(0 if anchor_field is None else len(anchor_field.params["anchors"])),
        ply_baked=ply_baked,
        pose_opt=cfg.pose_opt,
        gsplat_commit=GSPLAT_COMMIT,
        config=config_dict,
    )
```

The quality report counts primitives through `gaussians["means"]`, which scaffold does not have. Replace
that line and the log call's argument:

```python
    # Primitive count: gaussians for vanilla, anchors for scaffold (the decoded count is per view)
    n_gaussians = int(len(gaussians["anchors" if anchor_field is not None else "means"]))
```

`render_all_views` also needs the decode: give it the same `anchor_field=None` keyword and, inside its
per-view loop, use `anchor_field.decode(...)` + `render_gaussians(...)` when it is set, mirroring the
trainer branch (full SH is meaningless for scaffold — pass `sh_degree=None`).

- [ ] **Step 4: Run the tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/ -v`
Expected: PASS — including the two scaffold trainer tests from Task 10.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/splats/outputs.py tests/splats/test_outputs.py
git commit --only collab_splats/splats/outputs.py tests/splats/test_outputs.py \
  -m "feat(splats): baked ply and scaffold checkpoint outputs"
```

---

### Task 12: Config surface, docs, and the measurement configs

**Files:**
- Modify: `configs/base.yaml`, `docs/splats.md` (if absent, create it mirroring the code tree), `CLAUDE.md`
- Create: `docs/superpowers/plans/scaffold-runs/gopro_scaffold_3dgs.yaml`, `.../gopro_scaffold_2dgs.yaml`

- [ ] **Step 1: Document the axis in `configs/base.yaml`**

Under the `splats:` block, add the commented defaults:

```yaml
  # representation: vanilla (per-gaussian parameters) | scaffold (anchors + MLP decode).
  # scaffold ignores sh_degree — its colors come from mlp_color — and densifies anchors
  # rather than gaussians.
  representation: vanilla
  # scaffold:
  #   n_offsets: 10
  #   feat_dim: 32
  #   voxel_multiplier: 1.0     # x median kNN spacing of the seed points
  #   update_from: 1500
  #   update_until: 15000
  #   refine_every: 100
  #   grad_threshold: 2.0e-4
  #   min_opacity: 0.005
  #   appearance_dim: 0         # Scaffold's per-image embedding; independent of appearance_opt
```

- [ ] **Step 2: Write the run configs**

`docs/superpowers/plans/scaffold-runs/gopro_scaffold_3dgs.yaml` — copy `gopro_3dgs_c2f_overrides.yaml`
and add:

```yaml
splats:
  representation: scaffold
  appearance_opt: true
  scaffold:
    n_offsets: 10
    feat_dim: 32
    appearance_dim: 0
semantics:
  enabled: false
```

`gopro_scaffold_2dgs.yaml` — copy `gopro_lever_combined_2dgs.yaml` and add the same block with
`primitive: 2dgs`. The 2x2 appearance sweep is these two files with `appearance_dim: 0 | 32` crossed
with `appearance_opt: false | true` — four variants each, eight runs total.

- [ ] **Step 3: Run the suite one more time**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/ -v`
Expected: PASS

- [ ] **Step 4: Format**

Run: `black collab_splats/splats/ tests/splats/ && isort collab_splats/splats/ tests/splats/`

Never run repo-wide `black .` — the venv's black is newer than the pinned one and reformats unrelated files.

- [ ] **Step 5: Record the in-flight entry**

Add to `CLAUDE.md` under **In-Flight Work**:

```markdown
- **scaffold-gs** — Scaffold-GS anchors on gsplat ([spec](docs/superpowers/specs/2026-08-27-scaffold-gs-gsplat-design.md) · [plan](docs/superpowers/plans/2026-08-27-scaffold-gs-gsplat.md))
```

- [ ] **Step 6: Commit**

```bash
git add -f configs/base.yaml CLAUDE.md docs/superpowers/plans/scaffold-runs/
git commit --only configs/base.yaml CLAUDE.md docs/superpowers/plans/scaffold-runs/ \
  -m "feat(splats): scaffold config surface and measurement run configs"
```

---

### Task 13: Measurement (human-gated)

Do not start these runs without asking — they are ~10 minutes each on the shared GPU and the container
cgroup cap is 46.6 GB. Do not run anything else heavy in parallel.

**Baselines are already logged. Do not rerun them.** From
`docs/superpowers/specs/2026-08-26-psnr-levers-design.md`:

| primitive | scene | PSNR | SSIM | gaussians | train s |
|---|---|---|---|---|---|
| 3dgs | `GH010229` | 21.606 | 0.7169 | 1 000 000 | 601 |
| 2dgs | `GH010229_undist_r7` | 20.800 | 0.6780 | 1 875 613 | 803 |

- [ ] **Step 1: Run the 3dgs 2x2** (`GH010229`, 12k steps, c2f on) — `appearance_dim` {0, 32} x `appearance_opt` {false, true}
- [ ] **Step 2: Run the 2dgs 2x2** (`GH010229_undist_r7`, 12k steps, **c2f off** — `DefaultStrategy` OOMs with it and the scaffold path is untested there)
- [ ] **Step 3: Record** PSNR, SSIM, anchor count, decoded-Gaussian count, peak memory and wall time for all eight into a `## Measured` section appended to the spec
- [ ] **Step 4: Verdict** — parity within noise at materially fewer primitives, plus which appearance model wins and whether the two stack. Append to `docs/superpowers/CHANGELOG.md`.

Run hygiene: send outputs through the `/tmp` symlink — `/workspace` has ~8.7 GB of quota and
`zarr mode="w"` rmtree's a symlinked `splats.zarr` (dir-level symlink is the safe form).

---

## Notes for the implementer

- **Do not vendor Scaffold-GS or GS-SR source.** Reimplement from the paper; cite upstream file+line in comments.
- **`graphify update .`** after the code lands, to keep the knowledge graph current.
- **Commit with `--only`.** Other sessions share this git index; a bare `git commit -a` sweeps their work.
- **`sh_degree=None`** is what tells the rasterizer that `colors` are RGB rather than SH. Passing an integer with (N, 3) colors fails inside gsplat with a shape error, not a clear message.
