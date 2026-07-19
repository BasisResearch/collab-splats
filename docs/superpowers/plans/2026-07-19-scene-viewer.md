# Scene Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generic viser-based live 3D scene viewer (`Viewer`) plus a count-capped point subsampler, per `docs/superpowers/specs/2026-07-19-scene-viewer-design.md`. Loop-closure wiring is deferred.

**Architecture:** One domain-neutral class in `collab_splats/viewer.py` wrapping a viser websocket server — named scene nodes (points / single camera frustum / line segments), upsert-by-name, two GUI toggles. One helper `subsample_points` beside the existing downsamplers in `collab_splats/pointcloud/utils.py`.

**Tech Stack:** viser 1.0.29 (already installed in `/opt/venv/reconstruction`), numpy. Tests: pytest, flat functions.

**Conventions:** python = `/opt/venv/reconstruction/bin/python`. Imports at top; block-level comments; one-line docstrings; conventional commits.

---

### Task 1: Declare viser dependency

**Files:**
- Modify: `/workspace/collab-splats/pyproject.toml` (main `[project] dependencies` list)

- [ ] **Step 1: Add dep.** In the main `dependencies` list (around line 13), add in alphabetical position:

```toml
    # viser — websocket 3D viewer for live scene visualization (collab_splats/viewer.py)
    "viser>=1.0",
```

- [ ] **Step 2: Refresh lockfile.**

Run: `cd /workspace/collab-splats && uv lock`
Expected: lock succeeds; viser resolves to 1.0.x (already installed, so no env change needed).

- [ ] **Step 3: Commit.**

```bash
git add pyproject.toml uv.lock
git commit -m "build(deps): declare viser for scene viewer"
```

---

### Task 2: `subsample_points` (TDD)

**Files:**
- Modify: `/workspace/collab-splats/collab_splats/pointcloud/utils.py` (add after `voxel_downsample`, before the "Legacy cleaning utilities" divider)
- Create: `/workspace/collab-splats/tests/pointcloud/test_utils_subsample.py`

- [ ] **Step 1: Write failing tests.**

```python
"""Tests for subsample_points (confidence filter + random count cap)."""

import numpy as np

from collab_splats.pointcloud.utils import subsample_points


def _cloud(n):
    rng = np.random.default_rng(0)
    return rng.random((n, 3)).astype(np.float32), rng.integers(0, 256, (n, 3), dtype=np.uint8)


def test_caps_count():
    pts, cols = _cloud(1000)
    out_pts, out_cols = subsample_points(pts, cols, max_points=100)
    assert out_pts.shape == (100, 3)
    assert out_cols.shape == (100, 3)


def test_noop_under_budget():
    pts, cols = _cloud(50)
    out_pts, out_cols = subsample_points(pts, cols, max_points=100)
    np.testing.assert_array_equal(out_pts, pts)
    np.testing.assert_array_equal(out_cols, cols)


def test_conf_filter_drops_low_conf():
    pts, cols = _cloud(100)
    conf = np.zeros(100, dtype=np.float32)
    conf[:20] = 1.0  # only first 20 points are confident
    out_pts, _ = subsample_points(pts, cols, conf=conf, max_points=1000, conf_percentile=50.0)
    # Everything below the 50th percentile (the zeros) is dropped
    assert out_pts.shape[0] == 20
    assert set(map(tuple, out_pts)).issubset(set(map(tuple, pts[:20])))


def test_colors_stay_aligned():
    pts, _ = _cloud(500)
    cols = np.repeat(np.arange(500, dtype=np.uint8)[:, None] % 256, 3, axis=1)
    lookup = {tuple(p): c[0] for p, c in zip(pts, cols)}
    out_pts, out_cols = subsample_points(pts, cols, max_points=50)
    for p, c in zip(out_pts, out_cols):
        assert lookup[tuple(p)] == c[0]


def test_colors_none():
    pts, _ = _cloud(200)
    out_pts, out_cols = subsample_points(pts, None, max_points=50)
    assert out_pts.shape == (50, 3)
    assert out_cols is None
```

- [ ] **Step 2: Verify tests fail.**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_utils_subsample.py -v`
Expected: 5 errors — `ImportError: cannot import name 'subsample_points'`.

- [ ] **Step 3: Implement.** Add to `collab_splats/pointcloud/utils.py` (after `voxel_downsample`):

```python
def subsample_points(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    conf: Optional[np.ndarray] = None,
    max_points: int = 50_000,
    conf_percentile: float = 20.0,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Confidence-filter then randomly cap a point set to max_points.

    Unlike voxel_downsample (voxel-size-based, output count varies with scene
    extent), this guarantees an exact point budget — needed for scenes balanced
    across submaps. Returns (points, colors) index-aligned; colors may be None.
    """
    # Drop points at/below the conf cutoff; strict > so a cutoff equal to the
    # minimum still filters, while uniform conf (nothing above cutoff) keeps all.
    if conf is not None and len(conf) > 0:
        cutoff = np.percentile(conf, conf_percentile)
        above = conf > cutoff
        if above.any():
            points = points[above]
            colors = colors[above] if colors is not None else None

    # Random cap to the budget; seeded rng keeps results reproducible
    if len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx] if colors is not None else None

    return points, colors
```

(Sanity: conf = 80×0 + 20×1, percentile 50 → cutoff 0.0, keep `conf > 0` → 20 points. Uniform conf → nothing above cutoff → keep all.)

- [ ] **Step 4: Verify tests pass.**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_utils_subsample.py -v`
Expected: 5 passed.

- [ ] **Step 5: Format + commit.**

```bash
cd /workspace/collab-splats && black collab_splats/pointcloud/utils.py tests/pointcloud/test_utils_subsample.py && isort collab_splats/pointcloud/utils.py tests/pointcloud/test_utils_subsample.py
git add collab_splats/pointcloud/utils.py tests/pointcloud/test_utils_subsample.py
git commit -m "feat(pointcloud): add subsample_points count-capped downsampler"
```

---

### Task 3: `Viewer` class (TDD)

**Files:**
- Create: `/workspace/collab-splats/collab_splats/viewer.py`
- Create: `/workspace/collab-splats/tests/test_viewer.py`

- [ ] **Step 1: Write failing tests.**

```python
"""Smoke tests for the generic viser scene Viewer (no browser needed)."""

import socket

import numpy as np
import pytest

from collab_splats.viewer import Viewer


@pytest.fixture(scope="module")
def viewer():
    # Ephemeral free port so parallel test runs don't collide
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    v = Viewer(port=port)
    yield v
    v.server.stop()


def test_add_points_registers_node(viewer):
    pts = np.zeros((10, 3), dtype=np.float32)
    cols = np.full((10, 3), 128, dtype=np.uint8)
    viewer.add_points("cloud_a", pts, cols)
    assert "cloud_a" in viewer.points


def test_add_points_upserts_by_name(viewer):
    pts = np.ones((5, 3), dtype=np.float32)
    cols = np.zeros((5, 3), dtype=np.uint8)
    viewer.add_points("cloud_a", pts, cols)
    # Same name replaces: registry still holds one entry, with the new arrays
    assert viewer.points["cloud_a"][1].shape == (5, 3)


def test_add_frustum(viewer):
    pose = np.eye(4, dtype=np.float32)  # world-to-cam identity
    intrinsic = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float32)
    viewer.add_frustum("cams/frame_0", pose, intrinsic)
    assert "cams/frame_0" in viewer.frustums
    assert viewer.frustums["cams/frame_0"].visible


def test_add_frustum_with_image(viewer):
    pose = np.eye(4, dtype=np.float32)
    intrinsic = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float32)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    viewer.add_frustum("cams/frame_1", pose, intrinsic, image=image)
    assert "cams/frame_1" in viewer.frustums


def test_add_lines(viewer):
    segments = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.float32)  # (1, 2, 3)
    viewer.add_lines("loop_edges", segments)  # smoke: no exception
```

- [ ] **Step 2: Verify tests fail.**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_viewer.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'collab_splats.viewer'`.

- [ ] **Step 3: Implement `collab_splats/viewer.py`.**

```python
"""Generic viser-based 3D scene viewer for live visualization from running jobs.

Domain-neutral: arrays in, named scene nodes out. Nodes are upserted by name —
adding a node with an existing name replaces it, which is how callers refresh a
node (e.g. re-place a submap after pose-graph correction). View in a browser at
http://<host>:<port>. No GL/display needed on the host (websocket only).
"""

from __future__ import annotations

import logging
import zlib
from typing import Optional

import numpy as np
import viser
import viser.transforms as viser_tf

logger = logging.getLogger(__name__)

# Deterministic palette for the "Color by node" toggle; node -> color via name hash
_PALETTE_SEED = 100
_PALETTE_SIZE = 256

# Frustum image thumbnails are capped to this width before upload (bandwidth)
_MAX_IMAGE_WIDTH = 128


class Viewer:
    """Web-viewable live 3D scene: named point clouds, camera frusta, line sets."""

    def __init__(self, port: int = 8080):
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        logger.info("viser scene viewer on port %d", port)

        # Per-name state so GUI toggles can restyle nodes already in the scene.
        # points maps name -> (handle, points, colors, point_size); frusta -> handle.
        self.points: dict = {}
        self.frustums: dict = {}
        self.palette = np.random.default_rng(_PALETTE_SEED).integers(
            0, 256, size=(_PALETTE_SIZE, 3), dtype=np.uint8
        )

        # GUI: camera visibility toggle + flat per-node colors (shows node boundaries)
        self.show_cameras = self.server.gui.add_checkbox("Show cameras", initial_value=True)
        self.show_cameras.on_update(lambda _: self._apply_camera_visibility())
        self.color_by_node = self.server.gui.add_checkbox("Color by node", initial_value=False)
        self.color_by_node.on_update(lambda _: self._apply_point_colors())

    ########################################################
    ########## Scene nodes (upsert by name) ###############
    ########################################################

    def add_points(
        self,
        name: str,
        points: np.ndarray,
        colors: np.ndarray,
        point_size: float = 0.01,
    ) -> None:
        """Upsert a named point cloud; points (N, 3) float, colors (N, 3) uint8."""
        shown = self._flat_color(name, len(points)) if self.color_by_node.value else colors
        handle = self.server.scene.add_point_cloud(
            name, points=points, colors=shown, point_size=point_size
        )
        self.points[name] = (handle, points, colors, point_size)

    def add_frustum(
        self,
        name: str,
        pose: np.ndarray,
        intrinsic: np.ndarray,
        image: Optional[np.ndarray] = None,
        scale: float = 0.05,
    ) -> None:
        """Upsert one camera frustum; pose (4, 4) world-to-cam, intrinsic (3, 3).

        Callers loop over frames with hierarchical names (e.g. "submap_3/cams/frame_0").
        image is (H, W, 3) uint8, optional; downscaled before upload.
        """
        # Sensor size: from the image when given, else approximated from the
        # principal point (cx, cy) ~ image center.
        if image is not None:
            h, w = image.shape[:2]
            if w > _MAX_IMAGE_WIDTH:
                stride = int(np.ceil(w / _MAX_IMAGE_WIDTH))
                image = image[::stride, ::stride]
        else:
            w, h = 2.0 * float(intrinsic[0, 2]), 2.0 * float(intrinsic[1, 2])
        fov = 2.0 * float(np.arctan2(h / 2.0, float(intrinsic[1, 1])))

        # Repo poses are world-to-cam; viser frames are cam-to-world.
        T = viser_tf.SE3.from_matrix(np.linalg.inv(np.asarray(pose, dtype=np.float64)))
        handle = self.server.scene.add_camera_frustum(
            name,
            fov=fov,
            aspect=w / h,
            scale=scale,
            image=image,
            wxyz=T.rotation().wxyz,
            position=T.translation(),
        )
        handle.visible = self.show_cameras.value
        self.frustums[name] = handle

    def add_lines(
        self,
        name: str,
        segments: np.ndarray,
        color: tuple = (0, 255, 0),
        line_width: float = 2.0,
    ) -> None:
        """Upsert named line segments; segments (M, 2, 3) float."""
        colors = np.tile(np.asarray(color, dtype=np.uint8), (len(segments), 2, 1))
        self.server.scene.add_line_segments(
            name, points=np.asarray(segments), colors=colors, line_width=line_width
        )

    ########################################################
    ########## GUI toggle handlers ########################
    ########################################################

    def _apply_camera_visibility(self) -> None:
        """Show/hide every frustum to match the checkbox."""
        for handle in self.frustums.values():
            handle.visible = self.show_cameras.value

    def _apply_point_colors(self) -> None:
        """Re-upload each cloud with flat or original colors (no in-place recolor in viser)."""
        for name, (_, points, colors, point_size) in list(self.points.items()):
            self.add_points(name, points, colors, point_size=point_size)

    def _flat_color(self, name: str, n: int) -> np.ndarray:
        """Deterministic per-name palette color, broadcast to (n, 3)."""
        color = self.palette[zlib.crc32(name.encode()) % _PALETTE_SIZE]
        return np.tile(color, (n, 1))
```

- [ ] **Step 4: Verify tests pass.**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_viewer.py -v`
Expected: 5 passed. If `add_line_segments` rejects the `colors` shape, viser 1.x expects `(M, 2, 3)` — that is what `np.tile(..., (len(segments), 2, 1))` produces; if it instead demands a flat single color, pass `colors=color`.

- [ ] **Step 5: Format + commit.**

```bash
cd /workspace/collab-splats && black collab_splats/viewer.py tests/test_viewer.py && isort collab_splats/viewer.py tests/test_viewer.py
git add collab_splats/viewer.py tests/test_viewer.py
git commit -m "feat(viewer): generic viser live scene viewer (points/frustum/lines)"
```

---

### Task 4: Regression + spec touch-up

**Files:**
- Modify: `docs/superpowers/specs/2026-07-19-scene-viewer-design.md` (already edited in working tree: singular `add_frustum` signature)

- [ ] **Step 1: Run the affected test dirs** (full suite is slow; these cover touched modules):

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/test_viewer.py -q`
Expected: all pass (plus pre-existing known failures per `docs/known-test-failures.md`, none in these dirs).

- [ ] **Step 2: Commit spec edit.**

```bash
git add -f docs/superpowers/specs/2026-07-19-scene-viewer-design.md
git commit -m "docs(specs): scene viewer — add_frustum takes a single camera"
```
