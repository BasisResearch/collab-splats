# SemanticsPane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace in-memory `state.frames` with a lazy zarr store, add `extract_and_cache_from_zarr` to `BaseFeatureExtractor`, and implement `SemanticsPane` — a 4-column 2D feature extraction viewer with reactive frame display and text query support.

**Architecture:** PreprocessPane writes extracted frames to `output_dir/frames.zarr` (Blosc-compressed, one chunk per frame) and sets `state.frames_zarr_path`. SemanticsPane reads frames on demand from that zarr, runs up to 3 independent extractor columns (each writing their own feature zarr via `extract_and_cache_from_zarr`), and updates the display reactively on frame slider changes. Text query bar appears when any active column is a `BaseQueryableExtractor`.

**Tech Stack:** Panel 1.x, param, zarr, torch, PIL, matplotlib (viridis colormap), threading

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `collab_splats/dashboard/state.py` | Modify | Replace `frames` param with `frames_zarr_path` |
| `collab_splats/dashboard/panes/preprocess.py` | Modify | Write frames.zarr; set `frames_zarr_path` |
| `collab_splats/semantics/features/base.py` | Modify | Add `extract_and_cache_from_zarr` + `features_to_rgb` static method |
| `collab_splats/dashboard/panes/semantics.py` | Create | `SemanticsPane` — full Tab 2 implementation |
| `collab_splats/dashboard/app.py` | Modify | Wire `SemanticsPane` into tab 2 |
| `collab_splats/dashboard/__init__.py` | Modify | Export `SemanticsPane` |
| `tests/dashboard/test_state.py` | Modify | Update for `frames_zarr_path` |
| `tests/dashboard/test_preprocess.py` | Modify | Add `_write_frames_zarr` test |
| `tests/semantics/features/test_extract_from_zarr.py` | Create | Test `extract_and_cache_from_zarr` + `features_to_rgb` |
| `tests/dashboard/test_semantics.py` | Create | Test pure helpers in `SemanticsPane` |

---

## Task 1: AppState — replace `frames` with `frames_zarr_path`

**Files:**
- Modify: `collab_splats/dashboard/state.py`
- Modify: `tests/dashboard/test_state.py`

- [ ] **Step 1.1: Update test_state.py to reflect new param**

Replace `test_appstate_defaults`, `test_appstate_watch_fires_on_output_dir_change`, and the frames-related tests in `tests/dashboard/test_state.py`:

```python
from pathlib import Path
import numpy as np
from collab_splats.dashboard.state import AppState


def test_appstate_defaults():
    state = AppState()
    assert state.output_dir is None
    assert state.video_path is None
    assert state.frames_zarr_path is None
    assert state.feedforward_result is None
    assert state.feature_maps_path is None
    assert state.lifted_features_path is None


def test_appstate_watch_fires_on_output_dir_change():
    state = AppState()
    received = []
    state.param.watch(lambda e: received.append(e.new), "output_dir")
    state.output_dir = Path("/tmp/test_out")
    assert received == [Path("/tmp/test_out")]


def test_appstate_frames_zarr_path_accepts_path():
    state = AppState()
    p = Path("/workspace/outputs/birds/frames.zarr")
    state.frames_zarr_path = p
    assert state.frames_zarr_path == p


def test_appstate_watch_fires_on_frames_zarr_path_change():
    state = AppState()
    received = []
    state.param.watch(lambda e: received.append(e.new), "frames_zarr_path")
    p = Path("/tmp/frames.zarr")
    state.frames_zarr_path = p
    assert received == [p]


def test_appstate_feature_maps_path_accepts_path():
    state = AppState()
    p = Path("/workspace/outputs/birds/vggt_omega/features.zarr")
    state.feature_maps_path = p
    assert state.feature_maps_path == p
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_state.py -v
```

Expected: FAIL — `AppState` has no `frames_zarr_path`, still has `frames`.

- [ ] **Step 1.3: Update state.py**

Replace the full contents of `collab_splats/dashboard/state.py`:

```python
from __future__ import annotations

from pathlib import Path

import param


class AppState(param.Parameterized):
    """Shared data bus passed between all dashboard panes.

    Panes observe fields via param.watch — downstream panes auto-enable
    when upstream data arrives (e.g. output_dir set by PreprocessPane).
    """

    output_dir = param.Parameter(default=None)
    video_path = param.Parameter(default=None)
    frames_zarr_path = param.Parameter(default=None)   # Path to output_dir/frames.zarr
    feedforward_result = param.Parameter(default=None)
    feature_maps_path = param.Parameter(default=None)
    lifted_features_path = param.Parameter(default=None)
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_state.py -v
```

Expected: all PASS.

- [ ] **Step 1.5: Commit**

```bash
git add collab_splats/dashboard/state.py tests/dashboard/test_state.py
git commit -m "refactor(dashboard): replace state.frames list with frames_zarr_path"
```

---

## Task 2: PreprocessPane — write frames.zarr

**Files:**
- Modify: `collab_splats/dashboard/panes/preprocess.py`
- Modify: `tests/dashboard/test_preprocess.py`

- [ ] **Step 2.1: Add test for `_write_frames_zarr` helper in test_preprocess.py**

Append to `tests/dashboard/test_preprocess.py`:

```python
import tempfile
import zarr
from collab_splats.dashboard.panes.preprocess import _write_frames_zarr


def test_write_frames_zarr_shape_and_attrs():
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "frames.zarr"
        result = _write_frames_zarr(frames, zarr_path)
        assert result == zarr_path
        z = zarr.open(str(zarr_path), mode="r")
        arr = z["frames"]
        assert arr.shape == (5, 48, 64, 3)
        assert z.attrs["n_frames"] == 5
        assert z.attrs["height"] == 48
        assert z.attrs["width"] == 64


def test_write_frames_zarr_roundtrip():
    frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "frames.zarr"
        _write_frames_zarr([frame], zarr_path)
        z = zarr.open(str(zarr_path), mode="r")
        np.testing.assert_array_equal(z["frames"][0], frame)
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_preprocess.py::test_write_frames_zarr_shape_and_attrs tests/dashboard/test_preprocess.py::test_write_frames_zarr_roundtrip -v
```

Expected: FAIL — `_write_frames_zarr` not defined.

- [ ] **Step 2.3: Add `_write_frames_zarr` helper and update `_run_extraction` in preprocess.py**

At the top of `collab_splats/dashboard/panes/preprocess.py`, add `zarr` to imports (it is already available via the environment). Then add the helper function in the "Pure helpers" section after the existing helpers:

```python
def _write_frames_zarr(frames: list[np.ndarray], zarr_path: Path) -> Path:
    """Write extracted frames to a Blosc-compressed zarr store.

    Layout: frames (N, H, W, 3) uint8, chunks=(1, H, W, 3) — one chunk per frame.
    Returns the zarr store path.
    """
    import zarr
    from zarr.codecs import BloscCodec

    N = len(frames)
    H, W = frames[0].shape[:2]
    store = zarr.open(str(zarr_path), mode="w")
    store.attrs.update({"n_frames": N, "height": H, "width": W})
    arr = store.create_array(
        "frames",
        shape=(N, H, W, 3),
        chunks=(1, H, W, 3),
        dtype="uint8",
        fill_value=0,
        codecs=[BloscCodec(cname="lz4", clevel=5)],
    )
    for i, frame in enumerate(frames):
        arr[i] = frame
    return zarr_path
```

Then in `_run_extraction`, find the block that currently sets `self._state.frames = frames` and replace it with the zarr write + new state param. The relevant section (after window filter, before metrics):

```python
            self._selected_frames = frames
            self._selected_indices = list(range(len(frames)))

            # Write output_dir if not already set (session started from video path)
            if self._state.output_dir is None and self._state.video_path is not None:
                self._state.output_dir = Path("/workspace/outputs") / Path(self._state.video_path).stem

            # Write frames to zarr and update shared state
            output_dir = Path(self._state.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            frames_zarr_path = _write_frames_zarr(frames, output_dir / "frames.zarr")
            self._state.frames_zarr_path = frames_zarr_path
```

Note: remove `self._state.frames = frames` — the `frames` param no longer exists on `AppState`.

- [ ] **Step 2.4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_preprocess.py -v
```

Expected: all PASS.

- [ ] **Step 2.5: Run full dashboard test suite to check for regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/ -v
```

Expected: all PASS (some tests may skip if Panel can't headless — that's OK).

- [ ] **Step 2.6: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py tests/dashboard/test_preprocess.py
git commit -m "feat(preprocess): write frames to zarr; set state.frames_zarr_path"
```

---

## Task 3: BaseFeatureExtractor — `features_to_rgb` and `extract_and_cache_from_zarr`

**Files:**
- Modify: `collab_splats/semantics/features/base.py`
- Create: `tests/semantics/features/test_extract_from_zarr.py`

- [ ] **Step 3.1: Create test file**

Create `tests/semantics/features/test_extract_from_zarr.py`:

```python
"""Tests for BaseFeatureExtractor.features_to_rgb and extract_and_cache_from_zarr."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import zarr
from zarr.codecs import BloscCodec

from collab_splats.semantics.features.base import BaseFeatureExtractor


########################################################################
# Minimal concrete extractor for tests — no model weights needed
########################################################################

@BaseFeatureExtractor.register("_test_extractor")
class _TestExtractor(BaseFeatureExtractor):
    """Minimal extractor that returns constant (D, H_p, W_p) tensors."""

    patch_size = 16

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._D = 8
        self._H_p = 4
        self._W_p = 4

    def forward(self, images: list) -> list[torch.Tensor]:
        return [torch.ones(self._D, self._H_p, self._W_p) for _ in images]


def _make_frames_zarr(n: int, H: int, W: int, tmp_dir: str) -> Path:
    """Write a minimal frames.zarr with n random uint8 frames."""
    zarr_path = Path(tmp_dir) / "frames.zarr"
    store = zarr.open(str(zarr_path), mode="w")
    store.attrs.update({"n_frames": n, "height": H, "width": W})
    arr = store.create_array(
        "frames",
        shape=(n, H, W, 3),
        chunks=(1, H, W, 3),
        dtype="uint8",
        fill_value=0,
        codecs=[BloscCodec(cname="lz4", clevel=5)],
    )
    for i in range(n):
        arr[i] = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    return zarr_path


########################################################################
# features_to_rgb
########################################################################

def test_features_to_rgb_shape():
    feat = torch.randn(16, 6, 8)  # (D, H_p, W_p)
    rgb = BaseFeatureExtractor.features_to_rgb(feat)
    assert rgb.shape == (6, 8, 3)
    assert rgb.dtype == np.uint8


def test_features_to_rgb_range():
    feat = torch.randn(32, 4, 4)
    rgb = BaseFeatureExtractor.features_to_rgb(feat)
    assert rgb.min() >= 0
    assert rgb.max() <= 255


def test_features_to_rgb_constant_returns_midpoint():
    # All-identical feature vectors → PCA variance is zero → output is uniform mid-grey
    feat = torch.ones(16, 4, 4)
    rgb = BaseFeatureExtractor.features_to_rgb(feat)
    # All values are 0 after normalization (range=0, +eps), so result is 0
    assert rgb.max() == 0


########################################################################
# extract_and_cache_from_zarr
########################################################################

def test_extract_and_cache_from_zarr_creates_zarr():
    extractor = _TestExtractor()
    with tempfile.TemporaryDirectory() as tmp:
        frames_zarr = _make_frames_zarr(n=3, H=64, W=64, tmp_dir=tmp)
        cache_dir = Path(tmp) / "features" / "_test_extractor"
        result = extractor.extract_and_cache_from_zarr(frames_zarr, cache_dir)
        assert result.exists()
        z = zarr.open(str(result), mode="r")
        assert z["features"].shape[0] == 3  # N frames
        assert z.attrs["extractor"] == "_test_extractor"
        assert z.attrs["n_frames"] == 3


def test_extract_and_cache_from_zarr_feature_shape():
    extractor = _TestExtractor()
    with tempfile.TemporaryDirectory() as tmp:
        frames_zarr = _make_frames_zarr(n=4, H=64, W=64, tmp_dir=tmp)
        cache_dir = Path(tmp) / "features"
        result = extractor.extract_and_cache_from_zarr(frames_zarr, cache_dir)
        z = zarr.open(str(result), mode="r")
        N, D, H_p, W_p = z["features"].shape
        assert N == 4
        assert D == extractor._D
        assert H_p == extractor._H_p
        assert W_p == extractor._W_p


def test_extract_and_cache_from_zarr_skip_existing():
    extractor = _TestExtractor()
    with tempfile.TemporaryDirectory() as tmp:
        frames_zarr = _make_frames_zarr(n=2, H=64, W=64, tmp_dir=tmp)
        cache_dir = Path(tmp) / "features"
        # First call — creates cache
        result1 = extractor.extract_and_cache_from_zarr(frames_zarr, cache_dir)
        mtime1 = result1.stat().st_mtime
        # Second call — should skip and return same path without modifying mtime
        result2 = extractor.extract_and_cache_from_zarr(frames_zarr, cache_dir)
        assert result1 == result2
        assert result2.stat().st_mtime == mtime1
```

- [ ] **Step 3.2: Ensure test directory has `__init__.py`**

```bash
ls /workspace/collab-splats/tests/semantics/
```

If `features/` subdirectory is missing:

```bash
mkdir -p /workspace/collab-splats/tests/semantics/features
touch /workspace/collab-splats/tests/semantics/features/__init__.py
```

- [ ] **Step 3.3: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/features/test_extract_from_zarr.py -v
```

Expected: FAIL — `features_to_rgb` and `extract_and_cache_from_zarr` not defined.

- [ ] **Step 3.4: Add `features_to_rgb` static method to `BaseFeatureExtractor` in base.py**

Add after the `extract_and_cache` method (before `_build_positional_basis`):

```python
    @staticmethod
    def features_to_rgb(feat: "torch.Tensor") -> "np.ndarray":
        """Project a feature map onto its top-3 principal components to produce an RGB image.

        Args:
            feat: (D, H_p, W_p) float tensor — output of forward() for one frame.

        Returns:
            np.ndarray of shape (H_p, W_p, 3) dtype uint8.
        """
        import numpy as _np
        import torch as _torch

        D, H_p, W_p = feat.shape
        # Reshape to (N, D) where N = H_p * W_p patch locations
        E = feat.reshape(D, -1).T.float()  # (N, D)
        E = E - E.mean(dim=0, keepdim=True)  # center columns

        # SVD: right singular vectors are the principal components in feature space
        _, _, Vt = _torch.linalg.svd(E, full_matrices=False)  # Vt: (min(N,D), D)
        rgb = (E @ Vt[:3].T).detach().cpu().numpy()  # (N, 3)

        # Normalize to [0, 255]
        rgb -= rgb.min()
        rgb /= rgb.max() + 1e-8
        return (rgb.reshape(H_p, W_p, 3) * 255).astype(_np.uint8)
```

- [ ] **Step 3.5: Add `extract_and_cache_from_zarr` method to `BaseFeatureExtractor` in base.py**

Add after `extract_and_cache`:

```python
    def extract_and_cache_from_zarr(
        self,
        frames_zarr_path: Path,
        cache_dir: Path,
        batch_size: int = 1,
        skip_existing: bool = True,
    ) -> Path:
        """Extract patch features from a frames zarr store and write to cache_dir/{name}.zarr.

        Iterates frames lazily (one chunk at a time) — never holds all frames in RAM.
        Output zarr layout is identical to extract_and_cache: features (N, D, H_p, W_p),
        chunks=(1, D, H_p, W_p). Re-entrant: valid existing cache is reused when skip_existing=True.
        """
        import zarr
        from zarr.codecs import BloscCodec
        from PIL import Image as _PILImage

        zarr_path = Path(cache_dir) / f"{self.name}.zarr"
        frames_store = zarr.open(str(frames_zarr_path), mode="r")
        N = int(frames_store.attrs["n_frames"])

        # Validate existing cache
        if skip_existing and zarr_path.exists():
            try:
                z = zarr.open(str(zarr_path), mode="r")
                if z.attrs.get("extractor") == self.name and z.attrs.get("n_frames") == N:
                    logger.info("Feature cache valid, skipping extraction: %s", zarr_path)
                    return zarr_path
            except Exception:
                logger.warning("Cache at %s is corrupt or unreadable, re-extracting", zarr_path)

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Probe first frame to learn output shape (D, H_p, W_p)
        first_frame = _PILImage.fromarray(frames_store["frames"][0]).convert("RGB")
        with torch.no_grad():
            [first_feat] = self.forward([first_frame])
        D, H_p, W_p = first_feat.shape

        # Open output zarr and write metadata
        store = zarr.open(str(zarr_path), mode="w")
        store.attrs.update({
            "extractor": self.name,
            "patch_size": self.patch_size,
            "n_frames": N,
            "feature_dim": D,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        arr = store.create_array(
            "features",
            shape=(N, D, H_p, W_p),
            chunks=(1, D, H_p, W_p),
            dtype="float32",
            fill_value=0,
        )

        # Write first frame (already extracted for shape probe)
        arr[0] = first_feat.cpu().float().numpy()

        # Iterate remaining frames lazily from zarr
        for i in range(1, N):
            pil_img = _PILImage.fromarray(frames_store["frames"][i]).convert("RGB")
            with torch.no_grad():
                [feat] = self.forward([pil_img])
            arr[i] = feat.cpu().float().numpy()
            if i % 10 == 0:
                logger.info("extract_and_cache_from_zarr: %d/%d frames written", i + 1, N)

        logger.info("Feature cache written: %s  shape=%s", zarr_path, tuple(arr.shape))
        return zarr_path
```

- [ ] **Step 3.6: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/features/test_extract_from_zarr.py -v
```

Expected: all PASS.

- [ ] **Step 3.7: Run full semantics test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/ tests/test_semantics_logging.py -v
```

Expected: all PASS (maskclip tests may skip if maskclip_onnx not installed — that's OK).

- [ ] **Step 3.8: Commit**

```bash
git add collab_splats/semantics/features/base.py tests/semantics/features/test_extract_from_zarr.py tests/semantics/features/__init__.py
git commit -m "feat(semantics): add features_to_rgb and extract_and_cache_from_zarr to BaseFeatureExtractor"
```

---

## Task 4: SemanticsPane — skeleton, frame selector, enabled gate

**Files:**
- Create: `collab_splats/dashboard/panes/semantics.py`
- Create: `tests/dashboard/test_semantics.py`

- [ ] **Step 4.1: Create test file for pure helpers**

Create `tests/dashboard/test_semantics.py`:

```python
"""Tests for pure helpers in SemanticsPane."""
from __future__ import annotations

import numpy as np
import pytest

from collab_splats.dashboard.panes.semantics import _score_to_rgb, _load_frame_rgb


########################################################################
# _score_to_rgb
########################################################################

def test_score_to_rgb_shape():
    score = np.random.rand(8, 10).astype(np.float32)
    rgb = _score_to_rgb(score)
    assert rgb.shape == (8, 10, 3)
    assert rgb.dtype == np.uint8


def test_score_to_rgb_range():
    score = np.random.rand(4, 4).astype(np.float32)
    rgb = _score_to_rgb(score)
    assert rgb.min() >= 0
    assert rgb.max() <= 255


def test_score_to_rgb_uniform_input():
    # Uniform score should produce uniform output (no crash)
    score = np.ones((6, 6), dtype=np.float32) * 0.5
    rgb = _score_to_rgb(score)
    assert rgb.shape == (6, 6, 3)


########################################################################
# _load_frame_rgb
########################################################################

def test_load_frame_rgb_returns_array(tmp_path):
    import zarr
    from zarr.codecs import BloscCodec

    frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    zarr_path = tmp_path / "frames.zarr"
    store = zarr.open(str(zarr_path), mode="w")
    store.attrs.update({"n_frames": 1, "height": 48, "width": 64})
    arr = store.create_array(
        "frames", shape=(1, 48, 64, 3), chunks=(1, 48, 64, 3), dtype="uint8",
        codecs=[BloscCodec(cname="lz4", clevel=5)],
    )
    arr[0] = frame

    result = _load_frame_rgb(zarr_path, 0)
    assert result.shape == (48, 64, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, frame)
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_semantics.py -v
```

Expected: FAIL — module `semantics` doesn't exist yet.

- [ ] **Step 4.3: Create `collab_splats/dashboard/panes/semantics.py`**

```python
"""SemanticsPane — Tab 2: 2D feature extraction and comparison."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import panel as pn
import param
import torch

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.state import AppState
from collab_splats.semantics.features.base import BaseFeatureExtractor, BaseQueryableExtractor

logger = logging.getLogger(__name__)

########################################################################
# Pure helpers — testable without Panel
########################################################################


def _score_to_rgb(score: np.ndarray) -> np.ndarray:
    """Convert a (H, W) float score map to a viridis RGB image (H, W, 3) uint8."""
    import matplotlib.cm as cm

    score = score.astype(np.float32)
    lo, hi = score.min(), score.max()
    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score = np.zeros_like(score)
    rgba = cm.viridis(score)  # (H, W, 4) float64 in [0, 1]
    return (rgba[..., :3] * 255).astype(np.uint8)


def _load_frame_rgb(frames_zarr_path: Path, idx: int) -> np.ndarray:
    """Load frame idx from frames.zarr and return (H, W, 3) uint8 array."""
    import zarr

    z = zarr.open(str(frames_zarr_path), mode="r")
    return z["frames"][idx]


########################################################################
# ExtractorColumn — per-column state and widgets
########################################################################

_IDLE = "idle"
_RUNNING = "running"
_DONE = "done"
_ERROR = "error"


class ExtractorColumn(param.Parameterized):
    """One extractor column: dropdown, run button, status, image display."""

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any) -> None:
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._extractor: BaseFeatureExtractor | None = None
        self._feature_zarr_path: Path | None = None
        self._thread: threading.Thread | None = None
        self._status = _IDLE

        # Widgets
        self._method_dd = pn.widgets.Select(
            name="Extractor",
            options=list(BaseFeatureExtractor._registry.keys()),
            value=list(BaseFeatureExtractor._registry.keys())[0] if BaseFeatureExtractor._registry else None,
            width=200,
        )
        self._run_btn = pn.widgets.Button(
            name="▶ Run", button_type="primary", width=100, disabled=True
        )
        self._status_html = pn.pane.HTML(
            "<span style='color:#888;font-size:11px'>idle</span>", width=200
        )
        self._image_pane = pn.pane.PNG(None, width=320, height=240)

        # Wire callbacks
        self._method_dd.param.watch(self._on_method_change, "value")
        self._run_btn.on_click(self._on_run)
        state.param.watch(self._on_frames_ready, "frames_zarr_path")

        # Enable if frames already available
        if state.frames_zarr_path is not None:
            self._run_btn.disabled = False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_frames_ready(self, event: Any) -> None:
        """Enable Run button when frames zarr becomes available."""
        if event.new is not None:
            self._run_btn.disabled = False

    def _on_method_change(self, event: Any) -> None:
        """Unload current extractor and reset column when method changes."""
        if self._extractor is not None:
            del self._extractor
            self._extractor = None
            torch.cuda.empty_cache()
        self._feature_zarr_path = None
        self._image_pane.object = None
        self._set_status(_IDLE)

    def _on_run(self, event: Any) -> None:
        """Launch background extraction thread."""
        if self._thread and self._thread.is_alive():
            return
        if self._state.frames_zarr_path is None:
            return
        # Unload previous extractor before loading new one
        if self._extractor is not None:
            del self._extractor
            self._extractor = None
            torch.cuda.empty_cache()
        self._feature_zarr_path = None
        self._image_pane.object = None
        self._set_status(_RUNNING)
        self._run_btn.disabled = True
        method = self._method_dd.value
        self._thread = threading.Thread(
            target=self._run_extraction, args=(method,), daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run_extraction(self, method: str) -> None:
        """Background: instantiate extractor, run extract_and_cache_from_zarr, update display."""
        try:
            self._op_log.start_op(f"Extracting {method} features")
            extractor_cls = BaseFeatureExtractor.get(method)
            self._extractor = extractor_cls()
            output_dir = Path(self._state.output_dir)
            cache_dir = output_dir / "features" / method
            zarr_path = self._extractor.extract_and_cache_from_zarr(
                frames_zarr_path=Path(self._state.frames_zarr_path),
                cache_dir=cache_dir,
                on_progress=lambda i, n: self._op_log.update_progress(
                    int(i / n * 100), f"{i}/{n} frames"
                ) if n > 0 else None,
            )
            self._feature_zarr_path = zarr_path
            self._state.feature_maps_path = zarr_path
            self._set_status(_DONE)
            self._op_log.finish_op()
            # Display current frame
            self._refresh_display()
        except Exception as exc:
            logger.exception("Extraction failed for %s", method)
            self._set_status(f"{_ERROR}: {exc}")
            self._op_log.error_op(str(exc))
        finally:
            self._run_btn.disabled = False

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def refresh(self, frame_idx: int) -> None:
        """Called by SemanticsPane when the frame slider changes."""
        self._current_frame_idx = frame_idx
        if self._feature_zarr_path is not None and self._extractor is not None:
            self._refresh_display()

    def _refresh_display(self) -> None:
        """Load features for current frame and render PCA or query heatmap."""
        import zarr
        from io import BytesIO
        from PIL import Image as PILImage

        if self._feature_zarr_path is None:
            return
        idx = getattr(self, "_current_frame_idx", 0)
        z = zarr.open(str(self._feature_zarr_path), mode="r")
        feat = torch.from_numpy(z["features"][idx])  # (D, H_p, W_p)
        rgb = BaseFeatureExtractor.features_to_rgb(feat)
        buf = BytesIO()
        PILImage.fromarray(rgb).save(buf, format="PNG")
        self._image_pane.object = buf.getvalue()

    def render_query(self, frame_idx: int, query_text: str) -> None:
        """Render similarity heatmap for query_text on frame_idx (called from query bar thread)."""
        import zarr
        from io import BytesIO
        from PIL import Image as PILImage

        if self._feature_zarr_path is None or not isinstance(self._extractor, BaseQueryableExtractor):
            return
        z = zarr.open(str(self._feature_zarr_path), mode="r")
        feat = torch.from_numpy(z["features"][frame_idx].astype(np.float32))  # (D, H_p, W_p)
        score = self._extractor.score_queries(feat, positive=[query_text])  # (H_p, W_p)
        rgb = _score_to_rgb(score.cpu().numpy())
        buf = BytesIO()
        PILImage.fromarray(rgb).save(buf, format="PNG")
        self._image_pane.object = buf.getvalue()

    def is_queryable_and_ready(self) -> bool:
        """True when this column has a completed queryable extractor."""
        return (
            self._feature_zarr_path is not None
            and isinstance(self._extractor, BaseQueryableExtractor)
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def panel(self) -> pn.Column:
        """Return the column's Panel layout."""
        return pn.Column(
            pn.Row(self._method_dd, self._run_btn),
            self._status_html,
            self._image_pane,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_status(self, status: str) -> None:
        self._status = status
        color = {_IDLE: "#888", _RUNNING: "#f0c040", _DONE: "#50c050", _ERROR: "#e05050"}.get(
            status.split(":")[0], "#888"
        )
        self._status_html.object = (
            f"<span style='color:{color};font-size:11px'>{status}</span>"
        )


########################################################################
# SemanticsPane
########################################################################


class SemanticsPane(param.Parameterized):
    """Tab 2 — 2D feature extraction and comparison across up to 3 extractors."""

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any) -> None:
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._current_frame_idx = 0
        self._query_thread: threading.Thread | None = None

        # Frame selector
        self._frame_slider = pn.widgets.IntSlider(
            name="Frame", value=0, start=0, end=0, step=1, width=400, disabled=True
        )
        self._prev_btn = pn.widgets.Button(name="◀", width=50, disabled=True)
        self._next_btn = pn.widgets.Button(name="▶", width=50, disabled=True)
        self._frame_count_html = pn.pane.HTML("", width=200)

        # Original frame display (col 0)
        self._original_pane = pn.pane.PNG(None, width=320, height=240)

        # Up to 3 extractor columns
        self._columns: list[ExtractorColumn] = [
            ExtractorColumn(state=state, op_log=op_log) for _ in range(3)
        ]
        self._col_panels: list[pn.Column] = [c.panel() for c in self._columns]
        # Hide columns 2 and 3 initially (show via "+" button logic — simplified: all 3 visible)

        # Text query bar
        self._query_input = pn.widgets.TextInput(
            placeholder="e.g. chair, table, floor ...", width=400
        )
        self._query_btn = pn.widgets.Button(
            name="Query", button_type="success", width=100, disabled=True
        )
        self._query_bar = pn.Row(
            self._query_input, self._query_btn, visible=False
        )

        # Wire callbacks
        self._frame_slider.param.watch(self._on_slider_change, "value")
        self._prev_btn.on_click(self._on_prev)
        self._next_btn.on_click(self._on_next)
        self._query_btn.on_click(self._on_query)
        state.param.watch(self._on_frames_zarr_ready, "frames_zarr_path")

        # Activate if already populated (e.g. session loaded)
        if state.frames_zarr_path is not None:
            self._activate(Path(state.frames_zarr_path))

    # ------------------------------------------------------------------
    # State callbacks
    # ------------------------------------------------------------------

    def _on_frames_zarr_ready(self, event: Any) -> None:
        """Enable controls and set slider range when frames zarr is available."""
        if event.new is not None:
            self._activate(Path(event.new))

    def _activate(self, frames_zarr_path: Path) -> None:
        """Enable all frame selector controls and load first frame."""
        import zarr

        z = zarr.open(str(frames_zarr_path), mode="r")
        n = int(z.attrs["n_frames"])
        self._frame_slider.end = max(0, n - 1)
        self._frame_slider.disabled = False
        self._prev_btn.disabled = False
        self._next_btn.disabled = False
        self._frame_count_html.object = (
            f"<span style='font-size:11px;color:#aaa'>{n} frames</span>"
        )
        self._load_original_frame(0)

    # ------------------------------------------------------------------
    # Frame navigation
    # ------------------------------------------------------------------

    def _on_slider_change(self, event: Any) -> None:
        idx = int(event.new)
        self._current_frame_idx = idx
        self._load_original_frame(idx)
        for col in self._columns:
            col.refresh(idx)
        self._update_query_bar_visibility()

    def _on_prev(self, event: Any) -> None:
        if self._frame_slider.value > 0:
            self._frame_slider.value -= 1

    def _on_next(self, event: Any) -> None:
        if self._frame_slider.value < self._frame_slider.end:
            self._frame_slider.value += 1

    def _load_original_frame(self, idx: int) -> None:
        """Load frame idx from zarr and display in the original frame pane."""
        from io import BytesIO
        from PIL import Image as PILImage

        if self._state.frames_zarr_path is None:
            return
        frame = _load_frame_rgb(Path(self._state.frames_zarr_path), idx)
        buf = BytesIO()
        PILImage.fromarray(frame).save(buf, format="PNG")
        self._original_pane.object = buf.getvalue()

    # ------------------------------------------------------------------
    # Query bar
    # ------------------------------------------------------------------

    def _update_query_bar_visibility(self) -> None:
        """Show query bar when any column has a ready queryable extractor."""
        any_queryable = any(c.is_queryable_and_ready() for c in self._columns)
        self._query_bar.visible = any_queryable
        self._query_btn.disabled = not any_queryable

    def _on_query(self, event: Any) -> None:
        """Run text query across all ready queryable columns in background."""
        if self._query_thread and self._query_thread.is_alive():
            return
        query_text = self._query_input.value.strip()
        if not query_text:
            return
        self._query_btn.disabled = True
        frame_idx = self._current_frame_idx
        self._query_thread = threading.Thread(
            target=self._run_query, args=(frame_idx, query_text), daemon=True
        )
        self._query_thread.start()

    def _run_query(self, frame_idx: int, query_text: str) -> None:
        """Background: call render_query on each ready queryable column."""
        try:
            for col in self._columns:
                if col.is_queryable_and_ready():
                    col.render_query(frame_idx, query_text)
        except Exception as exc:
            logger.exception("Query failed")
            self._op_log.error_op(f"Query error: {exc}")
        finally:
            self._update_query_bar_visibility()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def panel(self) -> pn.Column:
        """Return the full SemanticsPane Panel layout."""
        frame_selector = pn.Row(
            self._prev_btn,
            self._frame_slider,
            self._next_btn,
            self._frame_count_html,
        )
        grid = pn.Row(
            pn.Column(
                pn.pane.HTML("<b style='color:#aaa;font-size:12px'>Original</b>"),
                self._original_pane,
            ),
            *self._col_panels,
        )
        return pn.Column(
            pn.pane.HTML("<h3 style='color:#7ec8e3;margin:0 0 8px 0'>Semantics</h3>"),
            frame_selector,
            grid,
            self._query_bar,
            sizing_mode="stretch_width",
        )
```

- [ ] **Step 4.4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_semantics.py -v
```

Expected: all PASS.

- [ ] **Step 4.5: Commit**

```bash
git add collab_splats/dashboard/panes/semantics.py tests/dashboard/test_semantics.py
git commit -m "feat(dashboard): implement SemanticsPane — Tab 2 feature extraction viewer"
```

---

## Task 5: Wire SemanticsPane into app.py and update exports

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Modify: `collab_splats/dashboard/__init__.py`

- [ ] **Step 5.1: Update app.py**

In `collab_splats/dashboard/app.py`, add the import:

```python
from collab_splats.dashboard.panes.semantics import SemanticsPane
```

Then in `_build_panes` (the dict inside `__init__` that maps tab names to panes), replace:

```python
"Semantics": PlaceholderPane("Semantics", "Coming in Phase 2 — 2D feature extraction and comparison"),
```

with:

```python
"Semantics": SemanticsPane(state=self._state, op_log=self._op_log),
```

- [ ] **Step 5.2: Update `__init__.py`**

In `collab_splats/dashboard/__init__.py`, add `SemanticsPane` to the import and `__all__`:

```python
"""collab_splats interactive dashboard."""

from collab_splats.dashboard.app import App, run_app
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.panes.semantics import SemanticsPane
from collab_splats.dashboard.state import AppState

__all__ = ["App", "run_app", "AppState", "OperationLog", "SemanticsPane"]
```

- [ ] **Step 5.3: Run smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_smoke.py tests/dashboard/test_app.py -v
```

Expected: all PASS.

- [ ] **Step 5.4: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/integration -x -q 2>&1 | tail -30
```

Expected: all PASS (skip/xfail OK for heavy model tests).

- [ ] **Step 5.5: Commit**

```bash
git add collab_splats/dashboard/app.py collab_splats/dashboard/__init__.py
git commit -m "feat(dashboard): wire SemanticsPane into Tab 2"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - `state.frames` → `frames_zarr_path` ✓ (Task 1)
  - PreprocessPane writes frames.zarr ✓ (Task 2)
  - `extract_and_cache_from_zarr` ✓ (Task 3)
  - `features_to_rgb` static method ✓ (Task 3)
  - SemanticsPane skeleton + frame selector ✓ (Task 4)
  - ExtractorColumn with Run flow ✓ (Task 4)
  - Extractor memory management on dropdown change ✓ (Task 4 — `_on_method_change`)
  - Reactive slider → zarr reload ✓ (Task 4 — `_on_slider_change`)
  - Text query bar visibility + query execution ✓ (Task 4)
  - Viridis heatmap for query results ✓ (`_score_to_rgb`)
  - Wire into app.py ✓ (Task 5)
  - Export from `__init__.py` ✓ (Task 5)

- [x] **Placeholder scan:** No TBD/TODO in plan. All code complete.

- [x] **Type consistency:**
  - `_load_frame_rgb(frames_zarr_path: Path, idx: int) -> np.ndarray` — used consistently
  - `ExtractorColumn.refresh(frame_idx: int)` → called as `col.refresh(idx)` ✓
  - `ExtractorColumn.render_query(frame_idx, query_text)` → called correctly ✓
  - `BaseFeatureExtractor.features_to_rgb(feat: torch.Tensor) -> np.ndarray` ✓
  - `_score_to_rgb(score: np.ndarray) -> np.ndarray` ✓

- [x] **Note:** `extract_and_cache_from_zarr` in `_run_extraction` passes an `on_progress` kwarg that the method doesn't define in Task 3. Remove the `on_progress` arg from `_run_extraction` in `semantics.py` — use periodic log calls in the method itself (already present). The call in `_run_extraction` should be:

```python
zarr_path = self._extractor.extract_and_cache_from_zarr(
    frames_zarr_path=Path(self._state.frames_zarr_path),
    cache_dir=cache_dir,
)
```
