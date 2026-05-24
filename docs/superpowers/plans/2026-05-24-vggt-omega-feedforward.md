# VGGT-Omega Feedforward Creator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `VGGTOmegaCreator` (uninstantiable due to wrong abstract method name), add missing `max_points`, expose `resize_mode`/`resolution`/`enable_text_alignment` fields, and rename `image_preproc` → `resize_mode` across all three feedforward creators for consistency.

**Architecture:** Four independent tasks — VGGTX rename, MapAnything new fields, VGGTOmega fix+extend, notebook update. Each task is TDD: tests first (verified failing), then implementation, then green. Tasks 1-3 are independent and can run in any order; Task 4 depends on Tasks 1-3.

**Tech Stack:** Python 3.11, PyTorch, `@dataclass` inheritance, `pytest`, Jupyter notebook JSON editing via Python.

**Spec:** `docs/superpowers/specs/2026-05-24-vggt-omega-feedforward-design.md`

---

## File Map

| File | Task | Change |
|------|------|--------|
| `collab_splats/pointcloud/feedforward/vggtx.py` | 1 | Rename `image_preproc`→`resize_mode`, `"ratio"`→`"max_size"` |
| `tests/pointcloud/test_vggtx_preproc.py` | 1 | Update 5 tests for renamed field+value |
| `collab_splats/pointcloud/feedforward/mapanything.py` | 2 | Add `resize_mode`, `resolution`, `__post_init__`, wire `_preprocess` |
| `tests/pointcloud/test_mapanything_creator.py` | 2 | Add 4 new tests, update `load_images` mock |
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | 3 | Fix `_reproject_ba`→`_reproject`, add `max_points`, add 3 fields, `__post_init__` |
| `tests/pointcloud/test_vggt_omega_creator.py` | 3 | Rename `_reproject_ba`→`_reproject` everywhere, add 9 new tests |
| `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` | 4 | Add §7-§9 (Omega + three-way comparison) |

**Files not touched:** `base.py`, `wrappers.py`, `feedforward/__init__.py`, `pointcloud/__init__.py`.

---

## Task 1: VGGTXCreator — rename `image_preproc` → `resize_mode`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py:127-133,163-164,188-192`
- Modify: `tests/pointcloud/test_vggtx_preproc.py`

### Step 1: Update `test_vggtx_preproc.py` to use the new names (tests will fail)

Replace the entire contents of `tests/pointcloud/test_vggtx_preproc.py`:

```python
import pytest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch


def _make_fake_images(n=2, size=518):
    return torch.zeros(n, 3, size, size), torch.zeros(n, 6)


def _make_image_dir(tmp_path, n=2):
    for i in range(n):
        (tmp_path / f"frame_{i:04d}.jpg").touch()
    return tmp_path


def test_vggtx_creator_default_preproc_is_max_size():
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    c = VGGTXCreator()
    assert c.resize_mode == "max_size"


def test_vggtx_creator_accepts_square_preproc():
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    c = VGGTXCreator(resize_mode="square")
    assert c.resize_mode == "square"


def test_vggtx_creator_rejects_invalid_preproc():
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    with pytest.raises(ValueError, match="resize_mode"):
        VGGTXCreator(resize_mode="invalid")


def test_preprocess_max_size_calls_ratio_fn(tmp_path):
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    img_dir = _make_image_dir(tmp_path)
    c = VGGTXCreator(resize_mode="max_size")
    fake = _make_fake_images()
    with patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images_ratio", return_value=fake) as m, \
         patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images_square") as ms:
        c._preprocess(img_dir)
        assert m.called
        assert not ms.called


def test_preprocess_square_calls_square_fn(tmp_path):
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    img_dir = _make_image_dir(tmp_path)
    c = VGGTXCreator(resize_mode="square")
    fake = _make_fake_images()
    with patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images_square", return_value=fake) as m, \
         patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images_ratio") as mr:
        c._preprocess(img_dir)
        assert m.called
        assert not mr.called
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggtx_preproc.py -v 2>&1 | tail -20
```

Expected: 5 FAILED — `AttributeError: 'VGGTXCreator' object has no attribute 'resize_mode'` or `assert c.image_preproc == 'max_size'` failures.

- [ ] **Step 3: Rename field and values in `vggtx.py`**

In `collab_splats/pointcloud/feedforward/vggtx.py`, make these four edits:

**Edit 1** — field declaration (line ~128):
```python
# Before:
    image_preproc: str = "ratio"

# After:
    resize_mode: str = "max_size"
```

**Edit 2** — `__post_init__` validation (lines ~130-133):
```python
# Before:
    def __post_init__(self) -> None:
        if self.image_preproc not in ("ratio", "square"):
            raise ValueError(
                f"image_preproc must be 'ratio' or 'square', got {self.image_preproc!r}"
            )

# After:
    def __post_init__(self) -> None:
        if self.resize_mode not in ("max_size", "square"):
            raise ValueError(
                f"resize_mode must be 'max_size' or 'square', got {self.resize_mode!r}"
            )
```

**Edit 3** — docstring references (lines ~162-163 inside `_preprocess` docstring):
```python
# Before:
        aspect ratio (``image_preproc='ratio'``) or square-cropping
        (``image_preproc='square'``).  Stores original dimensions in

# After:
        aspect ratio (``resize_mode='max_size'``) or square-cropping
        (``resize_mode='square'``).  Stores original dimensions in
```

**Edit 4** — `_preprocess` branch (line ~189):
```python
# Before:
        if self.image_preproc == "ratio":

# After:
        if self.resize_mode == "max_size":
```

Also update the `Attributes` docstring in the class body. Find:
```python
        image_preproc:        Preprocessing mode: ``"ratio"`` (default, resize longest
                              side to 518 preserving aspect ratio) or ``"square"``
                              (center-crop + resize to 518×518).
```
Replace with:
```python
        resize_mode:          Preprocessing mode.
                              ``"max_size"`` (default): resize longest side to
                              ``VGGTX_IMG_LOAD_RESOLUTION`` (518), preserving AR.
                              ``"square"``: center-crop + resize to 518×518.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggtx_preproc.py -v 2>&1 | tail -10
```

Expected: 5 PASSED.

- [ ] **Step 5: Run the full vggtx test suite to check no regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggtx_creator.py tests/pointcloud/test_vggtx_preproc.py -v 2>&1 | tail -15
```

Expected: all PASSED.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggtx.py tests/pointcloud/test_vggtx_preproc.py
git commit -m "$(cat <<'EOF'
refactor(feedforward): rename VGGTXCreator.image_preproc → resize_mode

Breaking rename for consistency with 2D feature models. Value "ratio" → "max_size"
(same longest-side behavior, aligned name). "square" unchanged.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: MapAnythingCreator — add `resize_mode` + `resolution`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py:106-110,119-127`
- Modify: `tests/pointcloud/test_mapanything_creator.py`

- [ ] **Step 1: Add failing tests to `test_mapanything_creator.py`**

Append to the end of `tests/pointcloud/test_mapanything_creator.py`:

```python
########################################################################
########## resize_mode + resolution ####################################
########################################################################

def test_mapanything_resize_mode_default():
    """Default resize_mode is 'fixed'."""
    c = MapAnythingCreator()
    assert c.resize_mode == "fixed"
    assert c.resolution == 518


def test_mapanything_invalid_resize_mode_raises():
    """__post_init__ raises ValueError for unknown resize_mode."""
    with pytest.raises(ValueError, match="resize_mode"):
        MapAnythingCreator(resize_mode="bogus")


def test_mapanything_preprocess_fixed_mode_calls_load_images_with_fixed_mapping(tmp_path):
    """resize_mode='fixed' passes resize_mode='fixed_mapping' + resolution_set to load_images."""
    from PIL import Image as PILImage
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(2):
        PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_dir / f"f{i}.jpg")

    fake_view = {"img": __import__("torch").zeros(1, 3, 64, 64), "data_norm_type": "imagenet"}
    fake_views = [fake_view, fake_view]

    c = MapAnythingCreator(resize_mode="fixed", resolution=518)
    with patch("collab_splats.pointcloud.feedforward.mapanything.load_images",
               return_value=fake_views) as mock_li, \
         patch("collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
               return_value=fake_views), \
         patch("collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
               return_value=fake_views):
        c._preprocess(img_dir)

    call_kwargs = mock_li.call_args[1]
    assert call_kwargs.get("resize_mode") == "fixed_mapping"
    assert call_kwargs.get("resolution_set") == 518
    assert "size" not in call_kwargs


def test_mapanything_preprocess_longest_side_calls_load_images_with_size(tmp_path):
    """resize_mode='longest_side' passes resize_mode='longest_side' + size= to load_images."""
    from PIL import Image as PILImage
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(2):
        PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_dir / f"f{i}.jpg")

    fake_view = {"img": __import__("torch").zeros(1, 3, 64, 64), "data_norm_type": "imagenet"}
    fake_views = [fake_view, fake_view]

    c = MapAnythingCreator(resize_mode="longest_side", resolution=512)
    with patch("collab_splats.pointcloud.feedforward.mapanything.load_images",
               return_value=fake_views) as mock_li, \
         patch("collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference",
               return_value=fake_views), \
         patch("collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference",
               return_value=fake_views):
        c._preprocess(img_dir)

    call_kwargs = mock_li.call_args[1]
    assert call_kwargs.get("resize_mode") == "longest_side"
    assert call_kwargs.get("size") == 512
    assert "resolution_set" not in call_kwargs
```

- [ ] **Step 2: Run to verify tests fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py::test_mapanything_resize_mode_default tests/pointcloud/test_mapanything_creator.py::test_mapanything_invalid_resize_mode_raises tests/pointcloud/test_mapanything_creator.py::test_mapanything_preprocess_fixed_mode_calls_load_images_with_fixed_mapping tests/pointcloud/test_mapanything_creator.py::test_mapanything_preprocess_longest_side_calls_load_images_with_size -v 2>&1 | tail -15
```

Expected: 4 FAILED — `AttributeError` on `c.resize_mode`.

- [ ] **Step 3: Add fields, `__post_init__`, and `_MODE_MAP` to `mapanything.py`**

**Edit 1** — add module-level mode map constant. Find the `########` section divider before `── Creator ──` and add above the `@dataclass` line:

```python
# Maps our public resize_mode values to mapanything's load_images resize_mode strings
_MA_RESIZE_MODE_MAP: dict[str, str] = {
    "fixed": "fixed_mapping",
    "longest_side": "longest_side",
    "square": "square",
}
```

**Edit 2** — add fields to the dataclass. Find:
```python
    minibatch_size: int = 1
    _processed_views: Any = field(default=None, init=False, repr=False)
```
Replace with:
```python
    minibatch_size: int = 1
    resize_mode: str = "fixed"   # "fixed" (aspect-ratio lookup table), "longest_side", "square"
    resolution: int = 518         # resolution_set= for "fixed"; size= for "longest_side"/"square"
    _processed_views: Any = field(default=None, init=False, repr=False)
```

**Edit 3** — add `__post_init__` method. Insert between the field declarations and `_load_model`. Find:
```python
    def _load_model(self, device: str) -> Any:
        # Load pretrained model, move to device, set eval mode
```
Insert before it:
```python
    def __post_init__(self) -> None:
        if self.resize_mode not in _MA_RESIZE_MODE_MAP:
            raise ValueError(
                f"resize_mode must be one of {sorted(_MA_RESIZE_MODE_MAP)}, got {self.resize_mode!r}"
            )

```

**Edit 4** — update `_preprocess` to wire `resize_mode` and `resolution`. Find:
```python
        views = load_images([str(p) for p in image_paths])
```
Replace with:
```python
        # Map our public resize_mode to load_images' upstream name; pass resolution as the right kwarg
        upstream_mode = _MA_RESIZE_MODE_MAP[self.resize_mode]
        if self.resize_mode == "fixed":
            views = load_images(
                [str(p) for p in image_paths],
                resize_mode=upstream_mode,
                resolution_set=self.resolution,
            )
        else:
            views = load_images(
                [str(p) for p in image_paths],
                resize_mode=upstream_mode,
                size=self.resolution,
            )
```

**Edit 5** — update the class docstring `Attributes` block to document the new fields. Find the `Attributes:` section in `MapAnythingCreator` and append after `minibatch_size`:
```
        resize_mode:              Image resize strategy for ``load_images``.
                                  ``"fixed"`` (default): auto-selects the best HxW from a
                                  lookup table of patch-size-compatible resolutions based on
                                  the batch's average aspect ratio. ``resolution`` selects
                                  the lookup table (518 = DINOv2-aligned, 512 = ViT).
                                  ``"longest_side"``: resize so the longest side equals
                                  ``resolution`` px, preserving aspect ratio. Use when GPU
                                  memory is constrained.
                                  ``"square"``: resize all images to ``resolution × resolution``.
        resolution:               Lookup-table selector for ``"fixed"`` (518 or 512); target
                                  size in pixels for ``"longest_side"`` and ``"square"``.
```

- [ ] **Step 4: Run new tests to verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py::test_mapanything_resize_mode_default tests/pointcloud/test_mapanything_creator.py::test_mapanything_invalid_resize_mode_raises tests/pointcloud/test_mapanything_creator.py::test_mapanything_preprocess_fixed_mode_calls_load_images_with_fixed_mapping tests/pointcloud/test_mapanything_creator.py::test_mapanything_preprocess_longest_side_calls_load_images_with_size -v 2>&1 | tail -10
```

Expected: 4 PASSED.

- [ ] **Step 5: Run full mapanything test suite for regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py tests/pointcloud/test_mapanything.py -v 2>&1 | tail -15
```

Expected: all PASSED (existing `test_mapanything_full_pipeline_cpu_mock` should still pass because it mocks `load_images` and the default `resize_mode="fixed"` routes to the `fixed_mapping` path which uses `resolution_set=518`, matching the existing mock that accepts any kwargs).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/mapanything.py tests/pointcloud/test_mapanything_creator.py
git commit -m "$(cat <<'EOF'
feat(feedforward): add resize_mode + resolution to MapAnythingCreator

Exposes load_images() resize_mode and resolution_set/size parameters.
Aligns MapAnythingCreator field name with VGGTXCreator and 2D feature models.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: VGGTOmegaCreator — fix + extend

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggt_omega.py`
- Modify: `tests/pointcloud/test_vggt_omega_creator.py`

- [ ] **Step 1: Update test file — rename `_reproject_ba` → `_reproject` everywhere**

In `tests/pointcloud/test_vggt_omega_creator.py`, do a global find-and-replace:
- `_reproject_ba` → `_reproject` (method calls and test function names)
- The section header `########## _reproject_ba` → `########## _reproject`

Affected locations (search for `_reproject_ba`):
- Section header comment
- `def test_reproject_ba_returns_pts3d_colors` → `def test_reproject_returns_pts3d_colors`
- `def test_reproject_ba_passes_conf_threshold` → `def test_reproject_passes_conf_threshold`
- `def test_reproject_ba_passes_extrinsics_to_unproject` → `def test_reproject_passes_extrinsics_to_unproject`
- Any `creator._reproject_ba(` → `creator._reproject(`

- [ ] **Step 2: Add new failing tests to `test_vggt_omega_creator.py`**

Append after the existing `_reproject_ba` section (now renamed `_reproject`):

```python
########################################################################
########## New fields: resolution, resize_mode, enable_text_alignment ##
########################################################################

def test_vggt_omega_defaults():
    """Default fields: resize_mode='balanced', enable_text_alignment=False, resolution resolves to 512."""
    c = VGGTOmegaCreator()
    assert c.resize_mode == "balanced"
    assert c.enable_text_alignment is False
    assert c.resolution == 512  # None sentinel resolved to 512 by __post_init__


def test_resolution_none_standard_resolves_to_512():
    """resolution=None without text alignment resolves to 512 after __post_init__."""
    c = VGGTOmegaCreator()
    assert c.resolution == 512


def test_resolution_none_text_aligned_resolves_to_256():
    """resolution=None with enable_text_alignment=True resolves to 256."""
    c = VGGTOmegaCreator(enable_text_alignment=True)
    assert c.resolution == 256


def test_explicit_resolution_not_overridden():
    """Explicit resolution= is never overridden, even with enable_text_alignment=True."""
    c = VGGTOmegaCreator(resolution=768, enable_text_alignment=True)
    assert c.resolution == 768


def test_invalid_resize_mode_raises():
    """__post_init__ raises ValueError for unknown resize_mode."""
    with pytest.raises(ValueError, match="resize_mode"):
        VGGTOmegaCreator(resize_mode="bilinear")


def test_resize_mode_max_size_accepted():
    """resize_mode='max_size' is valid."""
    c = VGGTOmegaCreator(resize_mode="max_size")
    assert c.resize_mode == "max_size"


def test_load_model_passes_enable_alignment_false(tmp_path):
    """Default path: VGGTOmega constructed with enable_alignment=False."""
    ckpt_path = tmp_path / "model.pt"
    state_dict = {}
    import torch
    torch.save(state_dict, ckpt_path)

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega") as MockOmega, \
         patch("collab_splats.pointcloud.feedforward.vggt_omega.torch.load", return_value=state_dict):
        mock_instance = MagicMock()
        mock_instance.eval.return_value = mock_instance
        mock_instance.to.return_value = mock_instance
        MockOmega.return_value = mock_instance
        creator = VGGTOmegaCreator(model_path=str(ckpt_path))
        creator._load_model("cpu")

    MockOmega.assert_called_once_with(enable_alignment=False)


def test_load_model_passes_enable_alignment_true(tmp_path):
    """enable_text_alignment=True: VGGTOmega constructed with enable_alignment=True."""
    ckpt_path = tmp_path / "model.pt"
    state_dict = {}
    import torch
    torch.save(state_dict, ckpt_path)

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega") as MockOmega, \
         patch("collab_splats.pointcloud.feedforward.vggt_omega.torch.load", return_value=state_dict):
        mock_instance = MagicMock()
        mock_instance.eval.return_value = mock_instance
        mock_instance.to.return_value = mock_instance
        MockOmega.return_value = mock_instance
        creator = VGGTOmegaCreator(model_path=str(ckpt_path), enable_text_alignment=True)
        creator._load_model("cpu")

    MockOmega.assert_called_once_with(enable_alignment=True)


def test_postprocess_passes_max_points():
    """_postprocess passes self.max_points to unproject_and_filter_points."""
    n = 2
    raw = _make_raw_outputs(n)
    pts = np.zeros((5, 3), dtype=np.float32)
    colors = np.zeros((5, 3), dtype=np.uint8)
    pixel_indices = np.zeros((5, 3), dtype=np.int32)

    creator = VGGTOmegaCreator(max_points=12345)
    creator.image_paths = [Path(f"f{i}.jpg") for i in range(n)]
    creator.original_coords = np.zeros((n, 6), dtype=np.float32)
    creator.original_coords[:, -2:] = 4.0  # orig_w, orig_h

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.unproject_and_filter_points",
               return_value=(pts, colors, pixel_indices)) as mock_unproject, \
         patch("collab_splats.pointcloud.feedforward.vggt_omega._raw_to_world_points",
               return_value=(None, None)):
        creator._postprocess(raw)

    call_kwargs = mock_unproject.call_args[1]
    assert call_kwargs.get("max_points") == 12345


def test_reproject_passes_max_points():
    """_reproject passes self.max_points to unproject_and_filter_points."""
    n = 2
    raw = _make_raw_outputs(n)
    extrinsics_3x4 = np.tile(np.eye(4)[:3], (n, 1, 1)).astype(np.float32)
    intrinsics = np.tile(np.eye(3), (n, 1, 1)).astype(np.float32)
    pts = np.zeros((5, 3), dtype=np.float32)
    colors = np.zeros((5, 3), dtype=np.uint8)
    pixel_indices = np.zeros((5, 3), dtype=np.int32)

    creator = VGGTOmegaCreator(max_points=99999)
    with patch("collab_splats.pointcloud.feedforward.vggt_omega.unproject_and_filter_points",
               return_value=(pts, colors, pixel_indices)) as mock_unproject:
        creator._reproject(raw, extrinsics_3x4, intrinsics)

    call_kwargs = mock_unproject.call_args[1]
    assert call_kwargs.get("max_points") == 99999
```

- [ ] **Step 3: Run tests to verify the new ones fail (and renamed ones now pass structurally)**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggt_omega_creator.py -v 2>&1 | tail -30
```

Expected: renamed `_reproject` tests still fail with `TypeError: Can't instantiate abstract class VGGTOmegaCreator with abstract method _reproject`; new field tests fail with `AttributeError`.

- [ ] **Step 4: Implement all changes in `vggt_omega.py`**

**Edit 1** — remove the now-unused `VGGT_OMEGA_DEFAULT_RESOLUTION` constant usage and update the field block. Find:

```python
VGGT_OMEGA_DEFAULT_RESOLUTION = 512
```

Keep the constant (useful for documentation), but update the dataclass.

**Edit 2** — replace the dataclass field declarations. Find:
```python
    camera_model: str = "PINHOLE"
    model_path: str | None = None
    model_repo: str = VGGT_OMEGA_HF_REPO
    model_filename: str = VGGT_OMEGA_DEFAULT_FILENAME
    image_resolution: int = VGGT_OMEGA_DEFAULT_RESOLUTION
    conf_threshold: float = 50.0
```
Replace with:
```python
    camera_model: str = "PINHOLE"
    model_path: str | None = None
    model_repo: str = VGGT_OMEGA_HF_REPO
    model_filename: str = VGGT_OMEGA_DEFAULT_FILENAME
    resolution: int | None = None        # None → 512 standard, 256 text-aligned; explicit overrides
    resize_mode: str = "balanced"        # "balanced" (default) or "max_size" (lower VRAM)
    conf_threshold: float = 50.0
    enable_text_alignment: bool = False  # VGGTOmega(enable_alignment=True); auto-sets resolution=256 when None
```

**Edit 3** — add `__post_init__` method. Insert between the field declarations and `_load_model`. Find:
```python
    def _load_model(self, device: str) -> Any:
        """Load VGGT-Omega from local path or HuggingFace, move to device."""
```
Insert before it:
```python
    def __post_init__(self) -> None:
        # Validate resize_mode before any I/O
        if self.resize_mode not in {"balanced", "max_size"}:
            raise ValueError(
                f"resize_mode must be 'balanced' or 'max_size', got {self.resize_mode!r}"
            )
        # Resolve None sentinel: standard checkpoint → 512, text-aligned → 256
        if self.resolution is None:
            self.resolution = 256 if self.enable_text_alignment else 512
            logger.debug("VGGTOmegaCreator: resolved resolution=%d", self.resolution)

```

**Edit 4** — update `_load_model` to pass `enable_alignment`. Find:
```python
        model = VGGTOmega()
```
Replace with:
```python
        model = VGGTOmega(enable_alignment=self.enable_text_alignment)
```

**Edit 5** — update `_preprocess` to use `resolution` and `resize_mode`. Find:
```python
        images = load_and_preprocess_images(image_names, image_resolution=self.image_resolution)
```
Replace with:
```python
        images = load_and_preprocess_images(
            image_names, image_resolution=self.resolution, mode=self.resize_mode
        )
```

**Edit 6** — update `_postprocess` to pass `max_points`. Find the `unproject_and_filter_points` call in `_postprocess`:
```python
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=raw_outputs["intrinsics_downsampled"],
            conf_threshold=self.conf_threshold,
        )
```
Replace with:
```python
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=raw_outputs["intrinsics_downsampled"],
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
        )
```

**Edit 7** — rename `_reproject_ba` → `_reproject` and add `max_points`. Find:
```python
    def _reproject_ba(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-derive world-space points using bundle-adjusted camera poses."""
        # Re-run depth unprojection with refined extrinsics and intrinsics
        pts3d, colors, _pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsics_3x4,
            intrinsic=intrinsics,
            conf_threshold=self.conf_threshold,
        )
        return pts3d, colors  # pixel_indices unused; post-BA uses stored indices
```
Replace with:
```python
    def _reproject(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-derive world-space points using bundle-adjusted camera poses."""
        # Re-run depth unprojection with refined extrinsics and intrinsics
        pts3d, colors, _pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsics_3x4,
            intrinsic=intrinsics,
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
        )
        return pts3d, colors  # pixel_indices unused; post-BA uses stored indices
```

**Edit 8** — update the class docstring `Attributes` block. Find:
```python
        image_resolution: Target resolution for ``load_and_preprocess_images``.
                          512 for the standard checkpoint, 256 for text-aligned.
        conf_threshold:  Depth confidence percentile cutoff (0–100).  Points
                         below this percentile are discarded.  50.0 = top 50%.
```
Replace with:
```python
        resolution:      Target resolution passed as ``image_resolution=`` to
                         ``load_and_preprocess_images``.  ``None`` auto-selects:
                         512 for the standard checkpoint, 256 for text-aligned.
        resize_mode:     Preprocessing mode passed as ``mode=`` to
                         ``load_and_preprocess_images``.
                         ``"balanced"`` (default): crops/pads to preserve AR.
                         ``"max_size"``: resize longest side to ``resolution``,
                         no crop; lower VRAM (~512×336 for 3:2 at res=512).
        conf_threshold:  Depth confidence percentile cutoff (0–100).  Points
                         below this percentile are discarded.  50.0 = top 50%.
        enable_text_alignment: When True, constructs ``VGGTOmega(enable_alignment=True)``
                         for the text-aligned checkpoint and auto-sets ``resolution=256``
                         when ``resolution`` is ``None``.
```

- [ ] **Step 5: Verify the test_vggt_omega_creator test suite passes**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggt_omega_creator.py -v 2>&1 | tail -30
```

Expected: all PASSED (previously-failing structural tests now pass because `_reproject` exists; new field tests pass; renamed `_reproject` tests pass).

- [ ] **Step 6: Quick import smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud.feedforward import VGGTOmegaCreator
c = VGGTOmegaCreator()
print(f'resolution={c.resolution}, resize_mode={c.resize_mode!r}, enable_text_alignment={c.enable_text_alignment}')
c2 = VGGTOmegaCreator(enable_text_alignment=True)
print(f'text-aligned resolution={c2.resolution}')
print('OK')
"
```

Expected:
```
resolution=512, resize_mode='balanced', enable_text_alignment=False
text-aligned resolution=256
OK
```

- [ ] **Step 7: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggt_omega.py tests/pointcloud/test_vggt_omega_creator.py
git commit -m "$(cat <<'EOF'
fix(feedforward): fix VGGTOmegaCreator + add resize_mode/resolution/enable_text_alignment

Critical: rename _reproject_ba → _reproject (abstract method mismatch blocked
all instantiation). Also: add max_points to both unproject calls, add
resolution/resize_mode/enable_text_alignment fields with __post_init__ validation.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Notebook — add VGGT-Omega §§7-9

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb`

The notebook is JSON. Edit it via Python script to avoid corrupting structure. Current cell layout:

| Index | Type | Content |
|-------|------|---------|
| 0 | markdown | Title (mentions VGGT-X and MapAnything) |
| 1 | code | `%load_ext autoreload` |
| 2 | code | imports (VGGTXCreator, MapAnythingCreator, ...) |
| 3 | code | config + device |
| 4 | markdown | §1 — VGGT-X Reconstruction |
| 5 | code | zarr cache-or-run VGGTX |
| 6 | markdown | §2 — VGGT-X Post-processing |
| 7 | code | clean_and_extract_result vggt |
| 8 | markdown | §3 — VGGT-X Pointcloud Viewer |
| 9 | code | visualize_splat vggt |
| 10 | markdown | §4 — MapAnything Reconstruction |
| 11 | code | zarr cache-or-run MapAnything |
| 12 | markdown | §5 — MapAnything Post-processing |
| 13 | code | clean_and_extract_result ma |
| 14 | markdown | §6 — MapAnything Pointcloud Viewer |
| 15 | code | visualize_splat ma |
| 16 | markdown | §7 — Side-by-side Comparison |
| 17 | code | stats table (two models) |
| 18 | markdown | §8 — Camera Pose Overlay |
| 19 | code | camera frustums (two models) |

Goal: insert 4 new cells after index 15; update cells 0, 2, 16-19.

- [ ] **Step 1: Update title cell (cell 0) and imports cell (cell 2)**

Run this Python script:

```bash
/opt/conda/envs/nerfstudio/bin/python3 - << 'PYEOF'
import json
from pathlib import Path

nb_path = Path("/workspace/collab-splats/docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb")
with open(nb_path) as f:
    nb = json.load(f)

# Cell 0: update title to mention all three methods
nb["cells"][0]["source"] = [
    "# Feedforward Pointcloud Reconstruction\n",
    "\n",
    "This tutorial runs **VGGT-X**, **MapAnything**, and **VGGT-Omega** on keyframes extracted from a real\n",
    "video, then compares the resulting pointclouds side by side.\n",
    "\n",
    "**Prerequisite:** [Keyframe Extraction](../preprocessing/keyframe_extraction.ipynb)\n",
    "— this notebook uses the same video and the same `score_all_frames` API to select frames.",
]

# Cell 2: add VGGTOmegaCreator to imports
src = "".join(nb["cells"][2]["source"])
src = src.replace(
    "from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator",
    "from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator",
)
nb["cells"][2]["source"] = src

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)
print("cells 0 and 2 updated")
PYEOF
```

- [ ] **Step 2: Insert 4 new cells after cell 15 (§7 Omega Reconstruction, §8 Omega Viewer)**

```bash
/opt/conda/envs/nerfstudio/bin/python3 - << 'PYEOF'
import json, uuid
from pathlib import Path

nb_path = Path("/workspace/collab-splats/docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb")
with open(nb_path) as f:
    nb = json.load(f)

def md_cell(source_lines):
    return {"cell_type": "markdown", "id": str(uuid.uuid4())[:8], "metadata": {}, "source": source_lines}

def code_cell(source_lines):
    return {"cell_type": "code", "execution_count": None, "id": str(uuid.uuid4())[:8],
            "metadata": {}, "outputs": [], "source": source_lines}

new_cells = [
    md_cell([
        "## §7 — VGGT-Omega Reconstruction\n",
        "\n",
        "VGGT-Omega jointly predicts camera poses and per-frame depth maps. Uses a balanced\n",
        "aspect-ratio-preserving resize strategy by default. Loads from zarr cache if available.\n",
    ]),
    code_cell([
        "_omega_cache = CACHE_DIR / \"vggt_omega\" / \"reconstruction.zarr\"\n",
        "_omega_cache.parent.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "if _omega_cache.exists():\n",
        "    result_omega = FeedforwardResult.load_zarr(_omega_cache)\n",
        "    print(f\"Loaded VGGT-Omega result from cache  ({result_omega.pts3d.shape[0]:,} pts)\")\n",
        "else:\n",
        "    result_omega = VGGTOmegaCreator().run(IMAGES, device)\n",
        "    result_omega.save_zarr(_omega_cache)\n",
        "    print(f\"VGGT-Omega done  →  saved to {_omega_cache}\")\n",
    ]),
    md_cell([
        "## §8 — VGGT-Omega Post-processing and Visualisation\n",
        "\n",
        "Removes outliers and downsamples, then renders the pointcloud with camera frustums.\n",
        "Green frustums show the VGGT-Omega predicted camera positions.\n",
    ]),
    code_cell([
        "pts3d_omega, colors_omega, conf_omega_mean, conf_omega_std = clean_and_extract_result(result_omega, \"VGGT-Omega\")\n",
        "\n",
        "cloud_omega = pointcloud_to_polydata(pts3d_omega, RGB=colors_omega)\n",
        "pl = visualize_splat(\n",
        "    cloud_omega,\n",
        "    aligned_cameras=extrinsics_to_c2w(result_omega.extrinsics),\n",
        "    mesh_kwargs=PCD_KWARGS,\n",
        "    camera_kwargs=CAMERA_KWARGS,\n",
        "    viz_kwargs=VIZ_KWARGS,\n",
        ")\n",
        "pl.show()\n",
    ]),
]

# Insert after cell 15 (§6 MapAnything Viewer code)
nb["cells"] = nb["cells"][:16] + new_cells + nb["cells"][16:]

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Inserted 4 cells. Total cells: {len(nb['cells'])}")
PYEOF
```

- [ ] **Step 3: Update old §7/§8 → §9 Three-way Comparison + §10 Camera Pose Overlay**

After insertion, the old §7/§8 are now at indices 20-23. Update them to include VGGT-Omega:

```bash
/opt/conda/envs/nerfstudio/bin/python3 - << 'PYEOF'
import json
from pathlib import Path

nb_path = Path("/workspace/collab-splats/docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb")
with open(nb_path) as f:
    nb = json.load(f)

# Old §7 markdown is now at index 20
nb["cells"][20]["source"] = [
    "## §9 — Three-way Comparison\n",
    "\n",
    "Tabulates raw vs. filtered point counts and confidence statistics for all three models.\n",
    "Higher confidence mean = more coherent depth predictions.\n",
]

# Old §7 code is now at index 21 — update to include Omega row
nb["cells"][21]["source"] = [
    "col = 14\n",
    "print(f\"{'Model':<{col}} {'Pts raw':>10} {'Pts filt':>10} {'Conf mean':>10} {'Conf std':>10}\")\n",
    "print(\"-\" * (col + 44))\n",
    "for name, res, pts, conf_mean, conf_std in [\n",
    "    (\"VGGT-X\",    result_vggt,  pts3d_vggt,  conf_vggt_mean,  conf_vggt_std),\n",
    "    (\"MapAnything\", result_ma,   pts3d_ma,    conf_ma_mean,    conf_ma_std),\n",
    "    (\"VGGT-Omega\",  result_omega, pts3d_omega, conf_omega_mean, conf_omega_std),\n",
    "]:\n",
    "    print(f\"{name:<{col}} {res.pts3d.shape[0]:>10,} {pts.shape[0]:>10,} {conf_mean:>10.3f} {conf_std:>10.3f}\")\n",
]

# Old §8 markdown is now at index 22
nb["cells"][22]["source"] = [
    "## §10 — Camera Pose Overlay\n",
    "\n",
    "Overlays camera frustums from all three models in a single scene — blue for VGGT-X,\n",
    "orange for MapAnything, green for VGGT-Omega. Misalignment between colours indicates\n",
    "scale or pose drift between methods.\n",
]

# Old §8 code is now at index 23 — update to include Omega frustums
nb["cells"][23]["source"] = [
    "pl = pv.Plotter()\n",
    "add_camera_frustums(pl, result_vggt.extrinsics, color=\"cornflowerblue\")\n",
    "add_camera_frustums(pl, result_ma.extrinsics, color=\"orange\")\n",
    "add_camera_frustums(pl, result_omega.extrinsics, color=\"mediumseagreen\")\n",
    "pl.show()\n",
]

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)
print("§9 and §10 updated")
PYEOF
```

- [ ] **Step 4: Verify notebook structure**

```bash
/opt/conda/envs/nerfstudio/bin/python3 -c "
import json
with open('/workspace/collab-splats/docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb') as f:
    nb = json.load(f)
for i, c in enumerate(nb['cells']):
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    print(f'cell {i:2d} [{c[\"cell_type\"]:8s}]: {src[:80].strip()!r}')
"
```

Expected output (24 cells total):
```
cell  0 [markdown]: '# Feedforward Pointcloud Reconstruction...'
cell  2 [code    ]: '...VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator...'
cell 16 [markdown]: '## §7 — VGGT-Omega Reconstruction...'
cell 17 [code    ]: '_omega_cache = CACHE_DIR / "vggt_omega" / "reconstruction.zarr"...'
cell 18 [markdown]: '## §8 — VGGT-Omega Post-processing and Visualisation...'
cell 19 [code    ]: 'pts3d_omega, colors_omega, ...'
cell 20 [markdown]: '## §9 — Three-way Comparison...'
cell 21 [code    ]: 'col = 14...'
cell 22 [markdown]: '## §10 — Camera Pose Overlay...'
cell 23 [code    ]: 'pl = pv.Plotter()...'
```

- [ ] **Step 5: Verify notebook is valid JSON**

```bash
/opt/conda/envs/nerfstudio/bin/python3 -c "
import json
with open('/workspace/collab-splats/docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb') as f:
    nb = json.load(f)
print(f'Valid JSON. {len(nb[\"cells\"])} cells, nbformat {nb[\"nbformat\"]}.{nb[\"nbformat_minor\"]}')
"
```

Expected: `Valid JSON. 24 cells, nbformat 4.X`

- [ ] **Step 6: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "$(cat <<'EOF'
feat(notebook): add VGGT-Omega §§7-9 to feedforward_methods.ipynb

Adds Omega reconstruction + postproc/viewer sections and expands
side-by-side comparison and camera overlay to all three models.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Final: Full test suite

- [ ] **Run the feedforward-related test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_vggtx_creator.py \
  tests/pointcloud/test_vggtx_preproc.py \
  tests/pointcloud/test_mapanything_creator.py \
  tests/pointcloud/test_vggt_omega_creator.py \
  tests/pointcloud/test_feedforward_shared.py \
  tests/pointcloud/test_feedforward_reproject.py \
  -v 2>&1 | tail -20
```

Expected: all PASSED, no regressions.

- [ ] **Verify import from package root**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
vx = VGGTXCreator()
ma = MapAnythingCreator()
vo = VGGTOmegaCreator()
assert vx.resize_mode == 'max_size'
assert ma.resize_mode == 'fixed' and ma.resolution == 518
assert vo.resize_mode == 'balanced' and vo.resolution == 512
print('All imports and defaults OK')
"
```

Expected: `All imports and defaults OK`
