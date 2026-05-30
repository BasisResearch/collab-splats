# VGGT-Omega Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `VGGTOmegaCreator` as a third feedforward backend alongside `VGGTXCreator` and `MapAnythingCreator`, installable via `setup/feedforward.sh`, supporting the full pipeline including BundleAdjustment, LoopClosure, and feature lifting.

**Architecture:** New standalone `collab_splats/pointcloud/feedforward/vggt_omega.py` extending `BaseFeedforwardCreator` via the 5-step template method. vggt-omega installed as a git submodule in `third_party/vggt-omega` (pinned commit) with `--no-deps` to bypass its `numpy<2` metadata constraint. Checkpoint auto-downloaded via `hf_hub_download` from the gated `facebook/VGGT-Omega` HF repo, or loaded from a local `.pt` path.

**Tech Stack:** Python 3.11, PyTorch 2.4+cu121, vggt-omega (`vggt_omega` package), huggingface-hub, pycolmap, numpy 2.4.6.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | **Create** | `_compute_omega_original_coords` helper + `VGGTOmegaCreator` |
| `collab_splats/pointcloud/feedforward/__init__.py` | **Modify** | Export `VGGTOmegaCreator` |
| `setup/feedforward.sh` | **Modify** | Add vggt-omega submodule install block + smoke test |
| `tests/pointcloud/test_vggt_omega_creator.py` | **Create** | Unit tests for all abstract methods |

---

## Task 1: Add vggt-omega git submodule

**Files:**
- Modify: `.gitmodules`
- Create: `third_party/vggt-omega/` (submodule)

- [ ] **Step 1: Add the submodule**

```bash
cd /workspace/collab-splats
git submodule add https://github.com/facebookresearch/vggt-omega.git third_party/vggt-omega
```

Expected output: `Cloning into '/workspace/collab-splats/third_party/vggt-omega'...`

- [ ] **Step 2: Verify submodule registered**

```bash
cat /workspace/collab-splats/.gitmodules | grep vggt-omega
```

Expected output contains:
```
[submodule "third_party/vggt-omega"]
	path = third_party/vggt-omega
	url = https://github.com/facebookresearch/vggt-omega.git
```

- [ ] **Step 3: Commit**

```bash
git add .gitmodules third_party/vggt-omega
git commit -m "chore(deps): add vggt-omega git submodule"
```

---

## Task 2: Update setup/feedforward.sh

**Files:**
- Modify: `setup/feedforward.sh`

The existing script has sections for vggt-x, mapanything, loop closure deps, and a smoke test. Add a new vggt-omega block after mapanything and update the smoke test.

- [ ] **Step 1: Read the current smoke test line**

```bash
grep -n "VGGTXCreator\|MapAnythingCreator\|smoke" /workspace/collab-splats/setup/feedforward.sh
```

Note the exact line numbers for the smoke test block.

- [ ] **Step 2: Add vggt-omega install block before the smoke test**

Find this block in `setup/feedforward.sh`:
```bash
# collab-splats feedforward extras (non-git deps declared in pyproject.toml)
$PIP install -e '.[feedforward]' --no-deps -q
```

Add the vggt-omega block immediately before it:
```bash
# vggt-omega
# --no-deps: skips numpy<2 metadata constraint; env has numpy 2.4.x (same bypass as VGGT-X)
echo "=== Installing vggt-omega ==="
if [ ! -f "$SCRIPT_DIR/third_party/vggt-omega/pyproject.toml" ]; then
    echo "ERROR: third_party/vggt-omega submodule not initialized"
    echo "       Run: git submodule update --init third_party/vggt-omega"
    exit 1
fi
$PIP install --no-deps -e "$SCRIPT_DIR/third_party/vggt-omega" -q

```

- [ ] **Step 3: Update the smoke test**

Find the existing smoke test line:
```bash
$PYTHON -c "
from collab_splats.pointcloud import VGGTXCreator, MapAnythingCreator
print('[OK] VGGTXCreator and MapAnythingCreator import successfully')
"
```

Replace with:
```bash
$PYTHON -c "
from collab_splats.pointcloud import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
print('[OK] VGGTXCreator, MapAnythingCreator, and VGGTOmegaCreator import successfully')
"
```

- [ ] **Step 4: Install vggt-omega in the dev env**

```bash
/opt/conda/envs/nerfstudio/bin/pip install --no-deps -e /workspace/collab-splats/third_party/vggt-omega -q
```

Expected: no errors, last line `Successfully installed vggt-omega-0.0.1`.

- [ ] **Step 5: Verify package is importable**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from vggt_omega.models import VGGTOmega; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add setup/feedforward.sh
git commit -m "feat(deps): add vggt-omega to setup/feedforward.sh"
```

---

## Task 3: Write failing tests for VGGTOmegaCreator

**Files:**
- Create: `tests/pointcloud/test_vggt_omega_creator.py`

Tests cover: defaults, inheritance, preprocessing (`_compute_omega_original_coords`), `_forward` raw output format, `_postprocess` output types, `_load_model` with explicit path, `extract_intermediate_features` hook/cleanup. All tests mock the model and checkpoint — no GPU or HF download required.

- [ ] **Step 1: Write the test file**

```python
"""Tests for VGGTOmegaCreator.

All tests mock vggt_omega.models and checkpoint loading — no GPU or HF download required.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator, FeedforwardResult
from collab_splats.pointcloud.feedforward.vggt_omega import (
    VGGTOmegaCreator,
    _compute_omega_original_coords,
)


########################################################################
########## Helpers #####################################################
########################################################################

def _make_raw_outputs(n: int = 2, h: int = 4, w: int = 4) -> dict:
    """Minimal raw_outputs dict matching VGGTOmegaCreator._forward output format."""
    return {
        "images": torch.zeros(n, 3, h, w),
        "extrinsic": np.tile(np.eye(4)[:3], (n, 1, 1)).astype(np.float32),
        "intrinsics": np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
        "intrinsics_downsampled": np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
        "depth": np.ones((n, h, w), dtype=np.float32),
        "depth_conf": np.ones((n, h, w), dtype=np.float32) * 100.0,
    }


def _make_mock_model(n_frames: int = 2, num_heads: int = 2, n_blocks: int = 2):
    """VGGTOmega-shaped mock with real qkv linears so hooks fire."""
    total_dim = num_heads * 8  # head_dim=8
    n_tokens = 10

    # Build inter_frame_blocks with real qkv linears
    inter_blocks = []
    for _ in range(n_blocks):
        block = MagicMock()
        block.attn.num_heads = num_heads
        block.attn.qkv = nn.Linear(total_dim, total_dim * 3, bias=False)
        inter_blocks.append(block)

    def mock_forward(images):
        # Trigger inter_frame_blocks[-1].attn.qkv so the hook fires
        B = 1
        x = torch.randn(B, n_tokens, total_dim)
        inter_blocks[-1].attn.qkv(x)
        return {
            "pose_enc": torch.zeros(B, n_frames, 9),
            "depth": torch.ones(B, n_frames, h, w),
            "depth_conf": torch.ones(B, n_frames, h, w) * 100.0,
            "images": images if images.ndim == 5 else images.unsqueeze(0),
        }

    model = MagicMock()
    model.aggregator.inter_frame_blocks = inter_blocks
    model.side_effect = mock_forward
    param = nn.Parameter(torch.zeros(1))
    model.parameters = lambda: iter([param])
    return model


########################################################################
########## Structural tests ############################################
########################################################################

def test_vggt_omega_creator_is_feedforward_creator():
    assert issubclass(VGGTOmegaCreator, BaseFeedforwardCreator)


def test_vggt_omega_creator_defaults():
    c = VGGTOmegaCreator()
    assert c.camera_model == "PINHOLE"
    assert c.model_path is None
    assert c.model_repo == "facebook/VGGT-Omega"
    assert c.model_filename == "vggt_omega_1b_512.pt"
    assert c.image_resolution == 512
    assert c.conf_threshold == 50.0


def test_vggt_omega_creator_missing_image_dir_raises(tmp_path):
    c = VGGTOmegaCreator()
    with pytest.raises(FileNotFoundError):
        c.reconstruct(tmp_path / "nonexistent", tmp_path / "out")


def test_vggt_omega_has_logger():
    import collab_splats.pointcloud.feedforward.vggt_omega as mod
    assert hasattr(mod, "logger")
    assert isinstance(mod.logger, logging.Logger)


########################################################################
########## _compute_omega_original_coords ##############################
########################################################################

def test_compute_omega_original_coords_normal_ar(tmp_path):
    """Square image → no crop; coords are full-frame."""
    img_path = tmp_path / "square.png"
    import PIL.Image
    PIL.Image.new("RGB", (64, 64), color=0).save(img_path)

    coords = _compute_omega_original_coords([img_path])
    assert coords.shape == (1, 6)
    assert coords.dtype == np.float32
    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    assert tl_x == pytest.approx(0.0)
    assert tl_y == pytest.approx(0.0)
    assert cr_x == pytest.approx(64.0)
    assert cr_y == pytest.approx(64.0)
    assert orig_w == pytest.approx(64.0)
    assert orig_h == pytest.approx(64.0)


def test_compute_omega_original_coords_tall_image_crops_height(tmp_path):
    """Tall image (AR > 2.0) → height is center-cropped."""
    img_path = tmp_path / "tall.png"
    import PIL.Image
    PIL.Image.new("RGB", (100, 400), color=0).save(img_path)  # AR = 4.0 > 2.0

    coords = _compute_omega_original_coords([img_path])
    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    # crop_h = 100 * 2.0 = 200; tl_y = (400 - 200) / 2 = 100; cr_y = 100 + 200 = 300
    assert tl_x == pytest.approx(0.0)
    assert tl_y == pytest.approx(100.0)
    assert cr_x == pytest.approx(100.0)
    assert cr_y == pytest.approx(300.0)
    assert orig_w == pytest.approx(100.0)
    assert orig_h == pytest.approx(400.0)


def test_compute_omega_original_coords_wide_image_crops_width(tmp_path):
    """Wide image (AR < 0.5) → width is center-cropped."""
    img_path = tmp_path / "wide.png"
    import PIL.Image
    PIL.Image.new("RGB", (400, 100), color=0).save(img_path)  # AR = 0.25 < 0.5

    coords = _compute_omega_original_coords([img_path])
    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    # crop_w = 100 / 0.5 = 200; tl_x = (400 - 200) / 2 = 100; cr_x = 100 + 200 = 300
    assert tl_x == pytest.approx(100.0)
    assert tl_y == pytest.approx(0.0)
    assert cr_x == pytest.approx(300.0)
    assert cr_y == pytest.approx(100.0)


def test_compute_omega_original_coords_multiple_images(tmp_path):
    """Multiple images → coords shape (N, 6)."""
    import PIL.Image
    paths = []
    for i, (w, h) in enumerate([(64, 64), (100, 400), (400, 100)]):
        p = tmp_path / f"img{i}.png"
        PIL.Image.new("RGB", (w, h), color=0).save(p)
        paths.append(p)

    coords = _compute_omega_original_coords(paths)
    assert coords.shape == (3, 6)
    assert coords.dtype == np.float32


########################################################################
########## _preprocess #################################################
########################################################################

def test_preprocess_empty_dir_raises(tmp_path):
    creator = VGGTOmegaCreator()
    with pytest.raises(FileNotFoundError, match="No images found"):
        creator._preprocess(tmp_path)


def test_preprocess_returns_correct_shapes(tmp_path):
    """_preprocess returns (views [N,3,H,W], paths, original_coords [N,6])."""
    import PIL.Image
    for i in range(3):
        PIL.Image.new("RGB", (64, 64), color=i * 80).save(tmp_path / f"frame_{i:04d}.jpg")

    creator = VGGTOmegaCreator(image_resolution=64)
    with patch("collab_splats.pointcloud.feedforward.vggt_omega.load_and_preprocess_images",
               return_value=torch.zeros(3, 3, 64, 64)) as mock_load:
        views, image_paths, original_coords = creator._preprocess(tmp_path)

    assert views.shape == (3, 3, 64, 64)
    assert len(image_paths) == 3
    assert original_coords.shape == (3, 6)
    assert original_coords.dtype == np.float32
    # Verify image_paths are sorted by name
    assert [p.name for p in image_paths] == sorted(p.name for p in image_paths)
    # Verify load_and_preprocess_images called with correct resolution
    mock_load.assert_called_once()
    _, kwargs = mock_load.call_args
    assert kwargs.get("image_resolution") == 64 or mock_load.call_args[0][1] == 64


########################################################################
########## _forward ####################################################
########################################################################

def test_forward_output_keys(tmp_path):
    """_forward returns dict with required keys for downstream processing."""
    n, h, w = 2, 8, 8

    mock_model = MagicMock()
    mock_model.side_effect = lambda imgs: {
        "pose_enc": torch.zeros(1, n, 9),
        "depth": torch.ones(1, n, h, w),
        "depth_conf": torch.ones(1, n, h, w) * 80.0,
        "images": imgs if imgs.ndim == 5 else imgs.unsqueeze(0),
    }
    param = nn.Parameter(torch.zeros(1))
    mock_model.parameters = lambda: iter([param])

    creator = VGGTOmegaCreator()
    creator.original_coords = np.array([[0, 0, w, h, w, h]] * n, dtype=np.float32)

    views = torch.zeros(n, 3, h, w)
    with patch("collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
               return_value=(torch.zeros(1, n, 3, 4), torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, n, 3, 3))):
        raw = creator._forward(mock_model, views)

    required_keys = {"images", "extrinsic", "intrinsics", "intrinsics_downsampled", "depth", "depth_conf"}
    assert required_keys.issubset(raw.keys())


def test_forward_extrinsic_shape(tmp_path):
    """extrinsic in raw_outputs is (N, 3, 4) float32 numpy."""
    n, h, w = 3, 8, 8

    mock_model = MagicMock()
    mock_model.side_effect = lambda imgs: {
        "pose_enc": torch.zeros(1, n, 9),
        "depth": torch.ones(1, n, h, w),
        "depth_conf": torch.ones(1, n, h, w),
        "images": imgs if imgs.ndim == 5 else imgs.unsqueeze(0),
    }
    param = nn.Parameter(torch.zeros(1))
    mock_model.parameters = lambda: iter([param])

    creator = VGGTOmegaCreator()
    creator.original_coords = np.array([[0, 0, w, h, w, h]] * n, dtype=np.float32)

    # encoding_to_camera returns (1, N, 3, 4) extrinsics after batch dim
    ext_tensor = torch.zeros(1, n, 3, 4)
    intr_tensor = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, n, 3, 3).contiguous()
    with patch("collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
               return_value=(ext_tensor, intr_tensor)):
        raw = creator._forward(mock_model, torch.zeros(n, 3, h, w))

    assert raw["extrinsic"].shape == (n, 3, 4)
    assert raw["extrinsic"].dtype == np.float32
    assert raw["intrinsics"].shape == (n, 3, 3)


########################################################################
########## _postprocess ################################################
########################################################################

def test_postprocess_returns_feedforward_result(tmp_path):
    n = 2
    image_paths = [tmp_path / f"frame_{i:04d}.jpg" for i in range(n)]
    raw = _make_raw_outputs(n)

    pts = np.zeros((5, 3), dtype=np.float32)
    colors = np.zeros((5, 3), dtype=np.uint8)
    pixel_indices = np.zeros((5, 3), dtype=np.int32)

    creator = VGGTOmegaCreator()
    creator.image_paths = image_paths
    creator.original_coords = np.zeros((n, 6), dtype=np.float32)
    creator.model = MagicMock()
    creator.model.parameters = lambda: iter([nn.Parameter(torch.zeros(1))])

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.unproject_and_filter_points",
               return_value=(pts, colors, pixel_indices)):
        result = creator._postprocess(raw)

    assert isinstance(result, FeedforwardResult)
    assert result.pts3d.shape == (5, 3)
    assert result.colors.shape == (5, 3)
    assert result.extrinsics.shape == (n, 4, 4)
    assert result.intrinsics.shape == (n, 3, 3)
    assert result.model_width == 4
    assert result.model_height == 4
    assert result.image_paths == image_paths


def test_postprocess_world_points_populated(tmp_path):
    """world_points always populated (needed by BundleAdjustment wrapper)."""
    n, h, w = 2, 4, 4
    raw = _make_raw_outputs(n, h, w)
    pts = np.zeros((5, 3), dtype=np.float32)
    colors = np.zeros((5, 3), dtype=np.uint8)
    pixel_indices = np.zeros((5, 3), dtype=np.int32)

    creator = VGGTOmegaCreator()
    creator.image_paths = [tmp_path / f"{i}.jpg" for i in range(n)]
    creator.original_coords = np.zeros((n, 6), dtype=np.float32)
    creator.model = MagicMock()
    creator.model.parameters = lambda: iter([nn.Parameter(torch.zeros(1))])

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.unproject_and_filter_points",
               return_value=(pts, colors, pixel_indices)):
        result = creator._postprocess(raw)

    assert result.world_points is not None
    assert result.world_points.shape == (n, h, w, 3)
    assert result.conf is not None
    assert result.images is not None


########################################################################
########## _load_model #################################################
########################################################################

def test_load_model_with_explicit_path(tmp_path):
    """_load_model with model_path skips hf_hub_download."""
    # Create a fake checkpoint file
    ckpt_path = tmp_path / "fake.pt"
    fake_state = {"aggregator.depth": torch.tensor(1.0)}
    torch.save(fake_state, ckpt_path)

    creator = VGGTOmegaCreator(model_path=str(ckpt_path))

    mock_model_instance = MagicMock()
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.to.return_value = mock_model_instance

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega",
               return_value=mock_model_instance) as mock_cls, \
         patch("collab_splats.pointcloud.feedforward.vggt_omega.hf_hub_download") as mock_hf:
        creator._load_model("cpu")

    mock_hf.assert_not_called()
    mock_cls.assert_called_once()
    mock_model_instance.load_state_dict.assert_called_once()
    mock_model_instance.eval.assert_called_once()


def test_load_model_nonexistent_path_raises(tmp_path):
    creator = VGGTOmegaCreator(model_path=str(tmp_path / "nonexistent.pt"))
    with pytest.raises(FileNotFoundError):
        creator._load_model("cpu")


def test_load_model_without_path_calls_hf_download(tmp_path):
    """_load_model with model_path=None downloads from HuggingFace."""
    ckpt_path = tmp_path / "downloaded.pt"
    torch.save({}, ckpt_path)

    creator = VGGTOmegaCreator()

    mock_model_instance = MagicMock()
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.to.return_value = mock_model_instance

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.VGGTOmega",
               return_value=mock_model_instance), \
         patch("collab_splats.pointcloud.feedforward.vggt_omega.hf_hub_download",
               return_value=str(ckpt_path)) as mock_hf:
        creator._load_model("cpu")

    mock_hf.assert_called_once_with(
        repo_id="facebook/VGGT-Omega",
        filename="vggt_omega_1b_512.pt",
    )


########################################################################
########## extract_intermediate_features ###############################
########################################################################

def test_extract_intermediate_features_returns_q_k_poses():
    """Hook fires on inter_frame_blocks[-1]; returns q, k, poses."""
    model = _make_mock_model(n_frames=2, h=16, w=16, num_heads=2, n_blocks=4)
    creator = VGGTOmegaCreator.__new__(VGGTOmegaCreator)
    creator.model = model

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
               return_value=(torch.zeros(1, 2, 3, 4), torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 2, 3, 3))):
        result = creator.extract_intermediate_features(
            torch.zeros(2, 3, 16, 16), layer_index=-1
        )

    assert "q" in result and "k" in result
    assert "poses" in result
    assert result["poses"].shape == (2, 4, 4)
    assert result["poses"].dtype == np.float32


def test_extract_intermediate_features_hook_removed_after_call():
    """Hook is removed after successful call."""
    model = _make_mock_model(n_frames=2, num_heads=2, n_blocks=4)
    creator = VGGTOmegaCreator.__new__(VGGTOmegaCreator)
    creator.model = model

    block = model.aggregator.inter_frame_blocks[-1]
    assert len(block.attn.qkv._forward_hooks) == 0

    with patch("collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
               return_value=(torch.zeros(1, 2, 3, 4), torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 2, 3, 3))):
        creator.extract_intermediate_features(torch.zeros(2, 3, 4, 4), layer_index=-1)

    assert len(block.attn.qkv._forward_hooks) == 0


def test_extract_intermediate_features_hook_removed_on_error():
    """Hook is removed even when model forward raises."""
    model = _make_mock_model(n_frames=2, num_heads=2, n_blocks=4)
    creator = VGGTOmegaCreator.__new__(VGGTOmegaCreator)

    def boom(images):
        raise RuntimeError("simulated failure")

    model.side_effect = boom
    creator.model = model

    block = model.aggregator.inter_frame_blocks[-1]
    with pytest.raises(RuntimeError, match="simulated failure"):
        with patch("collab_splats.pointcloud.feedforward.vggt_omega.encoding_to_camera",
                   return_value=(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 3))):
            creator.extract_intermediate_features(torch.zeros(2, 3, 4, 4), layer_index=-1)

    assert len(block.attn.qkv._forward_hooks) == 0


########################################################################
########## _reproject_ba ###############################################
########################################################################

def test_reproject_ba_returns_pts3d_colors(tmp_path):
    """_reproject_ba returns (pts3d, colors) tuple."""
    n = 2
    raw = _make_raw_outputs(n)
    extrinsics_3x4 = np.tile(np.eye(4)[:3], (n, 1, 1)).astype(np.float32)
    intrinsics = np.tile(np.eye(3), (n, 1, 1)).astype(np.float32)

    pts = np.zeros((5, 3), dtype=np.float32)
    colors = np.zeros((5, 3), dtype=np.uint8)
    pixel_indices = np.zeros((5, 3), dtype=np.int32)

    creator = VGGTOmegaCreator()
    with patch("collab_splats.pointcloud.feedforward.vggt_omega.unproject_and_filter_points",
               return_value=(pts, colors, pixel_indices)):
        result_pts, result_colors = creator._reproject_ba(raw, extrinsics_3x4, intrinsics)

    assert result_pts.shape == (5, 3)
    assert result_colors.shape == (5, 3)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggt_omega_creator.py -v 2>&1 | head -40
```

Expected: `ImportError` or `ModuleNotFoundError` for `vggt_omega.py` (not yet created). Tests should NOT pass.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/pointcloud/test_vggt_omega_creator.py
git commit -m "test(pointcloud): add failing tests for VGGTOmegaCreator"
```

---

## Task 4: Implement vggt_omega.py

**Files:**
- Create: `collab_splats/pointcloud/feedforward/vggt_omega.py`

- [ ] **Step 1: Write the implementation**

```python
"""VGGT-Omega feedforward backend: inference utilities and creator.

Provides:
  VGGT_OMEGA_HF_REPO            — default HuggingFace repo for checkpoint download
  VGGT_OMEGA_DEFAULT_FILENAME   — default checkpoint filename (512-res)
  VGGT_OMEGA_DEFAULT_RESOLUTION — default image resolution for inference
  _compute_omega_original_coords — compute original_coords for Omega's center-crop transform
  VGGTOmegaCreator              — feedforward creator using VGGT-Omega depth + pose estimation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from ..utils import lift_features
from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    _extrinsics_3x4_to_4x4,
    _raw_to_world_points,
    console,
)
from .vggtx import unproject_and_filter_points

logger = logging.getLogger(__name__)

########################################################################
########## Constants ###################################################
########################################################################

VGGT_OMEGA_HF_REPO = "facebook/VGGT-Omega"
VGGT_OMEGA_DEFAULT_FILENAME = "vggt_omega_1b_512.pt"
VGGT_OMEGA_DEFAULT_RESOLUTION = 512

########################################################################
########## Inference utilities #########################################
########################################################################

def _compute_omega_original_coords(image_paths: list[Path]) -> np.ndarray:
    """Compute original_coords after Omega's center-crop aspect-ratio enforcement.

    Mirrors the crop logic in vggt_omega.utils.load_fn._crop_to_supported_aspect_ratio
    so that _rescale_reconstruction_to_original_dimensions can invert the transform.

    Returns:
        (N, 6) float32 array [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h] per image.
    """
    # Must match vggt_omega.utils.load_fn._crop_to_supported_aspect_ratio exactly
    _MIN_AR = 0.5
    _MAX_AR = 2.0

    coords = []
    for p in image_paths:
        # Read image dimensions without decoding pixels
        with Image.open(p) as img:
            orig_w, orig_h = img.size
        ar = orig_h / max(orig_w, 1)

        # Default: no crop
        tl_x, tl_y = 0.0, 0.0
        cr_x, cr_y = float(orig_w), float(orig_h)

        # Center-crop height for tall images (AR > _MAX_AR)
        if ar > _MAX_AR:
            crop_h = orig_w * _MAX_AR
            tl_y = (orig_h - crop_h) / 2
            cr_y = tl_y + crop_h

        # Center-crop width for wide images (AR < _MIN_AR)
        elif ar < _MIN_AR:
            crop_w = orig_h / _MIN_AR
            tl_x = (orig_w - crop_w) / 2
            cr_x = tl_x + crop_w

        coords.append([tl_x, tl_y, cr_x, cr_y, float(orig_w), float(orig_h)])

    return np.array(coords, dtype=np.float32)


########################################################################
########## Creator #####################################################
########################################################################

@dataclass
class VGGTOmegaCreator(BaseFeedforwardCreator):
    """Pointcloud via VGGT-Omega feedforward pose + depth estimation.

    Uses VGGT-Omega to jointly predict camera poses and per-frame depth maps,
    which are then unprojected to a 3D point cloud.  Supports BundleAdjustment
    and LoopClosure wrappers via the standard BaseFeedforwardCreator interface.

    Attributes:
        camera_model:    pycolmap camera model.  Defaults to ``"PINHOLE"`` because
                         Omega predicts separate fx/fy via FoV encoding.
        model_path:      Local path to a ``vggt_omega_1b_512.pt`` checkpoint.
                         ``None`` → auto-download from HuggingFace on first run.
        model_repo:      HuggingFace repo ID for checkpoint download.
        model_filename:  Checkpoint filename to download from ``model_repo``.
        image_resolution: Target resolution for ``load_and_preprocess_images``.
                          512 for the standard checkpoint, 256 for text-aligned.
        conf_threshold:  Depth confidence percentile cutoff (0–100).  Points
                         below this percentile are discarded.  50.0 = top 50%.
    """

    camera_model: str = "PINHOLE"
    model_path: str | None = None
    model_repo: str = VGGT_OMEGA_HF_REPO
    model_filename: str = VGGT_OMEGA_DEFAULT_FILENAME
    image_resolution: int = VGGT_OMEGA_DEFAULT_RESOLUTION
    conf_threshold: float = 50.0

    def _load_model(self, device: str) -> Any:
        """Load VGGT-Omega from local path or HuggingFace, move to device.

        Uses bfloat16 on Ampere+ GPUs (compute capability >= 8), float16 otherwise.

        Args:
            device: Target device string (e.g. ``"cuda"`` or ``"cpu"``).

        Returns:
            VGGTOmega model in eval mode on the requested device.
        """
        # Choose dtype based on GPU capability: bfloat16 for Ampere+, float16 for older
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        # Resolve checkpoint path — local file or download from HuggingFace
        if self.model_path is not None:
            ckpt_path = Path(self.model_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        else:
            ckpt_path = Path(hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_filename,
            ))

        # Instantiate model, load checkpoint weights, move to device in eval mode
        model = VGGTOmega()
        model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
        model.eval()
        model = model.to(device, dtype=dtype)
        return model

    def _preprocess(self, image_dir: Path) -> tuple[Any, list[Path], np.ndarray]:
        """Load and preprocess images from directory into VGGT-Omega input format.

        Sorts images by filename, computes original_coords from Omega's center-crop
        AR enforcement, then calls load_and_preprocess_images for the model tensor.

        Args:
            image_dir: Directory containing ``.png``/``.jpg``/``.jpeg`` images.

        Returns:
            (images, image_paths, original_coords) where images is an (N, 3, H, W)
            float tensor and original_coords is an (N, 6) float32 array of
            [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h].
        """
        # Collect and sort image paths; reject non-image extensions
        image_paths = sorted([
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # Compute crop transform for each image (replicated from Omega's load_fn)
        original_coords = _compute_omega_original_coords(image_paths)

        # Load and preprocess images to model resolution via Omega's balanced resize
        image_names = [str(p) for p in image_paths]
        images = load_and_preprocess_images(image_names, image_resolution=self.image_resolution)

        return images, image_paths, original_coords

    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        """Run VGGT-Omega on the preprocessed image tensor; return raw predictions dict.

        VGGTOmega.forward handles autocast and batch-dim addition internally.

        Args:
            model: Loaded VGGTOmega model (from _load_model).
            views: (N, 3, H, W) float tensor from _preprocess.
            **kwargs: Unused; present for interface compatibility.

        Returns:
            dict with keys: ``images``, ``extrinsic``, ``intrinsics``,
            ``intrinsics_downsampled``, ``depth``, ``depth_conf``.
        """
        device = next(model.parameters()).device
        image_shape = views.shape[-2:]  # (H_model, W_model)
        orig_w, orig_h = self.original_coords[0, -2:]

        # Move images to model device; VGGTOmega adds the batch dim internally
        images = views.to(device)

        # Model forward handles bf16/f16 autocast internally
        with torch.no_grad():
            predictions = model(images)

        # Decode poses at model resolution (for BA track extraction)
        ext_ds, intr_ds = encoding_to_camera(predictions["pose_enc"], image_shape)
        # Decode poses at original image resolution (for final COLMAP output)
        ext, intr = encoding_to_camera(predictions["pose_enc"], (int(orig_h), int(orig_w)))

        # Move to CPU float32 for downstream numpy ops; squeeze the batch dim (always 1)
        extrinsic = ext.cpu().float().numpy().squeeze(0)         # (N, 3, 4)
        intrinsic = intr.cpu().float().numpy().squeeze(0)        # (N, 3, 3)
        intrinsic_ds = intr_ds.cpu().float().numpy().squeeze(0)  # (N, 3, 3)
        depth = predictions["depth"].squeeze(0).cpu().float().numpy()       # (N, H, W)
        depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy()  # (N, H, W)

        return {
            "images": images,
            "extrinsic": extrinsic,
            "intrinsics": intrinsic,
            "intrinsics_downsampled": intrinsic_ds,
            "depth": depth,
            "depth_conf": depth_conf,
        }

    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        """Unproject depth maps to world-space points and build FeedforwardResult.

        Lifts semantic features if extractor_name is set.  Populates BA fields
        (world_points, conf, images) so the BundleAdjustment wrapper can refine poses.

        Args:
            raw_outputs: Dict from _forward with depth, extrinsics, images.
            **kwargs:    Unused.

        Returns:
            FeedforwardResult with pts3d, colors, extrinsics, and BA fields populated.
        """
        extrinsic = raw_outputs["extrinsic"]   # (N, 3, 4) at original resolution
        intrinsic = raw_outputs["intrinsics"]  # (N, 3, 3) at original resolution

        # Unproject depth maps to filtered world-space points and per-point colors
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=raw_outputs["intrinsics_downsampled"],
            conf_threshold=self.conf_threshold,
        )

        # Lift semantic features to 3D if an extractor is configured
        if self.extractor_name:
            device = str(next(self.model.parameters()).device)
            features = lift_features(raw_outputs["images"], pixel_indices, self.extractor_name, device)
        else:
            features = None

        # Resolve model spatial dimensions; handle (N, H, W, 1) and (N, H, W) depth formats
        depth = raw_outputs["depth"]
        if depth.ndim == 4:
            depth = depth.squeeze(-1)
        model_h, model_w = int(depth.shape[1]), int(depth.shape[2])

        # Populate BA fields: subsampled world-point grid for track extraction
        world_pts_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)
        if world_pts_flat is not None:
            world_points = world_pts_flat.reshape(world_pts_flat.shape[0], model_h, model_w, 3)
        else:
            world_points = None

        # Populate BA fields: depth confidence map and preprocessed images
        conf = torch.from_numpy(raw_outputs["depth_conf"])
        images = raw_outputs["images"]

        extrinsic_4x4 = _extrinsics_3x4_to_4x4(extrinsic)

        # LC merged outputs carry deduped global poses — one entry per input frame
        extrinsic_4x4_out = raw_outputs.get("extrinsic_global_4x4", extrinsic_4x4)

        return FeedforwardResult(
            pts3d=pts3d,
            colors=colors,
            pixel_indices=pixel_indices,
            features=features,
            extrinsics=extrinsic_4x4_out,
            intrinsics=intrinsic,
            image_paths=self.image_paths,
            original_coords=self.original_coords,
            model_width=model_w,
            model_height=model_h,
            images=images,
            conf=conf,
            world_points=world_points,
        )

    def _reproject_ba(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-derive world-space points using bundle-adjusted camera poses.

        Called by the BundleAdjustment wrapper after refining extrinsics.

        Args:
            raw_outputs:     Raw predictions dict from _forward.
            extrinsics_3x4: (N, 3, 4) refined world-to-camera matrices.
            intrinsics:      (N, 3, 3) refined camera intrinsics.

        Returns:
            (pts3d, colors) — (P, 3) float32 and (P, 3) uint8.
        """
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

    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int = -1, **kwargs: Any
    ) -> dict[str, Any]:
        """Hook inter_frame_blocks[layer_index].attn.qkv; return {q, k, poses}.

        Registers a forward hook on the cross-frame attention block at layer_index
        in the Omega aggregator to capture q/k activations for loop-closure gating.
        Also decodes camera poses from the pose encoding.  Hook is removed in a
        finally block — guaranteed cleanup even if the forward raises.

        Args:
            frames:      (2, C, H, W) preprocessed frames on CPU or GPU.
            layer_index: Which inter_frame_block to tap.  -1 = last (default).
            **kwargs:    Unused (kept for interface compatibility).

        Returns:
            dict with keys:
              "q":     (B, heads, N_tokens, head_dim) query projections
              "k":     (B, heads, N_tokens, head_dim) key projections
              "poses": (2, 4, 4) float32 np.ndarray — decoded camera extrinsics
        """
        device = next(self.model.parameters()).device

        # VGGTOmega manages bf16/f16 autocast internally — device-only cast matches _forward
        images = frames.to(device)

        # Register per-call hook on inter-frame attention block's QKV projection
        block = self.model.aggregator.inter_frame_blocks[layer_index]
        C_nh = block.attn.num_heads
        captured: dict[str, torch.Tensor] = {}

        def _hook(module, _inp, out: torch.Tensor) -> None:
            B, N, C3 = out.shape
            hd = (C3 // 3) // C_nh
            qkv = out.detach().reshape(B, N, 3, C_nh, hd).permute(2, 0, 3, 1, 4)
            captured["q"], captured["k"] = qkv[0], qkv[1]

        hook = block.attn.qkv.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                predictions = self.model(images)
        finally:
            # Always remove the hook — no persistent state left on the model
            hook.remove()

        # Decode (2, 4, 4) camera extrinsics from Omega pose encoding
        image_shape = (frames.shape[-2], frames.shape[-1])
        ext_3x4, _ = encoding_to_camera(
            predictions["pose_enc"].detach(), image_shape
        )
        ext_3x4 = ext_3x4.cpu().float().numpy().squeeze(0)  # (2, 3, 4)
        captured["poses"] = _extrinsics_3x4_to_4x4(ext_3x4)  # (2, 4, 4)
        return captured
```

- [ ] **Step 2: Run failing tests — they should now pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggt_omega_creator.py -v
```

Expected: All tests PASS (or close — if any fail, fix the implementation before continuing).

- [ ] **Step 3: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggt_omega.py
git commit -m "feat(pointcloud): implement VGGTOmegaCreator feedforward backend"
```

---

## Task 5: Update feedforward __init__.py

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/__init__.py`

- [ ] **Step 1: Read the current __init__.py**

Read `collab_splats/pointcloud/feedforward/__init__.py` to see the current exports.

- [ ] **Step 2: Add VGGTOmegaCreator exports**

Add the `VGGTOmegaCreator` import in the `── Concrete creators` section and add it to `__all__`. The file should look like:

```python
"""Feedforward pointcloud creators: VGGT-X, MapAnything, and VGGT-Omega backends.

Import from here — submodule structure is an implementation detail.
"""
from __future__ import annotations

# ── Public types and utilities ─────────────────────────────────────────────────
from .base import BaseFeedforwardCreator, FeedforwardResult, build_pycolmap_reconstruction

# ── Concrete creators ──────────────────────────────────────────────────────────
from .vggtx import VGGTXCreator
from .mapanything import MapAnythingCreator
from .vggt_omega import VGGTOmegaCreator

# ── Internal helpers (re-exported for wrappers and tests) ──────────────────────
# _raw_to_world_points re-exported for wrappers.py BundleAdjustment, which calls it
# directly on raw VGGT-X outputs outside the normal postprocess pipeline.
from .base import _raw_to_world_points

# ── Test-patchable symbols ─────────────────────────────────────────────────────
# Explicitly re-exported so tests can patch
# collab_splats.pointcloud.feedforward.unproject_and_filter_points
# regardless of which submodule defines the function.
from .vggtx import unproject_and_filter_points
from .vggtx import _patch_vggtx_compute_similarity

__all__ = [
    "FeedforwardResult",
    "BaseFeedforwardCreator",
    "build_pycolmap_reconstruction",
    "VGGTXCreator",
    "MapAnythingCreator",
    "VGGTOmegaCreator",
    "unproject_and_filter_points",
]
```

**Important:** Do NOT change the existing exports or their order — only add the `VGGTOmegaCreator` line and update `__all__`. Read the file first to see the exact current content, then make minimal edits.

- [ ] **Step 3: Verify import works**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud.feedforward import VGGTOmegaCreator, BaseFeedforwardCreator
assert issubclass(VGGTOmegaCreator, BaseFeedforwardCreator)
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Verify the top-level pointcloud package also exposes it**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud import VGGTOmegaCreator
print('OK:', VGGTOmegaCreator)
"
```

If this fails, check `collab_splats/pointcloud/__init__.py` and add `VGGTOmegaCreator` to its imports and `__all__`.

- [ ] **Step 5: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggt_omega_creator.py tests/pointcloud/test_vggtx_creator.py tests/pointcloud/test_feedforward_shared.py -v 2>&1 | tail -20
```

Expected: All tests PASS. No regressions in existing feedforward tests.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/__init__.py
git commit -m "feat(pointcloud): export VGGTOmegaCreator from feedforward package"
```

---

## Task 6: Update top-level pointcloud __init__.py

**Files:**
- Modify: `collab_splats/pointcloud/__init__.py`

- [ ] **Step 1: Read the current __init__.py**

Read `collab_splats/pointcloud/__init__.py` to see how `VGGTXCreator` and `MapAnythingCreator` are currently exported.

- [ ] **Step 2: Add VGGTOmegaCreator**

Find the line that imports `VGGTXCreator` and `MapAnythingCreator` from feedforward, and add `VGGTOmegaCreator` alongside them. Also add it to `__all__` if present.

Example — if the current import looks like:
```python
from .feedforward import VGGTXCreator, MapAnythingCreator
```

Change to:
```python
from .feedforward import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
```

- [ ] **Step 3: Verify top-level import**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add collab_splats/pointcloud/__init__.py
git commit -m "feat(pointcloud): expose VGGTOmegaCreator at top-level package"
```

---

## Task 7: Run full test suite and verify no regressions

- [ ] **Step 1: Run all pointcloud tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v --tb=short 2>&1 | tail -40
```

Expected: All existing tests pass. The new `test_vggt_omega_creator.py` tests pass.

- [ ] **Step 2: Run broader test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --tb=short -q 2>&1 | tail -30
```

Expected: No new failures introduced by `VGGTOmegaCreator` or the `__init__.py` changes.

- [ ] **Step 3: Verify setup/feedforward.sh smoke test passes**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
print('[OK] VGGTXCreator, MapAnythingCreator, and VGGTOmegaCreator import successfully')
"
```

Expected: `[OK] VGGTXCreator, MapAnythingCreator, and VGGTOmegaCreator import successfully`

---

## Spec Coverage Self-Check

| Spec requirement | Covered by task |
|---|---|
| New `VGGTOmegaCreator` in `feedforward/vggt_omega.py` | Task 4 |
| `feedforward/__init__.py` exports `VGGTOmegaCreator` | Task 5 |
| `setup/feedforward.sh` vggt-omega block | Task 2 |
| `.gitmodules` submodule | Task 1 |
| `tests/pointcloud/test_vggt_omega_creator.py` | Task 3 |
| `PINHOLE` camera model | Task 4 (`camera_model = "PINHOLE"`) |
| `hf_hub_download` for checkpoint | Task 4 (`_load_model`) |
| `--no-deps` numpy<2 bypass | Task 2 |
| `original_coords` center-crop computation | Task 3+4 (`_compute_omega_original_coords`) |
| `extract_intermediate_features` with hook cleanup | Task 3+4 |
| `_reproject_ba` for BA wrapper | Task 3+4 |
| Full LC support via base `_verify_loop_candidate` | Inherited from base class — no task needed |
| No text-alignment, no `use_global_alignment` | Not implemented (out of scope) |
