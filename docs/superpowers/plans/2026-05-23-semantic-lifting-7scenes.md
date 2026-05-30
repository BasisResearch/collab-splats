# Semantic Lifting on 7-Scenes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the semantic lifting notebook to run on 7-Scenes chess with Zarr feature caching, correct pre-lifting compression, and a pure geometric `lift_features`.

**Architecture:** Extract MaskCLIP features → cache to Zarr (N, D, H_p, W_p) → train AE on all patches → encode per-frame → lift compressed maps to 3D via bilinear upsample. Compression before lifting: never allocates (P, 768).

**Tech Stack:** PyTorch, zarr, PIL, `BaseFeatureExtractor`, `FeatureAutoencoder`, `VGGTXCreator`, PyVista.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `evals/datasets.py` | Add `_load_bicycle`, `_load_video` to `_REGISTRY` |
| Modify | `collab_splats/semantics/features.py` | Add `name` property + `extract_and_cache()` to `BaseFeatureExtractor` |
| Modify | `collab_splats/pointcloud/utils.py` | Rename old `lift_features` → `extract_and_lift_features`; add pure geometric `lift_features` |
| Modify | `docs/semantics/semantic_lifting.ipynb` | Update cells §0, §1, §4–§8 |
| Modify | `tests/evals/test_datasets.py` | Add bicycle/video loader tests |
| Modify | `tests/semantics/test_features.py` | Add `extract_and_cache` + `name` tests |
| Modify | `tests/pointcloud/test_pointcloud_utils.py` | Add new `lift_features` tests |

---

## Task 0: Install zarr

**Files:**
- No file changes — environment setup only

- [ ] **Step 1: Install zarr**

```bash
/opt/conda/envs/nerfstudio/bin/pip install zarr
```

Expected: `Successfully installed zarr-...`

- [ ] **Step 2: Verify import**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import zarr; print(zarr.__version__)"
```

Expected: version string printed, no ImportError.

---

## Task 1: Dataset registry — add bicycle and video loaders

**Files:**
- Modify: `evals/datasets.py` (after the `_REGISTRY` dict, around line 244)
- Modify: `tests/evals/test_datasets.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/evals/test_datasets.py`:

```python
import pytest
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock


def test_get_dataset_bicycle_registered():
    from datasets import get_dataset
    loader = get_dataset("bicycle")
    assert callable(loader)


def test_load_bicycle_returns_eval_dataset(tmp_path):
    from datasets import get_dataset, EvalDataset
    # Create fake images_4 directory with png files
    img_dir = tmp_path / "images_4"
    img_dir.mkdir()
    for i in range(3):
        (img_dir / f"frame_{i:04d}.png").touch()
    result = get_dataset("bicycle")(tmp_path, max_frames=10)
    assert isinstance(result, EvalDataset)
    assert len(result.images) == 3
    assert result.gt_poses.shape == (3, 4, 4)
    assert result.gt_poses.dtype == np.float32


def test_load_bicycle_respects_max_frames(tmp_path):
    from datasets import get_dataset
    img_dir = tmp_path / "images_4"
    img_dir.mkdir()
    for i in range(10):
        (img_dir / f"frame_{i:04d}.png").touch()
    result = get_dataset("bicycle")(tmp_path, max_frames=5)
    assert len(result.images) == 5


def test_get_dataset_video_registered():
    from datasets import get_dataset
    loader = get_dataset("video")
    assert callable(loader)


def test_load_video_calls_sample_frames_fps(tmp_path):
    from datasets import get_dataset, EvalDataset
    fake_video = tmp_path / "clip.mp4"
    fake_video.touch()
    fake_frames = [tmp_path / f"frame_{i:04d}.jpg" for i in range(5)]

    with patch("collab_splats.utils.frame_sampling.sample_frames_fps", return_value=fake_frames) as mock_fps:
        result = get_dataset("video")(fake_video, max_frames=10, fps=2.0)

    mock_fps.assert_called_once()
    assert isinstance(result, EvalDataset)
    assert len(result.images) == 5
    assert result.gt_poses.shape == (5, 4, 4)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/evals/test_datasets.py::test_get_dataset_bicycle_registered tests/evals/test_datasets.py::test_load_bicycle_returns_eval_dataset tests/evals/test_datasets.py::test_get_dataset_video_registered -v 2>&1 | tail -20
```

Expected: FAILED with `KeyError: 'bicycle'`.

- [ ] **Step 3: Implement `_load_bicycle` and `_load_video`**

In `evals/datasets.py`, add before the `_REGISTRY` dict (around line 238):

```python
def _load_bicycle(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    """Load a LLFF/bicycle-format sequence from images_4/ subdirectory."""
    images = sorted((seq_dir / "images_4").glob("*.png"))[:max_frames]
    return EvalDataset(images=images, gt_poses=np.zeros((len(images), 4, 4), dtype=np.float32))


def _load_video(seq_dir: Path, max_frames: int = 500, fps: float = 1.0) -> EvalDataset:
    """Sample frames from a video file at the given fps and return as EvalDataset."""
    from collab_splats.utils.frame_sampling import sample_frames_fps
    frames_dir = seq_dir.parent / (seq_dir.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    images = sample_frames_fps(str(seq_dir), frames_dir, fps=fps)[:max_frames]
    return EvalDataset(images=images, gt_poses=np.zeros((len(images), 4, 4), dtype=np.float32))
```

Then extend `_REGISTRY`:

```python
_REGISTRY: dict[str, Callable[..., EvalDataset]] = {
    "7scenes": _load_7scenes,
    "tum": _load_tum,
    "kitti": _load_kitti,
    "waymo": _load_waymo,
    "co3dv2": _load_co3dv2,
    "bicycle": _load_bicycle,
    "video": _load_video,
}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/evals/test_datasets.py::test_get_dataset_bicycle_registered tests/evals/test_datasets.py::test_load_bicycle_returns_eval_dataset tests/evals/test_datasets.py::test_load_bicycle_respects_max_frames tests/evals/test_datasets.py::test_get_dataset_video_registered tests/evals/test_datasets.py::test_load_video_calls_sample_frames_fps -v 2>&1 | tail -15
```

Expected: all 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add evals/datasets.py tests/evals/test_datasets.py
git commit -m "feat(evals): add bicycle and video dataset loaders to registry"
```

---

## Task 2: BaseFeatureExtractor — add `name` property and `extract_and_cache()`

**Files:**
- Modify: `collab_splats/semantics/features.py`
- Modify: `tests/semantics/test_features.py`

`BaseFeatureExtractor` is at line ~47. The `extract_and_cache` method needs to:
1. Resolve zarr path: `cache_dir / f"{self.name}.zarr"`
2. If `skip_existing=True`, validate `.zattrs` and return early if valid
3. Run `self.forward([pil_img])` per image, write each frame as one zarr chunk
4. Store metadata in `.zattrs`

- [ ] **Step 1: Write failing tests**

Add to `tests/semantics/test_features.py`:

```python
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image


def test_base_extractor_name_property():
    """Registered name is recoverable from instance."""
    from collab_splats.semantics.features import MaskCLIPExtractor
    # Patch model loading so we don't need CLIP weights
    with patch("clip.load", return_value=(MagicMock(), MagicMock())):
        ext = MaskCLIPExtractor.__new__(MaskCLIPExtractor)
        ext._registry = MaskCLIPExtractor._registry
    assert ext.name == "maskclip"


def _make_fake_extractor(tmp_path, D=8, H_p=4, W_p=6):
    """Return a fake extractor that produces (D, H_p, W_p) tensors."""
    from collab_splats.semantics.features import BaseFeatureExtractor
    import torch.nn as nn
    from abc import abstractmethod

    class FakeExtractor(BaseFeatureExtractor):
        def __init__(self):
            nn.Module.__init__(self)
            self.patch_size = 14
            self._registry = BaseFeatureExtractor._registry
            self._D, self._H_p, self._W_p = D, H_p, W_p

        def forward(self, images):
            return [torch.zeros(self._D, self._H_p, self._W_p) for _ in images]

        @property
        def name(self):
            return "fake"

    return FakeExtractor()


def test_extract_and_cache_creates_zarr(tmp_path):
    import zarr
    ext = _make_fake_extractor(tmp_path)
    # Create 3 fake image files
    image_paths = []
    for i in range(3):
        p = tmp_path / f"frame_{i:04d}.png"
        Image.new("RGB", (64, 48)).save(p)
        image_paths.append(p)

    zarr_path = ext.extract_and_cache(image_paths, tmp_path)

    assert zarr_path == tmp_path / "fake.zarr"
    assert zarr_path.exists()
    z = zarr.open(str(zarr_path), mode="r")
    assert z["features"].shape == (3, 8, 4, 6)
    assert z["features"].chunks == (1, 8, 4, 6)
    assert z.attrs["extractor"] == "fake"
    assert z.attrs["n_frames"] == 3
    assert z.attrs["patch_size"] == 14


def test_extract_and_cache_skip_existing(tmp_path):
    import zarr
    ext = _make_fake_extractor(tmp_path)
    image_paths = []
    for i in range(2):
        p = tmp_path / f"frame_{i:04d}.png"
        Image.new("RGB", (64, 48)).save(p)
        image_paths.append(p)

    # First run
    ext.extract_and_cache(image_paths, tmp_path)

    # Patch forward to detect if called again
    call_count = []
    orig_forward = ext.forward
    def counting_forward(images):
        call_count.append(1)
        return orig_forward(images)
    ext.forward = counting_forward

    # Second run with skip_existing=True
    ext.extract_and_cache(image_paths, tmp_path, skip_existing=True)
    assert len(call_count) == 0, "forward() should not be called when cache is valid"


def test_extract_and_cache_reruns_when_n_frames_mismatch(tmp_path):
    import zarr
    ext = _make_fake_extractor(tmp_path)
    image_paths_2 = []
    for i in range(2):
        p = tmp_path / f"frame_{i:04d}.png"
        Image.new("RGB", (64, 48)).save(p)
        image_paths_2.append(p)

    # Cache with 2 frames
    ext.extract_and_cache(image_paths_2, tmp_path)

    # Now request 3 frames — should re-extract
    image_paths_3 = image_paths_2[:]
    p3 = tmp_path / "frame_0002.png"
    Image.new("RGB", (64, 48)).save(p3)
    image_paths_3.append(p3)

    zarr_path = ext.extract_and_cache(image_paths_3, tmp_path, skip_existing=True)
    z = zarr.open(str(zarr_path), mode="r")
    assert z.attrs["n_frames"] == 3
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_features.py::test_extract_and_cache_creates_zarr tests/semantics/test_features.py::test_extract_and_cache_skip_existing -v 2>&1 | tail -15
```

Expected: FAILED with `AttributeError: 'BaseFeatureExtractor' object has no attribute 'extract_and_cache'`.

- [ ] **Step 3: Add `name` property to `BaseFeatureExtractor`**

In `collab_splats/semantics/features.py`, inside `class BaseFeatureExtractor`, add after `__init__` (around line 65):

```python
    @property
    def name(self) -> str:
        """Registry key for this extractor class. Reverses the _registry lookup."""
        for key, cls in self._registry.items():
            if cls is type(self):
                return key
        raise AttributeError(
            f"{type(self).__name__} is not registered — use @BaseFeatureExtractor.register('name')"
        )
```

- [ ] **Step 4: Add `extract_and_cache()` to `BaseFeatureExtractor`**

In `collab_splats/semantics/features.py`, still inside `class BaseFeatureExtractor`, add after the `name` property:

```python
    def extract_and_cache(
        self,
        image_paths: list[Path],
        cache_dir: Path,
        batch_size: int = 1,
        skip_existing: bool = True,
    ) -> Path:
        """Extract patch features for all images and write to cache_dir/{name}.zarr.

        Returns the zarr store path. Re-entrant: if skip_existing=True and the
        cache exists with matching extractor name and frame count, returns immediately.
        Chunks=(1, D, H_p, W_p) so reading frame i loads exactly one disk chunk.
        """
        import zarr
        from datetime import datetime, timezone
        from PIL import Image as PILImage

        zarr_path = Path(cache_dir) / f"{self.name}.zarr"

        # Validate existing cache before extracting
        if skip_existing and zarr_path.exists():
            try:
                z = zarr.open(str(zarr_path), mode="r")
                if (
                    z.attrs.get("extractor") == self.name
                    and z.attrs.get("n_frames") == len(image_paths)
                ):
                    logger.info("Feature cache valid, skipping extraction: %s", zarr_path)
                    return zarr_path
            except Exception:
                logger.warning("Cache at %s is corrupt, re-extracting", zarr_path)

        # Extract first frame to determine feature shape
        first_img = PILImage.open(image_paths[0]).convert("RGB")
        with torch.no_grad():
            [first_feat] = self.forward([first_img])
        D, H_p, W_p = first_feat.shape
        N = len(image_paths)

        # Create zarr store with chunked array and metadata
        store = zarr.open(str(zarr_path), mode="w")
        store.attrs.update({
            "extractor": self.name,
            "patch_size": self.patch_size,
            "n_frames": N,
            "feature_dim": D,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        arr = store.zeros(
            "features",
            shape=(N, D, H_p, W_p),
            chunks=(1, D, H_p, W_p),
            dtype="float32",
        )
        arr[0] = first_feat.cpu().float().numpy()

        # Extract remaining frames one by one (batch_size reserved for future multi-GPU)
        for i in range(1, N):
            pil_img = PILImage.open(image_paths[i]).convert("RGB")
            with torch.no_grad():
                [feat] = self.forward([pil_img])
            arr[i] = feat.cpu().float().numpy()
            if i % 10 == 0:
                logger.info("extract_and_cache: %d/%d frames done", i + 1, N)

        logger.info("Feature cache written: %s  shape=%s", zarr_path, arr.shape)
        return zarr_path
```

Note: `logger` is already defined at module level in `features.py`. `torch` is already imported.

- [ ] **Step 5: Run all new feature tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_features.py::test_base_extractor_name_property tests/semantics/test_features.py::test_extract_and_cache_creates_zarr tests/semantics/test_features.py::test_extract_and_cache_skip_existing tests/semantics/test_features.py::test_extract_and_cache_reruns_when_n_frames_mismatch -v 2>&1 | tail -20
```

Expected: all 4 PASSED.

- [ ] **Step 6: Run existing semantics tests to check no regressions**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/ -v 2>&1 | tail -20
```

Expected: all pass (same as before).

- [ ] **Step 7: Commit**

```bash
git add collab_splats/semantics/features.py tests/semantics/test_features.py
git commit -m "feat(semantics): add name property and extract_and_cache() to BaseFeatureExtractor"
```

---

## Task 3: Pure geometric `lift_features` + backward-compat rename

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`
- Modify: `tests/pointcloud/test_pointcloud_utils.py`

Current state (from grep):
- `_assign_frame_features` helper at ~line 744 (uses `patch_size` floor-div)
- `lift_features` at ~line 778 (runs extractor, calls `_assign_frame_features`)

Plan:
1. Rename `lift_features` → `extract_and_lift_features` (same body, backward compat)
2. Add new `lift_features(feature_maps, pixel_indices, image_size) → (P, D) np.ndarray`

- [ ] **Step 1: Write failing tests**

Add to `tests/pointcloud/test_pointcloud_utils.py`:

```python
import pytest
import numpy as np
import torch


def test_lift_features_shape():
    """lift_features maps (D,H_p,W_p) maps to (P, D) point array."""
    from collab_splats.pointcloud.utils import lift_features
    D, H_p, W_p = 16, 4, 6
    feature_maps = [torch.randn(D, H_p, W_p) for _ in range(3)]
    pixel_indices = np.array(
        [[0, 100, 80], [0, 200, 300], [1, 50, 60], [2, 10, 10]],
        dtype=np.int32,
    )
    image_size = (480, 640)
    result = lift_features(feature_maps, pixel_indices, image_size=image_size)
    assert result.shape == (4, D)
    assert result.dtype == np.float32


def test_lift_features_correct_frame_routing():
    """Points from frame i use only feature_maps[i]."""
    from collab_splats.pointcloud.utils import lift_features
    D = 4
    # frame 0: all ones; frame 1: all twos
    feature_maps = [
        torch.ones(D, 2, 2),
        torch.full((D, 2, 2), 2.0),
    ]
    pixel_indices = np.array(
        [[0, 0, 0], [1, 0, 0]],  # one point per frame
        dtype=np.int32,
    )
    result = lift_features(feature_maps, pixel_indices, image_size=(4, 4))
    np.testing.assert_allclose(result[0], np.ones(D), atol=1e-5)
    np.testing.assert_allclose(result[1], np.full(D, 2.0), atol=1e-5)


def test_lift_features_any_latent_dim():
    """Works for both raw 768D and compressed 13D — no patch_size needed."""
    from collab_splats.pointcloud.utils import lift_features
    for D in [768, 13]:
        maps = [torch.randn(D, 8, 8)]
        idx = np.array([[0, 100, 100]], dtype=np.int32)
        result = lift_features(maps, idx, image_size=(256, 256))
        assert result.shape == (1, D)


def test_extract_and_lift_features_still_exists():
    """Backward-compat alias exists and is callable."""
    from collab_splats.pointcloud.utils import extract_and_lift_features
    assert callable(extract_and_lift_features)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_pointcloud_utils.py::test_lift_features_shape tests/pointcloud/test_pointcloud_utils.py::test_extract_and_lift_features_still_exists -v 2>&1 | tail -15
```

Expected: FAILED — `test_lift_features_shape` fails because current signature takes `images` tensor not `feature_maps`, and `test_extract_and_lift_features_still_exists` fails because `extract_and_lift_features` doesn't exist.

- [ ] **Step 3: Rename old `lift_features` to `extract_and_lift_features`**

In `collab_splats/pointcloud/utils.py`, find the line with `def lift_features(` (around line 778) and rename to `extract_and_lift_features`. The body stays identical.

- [ ] **Step 4: Add new pure geometric `lift_features`**

In `collab_splats/pointcloud/utils.py`, add the new function right before `extract_and_lift_features`:

```python
def lift_features(
    feature_maps: "list[torch.Tensor]",
    pixel_indices: np.ndarray,
    image_size: "tuple[int, int]",
) -> np.ndarray:
    """Map per-frame 2D feature maps to per-point features via bilinear upsampling.

    Upsamples each (D, H_p, W_p) map to (D, H, W) then indexes at pixel coords.
    Pure geometric operation: no extractor, no cache, no AE — works for any D.

    Args:
        feature_maps:  List of (D, H_p, W_p) tensors, one per frame.
        pixel_indices: (P, 3) int32 — [frame_id, row, col] per point (pixel space).
        image_size:    (H, W) — upsample target; use out.images.shape[-2:].

    Returns:
        (P, D) float32 array aligned with the point set.
    """
    import torch
    import torch.nn.functional as F

    P = len(pixel_indices)
    D = feature_maps[0].shape[0]
    features = np.zeros((P, D), dtype=np.float32)

    for i, fmap in enumerate(feature_maps):
        mask_i = pixel_indices[:, 0] == i
        if not mask_i.any():
            continue
        # Bilinear upsample: (1, D, H_p, W_p) → (1, D, H, W) → (D, H, W)
        upsampled = F.interpolate(
            fmap.unsqueeze(0).float(),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        rows = pixel_indices[mask_i, 1]
        cols = pixel_indices[mask_i, 2]
        # Clamp to valid image bounds (VGGT-X may produce coords at edge)
        rows = np.clip(rows, 0, image_size[0] - 1)
        cols = np.clip(cols, 0, image_size[1] - 1)
        features[mask_i] = upsampled[:, rows, cols].cpu().numpy().T

    return features
```

- [ ] **Step 5: Run new lift_features tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_pointcloud_utils.py::test_lift_features_shape tests/pointcloud/test_pointcloud_utils.py::test_lift_features_correct_frame_routing tests/pointcloud/test_pointcloud_utils.py::test_lift_features_any_latent_dim tests/pointcloud/test_pointcloud_utils.py::test_extract_and_lift_features_still_exists -v 2>&1 | tail -15
```

Expected: all 4 PASSED.

- [ ] **Step 6: Run existing pointcloud tests for regressions**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/pointcloud/utils.py tests/pointcloud/test_pointcloud_utils.py
git commit -m "refactor(pointcloud): pure geometric lift_features with bilinear upsample; rename old to extract_and_lift_features"
```

---

## Task 4: Update notebook `docs/semantics/semantic_lifting.ipynb`

**Files:**
- Modify: `docs/semantics/semantic_lifting.ipynb` (cells §0, §1, §4–§8)

Use the `NotebookEdit` tool or directly edit JSON. Each section below shows the full cell source to write.

- [ ] **Step 1: Update §0 — Configuration cell**

Replace the existing §0 configuration cell source with:

```python
import sys
sys.path.insert(0, "/workspace/collab-splats")
sys.path.insert(0, "/workspace/collab-splats/evals")

import numpy as np
import torch
import zarr
import pyvista as pv
from pathlib import Path

from datasets import get_dataset
from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
from collab_splats.pointcloud.utils import lift_features
from collab_splats.semantics.features import MaskCLIPExtractor
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.utils.visualization import pointcloud_to_polydata, visualize_splat, PCD_KWARGS

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_TYPE = "7scenes"
SEQ_DIR      = Path("/workspace/collab-splats/evals/data/7scenes/chess/chess/seq-01")
N_FRAMES     = 30
CAPTURE_FPS  = 30   # native capture rate of 7-Scenes
SAMPLE_FPS   = 1    # 1 fps → stride=30 → 30 frames out of 1000

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_DIR    = Path("/tmp/semantic_lifting/feature_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Semantic queries ──────────────────────────────────────────────────────────
QUERIES    = ["chess board", "chess pieces", "table", "chair", "wall"]
NEGATIVES  = ["background"]

# ── Runtime ───────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 13

# ── Visualization defaults ─────────────────────────────────────────────────────
VIZ_KWARGS = dict(
    position=(0.0, -3.0, 1.5),
    focal_point=(0.0, 0.0, 0.0),
    view_up=(0.0, 0.0, 1.0),
)

print(f"Device: {DEVICE}  |  Dataset: {DATASET_TYPE}  |  Frames: {N_FRAMES}")
```

- [ ] **Step 2: Update §1 — Frame loading cell**

Replace §1 source with:

```python
# Uniform load — works for 7scenes, bicycle, video, co3dv2 without if/else
stride = max(1, round(CAPTURE_FPS / SAMPLE_FPS))
frames = get_dataset(DATASET_TYPE)(SEQ_DIR, max_frames=9999).images[::stride][:N_FRAMES]
print(f"Loaded {len(frames)} frames  source={DATASET_TYPE}  stride={stride}")
```

- [ ] **Step 3: Leave §2 (VGGTXCreator) and §3 (Inspect) unchanged**

These cells produce `out.images`, `out.pixel_indices`, `out.pts3d`. No edits needed.

After §2/§3 run, bind local variables used by later cells:

If not already present, add a binding cell after §3:

```python
# Bind outputs for use in §4–§8
images        = out.images         # (N, 3, H, W)
pixel_indices = out.pixel_indices  # (P, 3) int32
pts3d         = out.pts3d          # (P, 3)
```

- [ ] **Step 4: Update §4 — Extract + cache**

```python
# Extract MaskCLIP patch features and write to Zarr cache
extractor = MaskCLIPExtractor(device=DEVICE)
zarr_path = extractor.extract_and_cache(frames, CACHE_DIR, skip_existing=True)
print(f"Feature cache: {zarr_path}")
```

- [ ] **Step 5: Update §5 — Load cache, train AE, encode per frame**

```python
# Load raw features from cache — (N, 768, H_p, W_p) chunks=(1,...)
z = zarr.open(str(CACHE_DIR / "maskclip.zarr"), mode="r")
raw_maps = [torch.from_numpy(np.array(z["features"][i])) for i in range(len(frames))]
# raw_maps: list of (768, H_p, W_p) CPU float32 tensors

# Flatten all patches for AE training: (N_total_patches, 768)
all_patches = torch.cat([f.flatten(1).T for f in raw_maps])   # (N_patches, 768)
print(f"Training AE on {all_patches.shape[0]:,} patches  dim=768 → {LATENT_DIM}")

ae = FeatureAutoencoder(input_dim=768, latent_dim=LATENT_DIM)
ae.fit(all_patches.to(DEVICE))
print(f"AE trained: 768D → {LATENT_DIM}D")

# Encode per frame: list of (LATENT_DIM, H_p, W_p)
compressed_maps = [ae.encode(f.to(DEVICE)).detach().cpu() for f in raw_maps]
print(f"Encoded {len(compressed_maps)} frames  shape={compressed_maps[0].shape}")
```

- [ ] **Step 6: Update §6 — Lift compressed features (2D → 3D)**

```python
# Geometric lift: bilinear upsample (LATENT_DIM, H_p, W_p) → (LATENT_DIM, H, W) → (P, LATENT_DIM)
# (P, 768) is never allocated — compression happened before lifting
codes = lift_features(
    compressed_maps,
    pixel_indices,
    image_size=images.shape[-2:],   # (H, W)
)
print(f"codes: {codes.shape}  dtype={codes.dtype}")  # expect (P, LATENT_DIM)
```

- [ ] **Step 7: Update §7 — Text query + visualization**

```python
# Decode codes back to 768D only at query time
feat_decoded = ae.per_point_decode(
    torch.from_numpy(codes).to(DEVICE)
).detach().cpu().numpy()   # (P, 768)

# Score each query independently → (P, Q)
scores = np.column_stack([
    extractor.score_queries(
        torch.from_numpy(feat_decoded).to(DEVICE),
        positive=[q],
        negative=NEGATIVES,
    ).cpu().numpy()
    for q in QUERIES
])
print(f"scores: {scores.shape}  range=[{scores.min():.3f}, {scores.max():.3f}]")

# Visualize combined score (max across all queries)
cloud = pointcloud_to_polydata(pts3d, similarity=scores.max(axis=1))
pl = visualize_splat(
    cloud,
    mesh_kwargs={**PCD_KWARGS, "scalars": "similarity", "cmap": "viridis", "rgb": False},
    viz_kwargs=VIZ_KWARGS,
)
pl.show()
```

- [ ] **Step 8: Update §8 — Per-query gallery**

```python
# Per-query tiled renders — one plot per semantic query
for query, q_scores in zip(QUERIES, scores.T):   # scores (P, Q), cols = queries
    cloud_q = pointcloud_to_polydata(pts3d, similarity=q_scores)
    pl = visualize_splat(
        cloud_q,
        mesh_kwargs={**PCD_KWARGS, "scalars": "similarity", "cmap": "viridis", "rgb": False},
        viz_kwargs=VIZ_KWARGS,
    )
    pl.show()
    print(f"Query: '{query}'  score range=[{q_scores.min():.3f}, {q_scores.max():.3f}]")
```

- [ ] **Step 9: Commit**

```bash
git add docs/semantics/semantic_lifting.ipynb
git commit -m "feat(docs): extend semantic_lifting notebook for 7scenes with Zarr cache and compressed lift"
```

---

## Task 5: End-to-end smoke test

- [ ] **Step 1: Run full test suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/integration 2>&1 | tail -30
```

Expected: all pass.

- [ ] **Step 2: Smoke-test dataset registry**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -c "
import sys; sys.path.insert(0, 'evals')
from datasets import get_dataset
loader = get_dataset('7scenes')
ds = loader('evals/data/7scenes/chess/chess/seq-01', max_frames=5)
print('7scenes ok, frames:', len(ds.images))
"
```

Expected: `7scenes ok, frames: 5`.

- [ ] **Step 3: Smoke-test extract_and_cache (tiny mock)**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -c "
import sys; sys.path.insert(0, '.')
from collab_splats.pointcloud.utils import lift_features, extract_and_lift_features
import torch, numpy as np
maps = [torch.randn(13, 4, 4) for _ in range(2)]
idx = np.array([[0,100,100],[1,50,50]], dtype=np.int32)
codes = lift_features(maps, idx, image_size=(480, 640))
print('lift_features ok:', codes.shape)
assert extract_and_lift_features is not None
print('extract_and_lift_features ok (backward compat)')
"
```

Expected:
```
lift_features ok: (2, 13)
extract_and_lift_features ok (backward compat)
```

---

## Acceptance Criteria Checklist

- [ ] `extract_and_cache()` writes Zarr; re-run with `skip_existing=True` skips (Task 2 tests)
- [ ] Zarr `.zattrs` contains `extractor`, `patch_size`, `n_frames` (Task 2 tests)
- [ ] `ae.fit(tensor)` unchanged (no compression.py changes)
- [ ] `lift_features(compressed_maps, pixel_indices, image_size)` returns `(P, latent_dim)` — bilinear upsample, no `(P, 768)` allocated (Task 3 tests)
- [ ] `extract_and_lift_features()` callable for backward compat (Task 3 tests)
- [ ] `visualize_splat()` called with `cmap="viridis"` and `VIZ_KWARGS` passthrough (Task 4 §7–§8)
- [ ] Notebook runs end-to-end on chess seq-01 with `N_FRAMES=30` (Task 5)
- [ ] `DATASET_TYPE = "video"` path runs (Task 1 + get_dataset registry)
