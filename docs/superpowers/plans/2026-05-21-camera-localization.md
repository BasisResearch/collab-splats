# Camera Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete `localization.py` with Stage 2 (DISK+LightGlue / XFeat local feature matching) and Stage 3 (OpenCV PnP) so `CameraLocalizer` can return a 4×4 world-to-camera pose for a query image given any reconstructed scene.

**Architecture:** `localization.py` gains two extractor classes (`DiskExtractor`, `XFeatExtractor`), a private `_build_frame_assignments` helper that projects 3D points into reference frames and assigns keypoints via nearest-neighbour, and `CameraLocalizer` which orchestrates extraction → matching across all N frames → PnP. No global retrieval — exhaustive matching against all N frames is used to avoid false exclusions from appearance shift.

**Tech Stack:** kornia 0.8.1 (DISK, LightGlue, match_mnn), OpenCV 4.6.0 (solvePnPRansac), third_party/xfeat (accelerated_features.XFeat), PyTorch, NumPy.

---

## Task 1: Vendor XFeat

**Files:**
- Create: `third_party/xfeat/` — git clone
- Modify: `vendor/README.md`

- [ ] **Step 1: Clone XFeat into vendor/**

```bash
cd /workspace/collab-splats
git clone https://github.com/verlab/accelerated_features third_party/xfeat
```

Expected: `third_party/xfeat/` created with `accelerated_features/` subdirectory inside.

- [ ] **Step 2: Verify import works**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import sys
sys.path.insert(0, 'third_party/xfeat')
from accelerated_features import XFeat
xf = XFeat()
print('XFeat ok:', xf)
"
```

Expected: prints `XFeat ok: ...` with no errors.

- [ ] **Step 3: Update vendor/README.md**

Read `vendor/README.md` first, then add an XFeat entry following the same format as existing entries.

- [ ] **Step 4: Commit**

```bash
git add third_party/xfeat vendor/README.md
git commit -m "chore(vendor): add verlab/accelerated_features (XFeat)"
```

---

## Task 2: Fix Stage 1 code quality

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`

- [ ] **Step 1: Write a failing import-and-log test**

Create `tests/pointcloud/test_localization.py`:

```python
import logging
import pytest
from collab_splats.pointcloud.localization import BaseRetrievalExtractor, DinoSaladExtractor


def test_module_has_logger():
    import collab_splats.pointcloud.localization as loc_mod
    assert hasattr(loc_mod, "logger")
    assert isinstance(loc_mod.logger, logging.Logger)


def test_registry_get_dino_salad():
    cls = BaseRetrievalExtractor.get("dino-salad")
    assert cls is DinoSaladExtractor


def test_registry_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        BaseRetrievalExtractor.get("nonexistent-model")
```

- [ ] **Step 2: Run to verify first test fails**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py::test_module_has_logger -v
```

Expected: FAIL — `AttributeError` or similar (no `logger` attribute yet).

- [ ] **Step 3: Add logging + fix T.Compose in localization.py**

Open `collab_splats/pointcloud/localization.py`. Make these changes:

**Add after the `from __future__ import annotations` line:**
```python
import logging
```

**Add after the imports block, before the first `########` divider:**
```python
logger = logging.getLogger(__name__)
```

**In `DinoSaladExtractor.__init__`, add `_transform` attribute:**
```python
self._transform = T.Compose([
    T.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**In `DinoSaladExtractor.forward`, replace the inline `T.Compose(...)` block:**
```python
# Before (inline, rebuilt every call):
_t = T.Compose([
    T.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
imgs = torch.stack([_t(img) for img in images])

# After (use cached self._transform):
imgs = torch.stack([self._transform(img) for img in images])
```

**Add a logger call in `DinoSaladExtractor.forward` before the `with torch.no_grad()` block:**
```python
logger.debug("DinoSaladExtractor: embedding batch of %d images", len(imgs))
```

- [ ] **Step 4: Run all three tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
git commit -m "fix(localization): add logging, cache T.Compose in DinoSaladExtractor"
```

---

## Task 3: Implement DiskExtractor

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`
- Modify: `tests/pointcloud/test_localization.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/pointcloud/test_localization.py`:

```python
import torch
import numpy as np
import pytest


@pytest.mark.slow
def test_disk_extractor_returns_keypoints_and_descriptors():
    """Requires network access to download DISK weights (~4 MB)."""
    from collab_splats.pointcloud.localization import DiskExtractor
    extractor = DiskExtractor()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    kpts, descs = extractor.extract(image)
    assert kpts.ndim == 2 and kpts.shape[1] == 2
    assert descs.ndim == 2 and descs.shape[1] == 128
    assert len(kpts) == len(descs)
    assert len(kpts) > 0


@pytest.mark.slow
def test_disk_extractor_match_returns_index_pairs():
    from collab_splats.pointcloud.localization import DiskExtractor
    extractor = DiskExtractor()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    kpts, descs = extractor.extract(img)
    # Self-match: all keypoints should match themselves
    matches = extractor.match(kpts, descs, kpts, descs, image_hw=(480, 640))
    assert matches.ndim == 2 and matches.shape[1] == 2
    # Self-match must include at least some diagonal matches
    diag = (matches[:, 0] == matches[:, 1]).sum()
    assert diag > 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "disk" -v
```

Expected: FAIL — `ImportError: cannot import name 'DiskExtractor'`.

- [ ] **Step 3: Implement DiskExtractor in localization.py**

Add the following section after the Stage 1 block and before the Stage 2 comment stub. Replace the existing Stage 2 comment block entirely:

```python
########################################################
########## Stage 2: Local feature extraction ###########
########################################################


class DiskExtractor:
    """DISK local feature extractor with LightGlue matcher.

    Detects and describes keypoints using DISK (kornia pretrained on depth),
    matches pairs with LightGlue. DISK and LightGlue are a matched pair —
    LightGlue was trained specifically for DISK descriptors.

    Weights: downloaded automatically to torch hub cache on first use (~4 MB).
    """

    def __init__(self, top_k: int = 1024, device: str | None = None):
        from kornia.feature import DISK, LightGlue

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._top_k = top_k
        self._disk = DISK.from_pretrained("depth").to(self._device).eval()
        self._lightglue = LightGlue(features="disk").to(self._device).eval()
        logger.debug("DiskExtractor: loaded DISK + LightGlue on %s", self._device)

    def extract(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract DISK keypoints and L2-normalised descriptors.

        Args:
            image: HxWx3 uint8 RGB image.

        Returns:
            keypoints:   (N, 2) float32 pixel coordinates [x, y].
            descriptors: (N, 128) float32 L2-normalised.
        """
        img_t = (
            torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        ).to(self._device)
        with torch.no_grad():
            features = self._disk(img_t, self._top_k, pad_if_not_divisible=True)
        kpts = features[0].keypoints.cpu()        # (N, 2)
        descs = features[0].descriptors.cpu()     # (N, 128)
        return kpts, descs

    def match(
        self,
        kpts_q: torch.Tensor,
        descs_q: torch.Tensor,
        kpts_db: torch.Tensor,
        descs_db: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Match query features against database features using LightGlue.

        Args:
            kpts_q:    (N, 2) query keypoints.
            descs_q:   (N, 128) query descriptors.
            kpts_db:   (M, 2) database keypoints.
            descs_db:  (M, 128) database descriptors.
            image_hw:  (H, W) of both images — required for LightGlue normalisation.

        Returns:
            matches: (K, 2) int64 — [query_idx, db_idx] pairs.
        """
        hw = torch.tensor([[image_hw[0], image_hw[1]]], dtype=torch.float32)
        data = {
            "image0": {
                "keypoints": kpts_q.unsqueeze(0).to(self._device),
                "descriptors": descs_q.unsqueeze(0).to(self._device),
                "image_size": hw.to(self._device),
            },
            "image1": {
                "keypoints": kpts_db.unsqueeze(0).to(self._device),
                "descriptors": descs_db.unsqueeze(0).to(self._device),
                "image_size": hw.to(self._device),
            },
        }
        with torch.no_grad():
            result = self._lightglue(data)

        # LightGlue returns matches0: (1, N) — index in image1 for each kpt in image0, -1 if unmatched
        matches0 = result["matches0"][0].cpu()   # (N,)
        valid = matches0 > -1
        idx_q = torch.where(valid)[0]
        idx_db = matches0[valid]
        if len(idx_q) == 0:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.stack([idx_q, idx_db], dim=1)
```

- [ ] **Step 4: Run the slow tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "disk" -v -m slow
```

Expected: both PASS. If LightGlue output key differs from `matches0`, inspect `result.keys()` and adjust accordingly.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
git commit -m "feat(localization): add DiskExtractor with DISK + LightGlue"
```

---

## Task 4: Implement XFeatExtractor

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`
- Modify: `tests/pointcloud/test_localization.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/pointcloud/test_localization.py`:

```python
@pytest.mark.slow
def test_xfeat_extractor_returns_keypoints_and_descriptors():
    from collab_splats.pointcloud.localization import XFeatExtractor
    extractor = XFeatExtractor()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    kpts, descs = extractor.extract(image)
    assert kpts.ndim == 2 and kpts.shape[1] == 2
    assert descs.ndim == 2 and descs.shape[1] == 64
    assert len(kpts) == len(descs)
    assert len(kpts) > 0


@pytest.mark.slow
def test_xfeat_extractor_match_returns_index_pairs():
    from collab_splats.pointcloud.localization import XFeatExtractor
    extractor = XFeatExtractor()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    kpts, descs = extractor.extract(img)
    matches = extractor.match(kpts, descs, kpts, descs)
    assert matches.ndim == 2 and matches.shape[1] == 2
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "xfeat" -v
```

Expected: FAIL — `ImportError: cannot import name 'XFeatExtractor'`.

- [ ] **Step 3: Add XFeat vendor path + XFeatExtractor to localization.py**

Add the vendor path injection after the SALAD vendor path block (top of file):

```python
# XFeat vendored at third_party/xfeat/ — no pip package available.
_XFEAT_VENDOR_PATH = str(pathlib.Path(__file__).parents[2] / "vendor" / "xfeat")
if _XFEAT_VENDOR_PATH not in sys.path:
    sys.path.insert(0, _XFEAT_VENDOR_PATH)
```

Add `XFeatExtractor` after `DiskExtractor` in the Stage 2 section:

```python
class XFeatExtractor:
    """XFeat local feature extractor with mutual nearest-neighbour matching.

    Lightweight learned features from verlab/accelerated_features, vendored at
    third_party/xfeat/. Faster than DISK; suitable for CPU or real-time use.
    Paired with kornia match_mnn for descriptor matching.
    """

    def __init__(self, top_k: int = 1024, device: str | None = None):
        from modules.xfeat import XFeat  # vendored: third_party/xfeat/modules/

        self._top_k = top_k
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._xfeat = XFeat()
        logger.debug("XFeatExtractor: loaded on %s", self._device)

    def extract(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract XFeat keypoints and descriptors.

        Args:
            image: HxWx3 uint8 RGB image.

        Returns:
            keypoints:   (N, 2) float32 pixel coordinates [x, y].
            descriptors: (N, 64) float32 L2-normalised.
        """
        img_t = (
            torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        )
        out = self._xfeat.detectAndCompute(img_t, top_k=self._top_k)
        kpts = out[0]["keypoints"].cpu()      # (N, 2)
        descs = out[0]["descriptors"].cpu()   # (N, 64)
        return kpts, descs

    def match(
        self,
        kpts_q: torch.Tensor,
        descs_q: torch.Tensor,
        kpts_db: torch.Tensor,
        descs_db: torch.Tensor,
        image_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Match query features against database features using mutual nearest neighbour.

        Args:
            kpts_q:   (N, 2) query keypoints (unused in matching, kept for interface parity).
            descs_q:  (N, 64) query descriptors.
            kpts_db:  (M, 2) database keypoints (unused in matching).
            descs_db: (M, 64) database descriptors.
            image_hw: ignored — MNN matching does not require image size.

        Returns:
            matches: (K, 2) int64 — [query_idx, db_idx] pairs.
        """
        from kornia.feature import match_mnn

        _, match_ids = match_mnn(descs_q.unsqueeze(0), descs_db.unsqueeze(0))
        # match_ids: (1, K, 2) — [:,0] = query indices, [:,1] = db indices
        return match_ids[0].cpu()
```

- [ ] **Step 4: Run the slow tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "xfeat" -v -m slow
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
git commit -m "feat(localization): add XFeatExtractor with MNN matching"
```

---

## Task 5: Implement `_build_frame_assignments`

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`
- Modify: `tests/pointcloud/test_localization.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/pointcloud/test_localization.py`:

```python
def test_build_frame_assignments_assigns_visible_points():
    """Identity camera: projected 3D points → keypoints at exact positions."""
    from collab_splats.pointcloud.localization import _build_frame_assignments

    # Two points in front of identity camera at z=5
    pts3d = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float32)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None]   # (1, 4, 4) identity
    intrinsics = K[None]                              # (1, 3, 3)

    # Expected projections: [0,0,5] → [320,240], [1,0,5] → [420,240]
    kpts = torch.tensor([[320.0, 240.0], [420.0, 240.0]])
    assignments = _build_frame_assignments(
        pts3d, extrinsics, intrinsics, [kpts], image_hw=(480, 640), radius=2.0
    )

    assert len(assignments) == 1
    assert assignments[0][0] == 0   # kpt 0 → pt3d 0
    assert assignments[0][1] == 1   # kpt 1 → pt3d 1


def test_build_frame_assignments_ignores_points_behind_camera():
    from collab_splats.pointcloud.localization import _build_frame_assignments

    # One point in front, one behind
    pts3d = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None]
    intrinsics = K[None]
    kpts = torch.tensor([[320.0, 240.0], [320.0, 241.0]])

    assignments = _build_frame_assignments(
        pts3d, extrinsics, intrinsics, [kpts], image_hw=(480, 640), radius=2.0
    )

    assert 0 in assignments[0]    # front point assigned
    assert 1 not in assignments[0]  # behind-camera point not assigned


def test_build_frame_assignments_respects_radius():
    from collab_splats.pointcloud.localization import _build_frame_assignments

    pts3d = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)  # projects to [320, 240]
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None]
    intrinsics = K[None]

    # Keypoint 50px away from projection — outside radius=10
    kpts = torch.tensor([[370.0, 240.0]])
    assignments = _build_frame_assignments(
        pts3d, extrinsics, intrinsics, [kpts], image_hw=(480, 640), radius=10.0
    )
    assert 0 not in assignments[0]
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "assignment" -v
```

Expected: FAIL — `ImportError: cannot import name '_build_frame_assignments'`.

- [ ] **Step 3: Implement `_build_frame_assignments`**

Add to `localization.py`, in the Stage 3 section (replace the existing comment stub):

```python
########################################################
########## Stage 3: Pose estimation ####################
########################################################


def _build_frame_assignments(
    pts3d: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    frame_keypoints: list[torch.Tensor],
    image_hw: tuple[int, int],
    radius: float = 8.0,
) -> list[dict[int, int]]:
    """For each reference frame, map keypoint indices to 3D point indices.

    Projects all 3D points into each frame and assigns each keypoint to its
    nearest visible projected point within `radius` pixels via torch.cdist.

    Args:
        pts3d:           (P, 3) world-space points.
        extrinsics:      (N, 4, 4) world-to-camera transforms.
        intrinsics:      (N, 3, 3) camera intrinsics K per frame.
        frame_keypoints: list of N tensors, each (K_i, 2) keypoints for frame i.
        image_hw:        (H, W) image dimensions for bounds filtering.
        radius:          max pixel distance for a keypoint to claim a 3D point.

    Returns:
        List of N dicts: {kpt_idx: pt3d_idx} for each frame.
    """
    H, W = image_hw
    assignments: list[dict[int, int]] = []

    for i in range(len(frame_keypoints)):
        R = extrinsics[i, :3, :3]  # (3, 3)
        t = extrinsics[i, :3, 3]   # (3,)
        K = intrinsics[i]           # (3, 3)

        # Project pts3d into frame i — world-to-camera: p_cam = R @ p_world + t
        pts_cam = pts3d @ R.T + t                          # (P, 3)
        visible_mask = pts_cam[:, 2] > 0                   # in front of camera
        visible_idx = np.where(visible_mask)[0]

        frame_assignment: dict[int, int] = {}
        kpts = frame_keypoints[i]

        if len(visible_idx) == 0 or len(kpts) == 0:
            assignments.append(frame_assignment)
            continue

        pts_cam_vis = pts_cam[visible_mask]                # (Q, 3)
        pts_proj = pts_cam_vis @ K.T                       # (Q, 3)
        pts_proj_2d = pts_proj[:, :2] / pts_proj[:, 2:3]  # (Q, 2)  pixel coords

        # Filter to image bounds
        in_bounds = (
            (pts_proj_2d[:, 0] >= 0) & (pts_proj_2d[:, 0] < W) &
            (pts_proj_2d[:, 1] >= 0) & (pts_proj_2d[:, 1] < H)
        )
        if not in_bounds.any():
            assignments.append(frame_assignment)
            continue

        pts_proj_valid = torch.from_numpy(pts_proj_2d[in_bounds]).float()  # (V, 2)
        valid_pt_idx = visible_idx[in_bounds]                               # (V,) — original pt3d indices

        # Nearest-neighbour via torch.cdist: (K_i, V)
        dists = torch.cdist(kpts.float(), pts_proj_valid)   # (K_i, V)
        min_dists, nearest = dists.min(dim=1)               # (K_i,)

        for kpt_idx in range(len(kpts)):
            if min_dists[kpt_idx].item() < radius:
                frame_assignment[kpt_idx] = int(valid_pt_idx[nearest[kpt_idx]])

        assignments.append(frame_assignment)

    return assignments
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "assignment" -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
git commit -m "feat(localization): add _build_frame_assignments via torch.cdist NN"
```

---

## Task 6: Implement `CameraLocalizer`

**Files:**
- Modify: `collab_splats/pointcloud/localization.py`
- Modify: `tests/pointcloud/test_localization.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/pointcloud/test_localization.py`:

```python
def _make_synthetic_scene(n_pts: int = 30, n_frames: int = 4):
    """Synthetic scene: n_pts points in front of n_frames cameras.

    Returns (pts3d, extrinsics, intrinsics, image_paths).
    All cameras are identity-ish (small rotations) looking along +z.
    """
    rng = np.random.default_rng(0)
    pts3d = rng.uniform(-1, 1, (n_pts, 3)).astype(np.float32)
    pts3d[:, 2] += 5.0  # push in front of cameras

    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    intrinsics = np.tile(K, (n_frames, 1, 1))  # (N, 3, 3)

    extrinsics = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    for i in range(n_frames):
        # Slight lateral shift per camera
        extrinsics[i, 0, 3] = i * 0.1

    return pts3d, extrinsics, intrinsics


class _MockExtractor:
    """Extractor that returns exact projected 3D positions as keypoints.

    descriptor[j] = one-hot-ish encoding of pt3d index j — matching is exact.
    """

    def __init__(self, pts3d, extrinsics, intrinsics):
        self._pts3d = pts3d
        self._extrinsics = extrinsics
        self._intrinsics = intrinsics
        self._call = 0
        n = len(pts3d)
        # Descriptors: identity matrix padded/truncated to 128 dims
        self._descs = torch.zeros(n, 128)
        for j in range(min(n, 128)):
            self._descs[j, j] = 1.0

    def extract(self, image):
        i = self._call % len(self._extrinsics)
        self._call += 1
        R = self._extrinsics[i, :3, :3]
        t = self._extrinsics[i, :3, 3]
        K = self._intrinsics[i]
        pts_cam = self._pts3d @ R.T + t
        visible = pts_cam[:, 2] > 0
        pts_proj = pts_cam[visible] @ K.T
        pts_proj = pts_proj[:, :2] / pts_proj[:, 2:3]
        kpts = torch.from_numpy(pts_proj).float()
        descs = self._descs[visible]
        return kpts, descs

    def match(self, kpts_q, descs_q, kpts_db, descs_db, image_hw=None):
        # Match by max cosine similarity (identity descriptors → diagonal)
        sim = descs_q @ descs_db.T   # (N, M)
        best = sim.argmax(dim=1)
        valid = sim.max(dim=1).values > 0.5
        idx_q = torch.where(valid)[0]
        idx_db = best[valid]
        if len(idx_q) == 0:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.stack([idx_q, idx_db], dim=1)


def test_camera_localizer_from_feedforward_classmethod():
    """from_feedforward classmethod constructs CameraLocalizer correctly."""
    from collab_splats.pointcloud.localization import CameraLocalizer
    from unittest.mock import MagicMock
    import tempfile, pathlib

    pts3d, extrinsics, intrinsics = _make_synthetic_scene()
    result = MagicMock()
    result.pts3d = pts3d
    result.extrinsics = extrinsics
    result.intrinsics = intrinsics

    # Create dummy image files so cv2.imread doesn't crash
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(len(extrinsics)):
            p = pathlib.Path(tmp) / f"frame_{i:04d}.png"
            import cv2 as _cv2
            _cv2.imwrite(str(p), np.zeros((480, 640, 3), dtype=np.uint8))
            paths.append(p)
        result.image_paths = paths

        extractor = _MockExtractor(pts3d, extrinsics, intrinsics)
        loc = CameraLocalizer.from_feedforward(result, extractor=extractor)
        assert loc is not None


def test_camera_localizer_recovers_known_pose():
    """CameraLocalizer should recover the identity pose for a camera at extrinsics[0]."""
    from collab_splats.pointcloud.localization import CameraLocalizer
    import tempfile, pathlib

    pts3d, extrinsics, intrinsics = _make_synthetic_scene()
    K = intrinsics[0]

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(len(extrinsics)):
            p = pathlib.Path(tmp) / f"frame_{i:04d}.png"
            import cv2 as _cv2
            _cv2.imwrite(str(p), np.zeros((480, 640, 3), dtype=np.uint8))
            paths.append(p)

        extractor = _MockExtractor(pts3d, extrinsics, intrinsics)
        loc = CameraLocalizer(pts3d, extrinsics, intrinsics, paths, extractor=extractor)

        # Query = camera 0 (identity extrinsic)
        query_image = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = loc.localize(query_image, K)

        assert pose is not None, "localize() returned None — PnP failed"
        assert pose.shape == (4, 4)
        # Recovered R should be close to identity
        np.testing.assert_allclose(pose[:3, :3], extrinsics[0, :3, :3], atol=0.05)
        np.testing.assert_allclose(pose[:3, 3], extrinsics[0, :3, 3], atol=0.05)
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "camera_localizer" -v
```

Expected: FAIL — `ImportError: cannot import name 'CameraLocalizer'`.

- [ ] **Step 3: Implement CameraLocalizer**

Add to `localization.py` after `_build_frame_assignments`:

```python
class CameraLocalizer:
    """Locates a query camera within a known 3D scene.

    Matches the query image against all N reference frames via local feature
    matching (exhaustive — no global retrieval) then solves PnP for the pose.
    Build once per scene; call localize() for each query image.

    Output convention matches FeedforwardResult.extrinsics: (4, 4) float32
    world-to-camera homogeneous transform.
    """

    def __init__(
        self,
        pts3d: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        image_paths: list,
        extractor=None,
        radius: float = 8.0,
    ):
        """Build feature index from scene data.

        Args:
            pts3d:       (P, 3) float32 world-space 3D points.
            extrinsics:  (N, 4, 4) float32 world-to-camera transforms.
            intrinsics:  (N, 3, 3) float32 camera intrinsics K per frame.
            image_paths: length-N list of source image paths.
            extractor:   local feature extractor; defaults to DiskExtractor().
            radius:      pixel radius for 2D→3D keypoint assignment.
        """
        import cv2 as _cv2

        self._pts3d = pts3d
        self._extrinsics = extrinsics
        self._intrinsics = intrinsics
        self._extractor = extractor or DiskExtractor()

        logger.info(
            "CameraLocalizer: building index for %d frames, %d 3D points",
            len(image_paths), len(pts3d),
        )

        # Extract local features for all reference frames
        self._frame_features: list[tuple[torch.Tensor, torch.Tensor]] = []
        first_hw: tuple[int, int] | None = None
        for path in image_paths:
            bgr = _cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(f"CameraLocalizer: cannot read {path}")
            rgb = bgr[..., ::-1].copy()
            if first_hw is None:
                first_hw = (rgb.shape[0], rgb.shape[1])
            kpts, descs = self._extractor.extract(rgb)
            self._frame_features.append((kpts, descs))
            logger.debug("  frame %s: %d keypoints", path.name if hasattr(path, 'name') else path, len(kpts))

        self._image_hw: tuple[int, int] = first_hw or (480, 640)

        # Build kpt→3D assignment maps
        self._assignments = _build_frame_assignments(
            pts3d=pts3d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            frame_keypoints=[f[0] for f in self._frame_features],
            image_hw=self._image_hw,
            radius=radius,
        )
        logger.info("CameraLocalizer: index built")

    @classmethod
    def from_feedforward(cls, result, **kwargs) -> "CameraLocalizer":
        """Construct from a FeedforwardResult."""
        return cls(
            pts3d=result.pts3d,
            extrinsics=result.extrinsics,
            intrinsics=result.intrinsics,
            image_paths=result.image_paths,
            **kwargs,
        )

    def localize(
        self,
        query_image: np.ndarray,
        query_intrinsics: np.ndarray,
    ) -> np.ndarray | None:
        """Estimate world-to-camera pose for a query image.

        Matches query against all N reference frames, collects 2D↔3D
        correspondences, solves PnP with RANSAC.

        Args:
            query_image:      HxWx3 uint8 RGB image.
            query_intrinsics: (3, 3) float32 or float64 camera matrix K.

        Returns:
            (4, 4) float32 world-to-camera transform, same convention as
            FeedforwardResult.extrinsics. None if PnP fails or too few matches.
        """
        import cv2 as _cv2

        kpts_q, descs_q = self._extractor.extract(query_image)
        logger.debug("CameraLocalizer.localize: query has %d keypoints", len(kpts_q))

        # Collect 2D↔3D pairs across all reference frames
        pts2d_list: list[np.ndarray] = []
        pts3d_list: list[np.ndarray] = []
        best_score: dict[int, float] = {}   # pt3d_idx → best confidence so far
        best_2d: dict[int, np.ndarray] = {} # pt3d_idx → corresponding query 2D point

        for i, (kpts_db, descs_db) in enumerate(self._frame_features):
            if len(kpts_db) == 0:
                continue
            matches = self._extractor.match(
                kpts_q, descs_q, kpts_db, descs_db, self._image_hw
            )
            if len(matches) == 0:
                continue

            assignment = self._assignments[i]
            for q_idx, db_idx in matches.numpy():
                db_idx = int(db_idx)
                if db_idx not in assignment:
                    continue
                pt3d_idx = assignment[db_idx]
                pt2d = kpts_q[int(q_idx)].numpy().astype(np.float32)
                score = 1.0  # uniform confidence; LightGlue scores unused here

                if pt3d_idx not in best_score or score > best_score[pt3d_idx]:
                    best_score[pt3d_idx] = score
                    best_2d[pt3d_idx] = pt2d

        if len(best_2d) < 4:
            logger.warning(
                "CameraLocalizer: only %d 2D↔3D correspondences — need ≥4 for PnP",
                len(best_2d),
            )
            return None

        pts2d = np.array(list(best_2d.values()), dtype=np.float32)
        pts3d = np.array(
            [self._pts3d[idx] for idx in best_2d.keys()], dtype=np.float32
        )

        success, rvec, tvec, inliers = _cv2.solvePnPRansac(
            objectPoints=pts3d[:, None],
            imagePoints=pts2d[:, None],
            cameraMatrix=query_intrinsics.astype(np.float64),
            distCoeffs=None,
            reprojectionError=8.0,
            confidence=0.999,
            iterationsCount=1000,
        )

        if not success or inliers is None or len(inliers) < 4:
            logger.warning(
                "CameraLocalizer: PnP failed (inliers=%d / %d correspondences)",
                len(inliers) if inliers is not None else 0, len(pts2d),
            )
            return None

        logger.info(
            "CameraLocalizer: localized — %d / %d inliers",
            len(inliers), len(pts2d),
        )

        R, _ = _cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = tvec[:, 0]
        return pose
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "camera_localizer" -v
```

Expected: all PASS. If `test_camera_localizer_recovers_known_pose` fails on PnP (too few correspondences), increase `n_pts` in `_make_synthetic_scene` or reduce `radius` assertion tolerance.

- [ ] **Step 5: Run the full test file (excluding slow)**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -v -m "not slow"
```

Expected: all non-slow tests PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
git commit -m "feat(localization): add CameraLocalizer with exhaustive matching + PnP"
```

---

## Task 7: Update exports and worklog

**Files:**
- Modify: `collab_splats/pointcloud/__init__.py`
- Modify: `worklog/STATE.md`

- [ ] **Step 1: Add CameraLocalizer to `__init__.py`**

In `collab_splats/pointcloud/__init__.py`, add to the imports:

```python
from .localization import BaseRetrievalExtractor, CameraLocalizer, DiskExtractor, XFeatExtractor
```

Add to `__all__`:

```python
"BaseRetrievalExtractor",
"CameraLocalizer",
"DiskExtractor",
"XFeatExtractor",
```

- [ ] **Step 2: Verify import**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud import CameraLocalizer, DiskExtractor, XFeatExtractor
print('exports ok')
"
```

Expected: `exports ok`.

- [ ] **Step 3: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v -m "not slow" --ignore=tests/examples
```

Expected: no regressions.

- [ ] **Step 4: Update STATE.md**

Add to the "Recently Completed" section:

```markdown
- **camera-localization** (2026-05-21) — [spec](specs/2026-05-21-camera-localization-design.md) · [plan](plans/2026-05-21-camera-localization.md)
  - `localization.py` complete: `DiskExtractor` (DISK+LightGlue), `XFeatExtractor` (XFeat+MNN), `_build_frame_assignments` (torch.cdist NN), `CameraLocalizer` (exhaustive match + PnP). XFeat vendored at `third_party/xfeat/`. No global retrieval — matches all N frames to avoid false exclusions. `from_feedforward()` classmethod for feedforward pipeline.
```

- [ ] **Step 5: Final commit**

```bash
git add collab_splats/pointcloud/__init__.py worklog/STATE.md
git commit -m "feat(localization): export CameraLocalizer, DiskExtractor, XFeatExtractor"
```

---

## Self-Review

**Spec coverage:**
- ✅ Code quality (logging, T.Compose fix) — Task 2
- ✅ Stage 2 — DiskExtractor — Task 3
- ✅ Stage 2 — XFeatExtractor — Task 4
- ✅ 2D→3D assignment builder — Task 5
- ✅ Stage 3 — PnP via OpenCV — Task 6 (inside CameraLocalizer)
- ✅ CameraLocalizer with from_feedforward classmethod — Task 6
- ✅ No global retrieval — documented in CameraLocalizer docstring
- ✅ XFeat vendored — Task 1
- ✅ Tests: slow-marked extractor tests, synthetic geometry tests — Tasks 3-6
- ✅ Exports — Task 7

**Type consistency:**
- `DiskExtractor.extract()` → `(Tensor (N,2), Tensor (N,128))` — used as `kpts, descs` in CameraLocalizer ✅
- `XFeatExtractor.extract()` → `(Tensor (N,2), Tensor (N,64))` — same interface ✅
- `_build_frame_assignments` takes `list[Tensor]` for `frame_keypoints` and returns `list[dict[int,int]]` — used correctly in CameraLocalizer ✅
- `CameraLocalizer.localize()` returns `np.ndarray | None` — (4,4) float32 ✅
- `from_feedforward` accesses `result.pts3d`, `result.extrinsics`, `result.intrinsics`, `result.image_paths` — all present in FeedforwardResult ✅
