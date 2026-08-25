# Splatfacto Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the quality gap to the retired nerfstudio/rade-gs pipeline by adopting splatfacto's undistortion preproc, permutation view sampling, coarse-to-fine schedule, and DefaultStrategy densification args — keeping our measured wins (depth loss, pose_opt, MCMC for 3dgs).

**Architecture:** One new preproc module (`collab_splats/preproc/undistort.py`: pycolmap OPENCV self-calibration + splatfacto-exact cv2 undistort/crop) wired into `extract_frames` behind one boolean `preproc.undistort`; three surgical trainer changes in `collab_splats/splats/trainer.py` (ViewSampler, per-step downscale, splatfacto DefaultStrategy args). A/B measured on the GH010229 300-frame instantsfm 2dgs scene against baseline 19.21 PSNR (sparse depth targets — working tree has no dense-align; do NOT compare against the 19.38 dense number).

**Tech Stack:** cv2 (undistort), pycolmap (self-calib), gsplat d2f5c0f (DefaultStrategy), existing FrameStore/Reconstructor plumbing.

**Spec:** `docs/superpowers/specs/2026-08-25-splatfacto-parity-design.md` (read it first — it records two plan-time verdicts: `_STALENESS_KEYS` no longer exists so toggling `preproc.undistort` needs `preprocess(overwrite=True)`; absgrad is NOT usable on the 2dgs `gradient_2dgs` strategy path, so 2dgs keeps `absgrad=False, grow_grad2d=2e-4`).

**Environment ground rules (repo-wide, non-negotiable):**
- Python is `/opt/venv/reconstruction/bin/python` (py3.11) — never bare `python`.
- The branch carries FOREIGN uncommitted work from concurrent sessions. Never `git add -A` / `git add .`. Commit with `git commit --only <paths> -m "..."` exactly as each task's commit step shows.
- GPU tasks (7–8) run in tmux, never in a notebook; don't start them while another GPU job runs (`nvidia-smi` first).
- All imports at top of file; block comments per logical block; `########` section dividers; one-line docstrings with `"""` on their own lines.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `collab_splats/preproc/undistort.py` | Create | `DistortionProfile`, `estimate_camera_distortion` (pycolmap), `undistort_frames` (cv2) |
| `collab_splats/preproc/__init__.py` | Modify | re-export the two public functions + profile (public-API test asserts exact set) |
| `collab_splats/wrapper/reconstructor.py` | Modify | `extract_frames` gains `undistort` param; both branches undistort before `FrameStore.create`; `preprocess` passes `pre_cfg["undistort"]` |
| `configs/base.yaml` | Modify | `preproc.undistort: false` + comment; splats block gains `num_downscales`, `resolution_schedule` |
| `collab_splats/splats/trainer.py` | Modify | `ViewSampler`, `downscale_factor`/`downscale_view`, `SplatsConfig` fields, `make_strategy(cfg, n_views)` splatfacto args |
| `tests/preproc/test_undistort.py` | Create | roundtrip, crop/K, provenance, estimation smoke |
| `tests/splats/test_trainer.py` | Modify | sampler, downscale, strategy-arg tests |

---

### Task 1: `undistort_frames` + `DistortionProfile` (pure cv2 half)

**Files:**
- Create: `collab_splats/preproc/undistort.py`
- Create: `tests/preproc/test_undistort.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/preproc/test_undistort.py`:

```python
"""
Tests for preproc undistortion: profile round-trip, cv2 undistort/crop path.
"""

import cv2
import numpy as np
import pytest

from collab_splats.preproc.undistort import DistortionProfile, undistort_frames


def _profile(width=640, height=480, k1=0.006, k2=-0.003):
    # GoPro-Linear-magnitude distortion, centred principal point
    return DistortionProfile(
        k1=k1, k2=k2, p1=0.0, p2=0.0,
        fx=500.0, fy=500.0, cx=width / 2, cy=height / 2,
        width=width, height=height,
    )


def _distort_image(image, profile):
    # Forward-apply the profile with cv2.undistort's inverse mapping trick:
    # remap through initUndistortRectifyMap built from the NEGATED... no —
    # apply distortion directly: for each undistorted pixel find its distorted
    # source via the OPENCV model, then remap.
    h, w = image.shape[:2]
    K = np.array([[profile.fx, 0, profile.cx], [0, profile.fy, profile.cy], [0, 0, 1]])
    dist = np.array([profile.k1, profile.k2, profile.p1, profile.p2])
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1)[:, None, :]

    # cv2.undistortPoints maps distorted->normalized-undistorted; projecting those
    # back through K gives, for each DISTORTED pixel, its undistorted location.
    # Remapping the clean image at those locations SAMPLES the clean image where
    # the undistorted content of each distorted pixel lives — i.e. it distorts it.
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(h, w, 2)
    return cv2.remap(image, und[..., 0], und[..., 1], cv2.INTER_LINEAR)


def test_profile_dict_roundtrip():
    profile = _profile()
    assert DistortionProfile.from_dict(profile.to_dict()) == profile


def test_undistort_recovers_synthetic_distortion():
    # Checkerboard so residuals are visible; distort with the known profile,
    # undistort with the same profile, compare against the clean original.
    profile = _profile()
    tile = np.kron(np.indices((12, 16)).sum(0) % 2, np.ones((40, 40))) * 255
    clean = np.repeat(tile.astype(np.uint8)[:, :, None], 3, axis=2)
    distorted = _distort_image(clean, profile)

    restored, K_new, roi = undistort_frames([distorted], profile)
    x, y, w, h = roi
    reference = clean[y : y + h, x : x + w]

    # Interior compare (border interpolation is lossy either way)
    diff = np.abs(restored[0][20:-20, 20:-20].astype(int) - reference[20:-20, 20:-20].astype(int))
    assert diff.mean() < 10.0


def test_crop_dims_even_and_k_consistent():
    profile = _profile(width=641, height=481)  # odd input dims force the even-crop path
    frames = [np.zeros((481, 641, 3), dtype=np.uint8)]
    out, K_new, roi = undistort_frames(frames, profile)
    x, y, w, h = roi

    assert w % 2 == 0 and h % 2 == 0
    assert out[0].shape == (h, w, 3)
    # Principal point shifted by the crop offset, still inside the crop
    assert 0 < K_new[0, 2] < w and 0 < K_new[1, 2] < h


def test_all_frames_same_shape():
    profile = _profile()
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
    out, _, _ = undistort_frames(frames, profile)
    assert len({f.shape for f in out}) == 1


def test_wrong_frame_dims_raise():
    profile = _profile(width=640, height=480)
    with pytest.raises(ValueError, match="dims"):
        undistort_frames([np.zeros((100, 100, 3), dtype=np.uint8)], profile)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_undistort.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'collab_splats.preproc.undistort'`

- [ ] **Step 3: Write the implementation**

Create `collab_splats/preproc/undistort.py`:

```python
"""
Camera undistortion at the frames.zarr boundary: OPENCV-model self-calibration
(pycolmap) + splatfacto-exact cv2 undistort/crop.

Every downstream consumer (feedforward backbones, InstantSfM SIFT, splat
trainer, localization DB export) assumes pinhole; undistorting once here fixes
all of them. Port of nerfstudio full_images_datamanager._undistort_image
(nerfstudio @ 50e0e3c): getOptimalNewCameraMatrix(alpha=0) -> cv2.undistort ->
ROI crop -> K rewritten by the crop offset.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


########################################
# Profile
########################################


@dataclass(frozen=True)
class DistortionProfile:
    """
    OPENCV camera model (k1 k2 p1 p2) + calibrated pinhole K + input dims.
    """

    k1: float
    k2: float
    p1: float
    p2: float
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_dict(self) -> dict:
        """
        JSON-serialisable dict (frames.zarr provenance payload).
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DistortionProfile":
        """
        Inverse of to_dict.
        """
        return cls(**d)

    @property
    def K(self) -> np.ndarray:
        """
        3x3 pinhole intrinsics of the calibration.
        """
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        """
        cv2-ordered distortion vector (k1, k2, p1, p2).
        """
        return np.array([self.k1, self.k2, self.p1, self.p2])


########################################
# Estimation (pycolmap self-calibration)
########################################


def estimate_camera_distortion(
    frames: list[np.ndarray], max_frames: int = 60
) -> DistortionProfile:
    """
    Self-calibrate one shared OPENCV camera from selected frames via pycolmap.

    - Runs SIFT + exhaustive matching + incremental mapping on <= max_frames
      evenly spaced frames (shared camera, ba_refine_extra_params on by default).
    - Raises ValueError when mapping fails or registers < 60% of the subset
      (too weak a solve to trust the distortion params).
    """
    import pycolmap  # heavy optional dep; hard-required only when undistort is on

    n = len(frames)
    idxs = np.unique(np.linspace(0, n - 1, min(max_frames, n)).round().astype(int))

    with tempfile.TemporaryDirectory(prefix="undistort_calib_") as tmp:
        tmp_path = Path(tmp)
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        out_dir = tmp_path / "sparse"
        out_dir.mkdir()
        database = tmp_path / "database.db"

        # Stage the calibration subset; store holds RGB, cv2 writes BGR
        for i in idxs:
            cv2.imwrite(
                str(image_dir / f"calib_{int(i):06d}.jpg"),
                cv2.cvtColor(frames[int(i)], cv2.COLOR_RGB2BGR),
            )

        # One shared OPENCV camera across the subset; mapper refines k1 k2 p1 p2
        pycolmap.extract_features(
            database,
            image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model="OPENCV",
        )
        pycolmap.match_exhaustive(database)
        reconstructions = pycolmap.incremental_mapping(database, image_dir, out_dir)

        if not reconstructions:
            raise ValueError(
                f"undistort: self-calibration failed — pycolmap registered no model "
                f"from {len(idxs)} frames"
            )
        recon = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        if recon.num_reg_images() < 0.6 * len(idxs):
            raise ValueError(
                f"undistort: self-calibration too weak — {recon.num_reg_images()}/"
                f"{len(idxs)} frames registered; distortion params not trustworthy"
            )

        # Shared camera: exactly the largest model's camera params, OPENCV order
        camera = next(iter(recon.cameras.values()))
        fx, fy, cx, cy, k1, k2, p1, p2 = (float(v) for v in camera.params)

    height, width = frames[0].shape[:2]
    profile = DistortionProfile(
        k1=k1, k2=k2, p1=p1, p2=p2,
        fx=fx, fy=fy, cx=cx, cy=cy,
        width=width, height=height,
    )
    logger.info(
        "undistort: calibrated k1=%.5f k2=%.5f p1=%.5f p2=%.5f over %d/%d frames",
        k1, k2, p1, p2, recon.num_reg_images(), len(idxs),
    )
    return profile


########################################
# Undistortion (splatfacto-exact cv2 path)
########################################


def undistort_frames(
    frames: list[np.ndarray], profile: DistortionProfile
) -> tuple[list[np.ndarray], np.ndarray, tuple[int, int, int, int]]:
    """
    Undistort frames with alpha=0 crop; returns (frames, K_new, roi).

    - roi is (x, y, w, h) with w/h forced even (codec/model friendliness);
      K_new is the optimal new camera matrix shifted by the crop offset.
    - Raises ValueError when frame dims disagree with the profile.
    """
    height, width = frames[0].shape[:2]
    if (width, height) != (profile.width, profile.height):
        raise ValueError(
            f"undistort: frame dims {width}x{height} != profile dims "
            f"{profile.width}x{profile.height}"
        )

    # alpha=0: zoom so the valid (distortion-free) region fills the ROI
    K_new, roi = cv2.getOptimalNewCameraMatrix(
        profile.K, profile.dist_coeffs, (width, height), 0
    )
    x, y, w, h = roi
    w -= w % 2
    h -= h % 2

    # Per-frame: remap to the new camera, crop to the even ROI
    out = [
        cv2.undistort(frame, profile.K, profile.dist_coeffs, None, K_new)[
            y : y + h, x : x + w
        ]
        for frame in frames
    ]

    # Principal point moves with the crop
    K_out = K_new.copy()
    K_out[0, 2] -= x
    K_out[1, 2] -= y
    return out, K_out, (x, y, w, h)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_undistort.py -v`
Expected: all 5 PASS. If `test_undistort_recovers_synthetic_distortion` fails on the mean-diff threshold, check the `_distort_image` direction comment first — the remap-through-undistortPoints trick is the standard forward-distortion construction; a systematically large diff means the implementation cropped or K-shifted wrongly, not that the tolerance is tight.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats
git commit --only collab_splats/preproc/undistort.py tests/preproc/test_undistort.py \
  -m "feat(preproc): DistortionProfile + splatfacto-exact undistort_frames"
```

---

### Task 2: `estimate_camera_distortion` smoke test (real frames)

The pycolmap half was written in Task 1 (single module, imports at top rule). This task proves it on real footage — synthetic multi-view of a plane is degenerate for self-calibration, so the test uses the tutorial scene.

**Files:**
- Modify: `tests/preproc/test_undistort.py`

- [ ] **Step 1: Write the failing (or first-run) test**

Append to `tests/preproc/test_undistort.py`:

```python
def test_estimate_camera_distortion_tutorial_smoke():
    # Real-footage smoke: SIFT + exhaustive + mapper on 20 tutorial frames.
    # Slow (~1-2 min CPU); asserts a sane shared-camera OPENCV solve, not
    # specific distortion values.
    pycolmap = pytest.importorskip("pycolmap")  # noqa: F841
    from collab_splats.preproc.frame_store import FrameStore
    from collab_splats.preproc.undistort import estimate_camera_distortion

    frames_zarr = Path("data/tutorial/frames.zarr")
    if not frames_zarr.exists():
        pytest.skip("data/tutorial/frames.zarr not present")

    store = FrameStore.open(frames_zarr)
    frames = [store.image(i) for i in range(len(store))]
    profile = estimate_camera_distortion(frames, max_frames=20)

    height, width = frames[0].shape[:2]
    assert (profile.width, profile.height) == (width, height)
    assert 0 < profile.fx < 4 * width and 0 < profile.fy < 4 * width
    assert abs(profile.k1) < 0.5 and abs(profile.k2) < 0.5
    assert np.isfinite([profile.k1, profile.k2, profile.p1, profile.p2]).all()
```

Also add to the imports at the top of the test file: `from pathlib import Path`.

- [ ] **Step 2: Check the tutorial store exists, then run**

Run: `ls data/tutorial/frames.zarr` — if missing, run `git lfs pull` or accept the skip (the test self-skips; note it in the task summary).
Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_undistort.py::test_estimate_camera_distortion_tutorial_smoke -v`
Expected: PASS in ~1–3 min (CPU SIFT + exhaustive on 20 frames), or SKIP when the store is absent. If pycolmap raises inside `incremental_mapping`, quote the error verbatim in the task summary — do not paper over it with a broader skip.

- [ ] **Step 3: Commit**

```bash
cd /workspace/collab-splats
git commit --only tests/preproc/test_undistort.py \
  -m "test(preproc): self-calibration smoke on tutorial frames"
```

---

### Task 3: Wire undistortion into `extract_frames` + config + re-export

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (`extract_frames` at :116, both `FrameStore.create` branches at :150 and :208; `preprocess` call site at :800)
- Modify: `configs/base.yaml` (preproc block)
- Modify: `collab_splats/preproc/__init__.py`
- Modify: `tests/preproc/test_undistort.py`, `tests/preproc/test_sampling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/preproc/test_undistort.py`:

```python
def test_provenance_roundtrip_through_frame_store(tmp_path):
    # The undistort provenance payload written by extract_frames must survive
    # zarr attrs json round-trip and rebuild an identical profile.
    from collab_splats.preproc.frame_store import FrameStore

    profile = _profile()
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
    out, K_new, roi = undistort_frames(frames, profile)
    prov = {
        "video_path": "v.mp4",
        "video_mtime": None,
        "method": "dir",
        "fps": None,
        "max_frames": None,
        "undistort": {
            "profile": profile.to_dict(),
            "K_new": K_new.tolist(),
            "roi": list(roi),
        },
    }
    store = FrameStore.create(
        tmp_path / "frames.zarr", out, [{"frame_idx": 0}], provenance=prov
    )

    stored = store._store.attrs["provenance"]["undistort"]
    assert DistortionProfile.from_dict(stored["profile"]) == profile
    assert stored["roi"] == list(roi)
    assert np.allclose(np.array(stored["K_new"]), K_new)


def test_extract_frames_dir_undistorts(tmp_path, monkeypatch):
    # Image-dir branch: undistort=True estimates once, crops every frame, and
    # stamps provenance. Estimation is monkeypatched — no pycolmap in this test.
    import collab_splats.wrapper.reconstructor as recon_mod
    from collab_splats.preproc.frame_store import FrameStore

    src = tmp_path / "imgs"
    src.mkdir()
    for i in range(3):
        cv2.imwrite(str(src / f"{i:03d}.jpg"), np.full((480, 640, 3), 128, np.uint8))

    profile = _profile()
    monkeypatch.setattr(
        recon_mod, "estimate_camera_distortion", lambda frames, **kw: profile
    )

    frames_zarr = tmp_path / "frames.zarr"
    n = recon_mod.extract_frames(
        input_path=src, frames_zarr=frames_zarr, frame_selection="fps",
        fps=None, min_frames=None, max_frames=None, undistort=True,
    )
    assert n == 3

    store = FrameStore.open(frames_zarr)
    prov = store._store.attrs["provenance"]
    x, y, w, h = prov["undistort"]["roi"]
    assert store.image(0).shape == (h, w, 3)
    assert DistortionProfile.from_dict(prov["undistort"]["profile"]) == profile


def test_extract_frames_dir_no_undistort_no_payload(tmp_path):
    # Default path unchanged: no undistort key in provenance, native dims kept.
    from collab_splats.preproc.frame_store import FrameStore
    from collab_splats.wrapper.reconstructor import extract_frames

    src = tmp_path / "imgs"
    src.mkdir()
    cv2.imwrite(str(src / "000.jpg"), np.zeros((480, 640, 3), np.uint8))

    frames_zarr = tmp_path / "frames.zarr"
    extract_frames(
        input_path=src, frames_zarr=frames_zarr, frame_selection="fps",
        fps=None, min_frames=None, max_frames=None,
    )
    store = FrameStore.open(frames_zarr)
    assert "undistort" not in store._store.attrs["provenance"]
    assert store.image(0).shape == (480, 640, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_undistort.py -v -k "provenance or extract_frames"`
Expected: `test_provenance_roundtrip_through_frame_store` PASSES already (pure FrameStore); the two `extract_frames` tests FAIL with `TypeError: extract_frames() got an unexpected keyword argument 'undistort'`.

- [ ] **Step 3: Implement the wiring**

In `collab_splats/wrapper/reconstructor.py`:

3a. Import at top (with the other preproc imports):

```python
from collab_splats.preproc.undistort import estimate_camera_distortion, undistort_frames
```

3b. Add the param to `extract_frames` (signature at :116):

```python
def extract_frames(
    input_path: Path,
    frames_zarr: Path,
    frame_selection: str,
    fps: float | None,
    min_frames: int | None,
    max_frames: int | None,
    n_workers: int = 1,
    undistort: bool = False,
) -> int:
```

3c. Add one helper directly above `extract_frames` (shared by both branches):

```python
def _apply_undistortion(frame_arrays: list, prov: dict) -> list:
    """
    Self-calibrate + undistort selected frames in place of the raw ones; stamp provenance.
    """
    profile = estimate_camera_distortion(frame_arrays)
    frame_arrays, K_new, roi = undistort_frames(frame_arrays, profile)
    prov["undistort"] = {
        "profile": profile.to_dict(),
        "K_new": K_new.tolist(),
        "roi": list(roi),
    }
    return frame_arrays
```

3d. In the image-dir branch, between the `prov = {...}` dict and `FrameStore.create` (currently :143–150):

```python
        if undistort:
            frame_arrays = _apply_undistortion(frame_arrays, prov)
        FrameStore.create(frames_zarr, frame_arrays, records, provenance=prov)
```

3e. Same two lines in the video branch, between `prov = {...}` and `FrameStore.create` (currently :201–208).

3f. In `Reconstructor.preprocess` (call site at :800), pass the flag:

```python
            n_workers=pre_cfg["n_workers"],
            undistort=pre_cfg["undistort"],
```

3g. In `configs/base.yaml`, append to the `preproc:` block after `n_workers`:

```yaml
  undistort: false            # self-calibrate one shared OPENCV camera (pycolmap, <=60 frames)
                              # and undistort every selected frame (cv2, alpha=0 crop) before
                              # frames.zarr is written. Off until the GH010229 A/B measures the
                              # win. frames.zarr reuse is by EXISTENCE — toggling this on an
                              # existing scene needs preprocess(overwrite=True). Localization
                              # QUERY images are NOT undistorted (they never pass through here).
```

3h. In `collab_splats/preproc/__init__.py`, add to the imports and `__all__`:

```python
from collab_splats.preproc.undistort import (
    DistortionProfile,
    estimate_camera_distortion,
    undistort_frames,
)
```

and extend `__all__` with `"DistortionProfile", "estimate_camera_distortion", "undistort_frames"`.

3i. `tests/preproc/test_sampling.py::test_public_api_surface` asserts EXACT set equality on `__all__` — add the same three names to its expected set.

- [ ] **Step 4: Run the tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -v`
Expected: all pass, including the updated `test_public_api_surface`. The estimation smoke from Task 2 may be skipped if the tutorial store is absent — fine.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats
git commit --only collab_splats/wrapper/reconstructor.py configs/base.yaml \
  collab_splats/preproc/__init__.py tests/preproc/test_undistort.py \
  tests/preproc/test_sampling.py \
  -m "feat(preproc): undistort flag wired into extract_frames (one boolean, provenance-stamped)"
```

---

### Task 4: Permutation view sampler

**Files:**
- Modify: `collab_splats/splats/trainer.py` (new class near the setup helpers; loop change at :318)
- Modify: `tests/splats/test_trainer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/splats/test_trainer.py`:

```python
def test_view_sampler_covers_every_view_once_per_epoch():
    from collab_splats.splats.trainer import ViewSampler

    sampler = ViewSampler(7, seed=42)
    epoch1 = [sampler.next() for _ in range(7)]
    epoch2 = [sampler.next() for _ in range(7)]

    assert sorted(epoch1) == list(range(7))
    assert sorted(epoch2) == list(range(7))
    # Reshuffle across epochs: identical order for 7 views has p = 1/5040
    assert epoch1 != epoch2


def test_view_sampler_deterministic_for_seed():
    from collab_splats.splats.trainer import ViewSampler

    a = [ViewSampler(5, seed=42).next() for _ in range(1)]
    runs = [[ViewSampler(5, seed=42).next() for _ in range(15)] for _ in range(2)]
    assert runs[0] == runs[1]
    assert a[0] == runs[0][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_trainer.py -v -k view_sampler`
Expected: FAIL with `ImportError: cannot import name 'ViewSampler'`

- [ ] **Step 3: Implement**

In `collab_splats/splats/trainer.py`, add `import random` at the top, then the class in the "Setup helpers" section (after `compute_scene_scale`):

```python
class ViewSampler:
    """
    Splatfacto's view schedule: seeded shuffled permutation, popped until empty, reshuffled.

    - Guarantees every view trains max_steps/n_views (+-1) times, vs +-17%
      spread from torch.randint sampling with replacement.
    - Port of nerfstudio @ 50e0e3c full_images_datamanager (random.Random shuffle + pop).
    """

    def __init__(self, n_views: int, seed: int = 42):
        self._rng = random.Random(seed)
        self._n_views = n_views
        self._pending: list[int] = []

    def next(self) -> int:
        """
        Next view index; reshuffles a fresh permutation when the epoch empties.
        """
        if not self._pending:
            self._pending = list(range(self._n_views))
            self._rng.shuffle(self._pending)
        return self._pending.pop()
```

Then in `train()`, replace the line at :318:

```python
        view = int(torch.randint(n_views, (1,)))
```

with:

```python
        view = view_sampler.next()
```

and add, just above the `for step in progress(...)` loop (beside `start_time`):

```python
    view_sampler = ViewSampler(n_views)
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_trainer.py -v`
Expected: all pass (existing config tests untouched).

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats
git commit --only collab_splats/splats/trainer.py tests/splats/test_trainer.py \
  -m "feat(splats): permutation view sampler (splatfacto schedule) replaces randint"
```

---

### Task 5: Coarse-to-fine resolution schedule

**Files:**
- Modify: `collab_splats/splats/trainer.py` (SplatsConfig; two helpers; loop wiring at :319–:343)
- Modify: `configs/base.yaml` (splats block)
- Modify: `tests/splats/test_trainer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/splats/test_trainer.py`:

```python
def test_downscale_factor_boundaries():
    from collab_splats.splats.trainer import downscale_factor

    # splatfacto defaults: num_downscales=2, resolution_schedule=3000
    assert downscale_factor(0, 2, 3000) == 4
    assert downscale_factor(2999, 2, 3000) == 4
    assert downscale_factor(3000, 2, 3000) == 2
    assert downscale_factor(5999, 2, 3000) == 2
    assert downscale_factor(6000, 2, 3000) == 1
    assert downscale_factor(29999, 2, 3000) == 1
    # 0 disables the schedule entirely
    assert downscale_factor(0, 0, 3000) == 1


def test_downscale_view_scales_image_and_k():
    import numpy as np
    import torch

    from collab_splats.splats.trainer import downscale_view

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    K = torch.tensor([[[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]])

    small, K_small = downscale_view(image, K, 4)
    assert small.shape == (120, 160, 3)
    assert torch.allclose(K_small[0, 0, 0], torch.tensor(125.0))
    assert torch.allclose(K_small[0, 0, 2], torch.tensor(80.0))
    assert torch.allclose(K_small[0, 2, 2], torch.tensor(1.0))

    same, K_same = downscale_view(image, K, 1)
    assert same is image and K_same is K


def test_splats_config_accepts_downscale_fields():
    from collab_splats.splats.trainer import SplatsConfig

    cfg = SplatsConfig.from_dict(
        {"enabled": True, "num_downscales": 1, "resolution_schedule": 100}
    )
    assert cfg.num_downscales == 1 and cfg.resolution_schedule == 100
    # Defaults are splatfacto's
    default = SplatsConfig()
    assert default.num_downscales == 2 and default.resolution_schedule == 3000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_trainer.py -v -k downscale`
Expected: FAIL with `ImportError: cannot import name 'downscale_factor'`

- [ ] **Step 3: Implement**

3a. `SplatsConfig` gains two fields (after `log_every`; `from_dict`'s `allowed_keys` picks them up automatically from `__dataclass_fields__`):

```python
    # Coarse-to-fine (splatfacto): start at 1/2^num_downscales resolution, double
    # every resolution_schedule steps. 0 disables.
    num_downscales: int = 2
    resolution_schedule: int = 3000
```

3b. Two helpers in the trainer, beside `prepare_training_target` (add `import cv2` at the top of trainer.py):

```python
def downscale_factor(step: int, num_downscales: int, resolution_schedule: int) -> int:
    """
    Coarse-to-fine divisor at a step: 2^max(0, num_downscales - step // resolution_schedule).
    """
    if num_downscales <= 0:
        return 1
    return 2 ** max(0, num_downscales - step // resolution_schedule)


def downscale_view(image: np.ndarray, intrinsics: Tensor, factor: int):
    """
    Image (bilinear) and K scaled by 1/factor; passthrough at factor 1.
    """
    if factor == 1:
        return image, intrinsics
    height, width = image.shape[:2]
    small = cv2.resize(
        image, (width // factor, height // factor), interpolation=cv2.INTER_LINEAR
    )
    K_small = intrinsics.clone()
    K_small[:, :2, :] /= factor
    return small, K_small
```

3c. Wire into `train()`'s loop. Replace the current block at :318–:323:

```python
        view = view_sampler.next()
        view_image = images[view]
        view_depth_target = None if depth_targets is None else depth_targets[view]
        target = prepare_training_target(view_image, view_depth_target, device)
        view_cam_to_world = cam_to_world[view : view + 1]
        view_intrinsics = intrinsics_gpu[view : view + 1]
```

with:

```python
        view = view_sampler.next()
        view_image = images[view]
        view_depth_target = None if depth_targets is None else depth_targets[view]

        # Coarse-to-fine: train at 1/4 -> 1/2 -> native resolution on the splatfacto
        # schedule. Depth targets follow the image automatically — prepare_training_target
        # nearest-resizes them to the (downscaled) image dims.
        factor = downscale_factor(step, cfg.num_downscales, cfg.resolution_schedule)
        view_intrinsics = intrinsics_gpu[view : view + 1]
        view_image, view_intrinsics = downscale_view(view_image, view_intrinsics, factor)
        step_height, step_width = view_image.shape[:2]

        target = prepare_training_target(view_image, view_depth_target, device)
        view_cam_to_world = cam_to_world[view : view + 1]
```

and in the `render_view(...)` call at :333–:343, replace the `width, height` arguments with `step_width, step_height`.

3d. `configs/base.yaml` splats block, after `grow_grad2d`:

```yaml
  num_downscales: 2          # coarse-to-fine (splatfacto): start at 1/4 res, double every
  resolution_schedule: 3000  # this many steps until native. 0 disables.
```

- [ ] **Step 4: Run the full splats test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/ -v`
Expected: all pass. Watch for tests that call `train()` with tiny synthetic scenes (`tests/splats/synthetic.py` fixtures): a 64px synthetic image at factor 4 is 16px — still renderable; if a shape assert trips inside gsplat, the test scene needs `num_downscales: 0` in its config dict, not a production-code special case.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats
git commit --only collab_splats/splats/trainer.py configs/base.yaml \
  tests/splats/test_trainer.py \
  -m "feat(splats): coarse-to-fine resolution schedule (splatfacto num_downscales/resolution_schedule)"
```

---

### Task 6: Splatfacto densification args on DefaultStrategy (2dgs)

Verdict recorded in the spec: absgrad is NOT usable on the `gradient_2dgs` path (backward sets `.absgrad` on `means2d` only; the strategy would read `info["gradient_2dgs"].absgrad` → AttributeError). 2dgs keeps `absgrad=False, grow_grad2d=2e-4`; this task lands the OTHER splatfacto args.

**Files:**
- Modify: `collab_splats/splats/trainer.py` (`make_strategy` at :203; call site at :295)
- Modify: `tests/splats/test_trainer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/splats/test_trainer.py`:

```python
def test_make_strategy_2dgs_splatfacto_args():
    from gsplat.strategy import DefaultStrategy

    from collab_splats.splats.trainer import SplatsConfig, make_strategy

    cfg = SplatsConfig(primitive="2dgs")
    strategy = make_strategy(cfg, n_views=300)

    assert isinstance(strategy, DefaultStrategy)
    # splatfacto (nerfstudio @ 50e0e3c) non-default args
    assert strategy.prune_opa == 0.1
    assert strategy.prune_scale3d == 0.5
    assert strategy.refine_scale2d_stop_iter == 4000
    assert strategy.pause_refine_after_reset == 400  # n_views + 100
    # Measured-good pair kept (absgrad unusable on gradient_2dgs — see spec)
    assert strategy.absgrad is False
    assert strategy.grow_grad2d == pytest.approx(2e-4)
    assert strategy.key_for_gradient == "gradient_2dgs"


def test_make_strategy_3dgs_untouched():
    from gsplat.strategy import MCMCStrategy

    from collab_splats.splats.trainer import SplatsConfig, make_strategy

    strategy = make_strategy(SplatsConfig(primitive="3dgs"), n_views=300)
    assert isinstance(strategy, MCMCStrategy)
    assert strategy.cap_max == 1_000_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/test_trainer.py -v -k make_strategy`
Expected: FAIL with `TypeError: make_strategy() got an unexpected keyword argument 'n_views'`

- [ ] **Step 3: Implement**

Replace `make_strategy` (trainer.py:203–209):

```python
def make_strategy(cfg: SplatsConfig, n_views: int) -> MCMCStrategy | DefaultStrategy:
    """
    MCMC for 3dgs (budgeted, no gradient heuristics); Default with splatfacto's args for 2dgs.
    """
    if cfg.primitive == "3dgs":
        return MCMCStrategy(cap_max=cfg.cap_max, verbose=False)

    # splatfacto (nerfstudio @ 50e0e3c) non-default DefaultStrategy args. absgrad stays
    # False: the 2dgs backward writes .absgrad on means2d only, never on the
    # gradient_2dgs densify tensor this strategy reads (see the parity spec's verdict),
    # and grow_grad2d 2e-4 is the measured-good non-absgrad threshold.
    return DefaultStrategy(
        absgrad=False,
        grow_grad2d=cfg.grow_grad2d,
        key_for_gradient="gradient_2dgs",
        prune_opa=0.1,
        prune_scale3d=0.5,
        refine_scale2d_stop_iter=4000,
        pause_refine_after_reset=n_views + 100,
        verbose=False,
    )
```

Update the call site in `train()` (:295):

```python
    strategy = make_strategy(cfg, n_views)
```

- [ ] **Step 4: Run the splats suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/splats/ -v`
Expected: all pass. Any other `make_strategy(` caller found by `grep -rn "make_strategy(" collab_splats/ tests/ evals/` gets the `n_views` argument too (known callers: trainer.py train(); check eval/analyze scripts).

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats
git commit --only collab_splats/splats/trainer.py tests/splats/test_trainer.py \
  -m "feat(splats): splatfacto DefaultStrategy args for 2dgs (prune_opa 0.1, pause n_views+100)"
```

---

### Task 7: A/B run 1 — undistortion only (GPU, tmux, human-gated)

Fresh scene (undistorted frames need a full instantsfm rebuild), trainer at Task 3 state semantics — but by now trainer changes are merged, so run 1 must PIN the old trainer behavior via config: `num_downscales: 0` disables coarse-to-fine; sampler and densification args cannot be config-disabled, so run 1 isolates undistortion + (sampler/densification) while run 2 adds coarse-to-fine. Report deltas accordingly — the report must name exactly which levers each run carries.

**Files:**
- Create: `/tmp/claude-0/-workspace-collab-splats/a554d3f6-ae28-4875-bc24-afcfa133dfb5/scratchpad/gopro_undist_overrides.yaml`
- Create: `/tmp/claude-0/-workspace-collab-splats/a554d3f6-ae28-4875-bc24-afcfa133dfb5/scratchpad/run_ab_undist.py`

- [ ] **Step 1: Check the GPU is free**

Run: `nvidia-smi --query-compute-apps=used_memory --format=csv,noheader`
Expected: empty (no compute apps). If busy, wait — do not co-run (46.6 GB cgroup cap, OOM risk).

- [ ] **Step 2: Write the overrides yaml**

`$SP/gopro_undist_overrides.yaml` (SP = the session scratchpad):

```yaml
# A/B run 1: undistortion lever. Same 2dgs config as the 19.21 baseline
# (300 frames, sparse depth targets, grow 2e-4, retriangulation on); ONLY
# preproc.undistort flips on and coarse-to-fine is pinned OFF so the trainer
# matches baseline behavior as closely as config allows (permutation sampler
# and densification args are not config-gated — the report says so).
preproc:
  fps: 2.0
  min_frames: null
  max_frames: 300
  n_workers: 4
  undistort: true
pointcloud:
  method: sfm
  backend: instantsfm
  instantsfm:
    retriangulation: true
semantics:
  enabled: false
splats:
  enabled: true
  primitive: 2dgs
  pose_opt: true
  max_steps: 30000
  grow_grad2d: 2.0e-4
  num_downscales: 0          # run 1 pins coarse-to-fine OFF; run 2 turns it on
  resolution_schedule: 3000
  losses:
    depth: {weight: 0.01}
    normal_consistency: {weight: 0.05, start: 7000}
    distortion: {weight: 0.01, start: 3000}
    # base.yaml's 3dgs regularisers survive the config deep-merge — zero explicitly
    opacity_reg: {weight: 0.0}
    scale_reg: {weight: 0.0}
mesh:
  enabled: false
```

- [ ] **Step 3: Write the driver**

`$SP/run_ab_undist.py`:

```python
"""
A/B run 1 driver: GH010229 300 frames, undistorted, instantsfm + 2dgs 30k.

Fresh scene dir (undistorted frames change every downstream artifact). Standard
pipeline: preprocess (undistort on) -> build_pointcloud (instantsfm + retri) ->
splats. Compare splats_quality_report.json against the 19.21/0.621 baseline.
"""

import faulthandler
import logging
import sys
from pathlib import Path

faulthandler.enable()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

REPO = Path("/workspace/collab-splats")
sys.path.insert(0, str(REPO))

import yaml

import collab_splats
from collab_splats.wrapper.reconstructor import Reconstructor

print(f"collab_splats from: {collab_splats.__file__}", flush=True)
assert str(REPO) in str(collab_splats.__file__), "wrong collab_splats on sys.path"

OVERRIDES = Path(__file__).parent / "gopro_undist_overrides.yaml"
SCENE = Path("/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229_undist")
VIDEO = Path("/workspace/outputs/2026_07_15-Goprosplat-GH010229/GH010229.mp4")

SCENE.mkdir(parents=True, exist_ok=True)

config = yaml.safe_load(OVERRIDES.read_text())
config["input_path"] = str(VIDEO)
config["output_path"] = str(SCENE)

recon = Reconstructor(config, config_dir=REPO / "configs")
recon.preprocess()
print("PREPROC DONE", flush=True)
recon.build_pointcloud()
print("POINTCLOUD DONE", flush=True)
recon.splats(overwrite=True)
print("RUN1 DONE", flush=True)
```

- [ ] **Step 4: Launch in tmux and monitor**

```bash
tmux new-session -d -s ab_undist \
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /opt/venv/reconstruction/bin/python \
   /tmp/claude-0/-workspace-collab-splats/a554d3f6-ae28-4875-bc24-afcfa133dfb5/scratchpad/run_ab_undist.py \
   > /tmp/claude-0/-workspace-collab-splats/a554d3f6-ae28-4875-bc24-afcfa133dfb5/scratchpad/run_ab_undist.log 2>&1"
```

Monitor: `tail -5 $SP/run_ab_undist.log` every ~10 min. Expected phases: undistort self-calib (~2–4 min), VDA depth + GPU SIFT + mapper (~30–45 min), splats 30k (~60–75 min). Disk check before splat phase: `df -h /workspace` — the quota is cumulative; if under ~3 GB free, clean before the run crashes at the zarr write (the 4fps run's exact failure).

- [ ] **Step 5: Record the result**

Read `SCENE/instantsfm/splats/splats_quality_report.json` → summary psnr/ssim/n_gaussians/seconds. Baseline: **19.21 / 0.621 @ 1.4M** (sparse targets, grow 2e-4). Record the delta and which levers run 1 actually carried (undistort + sampler + densification args, coarse-to-fine off). Do NOT delete anything; do NOT start run 2 in parallel.

---

### Task 8: A/B run 2 — full trainer stack (GPU, tmux, human-gated)

Reuses run 1's scene (same undistorted geometry — isolates coarse-to-fine): splats stage only, `num_downscales: 2`.

**Files:**
- Create: `$SP/gopro_undist_full_overrides.yaml` (copy of run 1's yaml with `num_downscales: 2`)
- Create: `$SP/run_ab_full.py`

- [ ] **Step 1: Write overrides + driver**

`gopro_undist_full_overrides.yaml`: identical to `gopro_undist_overrides.yaml` except:

```yaml
  num_downscales: 2          # run 2: full splatfacto stack
```

`$SP/run_ab_full.py` — same as `run_ab_undist.py` except: docstring says run 2, `OVERRIDES` points at the full yaml, and the pipeline body is:

```python
recon = Reconstructor(config, config_dir=REPO / "configs")
recon.preprocess()          # store exists -> skips
recon.build_pointcloud()    # artifacts exist -> loads off disk
print("POINTCLOUD READY", flush=True)
recon.splats(overwrite=True)
print("RUN2 DONE", flush=True)
```

**Before launching:** copy run 1's splat outputs aside, or run 2's `overwrite=True` destroys them:

```bash
mv /workspace/outputs/rerun_2026_08_23_instantsfm/GH010229_undist/instantsfm/splats \
   /workspace/outputs/rerun_2026_08_23_instantsfm/GH010229_undist/instantsfm/splats_run1
```

- [ ] **Step 2: Launch in tmux (GPU quiet first, same check as Task 7)**

```bash
tmux new-session -d -s ab_full \
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /opt/venv/reconstruction/bin/python \
   /tmp/claude-0/-workspace-collab-splats/a554d3f6-ae28-4875-bc24-afcfa133dfb5/scratchpad/run_ab_full.py \
   > /tmp/claude-0/-workspace-collab-splats/a554d3f6-ae28-4875-bc24-afcfa133dfb5/scratchpad/run_ab_full.log 2>&1"
```

~60–75 min (splats only; coarse-to-fine makes the first 6000 steps cheaper).

- [ ] **Step 3: Report**

Three-row table: baseline 19.21/0.621@1.4M, run 1 (undistort + sampler + densification, no c2f), run 2 (full stack). Per-lever attribution is partial by construction — say so. Then the decision the spec defers to this evidence: flip `preproc.undistort` default? (only on a clear win; user decides). Append the measured section to the spec and update memory (`project_instantsfm_backend.md` or a new parity memory file). Commit spec/plan updates:

```bash
cd /workspace/collab-splats
git commit --only docs/superpowers/specs/2026-08-25-splatfacto-parity-design.md \
  docs/superpowers/plans/2026-08-25-splatfacto-parity.md \
  -m "docs(specs): splatfacto parity A/B measured results"
```

**NO 875-frame runs — explicitly excluded by the user.**

---

## Execution notes

- Tasks 1–6 are CPU-testable and independent of the GPU; run them back-to-back. Tasks 7–8 are compute + human-gated: announce before launching, one at a time.
- After each code task, `graphify update .` is cheap and keeps the graph current (project rule).
- Formatting: `black` + `isort` on ONLY the touched files (never repo-wide — venv black is newer than CI's): `/opt/venv/reconstruction/bin/python -m black <files> && /opt/venv/reconstruction/bin/python -m isort <files>`. isort wraps at 88 — write long imports parenthesized.
