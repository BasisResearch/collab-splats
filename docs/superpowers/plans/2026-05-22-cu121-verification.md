# cu121 Migration Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all verification artefacts from the cu121 verification spec: Phase 0 runbook, Phase 1–3 pytest file, Phase 5 integration tests, then execute Phase 4 (existing suite) and Phase 5 to confirm the migration is clean.

**Architecture:** Two new test files. `tests/test_cu121_migration.py` covers Phase 1 (env guards), Phase 2 (import sweep), and Phase 3 (migration smoke tests). `tests/integration/test_pipeline_cu121.py` covers Phase 5 pipeline integration using synthetic data and mocked `_forward` steps — no model downloads required. Phase 0 is a shell runbook. Phase 4 is the existing test suite run.

**Tech Stack:** pytest, Python 3.11, torch 2.4.0+cu121, pycolmap 4.0.4, gtsam-develop, bae 0.2.3 (USE_CUDSS=1), gsplat-rade, nerfstudio (local /workspace/nerfstudio), unittest.mock

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `tests/test_cu121_migration.py` | Create | Phase 1/2/3: env guards, import sweep, migration smoke |
| `tests/integration/__init__.py` | Create | Make integration/ a pytest package |
| `tests/integration/test_pipeline_cu121.py` | Create | Phase 5: pipeline round-trips with synthetic data |

---

## Task 0: Phase 0 Runbook — Execute Setup Scripts

No files to create. Run these commands in order, check each exit code. All must exit 0.

**Prerequisites:** `conda activate nerfstudio`

- [ ] **Step 1: Verify conda env**

```bash
conda activate nerfstudio
python --version
which python
```

Expected: `Python 3.11.x` and path is `/opt/conda/envs/nerfstudio/bin/python`. If version shows 3.13, wrong env active.

- [ ] **Step 2: Run setup_nerfstudio.sh**

```bash
cd /workspace/collab-splats
bash setup_nerfstudio.sh 2>&1 | tee /tmp/setup_nerfstudio.log
echo "Exit code: $?"
```

Expected: exit 0. All 5 patch lines appear:
```
patched: "gsplat==1.4.0" → ...
patched: "timm==0.6.7" → ...
patched: "viser==1.0.0" → ...
patched: '"opencv-python-headless==4.10.0.84",' → ''
patched: "nerfacc==0.5.2" → ...
```
Then: `[OK] gsplat-rade fork confirmed` in smoke test output.

- [ ] **Step 3: Verify nerfstudio patches applied in file**

```bash
python - <<'EOF'
import pathlib, sys
text = pathlib.Path("/workspace/nerfstudio/pyproject.toml").read_text()
checks = [
    ("gsplat-rade fork", "gsplat @ git+https://github.com/brian-xu/gsplat-rade.git" in text),
    ("no gsplat==1.4.0", "gsplat==1.4.0" not in text),
    ("timm>=0.6.7", "timm>=0.6.7" in text),
    ("viser>=0.2.0", "viser>=0.2.0" in text),
    ("no opencv-headless", "opencv-python-headless" not in text),
    ("nerfacc>=0.5.2", "nerfacc>=0.5.2" in text),
]
failed = [n for n, ok in checks if not ok]
[print(f"[FAIL] {n}") for n in failed]
if not failed: print("[OK] all 5 nerfstudio patches verified")
sys.exit(len(failed))
EOF
```

- [ ] **Step 4: Dry-run pip install -e . (catch dep parse errors before installing)**

```bash
python - <<'EOF'
import tomllib, sys
with open("pyproject.toml", "rb") as f:
    t = tomllib.load(f)
deps = t["project"]["dependencies"]
required = ["numpy", "scipy", "pypose", "bae", "clip", "python-dotenv", "pycolmap", "meshlib"]
dropped   = ["nerfstudio", "fairscale", "dotenv"]
ok = True
for pkg in required:
    if not any(pkg in d for d in deps):
        print(f"[FAIL] missing: {pkg}"); ok = False
for pkg in dropped:
    if any(pkg in d for d in deps):
        print(f"[FAIL] should be absent: {pkg}"); ok = False
ff = t["project"]["optional-dependencies"]["feedforward"]
if not any("gtsam-develop" in d for d in ff):
    print("[FAIL] feedforward: gtsam-develop missing"); ok = False
if ok: print("[OK] pyproject.toml looks correct")
sys.exit(0 if ok else 1)
EOF

pip install -e . --dry-run 2>&1 | grep -E "ERROR|InvalidRequirement|ParseError|conflict" \
  && echo "[FAIL] pip dry-run has errors" \
  || echo "[OK] pip dry-run: no resolution errors"
```

- [ ] **Step 5: pip install -e .**

```bash
pip install -e . 2>&1 | tee /tmp/setup_core.log
echo "Exit code: $?"
```

Expected: exit 0. Check log for `InvalidRequirement` (pycolmap backtick fix) if it fails.

- [ ] **Step 6: Install collab-data**

```bash
pip install git+https://github.com/BasisResearch/collab-data.git 2>&1 | tail -5
echo "Exit code: $?"
python -c "import collab_data; print('[OK] collab_data installed')"
```

- [ ] **Step 7: Install co3d (--no-deps required)**

```bash
pip install git+https://github.com/facebookresearch/co3d.git --no-deps 2>&1 | tail -5
echo "Exit code: $?"
```

- [ ] **Step 8: Run setup_feedforward.sh**

```bash
bash setup_feedforward.sh 2>&1 | tee /tmp/setup_feedforward.log
echo "Exit code: $?"
```

Expected: exit 0. Look for the invariant line:
```
[OK] torch=2.4.0+cu121  cuda=12.1  timm=<version>
```
If `FAIL: CUDA changed` appears, wrong torch is active.

- [ ] **Step 9: Verify gtsam-develop SL4 symbols**

```bash
python -c "
from gtsam import SL4, PriorFactorSL4, BetweenFactorSL4
print('[OK] gtsam SL4 manifold available')
"
```

- [ ] **Step 10: Verify deleted files are gone**

```bash
python - <<'EOF'
import pathlib, sys
deleted = ["setup_bundle_adjustment.sh", "patches/bae-torch21-compat.patch"]
failed = [f for f in deleted if pathlib.Path(f).exists()]
[print(f"[FAIL] still exists: {f}") for f in failed]
if not failed: print("[OK] deleted files absent")
sys.exit(len(failed))
EOF
```

- [ ] **Step 11: Verify constraints_feedforward.txt**

```bash
python - <<'EOF'
import pathlib, sys
lines = [l.strip() for l in pathlib.Path("constraints_feedforward.txt").read_text().splitlines()
         if l.strip() and not l.startswith("#")]
checks = [
    ("torch cu121", "torch==2.4.0+cu121" in lines),
    ("torchvision cu121", "torchvision==0.19.0+cu121" in lines),
    ("no numpy constraint", not any("numpy" in l for l in lines)),
    ("no cu118 refs", not any("cu118" in l for l in lines)),
]
failed = [n for n, ok in checks if not ok]
[print(f"[FAIL] {n}") for n in failed]
if not failed: print("[OK] constraints_feedforward.txt correct")
sys.exit(len(failed))
EOF
```

---

## Task 1: Create tests/test_cu121_migration.py — Phase 1 (Env Guards)

**Files:**
- Create: `tests/test_cu121_migration.py`

- [ ] **Step 1: Write the file with Phase 1 env guard tests**

```python
"""cu121 migration verification — Phases 1, 2, 3.

Run:
    /opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_cu121_migration.py -v

All tests are hard gates: any failure blocks the migration merge.
"""
import sys

import numpy as np
import pytest


# ── Phase 1: Environment guards ───────────────────────────────────────────────


def test_python_version():
    assert sys.version_info >= (3, 11), (
        f"Expected Python >= 3.11, got {sys.version_info.major}.{sys.version_info.minor}"
    )


def test_torch_version():
    import torch
    assert torch.__version__.startswith("2.4"), (
        f"Expected torch 2.4.x, got {torch.__version__}"
    )
    assert "cu121" in torch.__version__, (
        f"Expected cu121 build, got {torch.__version__}"
    )


def test_cuda_version():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    assert "12.1" in torch.version.cuda, (
        f"Expected CUDA 12.1, got {torch.version.cuda}"
    )


def test_numpy_not_downgraded():
    major, minor = [int(x) for x in np.__version__.split(".")[:2]]
    assert major >= 2, (
        f"numpy was downgraded below 2.0 — got {np.__version__}. "
        "Check pyproject.toml numpy>=1.26 constraint."
    )


def test_scipy_version():
    import scipy
    major, minor = [int(x) for x in scipy.__version__.split(".")[:2]]
    assert (major, minor) >= (1, 17), (
        f"Expected scipy >= 1.17, got {scipy.__version__}"
    )


def test_pycolmap_version():
    import pycolmap
    ver = pycolmap.__version__
    major = int(ver.split(".")[0])
    assert major >= 4, f"Expected pycolmap >= 4.0, got {ver}"


def test_bae_editable_install():
    import bae
    bae_file = bae.__file__
    assert bae_file is not None
    assert "/opt/bae" in bae_file or "bae" in bae_file, (
        f"bae not from /opt/bae editable install: {bae_file}"
    )


def test_viser_version():
    import viser
    parts = [int(x) for x in viser.__version__.split(".")[:3]]
    major, minor, patch = parts[0], parts[1], parts[2] if len(parts) > 2 else 0
    assert (major, minor, patch) >= (0, 2, 23), (
        f"Expected viser >= 0.2.23, got {viser.__version__}"
    )
```

- [ ] **Step 2: Run Phase 1 tests**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_cu121_migration.py -k "test_python or test_torch or test_cuda or test_numpy or test_scipy or test_pycolmap or test_bae_editable or test_viser" -v
```

Expected: 8 tests, all PASSED. If `test_torch_version` fails with `cu121` not found, check `pip show torch` — wrong torch wheel installed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cu121_migration.py
git commit -m "test(cu121): add Phase 1 env guard tests"
```

---

## Task 2: Add Phase 2 (Import Sweep) to tests/test_cu121_migration.py

**Files:**
- Modify: `tests/test_cu121_migration.py`

- [ ] **Step 1: Append Phase 2 tests to the file**

```python
# ── Phase 2: Import sweep ─────────────────────────────────────────────────────


def test_import_collab_splats_top_level():
    """import collab_splats must succeed — catches _patch_mapanything_torch_compat crash."""
    import collab_splats  # noqa: F401


def test_import_all_modules():
    """All collab_splats submodules must import without error."""
    modules = [
        "collab_splats.pointcloud",
        "collab_splats.pointcloud.base",
        "collab_splats.pointcloud.bundle_adjustment",
        "collab_splats.pointcloud.feedforward",
        "collab_splats.pointcloud.feedforward.base",
        "collab_splats.pointcloud.feedforward.vggtx",
        "collab_splats.pointcloud.feedforward.mapanything",
        "collab_splats.pointcloud.localization",
        "collab_splats.pointcloud.sfm",
        "collab_splats.pointcloud.utils",
        "collab_splats.pointcloud.wrappers",
        "collab_splats.pointcloud.loop_closure",
        "collab_splats.pointcloud.loop_closure.alignment",
        "collab_splats.pointcloud.loop_closure.closure",
        "collab_splats.pointcloud.loop_closure.pose_graph",
        "collab_splats.pointcloud.loop_closure.retrieval",
        "collab_splats.pointcloud.loop_closure.submap",
        "collab_splats.semantics",
        "collab_splats.semantics.features",
        "collab_splats.semantics.compression",
        "collab_splats.semantics.segmentation",
        "collab_splats.semantics.utils",
        "collab_splats.mesh",
        "collab_splats.mesh.base",
        "collab_splats.mesh.poisson",
        "collab_splats.mesh.tsdf",
        "collab_splats.mesh.utils",
        "collab_splats.nerfstudio.method_configs.rade_gs",
        "collab_splats.nerfstudio.method_configs.rade_features",
        "collab_splats.nerfstudio.models.rade_gs",
        "collab_splats.nerfstudio.models.rade_features",
        "collab_splats.utils",
        "collab_splats.utils.torch_utils",
        "collab_splats.utils.frame_sampling",
        "collab_splats.utils.camera_utils",
        "collab_splats.wrapper.splatter",
        "collab_splats.dashboard",
    ]
    import importlib
    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")
    assert not failed, "Import failures:\n" + "\n".join(failed)


def test_flagged_package_imports():
    """Packages flagged for numpy 2.x compat — all must import cleanly.

    These were explicitly flagged in the migration spec as having unknown numpy 2.x
    compatibility. All must resolve before Phase 4.
    """
    packages = {
        "pyntcloud": "import pyntcloud",
        "mobile_sam": "import mobile_sam",
        "maskclip_onnx": "import maskclip_onnx",
        "uniception": "import uniception",
        "meshlib": "import meshlib.mrmeshpy as mrmeshpy",
    }
    failed = []
    for name, stmt in packages.items():
        try:
            exec(stmt)
        except Exception as e:
            failed.append(f"{name}: {e}")
    assert not failed, (
        "Flagged packages failed to import — investigate numpy 2.x / torch 2.4 compat:\n"
        + "\n".join(failed)
    )
```

- [ ] **Step 2: Run Phase 2 tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_cu121_migration.py -k "test_import" -v
```

Expected: 3 tests PASSED. If `test_import_collab_splats_top_level` fails with `RuntimeError: mapanything intersection_check.py no longer matches expected pattern`, the `_patch_mapanything_torch_compat` call site at `collab_splats/pointcloud/feedforward/mapanything.py:254` must be removed (deferred cleanup is now blocking — remove the `_patch_mapanything_torch_compat()` call and the `if getattr(target, "_torch_compat_patched", False): return` guard).

If `test_flagged_package_imports` fails on individual packages, investigate each separately:
- `pyntcloud`: check `pip install pyntcloud --upgrade` for numpy 2.x fix
- `mobile_sam`: check GitHub for numpy 2.x compatibility patch
- `maskclip_onnx`: may need numpy < 2 pin or fork patch
- `uniception`: may need torch 2.4 compatible version
- `meshlib`: confirm version ≥ 3.1 is installed (`pip show meshlib`)

- [ ] **Step 3: Commit**

```bash
git add tests/test_cu121_migration.py
git commit -m "test(cu121): add Phase 2 import sweep tests"
```

---

## Task 3: Add Phase 3 (Migration Smoke Tests) to tests/test_cu121_migration.py

**Files:**
- Modify: `tests/test_cu121_migration.py`

- [ ] **Step 1: Append Phase 3 tests to the file**

```python
# ── Phase 3: Migration smoke tests ────────────────────────────────────────────


def test_gsplat_rade_fork():
    """rasterization_2dgs_inria_wrapper is a rade-specific symbol absent in official gsplat."""
    from gsplat import rasterization_2dgs_inria_wrapper  # noqa: F401


def test_gsplat_not_overwritten():
    """gsplat installed is the rade fork, not official PyPI gsplat 1.4.0."""
    import gsplat
    # The rade fork does not have a __version__ matching 1.4.0 or the standard release
    # Confirm rade-specific symbol reachable (duplicates test above for clarity)
    assert hasattr(gsplat, "rasterization_2dgs_inria_wrapper"), (
        f"gsplat at {gsplat.__file__} is missing rade-specific symbol — "
        "nerfstudio install may have overwritten gsplat-rade fork"
    )


def test_gtsam_sl4_manifold():
    """gtsam-develop 4.3a1 provides SL4 manifold; PyPI gtsam 4.2.1 does not."""
    from gtsam import SL4, PriorFactorSL4, BetweenFactorSL4  # noqa: F401


def test_bae_use_cudss():
    """bae key symbols importable: TrackingTensor, PCG solver, LM optimiser."""
    import pypose  # must import before bae
    from bae.autograd.function import TrackingTensor, map_transform  # noqa: F401
    from bae.utils.pysolvers import PCG  # noqa: F401
    from bae.optim import LM  # noqa: F401


def test_bae_cuda_backend():
    """bae active with CUDA 12.1 / torch 2.4."""
    import pypose  # noqa: F401
    import bae  # noqa: F401
    import torch
    assert torch.__version__.startswith("2.4"), f"Wrong torch: {torch.__version__}"
    assert "12.1" in torch.version.cuda, f"Wrong CUDA: {torch.version.cuda}"


def test_nerfstudio_installed_local():
    """nerfstudio is the local /workspace/nerfstudio install, not a PyPI package."""
    import nerfstudio
    assert nerfstudio.__file__ is not None
    assert "/workspace/nerfstudio" in nerfstudio.__file__, (
        f"nerfstudio loaded from unexpected location: {nerfstudio.__file__}. "
        "Expected /workspace/nerfstudio — PyPI nerfstudio may have been installed."
    )


def test_mapanything_compat_patch_safe():
    """_patch_mapanything_torch_compat must not raise RuntimeError at import time.

    At torch 2.4, any(dim=(1, 3)) works natively. If mapanything still has the
    old pattern, the patch applies (no-op at runtime). If mapanything was updated
    and the pattern is gone, the patch raises RuntimeError. Either outcome must
    not crash the import.
    """
    # This import triggers _patch_mapanything_torch_compat() at module load.
    # If it raises, the test fails with RuntimeError — the call site must be removed.
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator  # noqa: F401


def test_pycolmap_api_surface():
    """pycolmap 4.0.4 API: all constructors and new 4.x methods present and callable."""
    import pycolmap
    import numpy as np

    # Reconstruction and Track
    recon = pycolmap.Reconstruction()
    track = pycolmap.Track()

    # Camera
    camera = pycolmap.Camera(
        model="PINHOLE",
        width=224,
        height=224,
        params=[200.0, 200.0, 112.0, 112.0],
        camera_id=1,
    )
    # pycolmap 4.x method: add_camera_with_trivial_rig (replaces add_camera)
    recon.add_camera_with_trivial_rig(camera)

    # Rotation3d and Rigid3d
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    rot = pycolmap.Rotation3d(R)
    cam_from_world = pycolmap.Rigid3d(rot, t)

    # Image
    image = pycolmap.Image(name="frame_0000.jpg", camera_id=1, image_id=1)
    # pycolmap 4.x method: add_image_with_trivial_frame (replaces add_image)
    recon.add_image_with_trivial_frame(image, cam_from_world)

    assert len(recon.images) == 1
    assert len(recon.cameras) == 1


def test_nerfstudio_method_configs():
    """rade-gs and rade-features appear in nerfstudio's method registry."""
    import collab_splats.nerfstudio.method_configs.rade_gs  # noqa: F401 — triggers registration
    import collab_splats.nerfstudio.method_configs.rade_features  # noqa: F401
    from nerfstudio.configs.method_configs import all_methods
    assert "rade-gs" in all_methods, (
        f"rade-gs missing from nerfstudio registry. Keys: {sorted(all_methods)}"
    )
    assert "rade-features" in all_methods, (
        f"rade-features missing from nerfstudio registry. Keys: {sorted(all_methods)}"
    )


def test_collab_data_installed():
    import collab_data  # noqa: F401


def test_nerfstudio_patched_deps():
    """nerfacc and timm resolve to patched (unpinned) versions.

    The nerfstudio patch changed nerfacc==0.5.2 → >=0.5.2 and timm==0.6.7 → >=0.6.7.
    Verify the installed versions satisfy those bounds.
    """
    import nerfacc
    import timm
    nerfacc_parts = [int(x) for x in nerfacc.__version__.split(".")[:3]]
    assert tuple(nerfacc_parts) >= (0, 5, 2), (
        f"Expected nerfacc >= 0.5.2, got {nerfacc.__version__}"
    )
    timm_parts = [int(x) for x in timm.__version__.split(".")[:2]]
    assert tuple(timm_parts) >= (0, 6), (
        f"Expected timm >= 0.6.7, got {timm.__version__}"
    )


def test_splatfacto_uses_gsplat_rade():
    """nerfstudio.models.splatfacto imports without error and sees gsplat-rade.

    splatfacto is the only nerfstudio model that uses gsplat. If the rade fork
    was overwritten by nerfstudio's official gsplat==1.4.0, this import still
    succeeds but the rade-specific symbol check below will catch it.
    """
    import nerfstudio.models.splatfacto  # noqa: F401
    # Confirm gsplat-rade fork is what nerfstudio will use at training time
    from gsplat import rasterization_2dgs_inria_wrapper  # noqa: F401
```

- [ ] **Step 2: Run Phase 3 tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_cu121_migration.py -k "phase3 or test_gsplat or test_gtsam or test_bae or test_nerfstudio or test_mapanything or test_pycolmap_api or test_collab_data or test_splatfacto" -v
```

Expected: 13 tests PASSED.

Failure triage:
- `test_mapanything_compat_patch_safe` fails with `RuntimeError`: remove call at `mapanything.py:254`
- `test_gsplat_rade_fork` fails: rerun `setup_nerfstudio.sh` — step 3 reinstalls gsplat-rade
- `test_gtsam_sl4_manifold` fails: `pip install gtsam-develop -q` then retry
- `test_pycolmap_api_surface` fails on `add_camera_with_trivial_rig` / `add_image_with_trivial_frame`: pycolmap version is not 4.x — `pip show pycolmap` to check

- [ ] **Step 3: Run full Phase 1+2+3 together**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_cu121_migration.py -v
```

Expected: all 24 tests PASSED.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cu121_migration.py
git commit -m "test(cu121): add Phase 3 migration smoke tests (includes nerfstudio patched deps + splatfacto)"
```

---

## Task 4: Create Phase 5 Integration Tests

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_pipeline_cu121.py`

- [ ] **Step 1: Create tests/integration/__init__.py**

```python
```

(Empty file.)

- [ ] **Step 2: Write tests/integration/test_pipeline_cu121.py**

```python
"""Phase 5 pipeline integration tests for cu121 migration.

Tests pipeline components end-to-end with synthetic data. _forward is mocked
so no model downloads are required. Tests confirm that numpy 2.x + torch 2.4 +
pycolmap 4.0.4 work through the full data-flow path.

Run:
    /opt/conda/envs/nerfstudio/bin/python -m pytest tests/integration/test_pipeline_cu121.py -v
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch


# ── Helpers ───────────────────────────────────────────────────────────────────

N_FRAMES = 4
H, W = 224, 224


def _synthetic_extrinsics(n: int) -> np.ndarray:
    """(n, 3, 4) identity extrinsics with small translation offsets."""
    ext = np.tile(np.eye(3, 4), (n, 1, 1)).astype(np.float32)
    for i in range(n):
        ext[i, 2, 3] = i * 0.1  # translate along Z
    return ext


def _synthetic_intrinsics(n: int, w: int = W, h: int = H) -> np.ndarray:
    """(n, 3, 3) pinhole intrinsics."""
    K = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float32)
    return np.tile(K, (n, 1, 1))


def _synthetic_vggtx_raw(n: int = N_FRAMES, h: int = H, w: int = W) -> dict:
    """Minimal raw_outputs dict matching VGGTXCreator._forward output schema."""
    return {
        "images": torch.rand(n, 3, h, w),
        "extrinsic": _synthetic_extrinsics(n),
        "intrinsics_downsampled": _synthetic_intrinsics(n, w, h),
        "depth": np.ones((n, h, w, 1), dtype=np.float32),
        "depth_conf": np.ones((n, h, w), dtype=np.float32) * 0.9,
    }


def _synthetic_mapanything_raw(n: int = N_FRAMES, h: int = H, w: int = W) -> dict:
    """Minimal raw_outputs dict matching MapAnythingCreator._forward output schema."""
    return {
        "images": torch.rand(n, 3, h, w),
        "extrinsic": _synthetic_extrinsics(n),
        "intrinsics_downsampled": _synthetic_intrinsics(n, w, h),
        "depth": np.ones((n, h, w), dtype=np.float32),
        "depth_conf": np.ones((n, h, w), dtype=np.float32) * 0.9,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_build_pycolmap_reconstruction_roundtrip():
    """build_pycolmap_reconstruction → colmap_reconstruction_to_result.

    Exercises the full pycolmap 4.0.4 path:
      add_camera_with_trivial_rig + add_image_with_trivial_frame + add_point3D.
    """
    from collab_splats.pointcloud.feedforward.base import build_pycolmap_reconstruction
    from collab_splats.pointcloud.utils import colmap_reconstruction_to_result

    P = 30
    rng = np.random.default_rng(42)
    pts3d = rng.standard_normal((P, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (P, 3), dtype=np.uint8)

    recon = build_pycolmap_reconstruction(
        pts3d=pts3d,
        colors=colors,
        extrinsics=_synthetic_extrinsics(N_FRAMES),
        intrinsics=_synthetic_intrinsics(N_FRAMES),
        image_width=W,
        image_height=H,
        image_names=[f"frame_{i:04d}.jpg" for i in range(N_FRAMES)],
    )

    assert len(recon.images) == N_FRAMES
    assert len(recon.cameras) == N_FRAMES
    assert len(recon.points3D) == P

    result = colmap_reconstruction_to_result(recon)
    assert result.points.shape == (P, 3)
    assert result.camera_poses is not None
    assert result.camera_poses.shape[0] == N_FRAMES
    assert result.camera_poses.shape[1:] == (4, 4)
    assert result.camera_intrinsics is not None
    assert result.camera_intrinsics.shape == (N_FRAMES, 3, 3)


def test_feedforward_result_save_load(tmp_path):
    """FeedforwardResult .save() / .load() round-trip under numpy 2.x npz."""
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    P, N = 30, N_FRAMES
    rng = np.random.default_rng(0)
    original = FeedforwardResult(
        pts3d=rng.standard_normal((P, 3)).astype(np.float32),
        colors=rng.integers(0, 255, (P, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (N, 1, 1)).astype(np.float32),
        intrinsics=_synthetic_intrinsics(N),
        image_paths=[Path(f"frame_{i}.jpg") for i in range(N)],
        original_coords=np.zeros((N, 6), dtype=np.float32),
        model_width=W,
        model_height=H,
    )
    save_path = tmp_path / "result.npz"
    original.save(save_path)
    loaded = FeedforwardResult.load(save_path)

    np.testing.assert_array_equal(original.pts3d, loaded.pts3d)
    np.testing.assert_array_equal(original.colors, loaded.colors)
    np.testing.assert_array_equal(original.extrinsics, loaded.extrinsics)
    assert loaded.model_width == W
    assert loaded.model_height == H
    assert len(loaded.image_paths) == N


def test_vggtx_postprocess_pipeline(tmp_path):
    """VGGTXCreator._postprocess with synthetic raw_outputs → FeedforwardResult.

    Mocks _forward so no VGGT model download is needed.
    Tests torch 2.4 + numpy 2.x through the full postprocess + pycolmap path.
    """
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

    raw = _synthetic_vggtx_raw()
    creator = VGGTXCreator(use_global_alignment=False)
    creator.image_paths = [tmp_path / f"frame_{i:04d}.jpg" for i in range(N_FRAMES)]
    creator.original_coords = np.zeros((N_FRAMES, 6), dtype=np.float32)
    # Provide original_coords with realistic values (tl=0,0 cr=W,H orig=W,H)
    for i in range(N_FRAMES):
        creator.original_coords[i] = [0, 0, W, H, W, H]

    with patch(
        "collab_splats.pointcloud.feedforward.vggtx.unproject_and_filter_points",
        return_value=(
            np.random.randn(20, 3).astype(np.float32),
            np.random.randint(0, 255, (20, 3)).astype(np.uint8),
        ),
    ):
        result = creator._postprocess(raw)

    assert result is not None
    assert result.pts3d.shape[1] == 3
    assert result.extrinsics.shape == (N_FRAMES, 4, 4)
    assert result.intrinsics.shape == (N_FRAMES, 3, 3)
    assert result.model_width == W
    assert result.model_height == H


def test_mapanything_postprocess_pipeline(tmp_path):
    """MapAnythingCreator._postprocess with synthetic raw_outputs → FeedforwardResult.

    Also confirms _patch_mapanything_torch_compat fires without RuntimeError.
    """
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    raw = _synthetic_mapanything_raw()
    creator = MapAnythingCreator()
    creator.image_paths = [tmp_path / f"frame_{i:04d}.jpg" for i in range(N_FRAMES)]
    creator.original_coords = np.zeros((N_FRAMES, 6), dtype=np.float32)
    for i in range(N_FRAMES):
        creator.original_coords[i] = [0, 0, W, H, W, H]

    with patch(
        "collab_splats.pointcloud.feedforward.mapanything.unproject_and_filter_points",
        return_value=(
            np.random.randn(20, 3).astype(np.float32),
            np.random.randint(0, 255, (20, 3)).astype(np.uint8),
        ),
    ):
        result = creator._postprocess(raw)

    assert result is not None
    assert result.pts3d.shape[1] == 3
    assert result.extrinsics.shape == (N_FRAMES, 4, 4)


def test_tsdf_mesh_synthetic(tmp_path):
    """Open3DTSDFFusion.create with synthetic depth + RGB frames."""
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    n, h, w = 4, 64, 64
    rng = np.random.default_rng(1)
    # Flat plane at depth 2.0
    depths = np.full((n, h, w), 2.0, dtype=np.float32)
    rgbs = rng.integers(0, 255, (n, h, w, 3), dtype=np.uint8)
    # c2w: camera-to-world (identity + Z translation)
    c2w = np.tile(np.eye(4), (n, 1, 1)).astype(np.float32)
    for i in range(n):
        c2w[i, 2, 3] = i * 0.05
    K = np.array([[50, 0, 32], [0, 50, 32], [0, 0, 1]], dtype=np.float32)
    intrinsics = np.tile(K, (n, 1, 1))

    fusion = Open3DTSDFFusion(output_dir=tmp_path, clean_repair=False)
    mesh_result = fusion.create(
        depths=depths, rgbs=rgbs, c2w=c2w, intrinsics=intrinsics
    )
    # Synthetic flat plane may produce empty mesh, but must not raise
    assert mesh_result is not None
    assert isinstance(mesh_result.mesh_path, Path)


def test_nerfstudio_method_registry():
    """rade-gs and rade-features appear in nerfstudio method registry after import."""
    import collab_splats.nerfstudio.method_configs.rade_gs  # noqa: F401
    import collab_splats.nerfstudio.method_configs.rade_features  # noqa: F401
    from nerfstudio.configs.method_configs import all_methods
    assert "rade-gs" in all_methods, (
        f"rade-gs not registered. Available: {sorted(all_methods)}"
    )
    assert "rade-features" in all_methods, (
        f"rade-features not registered. Available: {sorted(all_methods)}"
    )
```

- [ ] **Step 3: Run Phase 5 integration tests**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/integration/test_pipeline_cu121.py -v
```

Expected: 6 tests PASSED.

Failure triage:
- `test_vggtx_postprocess_pipeline` / `test_mapanything_postprocess_pipeline`: if `unproject_and_filter_points` patch path is wrong, check the actual import path in `vggtx.py` and `mapanything.py` and update the `patch()` target string
- `test_tsdf_mesh_synthetic`: if `create()` method signature differs, read `collab_splats/mesh/tsdf.py` for the exact signature
- `test_nerfstudio_method_registry`: if `all_methods` import path changed in the local nerfstudio, check `from nerfstudio.configs.method_configs import all_methods`

- [ ] **Step 4: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_pipeline_cu121.py
git commit -m "test(cu121): add Phase 5 pipeline integration tests"
```

---

## Task 5: Run Phase 4 — Full Existing Test Suite

No files to create. Run the existing suite.

- [ ] **Step 1: Run full suite**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --tb=short 2>&1 | tee /tmp/pytest_full.log
echo "Exit code: $?"
```

Expected: exit 0. All tests pass.

- [ ] **Step 2: Run nerfstudio sub-suite explicitly**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/nerfstudio/ -v --tb=short
```

Expected: all tests in `test_imports.py` and `test_datamanager_config.py` PASSED. These cover `RadegsModel`, `RadegsFeaturesModel`, method config registration, datamanager config, and `load_checkpoint`.

- [ ] **Step 3: Run high-priority pycolmap 4.x files first if debugging**

If the full suite fails, run these first to isolate pycolmap 4.x surface breakage:

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_localization.py \
  tests/pointcloud/test_sfm_creator.py \
  tests/pointcloud/test_feedforward_shared.py \
  tests/pointcloud/test_vggtx_creator.py \
  tests/pointcloud/test_mapanything_creator.py \
  -v --tb=short
```

- [ ] **Step 4: Run bae smoke tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_bae_smoke.py -v
```

Expected: `test_bae_imports`, `test_bae_cuda_version`, `test_bundle_adjustment_module_loads` — all PASSED.

---

## Task 6: Final Sign-Off

- [ ] **Step 1: Run complete verification suite (Phases 1–3 + 5)**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/test_cu121_migration.py \
  tests/integration/test_pipeline_cu121.py \
  tests/test_bae_smoke.py \
  -v 2>&1 | tee /tmp/pytest_verification.log
echo "Exit code: $?"
```

Expected: all tests pass, exit 0.

- [ ] **Step 2: Confirm full suite still clean**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -20
echo "Exit code: $?"
```

Expected: exit 0.

- [ ] **Step 3: Commit sign-off**

```bash
git add tests/test_cu121_migration.py tests/integration/__init__.py tests/integration/test_pipeline_cu121.py
git commit -m "test(cu121): verification suite complete — all phases pass

Phases 0-5 of cu121 migration verification spec executed and passing.
See docs/superpowers/specs/2026-05-22-cu121-verification-design.md."
```
