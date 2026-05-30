# cu121 Migration Verification Design

**Date:** 2026-05-22  
**Branch:** `refactor/cu121`  
**Status:** Ready for implementation  
**Follows:** `2026-05-22-cu121-migration-design.md`

## Goal

End-to-end verification that all packages and modules work correctly after the cu121 migration (CUDA 11.8/torch 2.1.2/Python 3.10 → CUDA 12.1/torch 2.4.0/Python 3.11). Hard gate: all phases must pass before the migration branch is merged.

All commands run inside `conda activate nerfstudio` unless noted.

---

## Known Risks Surfaced During Design

| Risk | Where | Severity |
|---|---|---|
| `_patch_mapanything_torch_compat()` still called at module load; raises `RuntimeError` if mapanything pattern changed | `mapanything.py:254` | **HIGH** — crashes `import collab_splats` |
| `pycolmap.estimate_and_refine_absolute_pose` API changed in 4.x | `localization.py:655` | Medium — test_localization.py covers this |
| gsplat-rade overwritten by nerfstudio install | install order | Medium — setup_nerfstudio.sh reinstalls last |
| viser version resolves below 0.2.23 | env resolution | Low — collab_splats doesn't use viser directly |

**Pre-condition before running any verification phase:** the `_patch_mapanything_torch_compat` call site at `mapanything.py:254` must be audited. If mapanything==1.1 still contains `batch_intersect.any(dim=(1, 3))`, the patch applies but is now a no-op (torch 2.4 supports it natively). If mapanything was updated past 1.1, the patch raises `RuntimeError`. Verify which case applies before proceeding to Phase 2. If it raises, remove the call at line 254 (the deferred cleanup becomes blocking).

---

## Phase 0 — Setup Script Execution

All three scripts must complete successfully. Break each into steps so failures are pinpointed. Run in order: nerfstudio first, then core, then feedforward.

### 0.1 — Verify conda env

```bash
conda activate nerfstudio
python --version          # must be 3.11.x
which python              # must be /opt/conda/envs/nerfstudio/bin/python
which pip                 # must be /opt/conda/envs/nerfstudio/bin/pip
```

Expected: Python 3.11.x from the nerfstudio conda env. If `python --version` shows 3.13, wrong env — check `CONDA_DEFAULT_ENV`.

### 0.2 — setup_nerfstudio.sh: patch nerfstudio/pyproject.toml

```bash
cd /workspace/collab-splats
bash setup_nerfstudio.sh 2>&1 | tee /tmp/setup_nerfstudio.log
echo "Exit code: $?"
```

Expected output contains all 5 patch confirmations:
```
patched: "gsplat==1.4.0" → "gsplat @ git+https://github.com/brian-xu/gsplat-rade.git"
patched: "timm==0.6.7" → "timm>=0.6.7"
patched: "viser==1.0.0" → "viser>=0.2.0"
patched: '"opencv-python-headless==4.10.0.84",' → ''
patched: "nerfacc==0.5.2" → "nerfacc>=0.5.2"
```

If any patch shows `WARNING: pattern not found` and the script was not already run, the nerfstudio pyproject may have changed upstream — inspect manually.

### 0.3 — setup_nerfstudio.sh: verify nerfstudio patches in file

```bash
python - <<'EOF'
import pathlib, sys
p = pathlib.Path("/workspace/nerfstudio/pyproject.toml")
text = p.read_text()

checks = [
    ("gsplat-rade fork installed", "gsplat @ git+https://github.com/brian-xu/gsplat-rade.git" in text),
    ("official gsplat==1.4.0 absent", "gsplat==1.4.0" not in text),
    ("timm unpinned", "timm>=0.6.7" in text),
    ("viser unpinned", "viser>=0.2.0" in text),
    ("opencv-python-headless absent", "opencv-python-headless" not in text),
    ("nerfacc unpinned", "nerfacc>=0.5.2" in text),
]
failed = [(name, ok) for name, ok in checks if not ok]
for name, _ in failed:
    print(f"FAIL: {name}")
if not failed:
    print("[OK] all 5 nerfstudio patches verified in file")
sys.exit(len(failed))
EOF
```

### 0.4 — setup_nerfstudio.sh: nerfstudio + gsplat-rade installed

```bash
python - <<'EOF'
import sys
import nerfstudio
import gsplat
import torch

gsplat_path = gsplat.__file__
print(f"[OK] nerfstudio installed")
print(f"[OK] gsplat path: {gsplat_path}")
print(f"[OK] torch={torch.__version__}  cuda={torch.version.cuda}")

try:
    from gsplat import rasterization_2dgs_inria_wrapper
    print("[OK] gsplat-rade fork confirmed (rasterization_2dgs_inria_wrapper present)")
except ImportError:
    print("[FAIL] gsplat-rade specific symbol absent — official gsplat may have overwritten fork")
    sys.exit(1)
EOF
```

### 0.5 — setup.sh step 1: dry-run pip install -e .

Before installing, confirm pyproject.toml parses and pip resolves without conflicts:

```bash
# Verify pyproject.toml parses
python - <<'EOF'
import tomllib, sys
with open("pyproject.toml", "rb") as f:
    t = tomllib.load(f)
deps = t["project"]["dependencies"]
present = {d.split("@")[0].split(">=")[0].split("==")[0].split("[")[0].strip() for d in deps}

required = {"numpy", "scipy", "pypose", "bae", "clip", "python-dotenv", "pycolmap", "meshlib"}
dropped  = {"nerfstudio", "fairscale", "dotenv", "recognize-anything", "feature-splatting"}

for pkg in required:
    ok = any(pkg in d for d in deps)
    print(f"{'[OK]' if ok else '[FAIL]'} present: {pkg}")
for pkg in dropped:
    ok = not any(pkg in d for d in deps)
    print(f"{'[OK]' if ok else '[FAIL]'} absent: {pkg}")

ff = t["project"]["optional-dependencies"]["feedforward"]
gtsam_ok = any("gtsam-develop" in d for d in ff)
gtsam_old = any(d.strip() == '"gtsam"' or d.strip() == "gtsam" for d in ff)
print(f"{'[OK]' if gtsam_ok else '[FAIL]'} feedforward: gtsam-develop present")
print(f"{'[OK]' if not gtsam_old else '[FAIL]'} feedforward: old gtsam absent")
EOF

# Dry-run resolution
pip install -e . --dry-run 2>&1 | grep -E "ERROR|InvalidRequirement|ParseError|conflict" || echo "[OK] dry-run: no resolution errors"
```

### 0.6 — setup.sh step 2: pip install -e .

```bash
pip install -e . 2>&1 | tee /tmp/setup_core.log
echo "Exit code: $?"
```

Expected: exits 0. If `InvalidRequirement` appears, a dep spec in pyproject.toml has a syntax error — check the backtick fix on pycolmap.

### 0.7 — setup.sh step 3: collab-data

```bash
pip install git+https://github.com/BasisResearch/collab-data.git 2>&1 | tail -5
echo "Exit code: $?"
python -c "import collab_data; print('[OK] collab_data installed')"
```

### 0.8 — setup.sh step 4: co3d (--no-deps required)

```bash
pip install git+https://github.com/facebookresearch/co3d.git --no-deps 2>&1 | tail -5
echo "Exit code: $?"
python -c "import co3d; print('[OK] co3d installed')"
```

### 0.9 — setup/feedforward.sh: full execution

```bash
bash setup/feedforward.sh 2>&1 | tee /tmp/setup_feedforward.log
echo "Exit code: $?"
```

Expected: exits 0. The script has internal invariant checks — look for its `[OK]` lines:
```
[OK] torch=2.4.0+cu121  cuda=12.1  timm=<version>
```

If `FAIL: CUDA changed` or `FAIL: torch upgraded` appears, the wrong torch is active.

### 0.10 — setup/feedforward.sh: gtsam-develop SL4 symbols

```bash
python - <<'EOF'
from gtsam import SL4, PriorFactorSL4, BetweenFactorSL4
print("[OK] gtsam SL4 manifold available")
import gtsam
print(f"[OK] gtsam version: {gtsam.__version__ if hasattr(gtsam, '__version__') else 'installed'}")
EOF
```

### 0.11 — Deleted files absent

```bash
python - <<'EOF'
import pathlib, sys
deleted = [
    "setup_bundle_adjustment.sh",
    "patches/bae-torch21-compat.patch",
]
failed = []
for f in deleted:
    if pathlib.Path(f).exists():
        print(f"[FAIL] still exists: {f}")
        failed.append(f)
    else:
        print(f"[OK] deleted: {f}")
sys.exit(len(failed))
EOF
```

### 0.12 — constraints_feedforward.txt correct

```bash
python - <<'EOF'
import pathlib, sys
text = pathlib.Path("constraints_feedforward.txt").read_text()
lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]
checks = [
    ("torch cu121 pin", "torch==2.4.0+cu121" in lines),
    ("torchvision cu121 pin", "torchvision==0.19.0+cu121" in lines),
    ("no numpy constraint", not any("numpy" in l for l in lines)),
    ("no cu118 reference", not any("cu118" in l for l in lines)),
]
failed = [(n, ok) for n, ok in checks if not ok]
for n, _ in failed:
    print(f"[FAIL] {n}")
if not failed:
    print("[OK] constraints_feedforward.txt correct")
sys.exit(len(failed))
EOF
```

---

## Phase 1 — Environment Guards

Run after all setup scripts pass.

```bash
python -m pytest tests/test_cu121_migration.py::test_python_version \
                 tests/test_cu121_migration.py::test_torch_version \
                 tests/test_cu121_migration.py::test_cuda_version \
                 tests/test_cu121_migration.py::test_numpy_not_downgraded \
                 tests/test_cu121_migration.py::test_scipy_version \
                 tests/test_cu121_migration.py::test_pycolmap_version \
                 tests/test_cu121_migration.py::test_bae_editable_install \
                 -v
```

New test file `tests/test_cu121_migration.py` covers:

| Test | Assert |
|---|---|
| `test_python_version` | `sys.version_info >= (3, 11)` |
| `test_torch_version` | `torch.__version__.startswith("2.4")` |
| `test_cuda_version` | `"12.1" in torch.version.cuda` |
| `test_numpy_not_downgraded` | `np.__version__ >= "2.0"` |
| `test_scipy_version` | `scipy.__version__ >= "1.17"` |
| `test_pycolmap_version` | `pycolmap.__version__ >= "4.0"` |
| `test_bae_editable_install` | bae installed from `/opt/bae`; `USE_CUDSS=1` flag active |
| `test_viser_version` | resolved viser `>= "0.2.23"` |

---

## Phase 2 — Import Sweep

Two groups: project modules (must all pass), flagged packages (failures need investigation).

### 2.1 — All collab_splats modules import clean

```bash
python -m pytest tests/test_cu121_migration.py::test_import_collab_splats_top_level \
                 tests/test_cu121_migration.py::test_import_all_modules \
                 -v
```

`test_import_collab_splats_top_level` — `import collab_splats` must succeed. **This catches the `_patch_mapanything_torch_compat` crash if it fires.**

`test_import_all_modules` — imports each submodule:
- `collab_splats.pointcloud`, `.pointcloud.feedforward`, `.pointcloud.bundle_adjustment`
- `collab_splats.pointcloud.loop_closure`, `.pointcloud.localization`, `.pointcloud.sfm`
- `collab_splats.semantics`, `.semantics.features`, `.semantics.compression`, `.semantics.segmentation`
- `collab_splats.mesh`, `.mesh.poisson`, `.mesh.tsdf`
- `collab_splats.nerfstudio.method_configs.rade_gs`, `.rade_features`
- `collab_splats.utils`, `.wrapper.splatter`, `.dashboard`

### 2.2 — Flagged numpy 2.x packages

These were explicitly flagged in the migration spec as having unknown numpy 2.x compatibility. Run as a separate suite — failures investigated as individual follow-up tickets:

```bash
python -m pytest tests/test_cu121_migration.py::test_flagged_package_imports -v -s
```

| Package | Import to attempt | Risk |
|---|---|---|
| `pyntcloud` | `import pyntcloud` | Uses removed `np.bool`/`np.int` aliases |
| `mobile_sam` | `import mobile_sam` | numpy 2.x compat unknown |
| `maskclip_onnx` | `import maskclip_onnx` | numpy 2.x compat unknown |
| `uniception` | `import uniception` | torch 2.4 compat unknown |
| `meshlib` | `import meshlib.mrmeshpy` | 3.0.6→3.1 upgrade |

Each failure: open a tracked ticket, investigate the specific package's numpy 2.x compat (check their changelog / GitHub issues), and either pin a fixed version or patch. All flagged packages must import cleanly before proceeding to Phase 4.

---

## Phase 3 — Migration Smoke Tests

Targeted checks for the specific things the cu121 migration changed.

```bash
python -m pytest tests/test_cu121_migration.py -k "phase3" -v
```

| Test | What it verifies |
|---|---|
| `test_gsplat_rade_fork` | `rasterization_2dgs_inria_wrapper` importable (rade-specific symbol, absent in official gsplat) |
| `test_gsplat_not_overwritten` | gsplat `__file__` path does not reference official pip-installed gsplat (confirms nerfstudio install didn't overwrite) |
| `test_gtsam_sl4_manifold` | `from gtsam import SL4, PriorFactorSL4, BetweenFactorSL4` — confirms gtsam-develop, not gtsam 4.2.1 |
| `test_bae_use_cudss` | `from bae.autograd.function import TrackingTensor, map_transform` + `from bae.utils.pysolvers import PCG` + `from bae.optim import LM` |
| `test_bae_cuda_backend` | `torch.version.cuda == "12.1"` at bae import time; USE_CUDSS=1 path active |
| `test_nerfstudio_installed_local` | `nerfstudio.__file__` is under `/workspace/nerfstudio/` (not a pip-installed PyPI nerfstudio) |
| `test_mapanything_compat_patch_safe` | `import collab_splats.pointcloud.feedforward.mapanything` does not raise; explicitly confirms `_patch_mapanything_torch_compat` didn't raise `RuntimeError` |
| `test_pycolmap_api_surface` | `pycolmap.Reconstruction()`, `pycolmap.Camera(...)`, `pycolmap.Rigid3d(...)`, `pycolmap.Rotation3d(...)`, `pycolmap.Image(...)`, `pycolmap.Track()` all constructable |
| `test_nerfstudio_method_configs` | `rade_gs` and `rade_features` visible in nerfstudio method config registry |
| `test_nerfstudio_patched_deps` | nerfacc `>= 0.5.2`, timm `>= 0.6.7` (both were exact-pinned; verify unpinned versions resolved correctly) |
| `test_splatfacto_uses_gsplat_rade` | `nerfstudio.models.splatfacto` imports without error; gsplat-rade fork symbol reachable from nerfstudio's gsplat |
| `test_collab_data_installed` | `import collab_data` |

---

## Phase 4 — Full Test Suite

Run after Phases 0–3 all pass.

```bash
cd /workspace/collab-splats
python -m pytest tests/ -v --tb=short 2>&1 | tee /tmp/pytest_full.log
echo "Exit: $?"
```

Hard requirement: exit 0. Any failure blocks merge.

### Nerfstudio sub-suite (run explicitly)

```bash
python -m pytest tests/nerfstudio/ -v --tb=short
```

Covers `RadegsModel`, `RadegsFeaturesModel`, method config registration, datamanager config, `load_checkpoint`. These directly test the nerfstudio integration points changed by the migration.

### High-priority test files (pycolmap 4.x surface)

These test modules that use the changed pycolmap API — run first if iterating on failures:

```bash
python -m pytest tests/pointcloud/test_localization.py \
                 tests/pointcloud/test_sfm_creator.py \
                 tests/pointcloud/test_feedforward_shared.py \
                 tests/pointcloud/test_vggtx_creator.py \
                 tests/pointcloud/test_mapanything_creator.py \
                 -v --tb=short
```

If these 5 pass, pycolmap 4.0.4 compatibility is confirmed.

### bae smoke tests (already in repo)

```bash
python -m pytest tests/test_bae_smoke.py -v
```

Expected: `test_bae_imports`, `test_bae_cuda_version`, `test_bundle_adjustment_module_loads` all PASSED.

---

## Phase 5 — Pipeline Integration

Minimal end-to-end exercise of the full stack. Not a full reconstruction — a forward-pass smoke test that confirms all components wire together under the new env.

New test file: `tests/integration/test_pipeline_cu121.py`

### 5.1 — VGGT-X forward pass (synthetic input)

```
4 synthetic RGB frames (224×224, random pixel values)
→ VGGTXCreator.create()
→ assert PointcloudResult: points not empty, cameras populated, confidence tensor present
```

### 5.2 — MapAnything forward pass (synthetic input)

```
Same 4 frames
→ MapAnythingCreator.create()
→ assert FeedforwardResult: points not empty, cameras populated
→ assert _patch_mapanything_torch_compat did NOT raise during this run
```

### 5.3 — Bundle adjustment round-trip

```
PointcloudResult with known synthetic poses
→ BundleAdjustmentWrapper.create()
→ assert poses refined, result.raw_outputs["ba_converged"] == True
```

### 5.4 — Semantic feature extraction

```
Single synthetic frame tensor (1, 3, 224, 224)
→ DINOv2FeatureExtractor.extract()
→ assert feature tensor shape correct, no CUDA OOM
```

### 5.5 — Mesh creation round-trip

```
Synthetic PointcloudResult with depth maps
→ TSDFMeshCreator.create()
→ assert mesh has vertices and faces
```

### 5.6 — Nerfstudio method config registration

```python
from collab_splats.nerfstudio.method_configs import rade_gs, rade_features
# confirm both appear in nerfstudio's method registry (not just import — verify registration side-effect)
from nerfstudio.configs.method_configs import all_methods
assert "rade-gs" in all_methods
assert "rade-features" in all_methods
```

---

## Pass Criteria Summary

| Phase | Command | Must exit |
|---|---|---|
| 0.1 | `python --version` | shows 3.11.x |
| 0.2–0.4 | `bash setup_nerfstudio.sh` | 0, all 5 patches applied |
| 0.5 | pyproject.toml checks + pip dry-run | 0, no errors |
| 0.6 | `pip install -e .` | 0 |
| 0.7 | collab-data install | 0 |
| 0.8 | co3d install --no-deps | 0 |
| 0.9–0.10 | `bash setup/feedforward.sh` | 0, invariants pass |
| 0.11 | deleted files check | 0 |
| 0.12 | constraints_feedforward.txt check | 0 |
| 1 | env guard tests | 0 (8 tests pass) |
| 2.1 | import sweep | 0 |
| 2.2 | flagged packages | 0 (all must resolve before Phase 4) |
| 3 | migration smoke tests | 0 (10 tests pass) |
| 4 | `pytest tests/` | 0 (full suite) |
| 5 | `pytest tests/integration/` | 0 (6 integration tests) |

---

## Deferred (not blocking this verification)

| Item | Status |
|---|---|
| hloc install | Deferred — pycolmap 4.0.4 vs hloc compat unresolved. pycolmap IS installed; confirm `import pycolmap` works. hloc NOT expected to be installed. |
| `_patch_mapanything_torch_compat` source removal | Deferred to cleanup PR — but call site must NOT raise (Phase 3 test_mapanything_compat_patch_safe confirms this) |
| `"feature-splatting"` string refs in `dashboard/config_panel.py`, `wrapper/splatter.py` | String method name refs, not imports — no functional impact |
