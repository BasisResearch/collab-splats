# cu121 Migration Design
**Date:** 2026-05-22  
**Branch:** `refactor/cu121`  
**Status:** Approved, ready for implementation

## Context

Porting from CUDA 11.8 / torch 2.1.2 / Python 3.10 to CUDA 12.1 / torch 2.4.0 / Python 3.11 via `Dockerfile.cu121`. The new Dockerfile pre-installs bae (editable, USE_CUDSS=1) and gsplat-rade, but does NOT install nerfstudio or hloc. Setup scripts and pyproject.toml contain stale cu118 references, wrong package names, and conflicting version pins that will break `pip install -e .`.

### Environment delta

| Component | Old (cu118) | New (cu121) |
|---|---|---|
| Python | 3.10 | 3.11 |
| torch | 2.1.2+cu118 | 2.4.0+cu121 |
| torchvision | 0.16.2+cu118 | 0.19.0+cu121 |
| numpy | <2.0 | 2.4.6 installed |
| scipy | 1.11.4 | 1.17.1 installed |
| nerfstudio | installed (Dockerfile) | NOT installed |
| hloc | v1.4 + pycolmap 0.4.0 | NOT installed (deferred) |
| pycolmap | 0.4.0 | 4.0.4 |
| bae | USE_CUDSS=0, compat patches | 0.2.3 at /opt/bae, USE_CUDSS=1 |
| gtsam | 4.2.1 (no SL4) | needs gtsam-develop (has SL4) |

---

## Approach

Targeted in-place fixes only. No new scripts beyond `setup_nerfstudio.sh`. No parallel cu121-suffixed files.

---

## Ordered Implementation Plan

Each phase depends on the previous. Complete in order.

### Phase 0 — Backup untracked file

`/workspace/nerfstudio/pyproject.toml` is in a separate git repo and will not be tracked by collab-splats git. Before touching it:

1. Copy original to `patches/nerfstudio-pyproject.orig.toml` (tracked in collab-splats)
2. Create `patches/nerfstudio-cu121-compat.patch` as a unified diff that documents exactly what changes

All other files (`setup.sh`, `setup/feedforward.sh`, `setup_bundle_adjustment.sh`, `constraints_feedforward.txt`, `pyproject.toml`) are git-tracked — git history is the backup.

### Phase 1 — `collab-splats/pyproject.toml`

Source of truth for all deps. Fix first so subsequent scripts and `pip install -e .` work correctly.

**Bug fixes:**
- `"pycolmap>=3.1\`"` → `"pycolmap>=3.1"` (stray backtick breaks PEP 508 parsing; `pip install -e .` fails today)
- `"dotenv"` → `"python-dotenv"` (wrong PyPI package name)

**numpy/scipy unpinning** (numpy 2.4.6 already installed; keeping `<2.0` causes active downgrade):
- `"numpy<2.0.0"` → `"numpy>=1.26"`
- `"scipy==1.11.4"` → `"scipy>=1.11"` (1.17.1 installed; old pin was paired with old numpy)

**Version update:**
- `"meshlib==3.0.6.229"` → `"meshlib>=3.1"` (old pin, numpy 2.x support uncertain for 3.0.6; latest is 3.1.2.192)

**Remove unused dep:**
- Remove `"nerfstudio"` — must be installed from local `/workspace/nerfstudio/` via `setup_nerfstudio.sh`; declaring here pulls PyPI and installs wrong version

**Add missing deps** (currently in `requirements.txt` or `setup.sh` only):
```toml
"pypose",
# bae requires --no-build-isolation at build time (setup.py imports torch).
# Pre-installed by Dockerfile.cu121 Stage 2. Fresh install: see Dockerfile.cu121.
"bae @ git+https://github.com/pypose/bae.git",
"clip @ git+https://github.com/openai/CLIP.git",
```

**Remove entirely** (no imports in `collab_splats/`, functionality superseded or unused):
- `recognize-anything` — zero imports in collab_splats; dropped.
- `fairscale` — only required by recognize-anything; zero direct imports; dropped with it.
- `feature-splatting` — zero `import feature_splatting` anywhere; string refs (`"feature-splatting"` as a method name) remain in code but reference the nerfstudio method name, not the Python package. Functionality replaced by `rade-features`. Remove pip dep.

**Also: clean up `requirements.txt`** — remove `fairscale`, `recognize-anything`, `feature-splatting` entries. `requirements.txt` should only keep packages that cannot be expressed in pyproject.

**Feedforward extras:**
- `"gtsam"` → `"gtsam-develop"` (PyPI gtsam 4.2.1 lacks `SL4`, `PriorFactorSL4`, `BetweenFactorSL4` required by VGGT-SLAM 2.0; gtsam-develop 4.3a1 has them and is numpy 2.x compatible)

**Packages flagged — validate post-install (separate task):**
- `pyntcloud` — may use removed `np.bool`/`np.int` aliases
- `mobile_sam @ git` — numpy 2.x compat unknown
- `maskclip_onnx @ git` — numpy 2.x compat unknown
- `uniception==0.1.7` — torch 2.4 compat unknown

### Phase 2 — `patches/nerfstudio-cu121-compat.patch`

Create as a unified diff tracked in collab-splats. `setup_nerfstudio.sh` applies this via `git apply`.

Five changes to `/workspace/nerfstudio/pyproject.toml`:

| Current | Change | Reason |
|---|---|---|
| `gsplat==1.4.0` | `gsplat @ git+https://github.com/brian-xu/gsplat-rade.git` | exact pin overwrites gsplat-rade fork |
| `timm==0.6.7` | `timm>=0.6.7` | blocks uniception which needs `>=0.9` |
| `viser==1.0.0` | `viser>=0.2.0` | conflicts with feedforward `viser>=0.2.23` |
| `opencv-python-headless==4.10.0.84` | **remove entirely** | `opencv-python` (in collab-splats) is a superset; cv2 imports work; having both in the same env causes pip conflicts |
| `nerfacc==0.5.2` | `nerfacc>=0.5.2` | CUDA extension; exact pin may fail with torch 2.4 |

### Phase 3 — `setup_nerfstudio.sh` (new file)

Must exist before `setup.sh` calls it.

```
1. Apply patches/nerfstudio-cu121-compat.patch to /workspace/nerfstudio/
2. pip install -e /workspace/nerfstudio/ --no-cache-dir
3. Reinstall gsplat-rade last (nerfstudio install may overwrite with official gsplat):
   pip install --no-build-isolation git+https://github.com/brian-xu/gsplat-rade.git
4. Smoke test: python -c "import nerfstudio; import gsplat; print(gsplat.__file__)"
   (verify gsplat path is gsplat-rade fork, not nerfstudio's official)
5. TODO block: hloc install deferred — pycolmap 4.0.4 vs hloc compatibility
   unresolved. hloc v1.4 requires pycolmap ~0.4; latest hloc master may support
   4.x but needs validation. Track as separate task.
```

### Phase 4 — `setup.sh` (modify)

Depends on Phase 3 (`setup_nerfstudio.sh`) existing.

```bash
# Before:
pip install -e .
pip install git+https://github.com/BasisResearch/collab-data.git
pip install git+https://github.com/openai/CLIP.git
pip install pypose
pip install bae
pip install git+https://github.com/facebookresearch/co3d.git --no-deps

# After:
bash setup_nerfstudio.sh              # local nerfstudio + gsplat-rade
pip install -e .                      # pulls pypose, bae, clip, recognize-anything,
                                      # feature-splatting from pyproject deps
pip install git+https://github.com/BasisResearch/collab-data.git  # private, stays here
pip install git+https://github.com/facebookresearch/co3d.git --no-deps  # evals only,
                                      # --no-deps required, stays in script
```

Removed: `pip install pypose` (now in pyproject), `pip install bae` (Dockerfile + pyproject), explicit CLIP install (now in pyproject).

### Phase 5 — `constraints_feedforward.txt` (modify)

Depends on Phase 1 (numpy constraint removed from pyproject). Must be correct before Phase 6.

```
# Before:
torch==2.1.2+cu118
torchvision==0.16.2+cu118
numpy>=1.26,<2.0

# After:
torch==2.4.0+cu121
torchvision==0.19.0+cu121
# numpy constraint removed: 2.4.6 installed; no feedforward dep needs <2.0
```

### Phase 6 — `setup/feedforward.sh` (modify)

Depends on Phase 5 (constraints file updated).

**Changes:**
1. Header comments: replace all `cu118`/`2.1.2`/`0.16.2` with `cu121`/`2.4.0`/`0.19.0`
2. Remove mapanything torch compat note block (lines 16–20) — sunset condition met; `torch>=2.4` installed. `_patch_mapanything_torch_compat` is now dead code (separate task to remove from Python source).
3. Remove `pip install 'timm>=1.0' -q` line — already declared in pyproject as `timm>=0.9,<2.0`
4. `pip install gtsam -q` → `pip install gtsam-develop -q`
5. Invariant check: `"11.8"` → `"12.1"`, `"2.1"` → `"2.4"`, `"1.26"` → remove numpy assert (2.4.6 ok)

### Phase 7 — Delete `setup_bundle_adjustment.sh`

Last, after confirming new structure is consistent.

`bae 0.2.3` is pre-installed at `/opt/bae` (editable, `USE_CUDSS=1`) by Dockerfile.cu121. `pypose 0.9.5` pre-installed. compat patches gone (torch 2.4 doesn't need them). Nothing the script installed isn't already handled.

Smoke test from setup_bundle_adjustment.sh is valuable — move to `tests/` as `tests/test_bae_smoke.py` before deleting the script.

```bash
git rm setup_bundle_adjustment.sh
# Also remove patches/bae-torch21-compat.patch (no longer applicable)
git rm patches/bae-torch21-compat.patch
```

---

## Deferred / Out of Scope

| Item | Reason deferred |
|---|---|
| `hloc` install | pycolmap 4.0.4 vs hloc compat unknown; needs investigation |
| `_patch_mapanything_torch_compat` removal from Python source | Separate cleanup task |
| numpy 2.x validation of `fairscale`, `pyntcloud`, `meshlib`, `mobile_sam`, `maskclip_onnx`, `uniception` | Needs full install to test; separate validation task post-migration |
| `feature-splatting` string refs in code | `"feature-splatting"` strings remain in dashboard/splatter.py as method name refs — separate cleanup task, no functional impact |

---

## File Change Summary

| File | Action | Backup needed |
|---|---|---|
| `pyproject.toml` | modify | git-tracked ✓ |
| `setup.sh` | modify | git-tracked ✓ |
| `setup/feedforward.sh` | modify | git-tracked ✓ |
| `setup_bundle_adjustment.sh` | delete | git-tracked ✓ |
| `constraints_feedforward.txt` | modify | git-tracked ✓ |
| `setup_nerfstudio.sh` | create (new) | n/a |
| `patches/nerfstudio-pyproject.orig.toml` | create (backup) | is the backup |
| `patches/nerfstudio-cu121-compat.patch` | create | tracked in collab-splats |
| `patches/bae-torch21-compat.patch` | delete | git-tracked ✓ |
| `tests/test_bae_smoke.py` | create (from setup_bundle_adjustment.sh smoke test) | n/a |
| `requirements.txt` | modify — remove fairscale, recognize-anything, feature-splatting | git-tracked ✓ |
| `/workspace/nerfstudio/pyproject.toml` | patch via setup_nerfstudio.sh | `patches/nerfstudio-pyproject.orig.toml` |
