# Docker Env Setup Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three first-run container failures and consolidate bae + nerfstudio into pyproject.toml so setup.sh reduces to a single `pip install -e .` call.

**Architecture:** All changes are to install scripts and `pyproject.toml`. No Python source changes. bae and nerfstudio become git-URL deps in pyproject, so `pip install -e .` handles everything in one shot with `CUDA_HOME` set. Feedforward constraint file deleted — stale torch pin was the root cause of the numpy crash.

**Tech Stack:** bash, pip, pyproject.toml (PEP 517/508 git URLs)

**Spec:** `docs/superpowers/specs/2026-05-26-docker-env-setup-cleanup-design.md`

---

### Task 1: Update `pyproject.toml` — add nerfstudio git URL, replace bae pin

**Files:**
- Modify: `pyproject.toml:59-74`

- [ ] **Step 1: Replace bae and nerfstudio entries**

  In `pyproject.toml`, replace the entire bae comment block + loose pin (lines 59-69) and the nvidia-cudss-cu12 comment (lines 70-74):

  ```toml
      # gsplat-rade fork (version 1.4.0) must be pre-installed with --no-build-isolation
      # before this package. See setup_nerfstudio.sh. Git URL removed here because pip
      # always rebuilds git-URL deps even if already installed, failing without build isolation.
      "gsplat>=1.4.0",
      "maskclip_onnx @ git+https://github.com/RogerQi/maskclip_onnx.git",
      "pypose",
      # bae requires --no-build-isolation at build time (setup.py imports torch at
      # top level to query CUDA ABI). Pre-installed by Dockerfile.cu121 builder stage.
      # Git URL removed: pip rebuilds git-URL deps even if installed, failing without
      # build isolation. bae>=0.2.4 is pre-installed at /opt/bae.
      "bae>=0.2.4",
      # cuDSS runtime dep for bae. Must be installed before `pip install -e bae`
      # so find_cudss_root() finds headers at site-packages/nvidia/cu12.
      # 0.6.0.5 and 0.7.1.6 both verified by bae README; pinned for ABI stability.
      "nvidia-cudss-cu12==0.6.0.5",
  ```

  Replace with:

  ```toml
      # gsplat-rade fork (version 1.4.0) must be pre-installed with --no-build-isolation.
      # Pre-installed in Dockerfile builder stage.
      "gsplat>=1.4.0",
      "maskclip_onnx @ git+https://github.com/RogerQi/maskclip_onnx.git",
      "pypose",
      # cuDSS runtime dep for bae. Must be declared before bae for clarity (pip installs
      # wheels before source builds regardless of order, so cudss headers are present when
      # bae's find_cudss_root() runs at compile time). Pinned for ABI stability.
      "nvidia-cudss-cu12==0.6.0.5",
      # bae: pypose BA CUDA extension. Pinned git URL avoids PyPI 'bae' (2.0.x) collision.
      # Built inline during `pip install -e .` (setup.sh). Requires:
      #   PIP_NO_BUILD_ISOLATION=1 (exported by setup.sh — setup.py imports torch at top level)
      #   CUDA_HOME set (setup.sh — nvcc not in PATH without conda activate)
      # USE_CUDSS defaults to "1" in bae 0.2.4.
      "bae @ git+https://github.com/pypose/bae.git@0.2.4",
      # nerfstudio: BasisResearch fork (our own) with gsplat-rade + cu121 patches applied.
      # @main tracks our fork's main branch — we control what lands there.
      "nerfstudio @ git+https://github.com/BasisResearch/nerfstudio.git@main",
  ```

- [ ] **Step 2: Verify pyproject.toml parses**

  ```bash
  /opt/conda/envs/reconstruction/bin/python -c "
  import tomllib
  with open('/workspace/collab-splats/pyproject.toml', 'rb') as f:
      tomllib.load(f)
  print('OK')
  "
  ```
  Expected: `OK`

- [ ] **Step 3: Commit**

  ```bash
  cd /workspace/collab-splats
  git add pyproject.toml
  git commit -m "feat(setup): add nerfstudio + bae as pinned git URL deps in pyproject

  - nerfstudio @ BasisResearch/nerfstudio@main (our fork, we control main)
  - bae @ pypose/bae@0.2.4 (pinned git URL prevents PyPI bae 2.0.x collision)
  - nvidia-cudss-cu12==0.6.0.5 kept as runtime dep (find_cudss_root() needs headers)

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

### Task 2: Rewrite `setup.sh` — collapse to single install step

**Files:**
- Modify: `setup.sh`

- [ ] **Step 1: Replace setup.sh content**

  Current `setup.sh` is 31 lines with Steps 1–5. Replace with:

  ```bash
  #!/bin/bash
  set -e

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PYTHON=/opt/conda/envs/reconstruction/bin/python
  PIP=/opt/conda/envs/reconstruction/bin/pip
  export PIP_NO_BUILD_ISOLATION=1

  echo "=== Step 1: install collab-splats (builds nerfstudio + bae CUDA extension inline) ==="
  CUDA_HOME=/opt/conda/envs/reconstruction \
      $PIP install -e "$SCRIPT_DIR"

  echo "=== Step 2: install collab-data (private) ==="
  $PIP install git+https://github.com/BasisResearch/collab-data.git

  echo "=== Step 3: install co3d eval dependency (--no-deps required) ==="
  $PIP install git+https://github.com/facebookresearch/co3d.git --no-deps
  ```

- [ ] **Step 2: Verify syntax**

  ```bash
  bash -n /workspace/collab-splats/setup.sh && echo "OK"
  ```
  Expected: `OK`

- [ ] **Step 3: Commit**

  ```bash
  cd /workspace/collab-splats
  git add setup.sh
  git commit -m "refactor(setup): collapse nerfstudio+bae steps into single pip install -e .

  nerfstudio and bae are now pyproject deps (git URLs). setup.sh goes from
  5 steps to 3 — the manual clone/install blocks are deleted.
  CUDA_HOME required: nvcc not in PATH without conda activate.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

### Task 3: Delete `setup_nerfstudio.sh`

**Files:**
- Delete: `setup_nerfstudio.sh`

- [ ] **Step 1: Delete the file**

  ```bash
  rm /workspace/collab-splats/setup_nerfstudio.sh
  ```

- [ ] **Step 2: Confirm no other scripts reference it**

  ```bash
  grep -r 'setup_nerfstudio' /workspace/collab-splats/ --include='*.sh' --include='*.md' --include='*.toml' 2>/dev/null
  ```
  Expected: no output (setup.sh reference was removed in Task 2)

- [ ] **Step 3: Commit**

  ```bash
  cd /workspace/collab-splats
  git add -u setup_nerfstudio.sh
  git commit -m "refactor(setup): delete setup_nerfstudio.sh

  All patches it applied are already in BasisResearch/nerfstudio@d99c8cd7.
  Nerfstudio is now a pyproject git URL dep installed by setup.sh Step 1.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

### Task 4: Delete `constraints_feedforward.txt`, clean `setup_feedforward.sh`

**Files:**
- Delete: `constraints_feedforward.txt`
- Modify: `setup_feedforward.sh`

- [ ] **Step 1: Delete the constraints file**

  ```bash
  rm /workspace/collab-splats/constraints_feedforward.txt
  ```

- [ ] **Step 2: Edit setup_feedforward.sh — remove CONSTRAINTS variable and all -c flags**

  Remove line 24 (`CONSTRAINTS=...`) and lines 27 (`echo "Using constraints: $CONSTRAINTS"`).

  Then replace every occurrence of `-c "$CONSTRAINTS"` with nothing. The affected install blocks become:

  ```bash
  # vggt-x
  # --no-deps: avoids torch==2.3.1 / torchvision==0.18.1 upgrade
  $PIP install "$SCRIPT_DIR/third_party/VGGT-X" --no-deps -q
  $PIP install "viser==0.2.23" evo pyliblzfse safetensors roma kornia -q

  # mapanything
  # --no-deps: avoids opencv-python-headless==4.10.0.84 conflict with opencv-python
  $PIP install 'git+https://github.com/facebookresearch/map-anything.git' --no-deps -q
  $PIP install huggingface_hub hydra-core natsort orjson pillow-heif plyfile \
      python-box requests tensorboard tqdm -q
  # uniception==0.1.7 is mapanything's core model dep
  $PIP install 'uniception==0.1.7' -q
  # colmap extra: needed for MapAnythingCreator reconstruction pipeline
  $PIP install 'lightglue @ git+https://github.com/cvg/LightGlue.git' --no-deps -q
  $PIP install open3d -q
  ```

  Also remove the header comment block that references the constraint file (lines 6–8 of the comment):
  ```
  # Env constraints enforced:
  #   torch==2.5.1+cu121    — CUDA 12.1 gsplat-rade kernels must not change
  #   torchvision==0.20.1+cu121
  ```

- [ ] **Step 3: Verify syntax**

  ```bash
  bash -n /workspace/collab-splats/setup_feedforward.sh && echo "OK"
  ```
  Expected: `OK`

- [ ] **Step 4: Verify no remaining references to constraints file**

  ```bash
  grep -n 'CONSTRAINTS\|constraints_feedforward' /workspace/collab-splats/setup_feedforward.sh
  ```
  Expected: no output

- [ ] **Step 5: Commit**

  ```bash
  cd /workspace/collab-splats
  git add -u setup_feedforward.sh constraints_feedforward.txt
  git commit -m "fix(setup): delete stale feedforward constraint file

  constraints_feedforward.txt pinned torch==2.4.0+cu121 — stale from cu118
  migration. Env has torch 2.5.1; the pin forced downgrade → old numpy
  source build → distutils.msvccompiler crash. All feedforward deps are
  floor-only; no constraint file needed.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

### Task 5: Fix `setup_hloc.sh` conda env name

**Files:**
- Modify: `setup_hloc.sh:5-6`

- [ ] **Step 1: Replace env name in both lines**

  Current lines 5–6:
  ```bash
  PIP="/opt/conda/envs/nerfstudio/bin/pip"
  PYTHON="/opt/conda/envs/nerfstudio/bin/python"
  ```

  Replace with:
  ```bash
  PIP="/opt/conda/envs/reconstruction/bin/pip"
  PYTHON="/opt/conda/envs/reconstruction/bin/python"
  ```

- [ ] **Step 2: Verify no remaining `nerfstudio` env references**

  ```bash
  grep -n 'envs/nerfstudio' /workspace/collab-splats/setup_hloc.sh
  ```
  Expected: no output

- [ ] **Step 3: Verify syntax**

  ```bash
  bash -n /workspace/collab-splats/setup_hloc.sh && echo "OK"
  ```
  Expected: `OK`

- [ ] **Step 4: Commit**

  ```bash
  cd /workspace/collab-splats
  git add setup_hloc.sh
  git commit -m "fix(setup): fix setup_hloc.sh conda env name nerfstudio→reconstruction

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

### Task 6: Update `vendor/README.md` — remove setup_bundle_adjustment.sh references

**Files:**
- Modify: `vendor/README.md`

- [ ] **Step 1: Remove setup_bundle_adjustment.sh from the Populating code block**

  Current (lines 26–31):
  ```markdown
  ```bash
  # Loop closure / feedforward (clones salad)
  bash setup_feedforward.sh

  # Bundle adjustment (clones + patches bae)
  bash setup_bundle_adjustment.sh
  ```
  ```

  Replace with:
  ```markdown
  ```bash
  # Loop closure / feedforward
  bash setup_feedforward.sh

  # Bundle adjustment deps (bae) are installed by setup.sh via pyproject.toml git URL
  ```
  ```

- [ ] **Step 2: Remove setup_bundle_adjustment.sh from the Policy section**

  Current (lines 40–42):
  ```markdown
  - Local patches applied after clone (see `setup_bundle_adjustment.sh` for the
    pattern) must be encoded in the install script, not applied by hand. Future
    reclones need to reproduce them.
  ```

  Replace with:
  ```markdown
  - Local patches applied after clone must be encoded in the install script, not
    applied by hand. Future reclones need to reproduce them.
  ```

- [ ] **Step 3: Remove bae row from the Current entries table**

  Current line 53:
  ```markdown
  | `bae/` | `pypose/bae@0.2` | `setup_bundle_adjustment.sh` | Bundle adjustment. CUDA 11.8 compat patches applied after clone (see script header). |
  ```

  Delete this line entirely. bae is no longer vendored — it is a pyproject git URL dep installed to site-packages by `pip install -e .`.

- [ ] **Step 4: Verify no remaining references**

  ```bash
  grep -n 'setup_bundle_adjustment' /workspace/collab-splats/vendor/README.md
  ```
  Expected: no output

- [ ] **Step 5: Commit**

  ```bash
  cd /workspace/collab-splats
  git add vendor/README.md
  git commit -m "docs(vendor): remove setup_bundle_adjustment.sh references

  Script was deleted in e7fa1c1. bae is now a pyproject git URL dep
  installed by setup.sh — no vendor/ clone needed.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

## Self-Review

**Spec coverage:**
- [x] Change 1 (nerfstudio git URL in pyproject, delete setup_nerfstudio.sh) → Tasks 1 + 3
- [x] Change 2 (delete constraints_feedforward.txt, remove --constraint flags) → Task 4
- [x] Change 3 (bae git URL in pyproject, delete bae step from setup.sh) → Tasks 1 + 2
- [x] Change 5 (fix setup_hloc.sh env name) → Task 5
- [x] Change 6 (vendor/README.md cleanup) → Task 6

**Placeholder scan:** No TBDs, no "implement later", all code blocks show actual content.

**Type consistency:** No function/type definitions — bash + TOML only.
