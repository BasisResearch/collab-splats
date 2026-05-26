# Test Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate collab_splats test suite after torch 2.4→2.5.1 upgrade, bae@0.2.4 git URL, and nerfstudio BasisResearch git URL.

**Architecture:** Four stale assertions in `test_cu121_migration.py` reference old install paths and torch version. Fix those first, then run fast suite and triage against `worklog/known-test-failures.md`, then verify all migration hard gates pass.

**Tech Stack:** pytest, `/opt/conda/envs/reconstruction/bin/python` (py3.11), torch 2.5.1+cu121

**Spec:** `docs/superpowers/specs/2026-05-26-test-validation-design.md`

---

## Files Changed

| File | Change |
|---|---|
| `tests/test_cu121_migration.py` | Fix 4 stale assertions + module docstring env name |
| `worklog/known-test-failures.md` | Update based on actual run results |

---

### Task 1: Fix torch version assertions in `test_cu121_migration.py`

Two tests hard-code `"2.4"` — both fail with torch 2.5.1.

**Files:**
- Modify: `tests/test_cu121_migration.py:4,25-27,193-199`

- [ ] **Step 1: Fix module docstring env name (line 4)**

  Change:
  ```python
  #     /opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_cu121_migration.py -v
  ```
  To:
  ```python
  #     /opt/conda/envs/reconstruction/bin/python -m pytest tests/test_cu121_migration.py -v
  ```

- [ ] **Step 2: Fix `test_torch_version` (lines 23-30)**

  Change:
  ```python
  def test_torch_version():
      import torch
      assert torch.__version__.startswith("2.4"), (
          f"Expected torch 2.4.x, got {torch.__version__}"
      )
      assert "cu121" in torch.__version__, (
          f"Expected cu121 build, got {torch.__version__}"
      )
  ```
  To:
  ```python
  def test_torch_version():
      import torch
      assert torch.__version__.startswith("2.5"), (
          f"Expected torch 2.5.x, got {torch.__version__}"
      )
      assert "cu121" in torch.__version__, (
          f"Expected cu121 build, got {torch.__version__}"
      )
  ```

- [ ] **Step 3: Fix `test_bae_cuda_backend` (lines 193-199)**

  Change:
  ```python
  def test_bae_cuda_backend():
      """bae active with CUDA 12.1 / torch 2.4."""
      import pypose  # noqa: F401
      import bae  # noqa: F401
      import torch
      assert torch.__version__.startswith("2.4"), f"Wrong torch: {torch.__version__}"
      assert "12.1" in torch.version.cuda, f"Wrong CUDA: {torch.version.cuda}"
  ```
  To:
  ```python
  def test_bae_cuda_backend():
      """bae active with CUDA 12.1 / torch 2.5."""
      import pypose  # noqa: F401
      import bae  # noqa: F401
      import torch
      assert torch.__version__.startswith("2.5"), f"Wrong torch: {torch.__version__}"
      assert "12.1" in torch.version.cuda, f"Wrong CUDA: {torch.version.cuda}"
  ```

- [ ] **Step 4: Run both tests to verify they pass**

  ```bash
  /opt/conda/envs/reconstruction/bin/python -m pytest tests/test_cu121_migration.py::test_torch_version tests/test_cu121_migration.py::test_bae_cuda_backend -v
  ```
  Expected: both PASS. If either fails, check actual `torch.__version__` with:
  ```bash
  /opt/conda/envs/reconstruction/bin/python -c "import torch; print(torch.__version__)"
  ```
  Update the `startswith` prefix to match the actual major.minor.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/test_cu121_migration.py
  git commit -m "fix(tests): update torch version assertions to 2.5.x

  Torch upgraded from 2.4 to 2.5.1 in the cu121 Dockerfile.
  test_torch_version and test_bae_cuda_backend both hard-coded startswith('2.4').

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

### Task 2: Fix `test_bae_editable_install` path assertion

bae was previously installed as editable from `/opt/bae`. It is now installed as a git URL dep in `pyproject.toml` via `pip install -e .`, landing in the conda env's site-packages.

**Files:**
- Modify: `tests/test_cu121_migration.py:64-70`

- [ ] **Step 1: Confirm actual bae install path**

  ```bash
  /opt/conda/envs/reconstruction/bin/python -c "import bae; print(bae.__file__)"
  ```
  Note the path. It should contain `/opt/conda/envs/reconstruction`.

- [ ] **Step 2: Fix `test_bae_editable_install` (lines 64-70)**

  Change:
  ```python
  def test_bae_editable_install():
      import bae
      bae_file = bae.__file__
      assert bae_file is not None
      assert "/opt/bae" in bae_file, (
          f"bae not from /opt/bae editable install: {bae_file}"
      )
  ```
  To:
  ```python
  def test_bae_editable_install():
      import bae
      bae_file = bae.__file__
      assert bae_file is not None
      assert "/opt/conda/envs/reconstruction" in bae_file, (
          f"bae not installed in reconstruction env: {bae_file}"
      )
  ```

  If Step 1 showed a different path, use that prefix instead.

- [ ] **Step 3: Run the test to verify it passes**

  ```bash
  /opt/conda/envs/reconstruction/bin/python -m pytest tests/test_cu121_migration.py::test_bae_editable_install -v
  ```
  Expected: PASS.

- [ ] **Step 4: Commit**

  ```bash
  git add tests/test_cu121_migration.py
  git commit -m "fix(tests): update bae path assertion for pyproject git URL install

  bae is no longer an editable install from /opt/bae — it is now a pyproject
  git URL dep installed into the reconstruction conda env site-packages.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

### Task 3: Fix `test_nerfstudio_installed_local` path assertion

nerfstudio was previously installed as editable from `/workspace/nerfstudio`. It is now installed via the BasisResearch fork git URL in `pyproject.toml`, landing in site-packages.

**Files:**
- Modify: `tests/test_cu121_migration.py:202-210`

- [ ] **Step 1: Confirm actual nerfstudio install path**

  ```bash
  /opt/conda/envs/reconstruction/bin/python -c "
  import nerfstudio.field_components.activations as p
  print(p.__file__)
  "
  ```
  Note the path — should contain `/opt/conda/envs/reconstruction`.

- [ ] **Step 2: Fix `test_nerfstudio_installed_local` (lines 202-210)**

  Change:
  ```python
  def test_nerfstudio_installed_local():
      """nerfstudio is the local /workspace/nerfstudio install, not a PyPI package."""
      import nerfstudio.field_components.activations as _ns_probe
      ns_file = _ns_probe.__file__
      assert ns_file is not None
      assert "/workspace/nerfstudio" in ns_file, (
          f"nerfstudio loaded from unexpected location: {ns_file}. "
          "Expected /workspace/nerfstudio — PyPI nerfstudio may have been installed."
      )
  ```
  To:
  ```python
  def test_nerfstudio_installed_local():
      """nerfstudio loads from reconstruction conda env (BasisResearch fork via pyproject)."""
      import nerfstudio.field_components.activations as _ns_probe
      ns_file = _ns_probe.__file__
      assert ns_file is not None
      assert "/opt/conda/envs/reconstruction" in ns_file, (
          f"nerfstudio loaded from unexpected location: {ns_file}. "
          "Expected site-packages under /opt/conda/envs/reconstruction — "
          "stale /workspace/nerfstudio or PyPI nerfstudio may be on sys.path."
      )
  ```

  If Step 1 showed a different path, use that prefix.

- [ ] **Step 3: Run the test to verify it passes**

  ```bash
  /opt/conda/envs/reconstruction/bin/python -m pytest tests/test_cu121_migration.py::test_nerfstudio_installed_local -v
  ```
  Expected: PASS.

- [ ] **Step 4: Commit**

  ```bash
  git add tests/test_cu121_migration.py
  git commit -m "fix(tests): update nerfstudio path assertion for BasisResearch fork install

  nerfstudio is no longer an editable install from /workspace/nerfstudio — it is
  now installed via git URL dep (BasisResearch fork) into the reconstruction env.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

### Task 4: Check pypose/bae version conflict

The known-test-failures.md Group 1 was caused by pypose requiring `bae==0.2` but bae 0.2.1 being installed. bae is now at 0.2.4. Check if pypose has been updated to accept 0.2.x.

**Files:**
- Possibly modify: `worklog/known-test-failures.md` (if Group 1 clears)

- [ ] **Step 1: Check pypose's bae requirement**

  ```bash
  /opt/conda/envs/reconstruction/bin/python -c "
  import importlib.metadata as m
  reqs = m.requires('pypose') or []
  bae_reqs = [r for r in reqs if 'bae' in r.lower()]
  print('pypose bae requirement:', bae_reqs)
  print('bae installed version:', m.version('bae'))
  "
  ```

- [ ] **Step 2: Interpret result**

  - If output shows `bae==0.2` requirement and bae is `0.2.4` → Group 1 failures **still pre-existing** (no action needed, stays in known-test-failures.md)
  - If output shows `bae>=0.2` or no bae requirement → Group 1 failures **cleared** → remove Group 1 from `worklog/known-test-failures.md`
  - If `ImportError: pypose not found` → pypose not installed → note this, Group 1 tests will skip/error

  No code change needed in this step — just record the finding for Task 7.

---

### Task 5: Stage 2 — Fast unit test run and triage

Run the full fast suite and compare to `worklog/known-test-failures.md`.

**Files:**
- Read: `worklog/known-test-failures.md` (reference during triage)
- Possibly modify: `tests/` (if new failures found and fixable)

- [ ] **Step 1: Run fast suite**

  ```bash
  /opt/conda/envs/reconstruction/bin/python -m pytest tests/ -m 'not slow' --tb=short -q 2>&1 | tee /tmp/pytest_stage2.txt
  tail -40 /tmp/pytest_stage2.txt
  ```

- [ ] **Step 2: Triage failures**

  For each FAILED test in the output, check `worklog/known-test-failures.md`:

  | Category | Action |
  |---|---|
  | Test name appears in known-failures Groups 1–5 | Expected — skip |
  | Test fails with an error NOT in known-failures | **New regression — fix now** |
  | `--doctest-modules` doctest failure | Check if it's from a module you modified — fix the docstring |
  | Collection error (ImportError at module level) | Critical — fix before proceeding |

  New failures NOT in known-failures.md: diagnose with:
  ```bash
  /opt/conda/envs/reconstruction/bin/python -m pytest <failing_test> -v --tb=long
  ```
  Then fix and re-run.

- [ ] **Step 3: Commit any fixes found during triage**

  For each fix:
  ```bash
  git add <files>
  git commit -m "fix(<scope>): <what broke and why>"
  ```

---

### Task 6: Stage 3 — Migration hard gates

All `test_cu121_migration.py` tests are merge-blocking. Run the full file after Tasks 1–3 fixes.

**Files:**
- No changes expected (fixes already committed in Tasks 1–3)

- [ ] **Step 1: Run migration hard gates**

  ```bash
  /opt/conda/envs/reconstruction/bin/python -m pytest tests/test_cu121_migration.py -v 2>&1 | tee /tmp/pytest_stage3.txt
  cat /tmp/pytest_stage3.txt
  ```
  Expected: all PASS. Zero failures allowed.

- [ ] **Step 2: If any test still fails, fix it**

  Each failure = a stale assertion or a real env problem. Fix the assertion if the test logic is wrong for the new env; escalate if the env is actually broken.

  Common remaining failure modes:
  - `test_collab_data_installed`: `collab_data` not installed → run `setup.sh` Step 2 first, or mark as known-failure with explanation
  - `test_gtsam_sl4_manifold`: gtsam-develop not installed → mark as known-failure if gtsam is not part of the base install
  - `test_flagged_package_imports`: one of mobile_sam/uniception/meshlib fails → check if it's a numpy 2.x compat issue vs missing install

  For each: fix or explicitly categorize as known-failure in `worklog/known-test-failures.md`.

---

### Task 7: Update `worklog/known-test-failures.md`

Reconcile the document with actual results from Tasks 4–6.

**Files:**
- Modify: `worklog/known-test-failures.md`

- [ ] **Step 1: Update the document**

  For each Group in the document:
  - **Group cleared** (tests now pass): delete the group entry, add a note `## Cleared [date]` with one line explaining why
  - **Group still present, same error**: keep as-is, update date to 2026-05-26
  - **New failure group found during Stage 2**: add as a new Group with same format (Affected, Error, Root cause, Fix options)

  Standard Group format:
  ```markdown
  ## Group N: <title> (<count> failures)

  **Affected:**
  - `tests/path/test_file.py::test_name`

  **Error:**
  ```
  ExactErrorMessage: exact text
  ```

  **Root cause:** One sentence.

  **Fix options:**
  1. ...
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add worklog/known-test-failures.md
  git commit -m "docs(worklog): update known-test-failures for cu121+torch2.5 env

  Reflects actual failures after: torch 2.4→2.5.1, bae@0.2.4 git URL,
  nerfstudio BasisResearch fork via pyproject.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

## Self-Review

**Spec coverage:**
- Stage 0 pre-fixes → Tasks 1–3 ✓
- pypose/bae conflict check → Task 4 ✓
- Stage 1 import sweep → already covered by `test_import_all_modules` inside Task 6 ✓
- Stage 2 fast unit tests + triage → Task 5 ✓
- Stage 3 migration hard gates → Task 6 ✓
- Update known-test-failures.md → Task 7 ✓

**Placeholder scan:** No TBDs. All code blocks complete. Conditional branching in Task 4 Step 2 and Task 6 Step 2 explicit.

**Type consistency:** No cross-task type dependencies. All assertions are string comparisons on version/path strings — no shared types.
