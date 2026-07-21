# Proportions-Seeded Query Intrinsics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the feedforward-model-per-query-image intrinsics estimate with a model-free COLMAP-standard proportions seed, refined by the pycolmap focal refinement already in `localize`.

**Architecture:** Seed query K from image proportions (`f = 1.2·max(W,H)`, centered principal point, square pixels) via a small helper in `localizer.py`; make `CameraLocalizer.localize`'s `query_intrinsics` optional (seed when `None`); carry the used K on `LocalizationResult`. Delete `localization/intrinsics.py` and its FF path. Dashboard keeps its `calibration_path` YAML override, loses only the FF fallback. Tutorial nb07 drops the `estimate_intrinsics` call.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), numpy, pycolmap, pytest.

Spec: `docs/superpowers/specs/2026-07-21-proportions-query-intrinsics-design.md`

---

### Task 1: Proportions seed helper + optional `query_intrinsics` in `localize`

**Files:**
- Modify: `collab_splats/localization/localizer.py`
- Test: `tests/localization/test_intrinsics_seed.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/localization/test_intrinsics_seed.py`:

```python
"""seed_intrinsics: model-free COLMAP-style proportions seed for query K."""

import numpy as np

from collab_splats.localization.localizer import seed_intrinsics


def test_seed_landscape_focal_and_center():
    K = seed_intrinsics(480, 640)  # H, W
    f = 1.2 * 640  # 1.2 * max(W, H)
    assert K.shape == (3, 3)
    assert np.isclose(K[0, 0], f)  # fx
    assert np.isclose(K[1, 1], f)  # fy (square pixels)
    assert np.isclose(K[0, 2], 320.0)  # cx = W/2
    assert np.isclose(K[1, 2], 240.0)  # cy = H/2
    assert K.dtype == np.float32


def test_seed_portrait_uses_max_dimension():
    K = seed_intrinsics(800, 600)  # H > W
    f = 1.2 * 800
    assert np.isclose(K[0, 0], f)
    assert np.isclose(K[1, 1], f)
    assert np.isclose(K[0, 2], 300.0)
    assert np.isclose(K[1, 2], 400.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_intrinsics_seed.py -v`
Expected: FAIL — `ImportError: cannot import name 'seed_intrinsics'`.

- [ ] **Step 3: Add the helper**

In `collab_splats/localization/localizer.py`, add near the top-level helpers (after the
imports / before `class LocalizationResult`):

```python
def seed_intrinsics(height: int, width: int) -> np.ndarray:
    """Model-free pinhole K seed from image proportions (COLMAP `1.2*max` rule).

    Focal cannot be recovered from proportions alone, so use COLMAP's default
    ``f = 1.2 * max(W, H)`` with a centered principal point and square pixels.
    pycolmap focal refinement solves the true focal from 2D<->3D correspondences.
    """
    f = 1.2 * max(width, height)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_intrinsics_seed.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Make `query_intrinsics` optional in `localize` + carry it on the result**

In `collab_splats/localization/localizer.py`:

Add the field to `LocalizationResult` (after `query_features`):

```python
    query_intrinsics: np.ndarray | None = None  # (3, 3) K used for PnP (seed or supplied)
```

Change the `localize` signature (~line 823):

```python
    def localize(
        self,
        query_image: np.ndarray,
        query_intrinsics: np.ndarray | None = None,
    ) -> LocalizationResult:
```

Update the docstring `Args` for `query_intrinsics`:

```
            query_intrinsics: (3, 3) K, or None to seed from image proportions
                              (COLMAP 1.2*max rule) — pycolmap refines focal during PnP.
```

Immediately after `query_feats = self._extractor.extract(query_image)` (~line 842), resolve K once so every return path can report it:

```python
        # Seed intrinsics from image proportions when the query camera is uncalibrated;
        # pycolmap refine_focal_length solves the true focal from correspondences below.
        if query_intrinsics is None:
            H, W = query_image.shape[:2]
            query_intrinsics = seed_intrinsics(H, W)
```

Then add `query_intrinsics=query_intrinsics` to **each** `LocalizationResult(...)`
returned by `localize` (the `< 4` correspondences early-return, the pycolmap-failure
return, and the success return).

- [ ] **Step 6: Test — optional seed path + explicit-K passthrough**

Append to `tests/localization/test_intrinsics_seed.py`:

```python
def test_localizationresult_carries_intrinsics_field():
    from collab_splats.localization.localizer import LocalizationResult

    K = seed_intrinsics(480, 640)
    r = LocalizationResult(
        pose=None, n_correspondences=0, n_inliers=0,
        pts2d=None, pts3d_matched=None, inlier_mask=None,
        query_intrinsics=K,
    )
    assert np.allclose(r.query_intrinsics, K)
```

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_intrinsics_seed.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add collab_splats/localization/localizer.py tests/localization/test_intrinsics_seed.py
git commit -m "feat(localization): proportions-seeded query intrinsics; optional query_intrinsics in localize"
```

---

### Task 2: Delete the feedforward intrinsics path

**Files:**
- Delete: `collab_splats/localization/intrinsics.py`
- Delete: `tests/localization/test_intrinsics.py`
- Modify: `collab_splats/localization/__init__.py`

- [ ] **Step 1: Remove the export**

In `collab_splats/localization/__init__.py`, delete the import line:

```python
from .intrinsics import estimate_intrinsics
```

and the `__all__` entry:

```python
    "estimate_intrinsics",
```

- [ ] **Step 2: Delete the files**

```bash
git rm collab_splats/localization/intrinsics.py tests/localization/test_intrinsics.py
```

- [ ] **Step 3: Verify no references remain**

Run: `grep -rn "estimate_intrinsics" collab_splats/ tests/`
Expected: no output (dashboard reference is removed in Task 3; if it still shows
`pipeline.py`, that is expected until Task 3 — otherwise no hits).

- [ ] **Step 4: Verify the package imports**

Run: `/opt/venv/reconstruction/bin/python -c "import collab_splats.localization as m; assert not hasattr(m, 'estimate_intrinsics'); print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/localization/__init__.py
git commit -m "refactor(localization): drop estimate_intrinsics feedforward-per-query path"
```

---

### Task 3: Dashboard — keep calibration, drop FF fallback, handle `None` K

**Files:**
- Modify: `collab_splats/dashboard/pipeline.py`
- Test: `tests/dashboard/test_run_localization.py:107`

- [ ] **Step 1: Rewrite `_resolve_query_intrinsics`**

In `collab_splats/dashboard/pipeline.py` (~426), replace the body so the calibration
branch stays and the feedforward fallback becomes `None`:

```python
def _resolve_query_intrinsics(
    frame: np.ndarray, config: LocalizationConfig, op_log: OperationLog
) -> np.ndarray | None:
    """User-supplied YAML calibration when configured; else None → proportions seed."""
    if config.calibration_path:
        data = yaml.safe_load(Path(config.calibration_path).read_text())
        return np.asarray(data["K"], dtype=np.float32).reshape(3, 3)
    # No calibration → let CameraLocalizer.localize seed K from image proportions.
    return None
```

(The `frame` arg is now unused but kept for signature stability with the monkeypatch and
call site; it documents intent and avoids churn.)

- [ ] **Step 2: Update the call site + provenance string**

At ~545:

```python
            K = _resolve_query_intrinsics(frame, config, op_log)
            intr_source = "calibration file" if config.calibration_path else "proportions seed"
```

At the `localize` call (~551), pass `K` (may be `None`) and recover the actual K used
for the DB append / output from the result:

```python
            with op_log.step("localize: matching + solving pose"):
                loc = localizer.localize(frame, K)
            # localize() seeds K from proportions when K is None — use what it actually used.
            K = loc.query_intrinsics if K is None else K
```

The existing `add_localized_frame(img_path, loc.pose, K, ...)` and
`LocalizationRunOutput(..., query_intrinsics=K, intrinsics_source=intr_source, ...)`
lines now receive a concrete K in all cases — no further change.

- [ ] **Step 3: Update the dataclass comment (optional, cosmetic)**

At `pipeline.py:327`, update the field comment:

```python
    intrinsics_source: str  # "calibration file" | "proportions seed"
```

- [ ] **Step 4: Fix the test monkeypatch**

In `tests/dashboard/test_run_localization.py:107`, the monkeypatch already returns a
concrete K — keep it returning a real matrix so the calibration-style path is exercised
(the fake `localizer.localize` ignores it). Change it to reflect the new optional return
type is fine as-is; no edit required unless the fake localizer asserts on `None`. Verify
by running the suite in Step 5. If the fake `localize` reads `loc.query_intrinsics`,
ensure the fake `LocalizationResult` sets it; update the fake result construction in this
file to include `query_intrinsics=np.eye(3, dtype=np.float32)` if the assertion added in
Step 2 (`loc.query_intrinsics if K is None`) is reached (it is not, because the
monkeypatch returns a non-None K).

- [ ] **Step 5: Run the dashboard localization tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_run_localization.py tests/dashboard/test_localize_page.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/pipeline.py tests/dashboard/test_run_localization.py
git commit -m "refactor(dashboard): proportions-seed query intrinsics, keep calibration_path override"
```

---

### Task 4: Tutorial nb07 — drop `estimate_intrinsics`

**Files:**
- Modify: `docs/source/tutorials/07_localization/localization.ipynb`

- [ ] **Step 1: Remove the import**

In the imports cell, change:

```python
from collab_splats.localization import CameraLocalizer, LomaExtractor, estimate_intrinsics, plot_correspondences
```

to:

```python
from collab_splats.localization import CameraLocalizer, LomaExtractor, plot_correspondences
```

- [ ] **Step 2: Remove the estimate call in §2**

In the §2 code cell, delete these two lines:

```python
# Estimate query intrinsics (experimental; refined by pycolmap focal refinement during PnP).
query_K = estimate_intrinsics(query_image)
```

and change the print that references `query_K`:

```python
print(f"query: {QUERY_IMAGE.name}  shape={query_image.shape}  fx≈{query_K[0, 0]:.0f}")
```

to:

```python
print(f"query: {QUERY_IMAGE.name}  shape={query_image.shape}")
```

- [ ] **Step 3: Update the `localize` call**

Find the later cell that calls `localizer.localize(query_image, query_K)` and change it to
seed from proportions (no K):

```python
loc = localizer.localize(query_image)
print(f"query intrinsics (proportions seed): fx≈{loc.query_intrinsics[0, 0]:.0f}")
```

(If the notebook stores the result under a different variable name, keep that name; only
drop the `query_K` argument and read `.query_intrinsics` off the result.)

- [ ] **Step 4: Update the §2 markdown**

Replace the §2 markdown that mentions `estimate_intrinsics` predicting a calibration
matrix with:

> Load the query image (a frame from a different video, not part of the reconstruction).
> Its intrinsics are unknown, so `CameraLocalizer.localize` seeds a pinhole K from the
> image proportions (COLMAP's `f = 1.2·max(W,H)` rule) and pycolmap refines the focal
> during PnP — no feedforward model touches the query image. Reference pixels are read
> from the canonical `frames.zarr` store.

- [ ] **Step 5: Execute the notebook end-to-end**

Run (tutorial data must be present):

```bash
/opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
  docs/source/tutorials/07_localization/localization.ipynb
```

Expected: executes with no error; no VGGT-X load in the localization path.

- [ ] **Step 6: Commit**

```bash
git add docs/source/tutorials/07_localization/localization.ipynb
git commit -m "docs(tutorial): nb07 uses proportions-seeded intrinsics, drops estimate_intrinsics"
```

---

### Task 5: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the localization + dashboard suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization tests/dashboard -q`
Expected: all pass (no `test_intrinsics.py` collected — it was deleted).

- [ ] **Step 2: Confirm no dangling references**

Run: `grep -rn "estimate_intrinsics\|intrinsics.py" collab_splats/ tests/ docs/source/tutorials/`
Expected: no hits.

- [ ] **Step 3: Dashboard smoke gate (mandatory pre-commit for dashboard changes)**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: prints `SMOKE PASS`.

- [ ] **Step 4: Update graphify**

```bash
graphify update .
```

- [ ] **Step 5: Final commit (if graphify or formatting changed anything)**

```bash
black collab_splats/localization/localizer.py collab_splats/dashboard/pipeline.py
isort collab_splats/localization/localizer.py collab_splats/dashboard/pipeline.py
git add -A && git commit -m "chore: format + graphify update for proportions intrinsics" || echo "nothing to commit"
```
