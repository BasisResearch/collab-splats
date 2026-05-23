# clean_pointcloud API Refactor + Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `clean_pointcloud` to remove redundant boolean flags, use `None`-as-skip kwargs with merge semantics, expose previously hardcoded outlier params, update the `max_distance` default to 50.0, and add DEBUG-level per-step point-count logging.

**Architecture:** Module-level default dicts serve as canonical defaults; each step merges user kwargs over them. `None` kwargs skip the step entirely. A module-level `logger` emits DEBUG lines before/after each active step and a summary at the end.

**Tech Stack:** Python, open3d, numpy, standard `logging`

---

## File Map

| File | Change |
|---|---|
| `collab_splats/pointcloud/utils.py` | Add logger, default dicts, refactor `clean_pointcloud` |
| `tests/pointcloud/test_pointcloud_utils.py` | Add new tests, update old flag-based tests |
| `collab_splats/utils/pointcloud.py` | No changes needed (re-export shim unaffected) |

---

### Task 1: Write failing tests for new API behaviors

**Files:**
- Modify: `tests/pointcloud/test_pointcloud_utils.py`

- [ ] **Step 1: Add tests for `None`-as-skip, merge semantics, and logging**

Append to `tests/pointcloud/test_pointcloud_utils.py` after line 201:

```python
# ---------------------------------------------------------------------------
# clean_pointcloud — new API (None=skip, kwargs merge)
# ---------------------------------------------------------------------------

def test_clean_pointcloud_skip_all_via_none():
    pcd = _make_pcd(100)
    result_pcd, indices = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
        distance_kwargs=None,
    )
    assert len(result_pcd.points) == 100
    assert len(indices) == 100


def test_clean_pointcloud_skip_downsample_via_none():
    pcd = _make_pcd(5000)
    # Without downsample, point count should be ~5000 (outlier/distance removal may still reduce slightly)
    result_pcd, indices = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
        distance_kwargs=None,
    )
    assert len(result_pcd.points) == 5000


def test_clean_pointcloud_kwargs_merge_preserves_method():
    """Passing only max_distance should keep method='radial' from defaults."""
    pcd = _make_pcd(300)
    result_pcd, _ = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
        distance_kwargs={"max_distance": 100.0},
    )
    # Should not raise — merge must have kept method='radial'
    assert isinstance(result_pcd, o3d.geometry.PointCloud)


def test_clean_pointcloud_outlier_kwargs_overridable():
    """outlier_kwargs should now be overridable (previously hardcoded)."""
    pcd = _make_pcd(300)
    result_pcd, indices = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs={"nb_neighbors": 5, "std_ratio": 1.0},
        distance_kwargs=None,
    )
    assert isinstance(result_pcd, o3d.geometry.PointCloud)
    assert len(indices) <= 300


def test_clean_pointcloud_logging(caplog):
    """DEBUG logs must emit point counts for each active step."""
    import logging
    pcd = _make_pcd(300)
    with caplog.at_level(logging.DEBUG, logger="collab_splats.pointcloud.utils"):
        clean_pointcloud(pcd)
    messages = caplog.text
    assert "downsample" in messages
    assert "outlier_removal" in messages
    assert "distance_removal" in messages
    assert "→" in messages


def test_clean_pointcloud_logging_skipped_step_absent(caplog):
    """Skipped steps must not appear in DEBUG logs."""
    import logging
    pcd = _make_pcd(300)
    with caplog.at_level(logging.DEBUG, logger="collab_splats.pointcloud.utils"):
        clean_pointcloud(pcd, downsample_kwargs=None, outlier_kwargs=None)
    messages = caplog.text
    assert "downsample" not in messages
    assert "outlier" not in messages
    assert "distance_removal" in messages
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_pointcloud_utils.py::test_clean_pointcloud_skip_all_via_none tests/pointcloud/test_pointcloud_utils.py::test_clean_pointcloud_kwargs_merge_preserves_method tests/pointcloud/test_pointcloud_utils.py::test_clean_pointcloud_logging -v 2>&1 | tail -20
```

Expected: FAIL (TypeError — unexpected keyword `outlier_kwargs` or similar)

---

### Task 2: Implement new `clean_pointcloud`

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`

- [ ] **Step 1: Add logger and default dicts at module level**

Near the top of `collab_splats/pointcloud/utils.py`, after the existing imports (find the `import` block, add below it):

```python
import logging

logger = logging.getLogger(__name__)

_DEFAULT_DOWNSAMPLE_KWARGS: dict = {"voxel_size": 0.015, "adaptive": True}
_DEFAULT_OUTLIER_KWARGS: dict    = {"nb_neighbors": 20, "std_ratio": 2.0}
_DEFAULT_DISTANCE_KWARGS: dict   = {"method": "radial", "max_distance": 50.0}
```

- [ ] **Step 2: Replace `clean_pointcloud` body**

Replace the entire `clean_pointcloud` function (lines 206–250) with:

```python
def clean_pointcloud(
    pcd,
    downsample_kwargs: Optional[dict] = _DEFAULT_DOWNSAMPLE_KWARGS,
    outlier_kwargs:    Optional[dict] = _DEFAULT_OUTLIER_KWARGS,
    distance_kwargs:   Optional[dict] = _DEFAULT_DISTANCE_KWARGS,
) -> tuple:
    """Clean an Open3D point cloud via composable filter steps.

    Applies up to three steps in order:
      1. Voxel downsampling (``voxel_downsample``)
      2. Statistical outlier removal (``pcd.remove_statistical_outlier``)
      3. Distance-based removal (``filter_distance``)

    Each step is enabled by passing a dict of kwargs (merged over module defaults)
    and disabled by passing ``None``.

    Args:
        pcd:               Open3D PointCloud to clean.
        downsample_kwargs: kwargs for ``voxel_downsample``, or None to skip.
        outlier_kwargs:    kwargs for ``remove_statistical_outlier``, or None to skip.
        distance_kwargs:   kwargs for ``filter_distance``, or None to skip.

    Returns:
        ``(cleaned_pcd, index_mapping)`` where ``index_mapping`` maps each output
        point back to its original index.
    """
    n_start = len(pcd.points)
    indices = np.arange(n_start)
    logger.debug("clean_pointcloud: start %d points", n_start)

    if downsample_kwargs is not None and len(pcd.points) > 0:
        n_before = len(pcd.points)
        kwargs = {**_DEFAULT_DOWNSAMPLE_KWARGS, **downsample_kwargs}
        pcd, idx_map = voxel_downsample(pcd, **kwargs)
        indices = indices[idx_map]
        logger.debug("downsample: %d → %d (%d removed)", n_before, len(pcd.points), n_before - len(pcd.points))

    if outlier_kwargs is not None and len(pcd.points) > 0:
        n_before = len(pcd.points)
        kwargs = {**_DEFAULT_OUTLIER_KWARGS, **outlier_kwargs}
        pcd, ind = pcd.remove_statistical_outlier(**kwargs)
        indices = indices[ind]
        logger.debug("outlier_removal: %d → %d (%d removed)", n_before, len(pcd.points), n_before - len(pcd.points))

    if distance_kwargs is not None and len(pcd.points) > 0:
        n_before = len(pcd.points)
        kwargs = {**_DEFAULT_DISTANCE_KWARGS, **distance_kwargs}
        pcd, mask = filter_distance(pcd, return_mask=True, **kwargs)
        indices = indices[mask]
        logger.debug("distance_removal: %d → %d (%d removed)", n_before, len(pcd.points), n_before - len(pcd.points))

    logger.debug(
        "clean_pointcloud: done %d → %d (%d total removed)",
        n_start, len(pcd.points), n_start - len(pcd.points),
    )
    return pcd, indices
```

- [ ] **Step 3: Run new tests — verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_pointcloud_utils.py::test_clean_pointcloud_skip_all_via_none tests/pointcloud/test_pointcloud_utils.py::test_clean_pointcloud_skip_downsample_via_none tests/pointcloud/test_pointcloud_utils.py::test_clean_pointcloud_kwargs_merge_preserves_method tests/pointcloud/test_pointcloud_utils.py::test_clean_pointcloud_outlier_kwargs_overridable tests/pointcloud/test_pointcloud_utils.py::test_clean_pointcloud_logging tests/pointcloud/test_pointcloud_utils.py::test_clean_pointcloud_logging_skipped_step_absent -v 2>&1 | tail -20
```

Expected: all 6 PASS

---

### Task 3: Update old flag-based tests

**Files:**
- Modify: `tests/pointcloud/test_pointcloud_utils.py`

The existing tests at lines 181–201 use boolean flags that no longer exist. Update them:

- [ ] **Step 1: Replace `test_clean_pointcloud_disable_all_stages`**

Old (line 181–185):
```python
def test_clean_pointcloud_disable_all_stages():
    pcd = _make_pcd(100)
    result_pcd, indices = clean_pointcloud(pcd, downsample=False, outlier_removal=False, distance_removal=False)
    assert len(result_pcd.points) == 100
    assert len(indices) == 100
```

New:
```python
def test_clean_pointcloud_disable_all_stages():
    pcd = _make_pcd(100)
    result_pcd, indices = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
        distance_kwargs=None,
    )
    assert len(result_pcd.points) == 100
    assert len(indices) == 100
```

- [ ] **Step 2: Replace `test_clean_pointcloud_downsample_kwargs`**

Old (lines 188–192):
```python
def test_clean_pointcloud_downsample_kwargs():
    pcd = _make_pcd(5000)
    result_pcd, _ = clean_pointcloud(pcd, outlier_removal=False, distance_removal=False,
                                     downsample_kwargs={"voxel_size": 0.5, "adaptive": False})
    assert len(result_pcd.points) < 5000
```

New:
```python
def test_clean_pointcloud_downsample_kwargs():
    pcd = _make_pcd(5000)
    result_pcd, _ = clean_pointcloud(
        pcd,
        outlier_kwargs=None,
        distance_kwargs=None,
        downsample_kwargs={"voxel_size": 0.5, "adaptive": False},
    )
    assert len(result_pcd.points) < 5000
```

- [ ] **Step 3: Replace `test_clean_pointcloud_distance_kwargs`**

Old (lines 195–201):
```python
def test_clean_pointcloud_distance_kwargs():
    pcd = _make_pcd(300)
    result_large, _ = clean_pointcloud(pcd, downsample=False, outlier_removal=False,
                                       distance_kwargs={"max_distance": 100.0})
    result_small, _ = clean_pointcloud(pcd, downsample=False, outlier_removal=False,
                                       distance_kwargs={"max_distance": 0.5})
    assert len(result_large.points) >= len(result_small.points)
```

New:
```python
def test_clean_pointcloud_distance_kwargs():
    pcd = _make_pcd(300)
    result_large, _ = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
        distance_kwargs={"max_distance": 100.0},
    )
    result_small, _ = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
        distance_kwargs={"max_distance": 0.5},
    )
    assert len(result_large.points) >= len(result_small.points)
```

- [ ] **Step 4: Run full test suite for this file**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_pointcloud_utils.py -v 2>&1 | tail -30
```

Expected: all tests PASS, no failures

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/utils.py tests/pointcloud/test_pointcloud_utils.py
git commit -m "feat(pointcloud): refactor clean_pointcloud — remove flags, add logging, fix max_distance default

- Replace bool flags (downsample/outlier_removal/distance_removal) with None-as-skip kwargs
- Add module-level default dicts (_DEFAULT_DOWNSAMPLE_KWARGS, etc.) for merge semantics
- Expose previously hardcoded outlier params via outlier_kwargs
- Add DEBUG-level per-step point-count logging
- Increase max_distance default from 1.0 to 50.0 for MapAnything scene scale
- Update tests to new API"
```

---

### Task 4: Verify re-export shim

**Files:**
- Read: `collab_splats/utils/pointcloud.py`

- [ ] **Step 1: Confirm shim imports still resolve**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.utils.pointcloud import clean_pointcloud, clean_pcd
import inspect
print(inspect.signature(clean_pointcloud))
print(inspect.signature(clean_pcd))
print('OK')
"
```

Expected output:
```
(pcd, downsample_kwargs=..., outlier_kwargs=..., distance_kwargs=...)
(pcd, downsample_kwargs=..., outlier_kwargs=..., distance_kwargs=...)
OK
```

- [ ] **Step 2: No changes needed — done**

The shim at `collab_splats/utils/pointcloud.py` re-exports by name; no edits required.
