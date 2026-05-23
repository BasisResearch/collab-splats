# DataManager Feature Cache Path Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `FeatureSplattingDataManager` re-extracting features on every notebook load by making the cache comparison robust to nerfstudio's `test_mode="inference"` downscale path changes and relative `data` paths.

**Architecture:** Add a `_cache_stems()` module-level helper that normalises filenames to sorted stems, then use it in `setup()` for both cache location (resolved absolute path) and cache key comparison. No changes to storage format — old caches compare correctly via stems.

**Tech Stack:** Python, pathlib, pytest, unittest.mock

---

### Task 1: Add `_cache_stems` helper and fix `setup()` — write failing tests first

**Files:**
- Modify: `collab_splats/nerfstudio/datamanagers/features.py`
- Test: `tests/nerfstudio/test_datamanager_config.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/nerfstudio/test_datamanager_config.py`:

```python
# ── Cache stem helper ─────────────────────────────────────────────────────────

def test_cache_stems_returns_sorted_names():
    from collab_splats.nerfstudio.datamanagers.features import _cache_stems
    paths = [
        "/data/preproc/images_2/frame_002.jpg",
        "/data/preproc/images_2/frame_001.jpg",
    ]
    assert _cache_stems(paths) == ["frame_001.jpg", "frame_002.jpg"]


def test_cache_stems_strips_downscale_folder():
    from collab_splats.nerfstudio.datamanagers.features import _cache_stems
    training_paths = ["/data/preproc/images_2/frame_001.jpg"]
    inference_paths = ["/data/preproc/images/frame_001.jpg"]
    assert _cache_stems(training_paths) == _cache_stems(inference_paths)


def test_cache_stems_strips_absolute_prefix():
    from collab_splats.nerfstudio.datamanagers.features import _cache_stems
    abs_paths = ["/machine_a/data/images/frame_001.jpg"]
    rel_paths = ["../../data/images/frame_001.jpg"]
    assert _cache_stems(abs_paths) == _cache_stems(rel_paths)


# ── setup() cache hit/miss ────────────────────────────────────────────────────

def _make_setup_manager(tmp_path, main_features="samclip", data_path=None):
    """Build a minimal FeatureSplattingDataManager with mocked datasets."""
    from collab_splats.nerfstudio.datamanagers.features import (
        FeatureSplattingDataManager,
        FeatureSplattingDataManagerConfig,
    )
    config = FeatureSplattingDataManagerConfig()
    config.main_features = main_features
    config.enable_cache = True
    config.dataparser = MagicMock()
    config.dataparser.data = data_path if data_path is not None else tmp_path

    manager = object.__new__(FeatureSplattingDataManager)
    manager.config = config
    return manager


def test_setup_cache_hit_same_stems(tmp_path):
    """Cache is returned when stems match, even if full paths differ (downscale folder)."""
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManager

    cached_filenames = [tmp_path / "images_2" / "frame_001.jpg"]
    cached_features = {"samclip": torch.zeros(1, 8, 8, 8)}
    cache_path = tmp_path / "feature-splatting_samclip-features.pt"
    torch.save({"image_filenames": cached_filenames, "features_dict": cached_features}, cache_path)

    manager = _make_setup_manager(tmp_path)
    # Inference-mode paths have no downscale folder
    manager.train_dataset = MagicMock(image_filenames=[tmp_path / "images" / "frame_001.jpg"])
    manager.eval_dataset = MagicMock(image_filenames=[])

    result = manager.setup()

    assert result is cached_features


def test_setup_cache_miss_different_stems(tmp_path):
    """Cache is invalidated when the image set has changed."""
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManager

    cached_filenames = [tmp_path / "images" / "frame_001.jpg"]
    cached_features = {"samclip": torch.zeros(1, 8, 8, 8)}
    cache_path = tmp_path / "feature-splatting_samclip-features.pt"
    torch.save({"image_filenames": cached_filenames, "features_dict": cached_features}, cache_path)

    manager = _make_setup_manager(tmp_path)
    manager.train_dataset = MagicMock(image_filenames=[tmp_path / "images" / "frame_999.jpg"])
    manager.eval_dataset = MagicMock(image_filenames=[])

    extracted = {"samclip": torch.ones(1, 8, 8, 8)}
    with patch.object(manager, "extract_features", return_value=extracted):
        result = manager.setup()

    assert result is extracted


def test_setup_cache_path_resolved(tmp_path, monkeypatch):
    """cache_dir resolves to absolute — relative data path still finds the cache."""
    # Write the cache under tmp_path
    cached_filenames = [tmp_path / "images" / "frame_001.jpg"]
    cached_features = {"samclip": torch.zeros(1, 8, 8, 8)}
    cache_path = tmp_path / "feature-splatting_samclip-features.pt"
    torch.save({"image_filenames": cached_filenames, "features_dict": cached_features}, cache_path)

    # Change CWD to tmp_path.parent so that tmp_path.name is a valid relative path
    monkeypatch.chdir(tmp_path.parent)
    relative_data = Path(tmp_path.name)  # resolves to tmp_path from new CWD

    manager = _make_setup_manager(tmp_path, data_path=relative_data)
    manager.train_dataset = MagicMock(image_filenames=[tmp_path / "images" / "frame_001.jpg"])
    manager.eval_dataset = MagicMock(image_filenames=[])

    result = manager.setup()

    assert result is cached_features
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/nerfstudio/test_datamanager_config.py::test_cache_stems_returns_sorted_names tests/nerfstudio/test_datamanager_config.py::test_cache_stems_strips_downscale_folder tests/nerfstudio/test_datamanager_config.py::test_cache_stems_strips_absolute_prefix tests/nerfstudio/test_datamanager_config.py::test_setup_cache_hit_same_stems tests/nerfstudio/test_datamanager_config.py::test_setup_cache_miss_different_stems tests/nerfstudio/test_datamanager_config.py::test_setup_cache_path_resolved -v 2>&1 | tail -20
```

Expected: 3 `ImportError` failures on `_cache_stems`, 3 failures on cache behavior.

- [ ] **Step 3: Add `_cache_stems` helper to `features.py`**

In `collab_splats/nerfstudio/datamanagers/features.py`, after the `_EXTRACTOR_NAME` line (line 34), add:

```python
def _cache_stems(filenames: list) -> list[str]:
    """Canonical cache key: sorted filename stems, independent of path prefix and downscale folder."""
    return sorted(Path(f).name for f in filenames)
```

- [ ] **Step 4: Fix `cache_dir` and comparison in `setup()`**

In `setup()`, replace lines 112–124:

```python
        # Set up cache path
        cache_dir = self.config.dataparser.data
        cache_path = (
            cache_dir / f"feature-splatting_{self.config.main_features}-features.pt"
        )

        # Try loading from cache if enabled
        if self.config.enable_cache and cache_path.exists():
            cache_dict = torch.load(cache_path)

            if cache_dict.get("image_filenames") != image_filenames:
                CONSOLE.print("Image filenames have changed, cache invalidated...")
            else:
                return cache_dict["features_dict"]
        else:
            CONSOLE.print("Cache does not exist, extracting features...")
```

With:

```python
        # Set up cache path — resolve to absolute so CWD changes don't affect it
        cache_dir = Path(self.config.dataparser.data).resolve()
        cache_path = (
            cache_dir / f"feature-splatting_{self.config.main_features}-features.pt"
        )

        # Try loading from cache if enabled
        if self.config.enable_cache and cache_path.exists():
            cache_dict = torch.load(cache_path)

            if _cache_stems(cache_dict.get("image_filenames", [])) != _cache_stems(image_filenames):
                CONSOLE.print("Image filenames have changed, cache invalidated...")
            else:
                return cache_dict["features_dict"]
        else:
            CONSOLE.print("Cache does not exist, extracting features...")
```

- [ ] **Step 5: Run new tests — expect all pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/nerfstudio/test_datamanager_config.py::test_cache_stems_returns_sorted_names tests/nerfstudio/test_datamanager_config.py::test_cache_stems_strips_downscale_folder tests/nerfstudio/test_datamanager_config.py::test_cache_stems_strips_absolute_prefix tests/nerfstudio/test_datamanager_config.py::test_setup_cache_hit_same_stems tests/nerfstudio/test_datamanager_config.py::test_setup_cache_miss_different_stems tests/nerfstudio/test_datamanager_config.py::test_setup_cache_path_resolved -v 2>&1 | tail -20
```

Expected: 6 PASSED.

- [ ] **Step 6: Run full test suite — no regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/nerfstudio/ -v 2>&1 | tail -20
```

Expected: all previously-passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/nerfstudio/datamanagers/features.py tests/nerfstudio/test_datamanager_config.py
git commit -m "fix(datamanager): resolve cache_dir to absolute, compare by filename stems

test_mode='inference' forces downscale_factor=1 in nerfstudio, changing
image_filenames from images_2/frame.jpg to images/frame.jpg. Comparing
sorted Path.name stems instead of full paths makes cache hits robust to
this and to relative data paths in config.yml."
```
