# Semantic Heatmap Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `compute_heatmap`, `plot_heatmap`, and `query_heatmap` to `collab_splats/utils/visualization.py` for overlaying semantic similarity maps on RGB images.

**Architecture:** Three functions live in the existing visualization module — `compute_heatmap` normalizes a raw similarity tensor and blends a viridis colormap over an RGB image, `plot_heatmap` handles display/export, and `query_heatmap` is a convenience wrapper that runs a full feature-extraction pipeline for a single text query. Separation keeps viz logic decoupled from extractors.

**Tech Stack:** numpy, matplotlib (already in visualization.py), cv2 (resize), torch (optional input type), PIL (extractor input conversion)

---

## File Map

| File | Action |
|------|--------|
| `collab_splats/utils/visualization.py` | **Modified** — three new functions added after imports |
| `tests/test_heatmap.py` | **Create** — unit tests for compute_heatmap and plot_heatmap |

---

### Task 1: Verify implementation is present

**Files:**
- Read: `collab_splats/utils/visualization.py`

- [ ] **Step 1: Confirm three functions exist**

```bash
grep -n "def compute_heatmap\|def plot_heatmap\|def query_heatmap" collab_splats/utils/visualization.py
```

Expected output:
```
8:def compute_heatmap(
35:def plot_heatmap(
51:def query_heatmap(
```

---

### Task 2: Write and run tests for `compute_heatmap`

**Files:**
- Create: `tests/test_heatmap.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_heatmap.py`:

```python
import numpy as np
import pytest
from collab_splats.utils.visualization import compute_heatmap, plot_heatmap


def _make_image(h=100, w=120):
    rng = np.random.default_rng(0)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def _make_sim_map(h=100, w=120):
    rng = np.random.default_rng(1)
    return rng.random((h, w)).astype(np.float32)


class TestComputeHeatmap:
    def test_output_shape_same_size(self):
        image = _make_image(100, 120)
        sim_map = _make_sim_map(100, 120)
        result = compute_heatmap(image, sim_map)
        assert result.shape == (100, 120, 3)

    def test_output_dtype_uint8(self):
        image = _make_image()
        sim_map = _make_sim_map()
        result = compute_heatmap(image, sim_map)
        assert result.dtype == np.uint8

    def test_output_values_in_range(self):
        image = _make_image()
        sim_map = _make_sim_map()
        result = compute_heatmap(image, sim_map)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_resize_sim_map_to_image_size(self):
        image = _make_image(100, 120)
        sim_map = _make_sim_map(20, 24)  # patch grid, smaller
        result = compute_heatmap(image, sim_map)
        assert result.shape == (100, 120, 3)

    def test_squeeze_hw1_input(self):
        image = _make_image(100, 120)
        sim_map = _make_sim_map(100, 120).reshape(100, 120, 1)
        result = compute_heatmap(image, sim_map)
        assert result.shape == (100, 120, 3)

    def test_torch_tensor_input(self):
        import torch
        image = _make_image()
        sim_map = torch.from_numpy(_make_sim_map())
        result = compute_heatmap(image, sim_map)
        assert result.shape == (100, 120, 3)
        assert result.dtype == np.uint8

    def test_alpha_zero_returns_image(self):
        image = _make_image()
        sim_map = _make_sim_map()
        result = compute_heatmap(image, sim_map, alpha=0.0)
        np.testing.assert_array_equal(result, image)

    def test_alpha_one_returns_heatmap_only(self):
        image = _make_image()
        sim_map = _make_sim_map()
        result_a1 = compute_heatmap(image, sim_map, alpha=1.0)
        result_a0 = compute_heatmap(image, sim_map, alpha=0.0)
        # With alpha=1, result should differ from the raw image
        assert not np.array_equal(result_a1, result_a0)

    def test_constant_sim_map_no_crash(self):
        image = _make_image()
        sim_map = np.ones((100, 120), dtype=np.float32)
        result = compute_heatmap(image, sim_map)
        assert result.shape == (100, 120, 3)

    def test_custom_colormap(self):
        image = _make_image()
        sim_map = _make_sim_map()
        result = compute_heatmap(image, sim_map, colormap="plasma")
        assert result.shape == (100, 120, 3)
```

- [ ] **Step 2: Run tests to verify they fail (or pass — implementation is pre-written)**

```bash
cd /workspace/collab-splats && python -m pytest tests/test_heatmap.py::TestComputeHeatmap -v
```

Expected: all pass (implementation already present). If failures occur, see Task 3.

---

### Task 3: Write and run tests for `plot_heatmap`

**Files:**
- Modify: `tests/test_heatmap.py`

- [ ] **Step 1: Append plot_heatmap tests**

Append to `tests/test_heatmap.py`:

```python
import tempfile
import os
import matplotlib.pyplot as plt


class TestPlotHeatmap:
    def _heatmap(self):
        return compute_heatmap(_make_image(), _make_sim_map())

    def test_saves_to_disk(self):
        heatmap = self._heatmap()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_heatmap(heatmap, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)
            plt.close("all")

    def test_accepts_external_ax(self):
        heatmap = self._heatmap()
        fig, ax = plt.subplots()
        plot_heatmap(heatmap, ax=ax, title="test")
        assert ax.get_title() == "test"
        plt.close(fig)

    def test_no_crash_no_ax_no_save(self):
        import matplotlib
        matplotlib.use("Agg")
        heatmap = self._heatmap()
        plot_heatmap(heatmap)
        plt.close("all")
```

- [ ] **Step 2: Run tests**

```bash
cd /workspace/collab-splats && python -m pytest tests/test_heatmap.py::TestPlotHeatmap -v
```

Expected: all pass.

---

### Task 4: Smoke-test `query_heatmap` interface

**Files:**
- Read: `collab_splats/semantics/features.py`

- [ ] **Step 1: Verify `query_heatmap` signature matches extractor API**

Check that `BaseFeatureExtractor` subclasses expose `encode_text`, `forward`, and `compute_similarity`:

```bash
grep -n "def encode_text\|def forward\|def compute_similarity" collab_splats/semantics/features.py
```

Expected: all three found in at least one extractor class.

- [ ] **Step 2: Add import smoke test**

Append to `tests/test_heatmap.py`:

```python
def test_query_heatmap_importable():
    from collab_splats.utils.visualization import query_heatmap
    assert callable(query_heatmap)
```

- [ ] **Step 3: Run**

```bash
cd /workspace/collab-splats && python -m pytest tests/test_heatmap.py::test_query_heatmap_importable -v
```

Expected: PASS.

---

### Task 5: Commit

- [ ] **Step 1: Stage and commit**

```bash
cd /workspace/collab-splats
git add collab_splats/utils/visualization.py tests/test_heatmap.py
git commit -m "feat(viz): add compute_heatmap, plot_heatmap, query_heatmap

Blends viridis-mapped semantic similarity tensor over RGB image.
Supports numpy and torch inputs, auto-resize for patch-grid sim maps.
query_heatmap wraps full extractor pipeline for single text query."
```

---

## Verification Checklist

- [ ] `pytest tests/test_heatmap.py -v` — all tests green
- [ ] `grep "def compute_heatmap\|def plot_heatmap\|def query_heatmap" collab_splats/utils/visualization.py` — all three present
- [ ] Notebook smoke test: `compute_heatmap(img, sim)` → display inline with `plot_heatmap`
