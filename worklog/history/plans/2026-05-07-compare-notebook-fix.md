# compare_maskclip_talk2dino Notebook Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four issues in the maskclip vs talk2dino comparison notebook: missing `splats.ply` for T2D, raw visualization code, non-interactive plots, and flat similarity maps due to missing temperature control.

**Architecture:** Two Python changes (`splatter.py`: add `_export_gaussian_splats()` + `temperature` param to `query_mesh()`), then a full notebook rewrite using those APIs plus `pv.set_jupyter_backend("trame")` and `ipywidgets`.

**Tech Stack:** Python 3.10, nerfstudio `ns-export gaussian-splat`, PyVista trame, ipywidgets 8.1.7, pytest, open3d

---

## File Map

| File | What changes |
|------|-------------|
| `collab_splats/wrapper/splatter.py` | Add `_export_gaussian_splats()`; call from both `mesh()` branches; add `temperature` to `query_mesh()`, fix return type |
| `docs/splats/compare_maskclip_talk2dino.ipynb` | Full rewrite: trame backend, `Splatter` methods, ipywidgets temperature slider, `visualize_splat` for point cloud |
| `tests/wrapper/test_splatter_query.py` | New: unit tests for `query_mesh()` temperature forwarding + return type |
| `tests/wrapper/test_splatter_mesh.py` | New: unit tests for `_export_gaussian_splats()` + `mesh()` splats key |
| `tests/wrapper/__init__.py` | New: empty init |

---

## Task 1: `query_mesh()` — add `temperature`, fix return type

**Files:**
- Modify: `collab_splats/wrapper/splatter.py:498-558`
- Create: `tests/wrapper/__init__.py`
- Create: `tests/wrapper/test_splatter_query.py`

- [ ] **Step 1: Create test file (will fail — `temperature` not in signature yet)**

Create `tests/wrapper/__init__.py` (empty):
```python
```

Create `tests/wrapper/test_splatter_query.py`:
```python
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


def _make_splatter_with_model(tmp_path, n_verts=50):
    """Return a Splatter instance with a fake model wired in — no disk I/O."""
    from collab_splats.wrapper.splatter import Splatter

    mesh_pt = tmp_path / "mesh_features.pt"
    mesh_ply = tmp_path / "mesh_tsdf_clean.ply"
    mesh_ply.touch()

    # Build a minimal Splatter without calling __init__ validation
    s = object.__new__(Splatter)
    s.config = {
        "output_path": tmp_path,
        "method": "rade-features",
        "model_config_path": str(tmp_path / "config.yml"),
        "mesh_info": {
            "mesh": mesh_ply,
            "features": mesh_pt,
        },
    }

    # Fake model
    mock_model = MagicMock()
    mock_model.main_features_name = "distill_features"
    mock_model.device = torch.device("cpu")

    fake_features_pt = torch.zeros(n_verts, 64)
    decoded = {"distill_features": torch.zeros(n_verts, 64)}
    mock_model.decoder.per_gaussian_forward.return_value = decoded
    mock_model.similarity_fx.return_value = torch.zeros(n_verts, 1)

    s.model = mock_model
    return s, mock_model, fake_features_pt


def test_query_mesh_returns_ndarray(tmp_path):
    s, mock_model, fake_features = _make_splatter_with_model(tmp_path)
    with patch("torch.load", return_value=fake_features):
        result = s.query_mesh(positive_queries=["feeder"], negative_queries=["ground"])
    assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"


def test_query_mesh_temperature_forwarded(tmp_path):
    s, mock_model, fake_features = _make_splatter_with_model(tmp_path)
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return torch.zeros(50, 1)

    mock_model.similarity_fx.side_effect = capture

    with patch("torch.load", return_value=fake_features):
        s.query_mesh(positive_queries=["feeder"], negative_queries=["ground"], temperature=0.01)

    assert "temperature" in captured, "temperature was not forwarded to similarity_fx"
    assert captured["temperature"] == pytest.approx(0.01)


def test_query_mesh_default_temperature_is_005(tmp_path):
    s, mock_model, fake_features = _make_splatter_with_model(tmp_path)
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return torch.zeros(50, 1)

    mock_model.similarity_fx.side_effect = capture

    with patch("torch.load", return_value=fake_features):
        s.query_mesh(positive_queries=["feeder"])

    assert captured.get("temperature") == pytest.approx(0.05)


def test_query_mesh_output_fn_writes_ply(tmp_path):
    """When output_fn is given, a PLY is written AND np.ndarray is still returned."""
    import open3d as o3d

    # Create a minimal valid PLY so o3d.io.read_triangle_mesh doesn't fail
    mesh_ply = tmp_path / "mesh_tsdf_clean.ply"
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.zeros((50, 3)))
    m.triangles = o3d.utility.Vector3iVector(np.zeros((1, 3), dtype=np.int32))
    o3d.io.write_triangle_mesh(str(mesh_ply), m)

    s, mock_model, fake_features = _make_splatter_with_model(tmp_path)
    s.config["mesh_info"]["mesh"] = mesh_ply

    with patch("torch.load", return_value=fake_features):
        result = s.query_mesh(
            positive_queries=["feeder"],
            negative_queries=["ground"],
            output_fn="query-feeder.ply",
        )

    assert isinstance(result, np.ndarray)
    assert (tmp_path / "query-feeder.ply").exists()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_splatter_query.py -v 2>&1 | tail -20
```

Expected: all 4 tests FAIL with `TypeError: query_mesh() got unexpected keyword argument 'temperature'` or similar.

- [ ] **Step 3: Modify `query_mesh()` in splatter.py**

Open `collab_splats/wrapper/splatter.py`. Find `def query_mesh` at line ~498. Replace the entire method:

```python
def query_mesh(
    self,
    positive_queries: List[str] = [""],
    negative_queries: List[str] = ["object"],
    output_fn: Optional[str] = None,
    temperature: float = 0.05,
) -> np.ndarray:
    """Query the mesh for features.

    Returns:
        similarity_colors: (N_vertices, 3) float64 array. R channel holds the
        normalised similarity score [0, 1]; G and B are zero unless multiple
        positive queries are provided.
    """

    if not self.config.get("model_config_path"):
        self._select_run()
    elif getattr(self, "model", None) is None:
        print(f"Loading model from {self.config['model_config_path']}")
        from nerfstudio.utils.eval_utils import eval_setup

        _, pipeline, _, _ = eval_setup(Path(self.config["model_config_path"]))
        self.model = pipeline.model

    mesh_info = self.config.get("mesh_info")
    if mesh_info is None:
        raise ValueError("Mesh information not found. Please run mesh() first.")
    elif mesh_info.get("features") is None:
        raise ValueError("Features not found. Please run mesh() with features_name specified.")

    features = torch.load(self.config["mesh_info"]["features"])

    decoded_features = self.model.decoder.per_gaussian_forward(features.to(self.model.device).to(torch.float32))

    similarity_map = (
        self.model.similarity_fx(
            features=decoded_features[self.model.main_features_name].unsqueeze(0).permute(2, 1, 0),
            positive=positive_queries,
            negative=negative_queries,
            temperature=temperature,
        )
        .squeeze(-1)
        .detach()
        .cpu()
        .numpy()
    )

    del features

    # Normalise and pack into (N, 3) — R = score, G = B = 0
    similarity_colors = np.zeros((len(similarity_map), 3))
    similarity_cast = similarity_map.astype(np.float64)
    if similarity_cast.ndim == 1:
        similarity_cast = similarity_cast[:, np.newaxis]
    if np.max(similarity_cast) > 0:
        similarity_cast /= np.max(similarity_cast)
    similarity_colors[:, : similarity_cast.shape[1]] = similarity_cast

    if output_fn is not None:
        output_dir = self.config["mesh_info"]["mesh"].parent
        output_path = output_dir / output_fn
        mesh = o3d.io.read_triangle_mesh(self.config["mesh_info"]["mesh"])
        mesh.vertex_colors = o3d.utility.Vector3dVector(similarity_colors)
        o3d.io.write_triangle_mesh(output_path, mesh)

    return similarity_colors
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_splatter_query.py -v 2>&1 | tail -20
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/splatter.py tests/wrapper/__init__.py tests/wrapper/test_splatter_query.py
git commit -m "feat(splatter): add temperature param to query_mesh, fix return type to np.ndarray"
```

---

## Task 2: `_export_gaussian_splats()` + wire into `mesh()`

**Files:**
- Modify: `collab_splats/wrapper/splatter.py` (add method, modify `mesh()`)
- Create: `tests/wrapper/test_splatter_mesh.py`

- [ ] **Step 1: Write failing tests**

Create `tests/wrapper/test_splatter_mesh.py`:
```python
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


def _bare_splatter(tmp_path):
    from collab_splats.wrapper.splatter import Splatter

    s = object.__new__(Splatter)
    s.config = {
        "output_path": tmp_path,
        "method": "rade-features",
        "model_config_path": str(tmp_path / "config.yml"),
    }
    return s


def test_export_gaussian_splats_calls_ns_export(tmp_path):
    """_export_gaussian_splats runs ns-export and returns path to splats.ply."""
    s = _bare_splatter(tmp_path)
    mesh_dir = tmp_path / "rade-features" / "mesh"
    mesh_dir.mkdir(parents=True)

    def fake_run(cmd, check, **kwargs):
        # Simulate ns-export writing splats.ply
        (mesh_dir / "splats.ply").write_bytes(b"ply")

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        result = s._export_gaussian_splats(mesh_dir, overwrite=False)

    assert result == mesh_dir / "splats.ply"
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "ns-export" in call_args
    assert "gaussian-splat" in call_args


def test_export_gaussian_splats_skips_if_exists(tmp_path):
    """_export_gaussian_splats does not call ns-export when splats.ply exists."""
    s = _bare_splatter(tmp_path)
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    existing = mesh_dir / "splats.ply"
    existing.write_bytes(b"ply")

    with patch("subprocess.run") as mock_run:
        result = s._export_gaussian_splats(mesh_dir, overwrite=False)

    mock_run.assert_not_called()
    assert result == existing


def test_export_gaussian_splats_overwrite_reruns(tmp_path):
    """overwrite=True re-runs ns-export even when splats.ply exists."""
    s = _bare_splatter(tmp_path)
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    (mesh_dir / "splats.ply").write_bytes(b"old")

    def fake_run(cmd, check, **kwargs):
        (mesh_dir / "splats.ply").write_bytes(b"new")

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        s._export_gaussian_splats(mesh_dir, overwrite=True)

    mock_run.assert_called_once()


def test_mesh_else_branch_sets_splats_key(tmp_path):
    """mesh(overwrite=False) sets mesh_info['splats'] when splats.ply exists."""
    from collab_splats.wrapper.splatter import Splatter

    s = object.__new__(Splatter)
    mesh_dir = tmp_path / "rade-features" / "mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "mesh_tsdf_clean.ply").write_bytes(b"ply")
    (mesh_dir / "splats.ply").write_bytes(b"ply")
    (mesh_dir / "mesh_features.pt").write_bytes(b"pt")

    s.config = {
        "output_path": tmp_path,
        "method": "rade-features",
        "model_config_path": str(tmp_path / "config.yml"),
    }

    # Patch _select_run so it doesn't look for real config.yml
    with patch.object(Splatter, "_select_run"):
        s.mesh(overwrite=False)

    assert "splats" in s.config["mesh_info"]
    assert s.config["mesh_info"]["splats"] == mesh_dir / "splats.ply"


def test_mesh_else_branch_no_splats_key_when_missing(tmp_path):
    """mesh(overwrite=False) does not set mesh_info['splats'] when splats.ply absent."""
    from collab_splats.wrapper.splatter import Splatter

    s = object.__new__(Splatter)
    mesh_dir = tmp_path / "rade-features" / "mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "mesh_tsdf_clean.ply").write_bytes(b"ply")

    s.config = {
        "output_path": tmp_path,
        "method": "rade-features",
        "model_config_path": str(tmp_path / "config.yml"),
    }

    with patch.object(Splatter, "_select_run"):
        s.mesh(overwrite=False)

    assert "splats" not in s.config["mesh_info"]
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_splatter_mesh.py -v 2>&1 | tail -20
```

Expected: all 5 tests FAIL (`AttributeError: _export_gaussian_splats` or missing `splats` key).

- [ ] **Step 3: Add `_export_gaussian_splats()` method to splatter.py**

In `collab_splats/wrapper/splatter.py`, after the `_extract_mesh_features` method (currently around line 474), add:

```python
def _export_gaussian_splats(self, mesh_dir: Path, overwrite: bool = False) -> Path:
    """Export the trained Gaussian splat cloud to splats.ply via ns-export.

    Skipped (returns existing path) if splats.ply already exists and overwrite is False.
    """
    out = mesh_dir / "splats.ply"
    if out.exists() and not overwrite:
        return out
    if not self.config.get("model_config_path"):
        self._select_run()
    try:
        subprocess.run(
            [
                "ns-export", "gaussian-splat",
                "--load-config", str(self.config["model_config_path"]),
                "--output-dir", str(mesh_dir),
                "--output-filename", "splats.ply",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[warning] ns-export gaussian-splat failed ({e}); splats.ply not written")
    return out
```

- [ ] **Step 4: Wire into `mesh()` — both branches**

In `mesh()`, after the new-mesh branch writes `mesh_info` (after `_extract_mesh_features` call, around line 461), add:

```python
            # Export Gaussian splat point cloud alongside the TSDF mesh
            self._export_gaussian_splats(mesh_dir, overwrite=overwrite)
            splats_path = mesh_dir / "splats.ply"
            if splats_path.exists():
                self.config["mesh_info"]["splats"] = splats_path
```

In the existing-mesh `else` branch (after the `features_path` block, around line 472), add:

```python
            splats_path = mesh_dir / "splats.ply"
            if splats_path.exists():
                self.config["mesh_info"]["splats"] = splats_path
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_splatter_mesh.py -v 2>&1 | tail -20
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Run full wrapper test suite to check for regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/ -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/wrapper/splatter.py tests/wrapper/test_splatter_mesh.py
git commit -m "feat(splatter): export gaussian splats as splats.ply in mesh(); add _export_gaussian_splats()"
```

---

## Task 3: Notebook rewrite

**Files:**
- Modify: `docs/splats/compare_maskclip_talk2dino.ipynb`

This task rewrites the notebook cells in-place. The JSON structure stays the same; only cell `source` arrays change. Use `NotebookEdit` or direct JSON edit.

- [ ] **Step 1: Fix backend — Cell 1 (imports)**

Replace Cell 1 source. The only change is `pv.start_xvfb()` → `pv.set_jupyter_backend("trame")`:

```python
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pyvista as pv
import ipywidgets as widgets
from IPython.display import display

pv.set_jupyter_backend("trame")

sys.path.insert(0, "/workspace/collab-splats")

FIELDWORK = Path("/workspace/fieldwork-data/birds/2024-02-06/environment")
CONFIG_DIR = Path("/workspace/collab-splats/docs/splats/configs")
DATASET = "birds_date-02062024_video-C0043"

MC_DIR = FIELDWORK / "C0043"
T2D_DIR = FIELDWORK / "C0043_talk2dino"

MC_MESH_DIR = MC_DIR / "rade-features/mesh"
T2D_MESH_DIR = T2D_DIR / "rade-features/mesh"

CAM_POS = [(2, 2, 1), (0, 0, 0), (0, 0, 1)]
WINDOW_SIZE = (1600, 800)
```

- [ ] **Step 2: Replace visualization helper Cell 8**

The existing `render_mesh` / `render_query_mesh` / `compare` helpers used `off_screen=True` rendering. Replace with helpers that use the Splatter API and built-in visualization utils:

```python
from collab_splats.utils.visualization import visualize_splat, compute_heatmap

def render_similarity_mesh(splatter, colors: np.ndarray, cam_pos=CAM_POS) -> np.ndarray:
    """Render a mesh with per-vertex similarity colours to a numpy image."""
    import pyvista as pv
    mesh_path = splatter.config["mesh_info"]["mesh"]
    mesh = pv.read(str(mesh_path))
    mesh.point_data["similarity"] = colors[:, 0]   # R channel = score
    pl = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    pl.add_mesh(mesh, scalars="similarity", cmap="hot", clim=[0.0, 1.0])
    pl.camera_position = cam_pos
    return pl.screenshot(return_img=True)


def compare_similarity(splatter_mc, splatter_t2d, query_label: str,
                       pos: list, neg: list, temperature: float = 0.05) -> None:
    """Query both splatters and show side-by-side matplotlib figure."""
    colors_mc  = splatter_mc.query_mesh(positive_queries=pos, negative_queries=neg,
                                        temperature=temperature)
    colors_t2d = splatter_t2d.query_mesh(positive_queries=pos, negative_queries=neg,
                                         temperature=temperature)
    img_mc  = render_similarity_mesh(splatter_mc,  colors_mc)
    img_t2d = render_similarity_mesh(splatter_t2d, colors_t2d)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(img_mc);  axes[0].set_title("maskclip");  axes[0].axis("off")
    axes[1].imshow(img_t2d); axes[1].set_title("talk2dino"); axes[1].axis("off")
    fig.suptitle(f'Query: "{query_label}"  |  temperature={temperature:.3f}')
    plt.tight_layout()
    plt.show()
```

- [ ] **Step 3: Replace RGB mesh render cells (Cells 10–11)**

Replace the raw `render_mesh(...)` + `compare(...)` calls with direct `plot_mesh()` calls
(each in its own cell so the trame widget renders separately):

Cell 10 source:
```python
print("maskclip — RGB mesh")
splatter_mc.plot_mesh(rgb=True)
```

Cell 11 source:
```python
print("talk2dino — RGB mesh")
splatter_t2d.plot_mesh(rgb=True)
```

- [ ] **Step 4: Add temperature exploration cell (new cell, after Cell 7 setup)**

Insert a new cell after the query-generation setup cells and before the per-query comparison cells:

```python
# ── Temperature exploration ────────────────────────────────────────────────
# Drag the slider to find the temperature that gives the sharpest separation.
# continuous_update=False prevents firing on every tick (inference is slow).

QUERIES = [
    ("feeder",  ["feeder"],          ["ground", "leaves", "rocks"]),
    ("tree",    ["tree", "bark"],    ["ground", "sky"]),
    ("ground",  ["ground", "grass"], ["sky", "leaves", "rocks"]),
    ("sky",     ["sky"],             ["ground", "tree", "feeder"]),
]

@widgets.interact(
    query=widgets.Dropdown(
        options=[q[0] for q in QUERIES],
        description="Query:",
        style={"description_width": "initial"},
    ),
    temperature=widgets.FloatSlider(
        min=0.001, max=0.2, step=0.005, value=0.05,
        continuous_update=False,
        description="Temperature:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    ),
)
def explore_temperature(query, temperature):
    _, pos, neg = next(q for q in QUERIES if q[0] == query)
    compare_similarity(splatter_mc, splatter_t2d, query, pos, neg, temperature)
```

- [ ] **Step 5: Replace per-query comparison cells (Cells 12–19)**

Replace each raw `render_query_mesh + compare` pair with calls to `compare_similarity`.
Use pre-computed query PLY files when they already exist by loading directly, OR just call
`compare_similarity` (which calls `query_mesh` and re-derives the colours in memory —
no disk write needed for display):

Cell 12 (tree/bark):
```python
compare_similarity(splatter_mc, splatter_t2d, "tree / bark",
                   pos=["tree", "bark"], neg=["ground", "sky"])
```

Cell 14 (feeder):
```python
compare_similarity(splatter_mc, splatter_t2d, "feeder",
                   pos=["feeder"], neg=["ground", "leaves", "rocks"])
```

Cell 16 (ground):
```python
compare_similarity(splatter_mc, splatter_t2d, "ground / grass",
                   pos=["ground", "grass"], neg=["sky", "leaves"])
```

Cell 18 (sky):
```python
compare_similarity(splatter_mc, splatter_t2d, "sky / background",
                   pos=["sky"], neg=["ground", "tree", "feeder"])
```

Remove the corresponding markdown placeholder cells for each query (Cells 13, 15, 17, 19)
or keep them as section labels — they're markdown, not broken code.

- [ ] **Step 6: Replace point cloud cell (Cell 20)**

```python
# ── Gaussian splat point clouds ───────────────────────────────────────────
from collab_splats.utils.visualization import visualize_splat

print("maskclip — Gaussian splats")
visualize_splat(str(splatter_mc.config["mesh_info"]["splats"]))
```

New cell below:
```python
print("talk2dino — Gaussian splats")
visualize_splat(str(splatter_t2d.config["mesh_info"]["splats"]))
```

- [ ] **Step 7: Generate `splats.ply` for the T2D run**

The T2D mesh dir exists but has no `splats.ply`. Force-export it now (one-off):

```bash
ns-export gaussian-splat \
  --load-config /workspace/fieldwork-data/birds/2024-02-06/environment/C0043_talk2dino/rade-features/2026-05-06_233516/config.yml \
  --output-dir /workspace/fieldwork-data/birds/2024-02-06/environment/C0043_talk2dino/rade-features/mesh \
  --output-filename splats.ply
```

Expected: `splats.ply` appears in T2D mesh dir (~100MB). Confirm:

```bash
ls -lh /workspace/fieldwork-data/birds/2024-02-06/environment/C0043_talk2dino/rade-features/mesh/splats.ply
```

- [ ] **Step 8: Commit**

```bash
git add docs/splats/compare_maskclip_talk2dino.ipynb
git commit -m "docs(notebook): rewrite compare_maskclip_talk2dino — trame, Splatter API, ipywidgets temperature slider"
```

---

## Verification Checklist

- [ ] `pytest tests/wrapper/ -v` — all pass
- [ ] `splatter_t2d.mesh(overwrite=False)` sets `mesh_info["splats"]` to existing T2D `splats.ply`
- [ ] `splatter_t2d.query_mesh(positive_queries=["feeder"], temperature=0.02)` returns `np.ndarray` shape `(N, 3)`
- [ ] Open notebook in Jupyter, run all cells — no errors
- [ ] RGB mesh cells show trame 3D widget (rotatable)
- [ ] Temperature slider fires `compare_similarity` on release; similarity maps update in output widget
- [ ] At temperature ≈ 0.02–0.05, feeder / tree / ground maps show clearly distinct regions
- [ ] Point cloud cells render `splats.ply` for both MC and T2D via `visualize_splat`
