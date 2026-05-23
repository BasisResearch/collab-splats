# Design: compare_maskclip_talk2dino notebook corrections

**Date:** 2026-05-07  
**Branch:** refactor/core-modules  
**Notebook:** `docs/splats/compare_maskclip_talk2dino.ipynb`

## Context

The `compare_maskclip_talk2dino` notebook compares MaskCLIP/samclip vs Talk2DINO feature
backends on the C0043 birds scene. Four issues block it from being a reliable diagnostic tool:

1. `splats.ply` (Gaussian point cloud, ~125 MB) exists for the maskclip run (`C0043/rade-features/mesh/`) but not for the Talk2DINO run (`C0043_talk2dino/`). No code in the repo generates it — it was created manually for the old MC run and was never added to the pipeline.
2. The notebook uses raw PyVista/open3d/matplotlib instead of `Splatter.plot_mesh()` and `collab_splats.utils.visualization` helpers that already exist.
3. Plots are non-interactive: `pv.start_xvfb()` renders offscreen. The LC eval notebook already uses `pv.set_jupyter_backend("trame")` — that's the established pattern.
4. `query_mesh()` does not expose the `temperature` parameter. `score_queries()` accepts it (default 0.05), but it is hardwired. Similarity maps look flat/indistinguishable — the contrastive softmax needs tuning per-backend.

## Goals

- `splats.ply` generated automatically for any new `mesh()` call (both backends).
- Notebook uses `Splatter` + built-in viz — no raw o3d/plotter boilerplate.
- 3D views are interactive (trame widget) in Jupyter.
- Similarity maps tunable via ipywidgets slider without re-running full cells.

## Design

### 1. `Splatter.mesh()` — add Gaussian splat export

Add private method `_export_gaussian_splats(mesh_dir: Path, overwrite: bool) -> Path`:

```python
def _export_gaussian_splats(self, mesh_dir: Path, overwrite: bool = False) -> Path:
    out = mesh_dir / "splats.ply"
    if out.exists() and not overwrite:
        return out
    # Ensure model_config_path is set
    if not self.config.get("model_config_path"):
        self._select_run()
    # Export Gaussian means + colors via ns-export gaussian-splat CLI
    import subprocess
    subprocess.run(
        [
            "ns-export", "gaussian-splat",
            "--load-config", str(self.config["model_config_path"]),
            "--output-dir", str(mesh_dir),
        ],
        check=True,
    )
    # ns-export may write splat.ply or point_cloud.ply — find and rename
    for candidate_name in ("splat.ply", "point_cloud.ply", "gaussians.ply"):
        candidate = mesh_dir / candidate_name
        if candidate.exists():
            candidate.rename(out)
            break
    return out
```

Call site inside `mesh()`, after `mesher.main()` returns:

```python
splats_path = self._export_gaussian_splats(mesh_dir, overwrite=overwrite)
self.config["mesh_info"]["splats"] = splats_path
```

**Why `ns-export` over manual open3d write:** nerfstudio already handles SH→RGB
conversion and opacity filtering for rade-gs models. Rolling it by hand would
duplicate that logic and risk subtle colour errors.

**Fallback:** if `ns-export gaussian-splat` fails (not all nerfstudio builds support it),
catch and log a warning; do not fail `mesh()`. `mesh_info["splats"]` is left unset.

**Files changed:** `collab_splats/wrapper/splatter.py`

---

### 2. `Splatter.query_mesh()` — expose temperature, always return array

Current signature:
```python
def query_mesh(self, positive_queries, negative_queries, output_fn=None) -> None
```

New signature:
```python
def query_mesh(
    self,
    positive_queries: List[str] = [""],
    negative_queries: List[str] = ["object"],
    output_fn: Optional[str] = None,
    temperature: float = 0.05,
) -> np.ndarray:
```

Pass `temperature` to `similarity_fx`:
```python
similarity_map = (
    self.model.similarity_fx(
        features=...,
        positive=positive_queries,
        negative=negative_queries,
        temperature=temperature,
    )
    ...
)
```

Always return `similarity_colors` (currently only computed when `output_fn` is set;
move computation before the `if output_fn` block so the caller always gets it).

**Files changed:** `collab_splats/wrapper/splatter.py`

---

### 3. Notebook rewrite

#### Backend
Replace:
```python
pv.start_xvfb()
```
With:
```python
pv.set_jupyter_backend("trame")
```

#### Splatter init
Keep existing `make_splatter()` helper (already uses `Splatter.from_config_file` +
`mesh(overwrite=False)`). No change needed — once `mesh()` exports splats, both
splatters will have `mesh_info["splats"]` populated.

#### RGB mesh renders (Cells 10–11)
Replace raw `render_mesh()` / `pv.Plotter(off_screen=True)` blocks with:
```python
splatter_mc.plot_mesh(rgb=True)   # interactive trame widget (rotatable in Jupyter)
splatter_t2d.plot_mesh(rgb=True)
```

Note: `pv.set_jupyter_backend("trame")` at import time makes `mesh.plot()` render
as an interactive trame widget. This applies to standalone cells. The temperature
exploration widget (below) uses `off_screen=True` + matplotlib because PyVista trame
widgets do not compose reliably inside `ipywidgets.Output` widgets.

#### Query similarity renders (Cells 12–19)
Replace `render_query_mesh()` + `compare()` with:
```python
from collab_splats.utils.visualization import compute_heatmap, plot_heatmap

def show_query_comparison(splatter_mc, splatter_t2d, query, pos, neg, temperature):
    colors_mc  = splatter_mc.query_mesh(positive_queries=pos, negative_queries=neg, temperature=temperature)
    colors_t2d = splatter_t2d.query_mesh(positive_queries=pos, negative_queries=neg, temperature=temperature)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # similarity_colors is (N_vertices, 3) — render as similarity-colored mesh via PyVista
    for ax, colors, label in [(axes[0], colors_mc, "maskclip"), (axes[1], colors_t2d, "talk2dino")]:
        mesh = pv.read(str(splatter_mc.config["mesh_info"]["mesh"] if label == "maskclip"
                           else splatter_t2d.config["mesh_info"]["mesh"]))
        mesh.point_data["similarity"] = colors[:, 0]  # R channel = similarity score
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(mesh, scalars="similarity", cmap="hot")
        ax.imshow(pl.screenshot(return_img=True))
        ax.set_title(label)
    fig.suptitle(f'Query: "{query}" — temperature={temperature:.3f}')
    plt.show()
```

Note: similarity colors are per-vertex (not per-pixel image). Visualization renders
the similarity-colored mesh via `pv.PolyData` vertex scalars, same as current
`render_query_mesh` does, but using `visualize_splat` from utils.

#### Temperature exploration cell
New cell after the setup block, before the per-query comparison cells:

```python
import ipywidgets as widgets

QUERIES = [
    ("feeder",  ["feeder"],          ["ground", "leaves", "rocks"]),
    ("tree",    ["tree", "bark"],    ["ground", "sky"]),
    ("ground",  ["ground", "grass"], ["sky", "leaves"]),
]

@widgets.interact(
    query=widgets.Dropdown(options=[q[0] for q in QUERIES], description="Query"),
    temperature=widgets.FloatSlider(min=0.001, max=0.2, step=0.005, value=0.05,
                                    continuous_update=False, description="Temperature"),
)
def explore(query, temperature):
    _, pos, neg = next(q for q in QUERIES if q[0] == query)
    show_query_comparison(splatter_mc, splatter_t2d, query, pos, neg, temperature)
```

`continuous_update=False` prevents firing on every drag tick (inference is slow).

#### Point cloud cell (Cell 20)
Replace raw `pv.Plotter` with:
```python
from collab_splats.utils.visualization import visualize_splat

visualize_splat(str(splatter_mc.config["mesh_info"]["splats"]))
visualize_splat(str(splatter_t2d.config["mesh_info"]["splats"]))
```

---

## Files Changed

| File | Change |
|------|--------|
| `collab_splats/wrapper/splatter.py` | Add `_export_gaussian_splats()`, call from `mesh()`; add `temperature` to `query_mesh()`, fix return type |
| `docs/splats/compare_maskclip_talk2dino.ipynb` | Replace offscreen rendering with trame; use Splatter methods + utils viz; add ipywidgets temperature slider |

## Verification

1. Run `splatter_mc.mesh(overwrite=False)` — verify `splats.ply` already exists, no re-export.
2. Run `splatter_t2d.mesh(overwrite=True)` — verify `splats.ply` written to `C0043_talk2dino/rade-features/mesh/`.
3. Run `splatter_t2d.query_mesh(positive_queries=["feeder"], temperature=0.02)` — verify returns non-None numpy array with shape `(N, 3)`.
4. Open notebook, run all cells — verify trame 3D widget renders for both RGB mesh cells.
5. Drag temperature slider — verify similarity maps update without reloading model.
6. Compare feeder/tree/ground similarity maps at temperature=0.02 vs 0.05 vs 0.1 — verify one value produces clearly distinguishable regions vs background.
