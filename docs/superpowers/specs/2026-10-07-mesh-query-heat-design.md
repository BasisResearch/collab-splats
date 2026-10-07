# Mesh query heat

Date: 2026-10-07 · Status: implemented (browser check: offset default 0.25, opacity dropped — alpha ignored)

## Problem

Semantic queries on the mesh viewer (`docs/examples/ocr_lens_viewer.py`) read as scattered
triangles and dots, not the smooth heatmap a queried 2D image gives.

- `search` thresholds the score (`score >= min_prob`) into a boolean mask
- `Viewer.highlight` draws the masked vertices as a points overlay
- label-list buttons go through the same `highlight` path
- the label list ranks top-1 word counts, after optional kNN smoothing (`Smoothing k`)

The scores themselves are already smooth. Measured on GH010229 (scratch `compare.py`):

- `lift_features` per-vertex probabilities, raw (no smoothing), drawn as a continuous
  colormap over faces, already looks like the 2D query
- extra Laplacian / kNN smoothing adds little; a mesh-raster projection is faster to bake
  (2.2 s vs 14.2 s) but sees fewer vertices (64% vs 97%)
- texel-space heat on the 8192 view-chart atlas fails: the atlas is too fragmented

So the defect is display only: a thresholded points overlay instead of a continuous one.

## Decision

Draw the continuous per-vertex score as vertex colors on a mesh overlay. The GPU blends
vertex colors across each face, which is the smoothness. The floor hides faces, not shading.

Out of scope:

- no change to `lift_features`, the bake, the vertex cache or the mesh module
- no smoothing step (raw lifted probabilities are smooth enough)
- no texture or albedo baking of the heat
- the dashboard's kNN `_transfer_mesh_features` path is not touched

## Design

### `Viewer.show_heat` (`collab_splats/viewer.py`)

Replaces `Viewer.highlight`. `viewer.py` is in `tests/test_docstring_contract.py` `MODULES`:
`show_heat` and the new `add_label_list` carry full `Args:`; the module, class and `add_mesh`
docstrings say heat overlay where they say highlights.

```python
def show_heat(
    self,
    name: str,
    scores: Optional[np.ndarray],
    floor: float,
    *,
    opacity: float = 0.6,
    offset: float = 0.0,
) -> None:
```

- `scores`: (V,) per-vertex score over mesh `name`'s vertices; `None` clears the overlay
- drops the previous `<name>/heat` node, if any, before drawing
- keeps faces where any vertex's score is at or above `floor`
- sub-mesh: `np.unique(faces[keep], return_inverse=True)` remaps to the used vertices only
- colors: viridis over `scores[used]` min-max normalized (what `apply_viridis` does and the
  prototype showed), via `matplotlib.colormaps["viridis"]`, plus an alpha channel at `opacity`
- `offset`: shift along the sub-mesh's vertex normals, in median edge lengths, to avoid
  z-fighting with the base mesh; 0 sends the positions unchanged
- sends one `trimesh.Trimesh(..., vertex_colors=rgba, process=False)` through
  `self.server.scene.add_mesh_trimesh(f"{name}/heat", ...)` — same path `add_mesh` uses
  (viser 1.0.29 `add_mesh_simple` takes a single color only)
- no face reaches the floor: overlay cleared, nothing sent
- handle kept in a `self.heats: dict[str, handle]`, mirroring how `highlight` tracked its node
- the base mesh is never resent; clicks still raycast `self.meshes`, so picking is unchanged

`opacity` and `offset` stay keyword defaults (no GUI sliders). Plan task 1 checks both
in the browser on GH010229:

- alpha ignored by viser's GLB path: drop `opacity` and send RGB
- z-fighting at offset 0: set the default to the smallest offset that removes it

### Colormap: matplotlib directly, `apply_viridis` stays put

`apply_viridis` lives in `collab_splats/dashboard/viz_utils.py`; moving it to
`collab_splats/utils/visualization.py` was the first plan. Rejected on import cost:

- `import collab_splats.viewer`: 3.7 s today, no matplotlib / pyvista / torch loaded
- `import collab_splats.utils.visualization`: 11.1 s (pyvista, torch, geometry incl. BA)
- `viz_utils` imports `utils.visualization`, so importing it from there costs the same
- `matplotlib` on top of the viewer: 0.16 s

So `show_heat` calls the matplotlib colormap itself (two lines). Dashboard untouched.

### `Viewer.add_label_list` takes weighted words

```python
def add_label_list(
    self,
    name: str,
    words: Sequence[str],
    weights: np.ndarray,
    on_select: Callable[[Optional[str]], None],
    top_n: int = 20,
) -> None:
```

- ranks `words` by `weights`, descending (stable), keeps `top_n`
- button label: word plus weight as expected vertex count, e.g. `grass (~28.7k)`
- word button calls `on_select(word)`; `Clear` calls `on_select(None)`
- re-adding replaces the folder, as today
- no per-vertex labels array, no counting inside the viewer

Weight is probability mass: the sum of p(word) over vertices. Each observed vertex's
probabilities sum to 1 (measured mean 0.99999), so mass is the expected vertex count —
the total of the heat a click draws. GH010229 check vs top-1 counts: 18 of 20 words shared;
top-1 inflates winners (grass 67.7k top-1 vs 28.7k mass).

### `docs/examples/ocr_lens_viewer.py`

- `search` calls `viewer.show_heat("mesh", score, min_prob.value)`; empty or unknown query
  passes `None`
- `Query min p` slider is the floor
- label list built once at load:
  `viewer.add_label_list("mesh", vocab.words, full.sum(axis=0, dtype=np.float64), on_select)`
  (unobserved rows are zero, so no mask; float64 avoids float16 accumulation)
- `on_select(word)` sets the query box to `word` (or `""`) and runs `search`, so a click and a
  typed query draw the same heat
- removed: `Smoothing k` slider, its `Apply` button, `relabel`, `transfer_features` import,
  `shown_probs`, per-vertex `labels`, `seen`, and `lock` (it only kept relabels and probes
  from interleaving; nothing mutates shared state once relabel is gone)
- probe chart reads `term_probs` directly (was the smoothed copy)
- module and `main` docstrings drop the top-1 / smoothing wording; query bullet says heat
- `transfer_features` stays in `collab_splats/semantics/lifting.py`; other code calls it

## Testing

`tests/test_viewer.py`, mocked server as today. Replace the `highlight` test and the
label-count test; `test_clicks_still_pick_after_a_highlight` becomes
`test_clicks_still_pick_after_show_heat`.

- `show_heat` keeps only faces with a vertex at or above the floor, remapped to used vertices
- vertex colors follow the score order (lowest score darkest viridis)
- `None` clears; a second call replaces the first node
- all scores below floor: nothing sent, previous overlay gone
- base mesh not resent (`add_mesh_trimesh` called once, for `<name>/heat` only)
- `add_label_list` orders by weight, cuts at `top_n`, labels `~N` counts, button click hands
  the word to `on_select`, `Clear` hands `None`, re-adding replaces
- clicks still pick vertices after `show_heat`

Manual: GH010229 in the browser — typed query and label click give the same heat;
heat shading reads like the 2D query.

## Files

| File | Change |
|---|---|
| `collab_splats/viewer.py` | `show_heat` replaces `highlight`; `add_label_list` takes words + weights + `on_select` |
| `docs/examples/ocr_lens_viewer.py` | heat search, mass-ranked list, smoothing + lock removed |
| `tests/test_viewer.py` | tests above |
