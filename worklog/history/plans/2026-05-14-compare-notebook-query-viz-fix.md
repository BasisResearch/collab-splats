# compare_maskclip_talk2dino Query-Testing Viz Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Plan storage:** Once plan mode exits, copy this plan to `worklog/history/plans/2026-05-14-compare-notebook-query-viz-fix.md` before starting execution.

**Goal:** Make the "Query testing" cell in `docs/splats/compare_maskclip_talk2dino.ipynb` actually render the side-by-side interactive trame viewer when the **Run query** button is clicked.

**Architecture:** Two surgical edits inside the notebook. (1) `compare_interactive` switches from implicit `pl.show()` to `display(pl.show(jupyter_backend="trame", return_viewer=True))` so the trame viewer widget is explicitly displayed instead of relying on Jupyter's last-expression auto-display (which does not fire inside button callbacks). (2) `_on_run` moves the `compare_interactive(...)` call **inside** the `with output:` block so the viewer renders into the `ipywidgets.Output` widget that lives under the controls. No source code under `collab_splats/` changes.

**Interactivity parity with existing cells:** the working `visualize_splat(mesh, ...).show()` cells (e.g. `rgb-compare`) succeed because `visualize_splat` returns a `Plotter` (visualization.py:307) and Jupyter auto-displays the trame viewer returned by `Plotter.show()` as the cell's last expression. Inside a button callback there is no "last expression" — return values are discarded. `return_viewer=True` + `display(viewer)` is the explicit form of the same auto-display pipeline, so the resulting widget is the identical trame viewer with identical drag-rotate / zoom / pan interactivity. `pl.link_views()` (Plotter-side, backend-agnostic) additionally syncs rotation across the two subplots.

**Tech Stack:** Jupyter, ipywidgets, pyvista (trame jupyter backend), `Splatter.query_mesh`, plyfile.

---

## Context

The "Query testing" cell builds an ipywidgets panel (preset dropdown, positive/negative text fields, temperature slider, **Run query** button, `Output` widget). On click, `_on_run` runs `splatter.query_mesh(...)` for both maskclip and talk2dino splatters and then calls `compare_interactive(...)` to render a 2-subplot pyvista trame viewer. **Nothing ever appears.**

Root cause — two compounding bugs in `_on_run` / `compare_interactive`:

1. `compare_interactive(...)` is called **outside** the `with output:` block in `_on_run`. Anything that would render does not go through the `Output` widget, so the viewer is orphaned in the active cell display context (which a button callback does not have).
2. `compare_interactive` ends with a bare `pl.show()`. With `pv.set_jupyter_backend("trame")`, `Plotter.show()` returns a trame viewer widget. At cell top-level that return value is auto-displayed as the last expression — but inside a callback the return value is discarded.

Confirmed via inspection:
- `compare_interactive` is defined in cell `viz-helpers` of `docs/splats/compare_maskclip_talk2dino.ipynb`.
- `_on_run` is defined in cell `query-testing` of the same notebook.
- `collab_splats/utils/visualization.py:visualize_splat` returns the `Plotter` object (line 307) and never calls `.show()` — caller-managed display is already the pattern, so the fix here aligns with existing conventions.
- `return_viewer=True` is not used anywhere in the repo yet — first use is fine, it is a stable upstream pyvista API.

---

## File Structure

**Files modified:**
- `docs/splats/compare_maskclip_talk2dino.ipynb` — two cells only (`viz-helpers`, `query-testing`).

**Files NOT modified:**
- `collab_splats/utils/visualization.py` — `visualize_splat` is not on the bug path.
- Any `.py` source under `collab_splats/`.

No new files. No deletions.

---

## Task 1: Fix `compare_interactive` to explicitly display the trame viewer

**Files:**
- Modify: `docs/splats/compare_maskclip_talk2dino.ipynb` — cell `viz-helpers`, the `compare_interactive` function definition.

- [ ] **Step 1: Open the notebook and locate the cell**

Open `docs/splats/compare_maskclip_talk2dino.ipynb`. Find the cell whose id is `viz-helpers` (contains `def compare_interactive(...)`). The current function ends with:

```python
    pl.link_views()
    pl.show()
```

- [ ] **Step 2: Replace the function body with the explicit-display version**

Replace the entire `compare_interactive` definition with:

```python
def compare_interactive(splatter_mc, splatter_t2d, colors_mc, colors_t2d,
                        query_label: str) -> None:
    """Side-by-side interactive pyvista viewer (trame, drag to rotate)."""
    from IPython.display import display

    mesh_mc  = pv.read(str(splatter_mc.config["mesh_info"]["mesh"]))
    mesh_t2d = pv.read(str(splatter_t2d.config["mesh_info"]["mesh"]))
    mesh_mc["similarity"]  = colors_mc[:, 0]
    mesh_t2d["similarity"] = colors_t2d[:, 0]

    pl = pv.Plotter(shape=(1, 2), window_size=list(WINDOW_SIZE), notebook=True)
    pl.subplot(0, 0)
    pl.add_mesh(mesh_mc,  scalars="similarity", cmap="hot", clim=[0.0, 1.0])
    pl.add_text(f"maskclip — {query_label}", font_size=12)
    pl.camera_position = CAM_POS
    pl.subplot(0, 1)
    pl.add_mesh(mesh_t2d, scalars="similarity", cmap="hot", clim=[0.0, 1.0])
    pl.add_text(f"talk2dino — {query_label}", font_size=12)
    pl.camera_position = CAM_POS
    pl.link_views()

    viewer = pl.show(jupyter_backend="trame", return_viewer=True)
    display(viewer)
```

Why each change matters:
- `notebook=True` + explicit `jupyter_backend="trame"` + `return_viewer=True` → returns the viewer widget unconditionally, independent of any global backend state.
- `display(viewer)` → renders into whichever output context is active (the `Output` widget once Task 2 lands).
- `query_label` interpolated into both subplot titles so each run labels itself.

- [ ] **Step 3: Sanity-check the cell parses (no execution yet)**

In the kernel, run just the `viz-helpers` cell. Expected: no output, no exception. The function is redefined.

- [ ] **Step 4: Commit**

```bash
git add docs/splats/compare_maskclip_talk2dino.ipynb
git commit -m "fix(docs): make compare_interactive explicitly display trame viewer"
```

---

## Task 2: Move `compare_interactive(...)` inside `with output:` in `_on_run`

**Files:**
- Modify: `docs/splats/compare_maskclip_talk2dino.ipynb` — cell `query-testing`, the `_on_run` callback.

- [ ] **Step 1: Locate `_on_run` in the `query-testing` cell**

Current body:

```python
def _on_run(b):
    pos = [p.strip() for p in pos_widget.value.split(",") if p.strip()]
    neg = [n.strip() for n in neg_widget.value.split(",") if n.strip()]
    with output:
        output.clear_output(wait=True)
        colors_mc  = splatter_mc.query_mesh(positive_queries=pos, negative_queries=neg,
                                            temperature=temp_widget.value)
        colors_t2d = splatter_t2d.query_mesh(positive_queries=pos, negative_queries=neg,
                                             temperature=temp_widget.value)
    compare_interactive(splatter_mc, splatter_t2d, colors_mc, colors_t2d,
                        query_label=pos_widget.value)
```

- [ ] **Step 2: Replace with the corrected version**

```python
def _on_run(b):
    pos = [p.strip() for p in pos_widget.value.split(",") if p.strip()]
    neg = [n.strip() for n in neg_widget.value.split(",") if n.strip()]
    with output:
        output.clear_output(wait=True)
        print(f"Running: + {pos}  − {neg}  (T={temp_widget.value:.3f})")
        colors_mc  = splatter_mc.query_mesh(positive_queries=pos, negative_queries=neg,
                                            temperature=temp_widget.value)
        colors_t2d = splatter_t2d.query_mesh(positive_queries=pos, negative_queries=neg,
                                             temperature=temp_widget.value)
        compare_interactive(splatter_mc, splatter_t2d, colors_mc, colors_t2d,
                            query_label=pos_widget.value)
```

Two changes:
- `compare_interactive(...)` now lives inside the `with output:` block — its `display()` call lands in the `Output` widget.
- Added a `print(...)` status line so the user gets immediate feedback (queries take seconds; without feedback the UI looks dead).

- [ ] **Step 3: Re-run the cell to rebind the button handler**

Run the `query-testing` cell. Expected: the same control panel renders. No exception.

- [ ] **Step 4: Commit**

```bash
git add docs/splats/compare_maskclip_talk2dino.ipynb
git commit -m "fix(docs): render query viewer inside Output widget in compare notebook"
```

---

## Task 3: End-to-end manual verification

**Files:** none modified — verification only.

> Notebook viz cannot be unit-tested; verification is manual rendering in Jupyter/VS Code with the `nerfstudio` env kernel.

- [ ] **Step 1: Restart kernel, run top-to-bottom through `inspect`**

Run cells `imports` → `setup-splatters` → `gen-mc-feeder` → `setup-t2d` → `gen-t2d-extra` → `inspect`.
Expected: both splatters load, missing queries generate (if any), ply listings print.

- [ ] **Step 2: Confirm trame backend works for static RGB cells**

Run `rgb-compare` and the cell with id `4564cfd6` (talk2dino mesh RGB). Expected: an interactive trame viewer renders for each (drag rotates). This isolates trame-vs-callback: if these fail, the issue is environmental (port forwarding, kernel), not this fix.

- [ ] **Step 3: Run the `query-testing` cell**

Expected: widget panel renders (preset dropdown, positive/negative text inputs, temperature slider, **Run query** button, empty `Output` widget below).

- [ ] **Step 4: Preset = "feeder" → click Run query**

Expected inside the `Output` widget, in order:
1. Status line: `Running: + ['feeder']  − ['ground', 'leaves', 'rocks']  (T=0.050)`
2. Side-by-side trame viewer (1600×800), left subplot titled `maskclip — feeder`, right titled `talk2dino — feeder`.
3. Dragging the left view rotates the right view in sync (linked views).
4. Mouse-wheel zoom works.

- [ ] **Step 5: Switch preset to "tree / bark" → click Run query**

Expected: prior viewer is disposed (`clear_output(wait=True)`), new status line + new viewer appears. Titles update to `... — tree, bark` (whatever the positive field shows).

- [ ] **Step 6: Edit positives manually + drop temperature**

Set positives to `feeder`, negatives to `tree, sky`, temperature slider to `0.010`. Click Run query.
Expected: new colors render (the lower temperature sharpens the similarity peak — the feeder should appear visibly more saturated).

- [ ] **Step 7: If any step fails**

- If nothing renders but Step 2's static viewers did render → callback fix incomplete; re-check that `display(viewer)` is inside `with output:`.
- If Step 2's static viewers also fail → environmental (trame proxy / VS Code port forwarding). See README troubleshooting note; this plan does not address that.
- If `clim=[0.0, 1.0]` saturates and you see only black → check `colors_mc[:, 0]` range; not a bug in this fix.

- [ ] **Step 8: Final commit (if any tweaks during verification)**

```bash
git status
# only if there are uncommitted tweaks from verification:
git add docs/splats/compare_maskclip_talk2dino.ipynb
git commit -m "fix(docs): verification polish for query-testing viz"
```

---

## Self-Review Checklist (already applied)

- Spec coverage: bug ("viz never renders") → Task 1 (display the widget) + Task 2 (route it to Output). Stretch goal "interactive, allows scene exploration" → trame backend with linked subplots, drag-rotate, scroll-zoom (Task 3 verifies).
- No placeholders: every code block is the literal target source.
- Type consistency: `compare_interactive(splatter_mc, splatter_t2d, colors_mc, colors_t2d, query_label: str)` signature unchanged between Task 1 and Task 2.
- No source code under `collab_splats/` touched — pure notebook fix, low blast radius.
