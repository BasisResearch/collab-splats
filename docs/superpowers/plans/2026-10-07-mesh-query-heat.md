# Mesh Query Heat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mesh viewer queries and label clicks draw a continuous per-vertex heat overlay instead of thresholded points; the label list ranks words by probability mass.

**Architecture:** `Viewer.show_heat` sends one vertex-colored sub-mesh (`<name>/heat`) per query; the GPU blends vertex colors across faces. `Viewer.add_label_list` takes words + weights + an `on_select` callback. `ocr_lens_viewer.py` wires both and drops kNN smoothing.

**Tech Stack:** viser 1.0.29 (`add_mesh_trimesh`), trimesh 4.12, matplotlib 3.10 colormaps, numpy 2.1.

Spec: `docs/superpowers/specs/2026-10-07-mesh-query-heat-design.md`

---

## File map

| File | Responsibility |
|---|---|
| `collab_splats/viewer.py` | `show_heat` (replaces `highlight`), `add_label_list(words, weights, on_select)`, `_compact_count` |
| `tests/test_viewer.py` | heat + label-list tests (replace `highlight` / label-count tests) |
| `docs/examples/ocr_lens_viewer.py` | heat search, mass-ranked list, smoothing / lock / Apply removed |

Conventions (CLAUDE.md + memory): docstring `"""` on own lines, `Args:` for public; block comments one plain line; blank line around blocks; imports at top, isort groups. `viewer.py` is in `tests/test_docstring_contract.py` `MODULES`.

Python: `/opt/venv/reconstruction/bin/python`. Commit with `git commit --only <paths>` (shared index).

---

### Task 1: `Viewer.show_heat` replaces `highlight`

**Files:**
- Modify: `collab_splats/viewer.py` (imports; `__init__` state; `highlight` at ~226-250 → `show_heat`; docstrings of module, class, `add_mesh`)
- Test: `tests/test_viewer.py` (replace `test_highlight_overlays_the_masked_vertices_without_resending_the_mesh`; rename + adapt `test_clicks_still_pick_after_a_highlight`)

- [ ] **Step 1: Write the failing tests**

Replace `test_highlight_overlays_the_masked_vertices_without_resending_the_mesh` with:

```python
def _capture_heat(viewer, monkeypatch):
    """Record (name, mesh) for every add_mesh_trimesh; handles get a no-op remove."""
    sent = []

    def add(name, mesh):
        sent.append((name, mesh))
        return SimpleNamespace(remove=lambda: None)

    monkeypatch.setattr(viewer.server.scene, "add_mesh_trimesh", add)
    return sent


def test_show_heat_keeps_faces_reaching_the_floor_on_their_used_vertices(viewer, monkeypatch):
    vertices, faces, colors = _quad()
    viewer.add_mesh("quad", vertices, faces, colors)
    sent = _capture_heat(viewer, monkeypatch)

    # Only vertex 3 reaches the floor: face [1, 3, 2] stays, face [0, 1, 2] goes
    viewer.show_heat("quad", np.array([0.0, 0.1, 0.2, 0.9]), floor=0.5)

    assert [name for name, _ in sent] == ["quad/heat"]
    heat = sent[0][1]
    assert heat.vertices.tolist() == vertices[[1, 2, 3]].tolist()
    assert heat.faces.tolist() == [[0, 2, 1]]


def test_show_heat_colors_follow_the_score_with_opacity_alpha(viewer, monkeypatch):
    viewer.add_mesh("quad", *_quad())
    sent = _capture_heat(viewer, monkeypatch)

    viewer.show_heat("quad", np.array([0.6, 0.7, 0.8, 0.9]), floor=0.5, opacity=0.5)

    rgba = sent[0][1].visual.vertex_colors
    # Viridis runs dark purple to yellow: brightness rises with the score
    assert np.all(np.diff(rgba[:, :3].astype(int).sum(axis=1)) > 0)
    assert rgba[:, 3].tolist() == [128] * 4


def test_show_heat_offset_lifts_along_normals(viewer, monkeypatch):
    vertices, faces, colors = _quad()
    viewer.add_mesh("quad", vertices, faces, colors)
    sent = _capture_heat(viewer, monkeypatch)

    viewer.show_heat("quad", np.ones(4), floor=0.5, offset=0.5)

    # Flat quad, normals along +z, median edge 1: every vertex moves 0.5 in z
    assert np.allclose(np.abs(sent[0][1].vertices[:, 2]), 0.5)
    assert np.allclose(sent[0][1].vertices[:, :2], vertices[:, :2])


def test_show_heat_replaces_clears_and_sends_nothing_below_the_floor(viewer, monkeypatch):
    viewer.add_mesh("quad", *_quad())
    removed = []

    def add(name, mesh):
        return SimpleNamespace(remove=lambda: removed.append(name))

    monkeypatch.setattr(viewer.server.scene, "add_mesh_trimesh", add)

    # A second call drops the first overlay
    viewer.show_heat("quad", np.ones(4), floor=0.5)
    viewer.show_heat("quad", np.ones(4), floor=0.5)
    assert removed == ["quad/heat"]

    # All below the floor: previous overlay gone, nothing new kept
    viewer.show_heat("quad", np.zeros(4), floor=0.5)
    assert removed == ["quad/heat"] * 2
    assert "quad" not in viewer.heats

    # None clears
    viewer.show_heat("quad", np.ones(4), floor=0.5)
    viewer.show_heat("quad", None, floor=0.5)
    assert removed == ["quad/heat"] * 3
    assert "quad" not in viewer.heats
```

Replace `test_clicks_still_pick_after_a_highlight` with:

```python
def test_clicks_still_pick_after_show_heat(viewer, monkeypatch):
    monkeypatch.setattr(viewer, "mesh_clicks", {})
    vertices, faces, colors = _quad()
    picked = []

    viewer.add_mesh("quad", vertices, faces, colors)
    viewer.on_click("quad", picked.append)
    _capture_heat(viewer, monkeypatch)
    viewer.show_heat("quad", np.ones(4), floor=0.5)

    event = SimpleNamespace(ray_origin=(0.9, 0.9, 1.0), ray_direction=(0.0, 0.0, -1.0))
    viewer._dispatch_click(event)
    assert picked == [3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_viewer.py -k "show_heat" -v`
Expected: FAIL, `AttributeError: 'Viewer' object has no attribute 'show_heat'`

- [ ] **Step 3: Implement**

Imports (third-party group, alphabetical):

```python
import matplotlib
import numpy as np
import open3d as o3d
import trimesh
import viser
import viser.transforms as viser_tf
```

`__init__`, after `self.label_lists`:

```python
        self.heats: dict[str, viser.GlbHandle] = {}
```

(`add_mesh_trimesh` returns `GlbHandle` in viser 1.0.29.)

Replace `highlight` with:

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
        """
        Overlay a per-vertex score on a mesh as a vertex-colored `<name>/heat` sub-mesh.

        - faces with any vertex at or above `floor` are drawn; colors blend across each face
        - viridis over the drawn vertices' scores, min-max normalized
        - the mesh itself is never re-sent: only the drawn sub-mesh goes over the wire

        Args:
            name: mesh node name, already added with add_mesh.
            scores: (V,) score per mesh vertex; None clears the overlay.
            floor: score a face needs on one vertex to be drawn.
            opacity: overlay alpha, 0-1.
            offset: shift along vertex normals, in median edge lengths, against z-fighting.
        """
        # Drop the previous overlay
        if name in self.heats:
            self.heats.pop(name).remove()

        if scores is None:
            return

        # Faces reaching the floor, remapped onto the vertices they use
        vertices, faces, _ = self.meshes[name]
        keep = (scores[faces] >= floor).any(axis=1)

        if not keep.any():
            return

        used, inverse = np.unique(faces[keep], return_inverse=True)
        sub_faces = inverse.reshape(-1, 3)

        # Viridis over the drawn scores, alpha at opacity
        values = scores[used].astype(np.float64)
        span = values.max() - values.min()
        normalized = (values - values.min()) / span if span > 0 else np.zeros_like(values)
        rgba = (matplotlib.colormaps["viridis"](normalized) * 255).astype(np.uint8)
        rgba[:, 3] = round(255 * opacity)
        heat = trimesh.Trimesh(vertices[used], sub_faces, vertex_colors=rgba, process=False)

        # Lift off the base surface along vertex normals
        if offset > 0:
            shift = offset * np.median(heat.edges_unique_length) * heat.vertex_normals
            heat.vertices = heat.vertices + shift

        self.heats[name] = self.server.scene.add_mesh_trimesh(f"{name}/heat", heat)
```

Docstring wording: module bullet "meshes add vertex picking, label lists and a heat overlay"; `add_mesh` summary "Upsert a named mesh; picking, labels and the heat overlay index its vertices."

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_viewer.py -v -p no:randomly`
Expected: all PASS. `add_label_list` still names `self.highlight` inside its click lambdas, never invoked by the old label test; Task 2 replaces it.

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/viewer.py tests/test_viewer.py -m "feat(viewer): show_heat — continuous per-vertex heat overlay replaces highlight points"
```

---

### Task 2: `add_label_list` takes weighted words + `on_select`

**Files:**
- Modify: `collab_splats/viewer.py` (`add_label_list` ~188-224; new module-level `_compact_count`; `typing` import gains `Sequence`)
- Test: `tests/test_viewer.py` (replace `test_add_label_list_counts_labels_most_common_first_and_skips_empty`)

- [ ] **Step 1: Write the failing test**

```python
def test_add_label_list_ranks_by_weight_and_hands_words_to_on_select(viewer):
    viewer.add_mesh("quad", *_quad())
    selected = []
    words = ["rock", "tree", "sky"]
    weights = np.array([12.0, 28_718.4, 950.0])

    viewer.add_label_list("quad", words, weights, selected.append, top_n=5)
    _, buttons = viewer.label_lists["quad"]
    assert [b.label for b in buttons] == ["Clear", "tree (~28.7k)", "sky (~950)", "rock (~12)"]

    # Word buttons hand their word over; Clear hands None
    buttons[1]._impl.update_cb[0](None)
    buttons[0]._impl.update_cb[0](None)
    assert selected == ["tree", None]

    # Re-adding replaces the list rather than stacking a second one
    viewer.add_label_list("quad", words, weights, selected.append, top_n=1)
    _, buttons = viewer.label_lists["quad"]
    assert [b.label for b in buttons] == ["Clear", "tree (~28.7k)"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_viewer.py -k add_label_list -v`
Expected: FAIL, `TypeError` (unexpected positional args)

- [ ] **Step 3: Implement**

`from typing import Callable, Optional, Sequence`

Replace `add_label_list`:

```python
    def add_label_list(
        self,
        name: str,
        words: Sequence[str],
        weights: np.ndarray,
        on_select: Callable[[Optional[str]], None],
        top_n: int = 20,
    ) -> None:
        """
        GUI buttons for a mesh's heaviest words; a click hands the word to `on_select`.

        - "Clear" hands None; re-adding replaces the mesh's earlier list
        - each button shows its weight rounded, e.g. `grass (~28.7k)`

        Args:
            name: mesh node name; titles the folder.
            words: candidate words.
            weights: (len(words),) weight per word, e.g. summed probability over vertices.
            on_select: called with the clicked word, or None from "Clear".
            top_n: words listed, heaviest first.
        """
        # Heaviest words first
        order = np.argsort(-np.asarray(weights), kind="stable")[:top_n]

        # Replace an earlier list for this mesh
        if name in self.label_lists:
            self.label_lists[name][0].remove()

        folder = self.server.gui.add_folder(f"Labels: {name}")

        with folder:
            clear = self.server.gui.add_button("Clear")
            clear.on_click(lambda _: on_select(None))
            buttons = [clear]

            for i in order:
                word = str(words[i])
                button = self.server.gui.add_button(f"{word} (~{_compact_count(float(weights[i]))})")
                button.on_click(lambda _, word=word: on_select(word))
                buttons.append(button)

        self.label_lists[name] = (folder, buttons)
```

Module-level, after the class (own `########` section "Helpers"):

```python
def _compact_count(value: float) -> str:
    """
    Short count for a button label: 950 -> "950", 28718 -> "28.7k".
    """
    if value < 1000:
        return f"{value:.0f}"

    return f"{value / 1000:.1f}k"
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_viewer.py tests/test_docstring_contract.py tests/test_import_style.py -v -p no:randomly`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/viewer.py tests/test_viewer.py -m "feat(viewer): label list ranks weighted words and hands clicks to on_select"
```

---

### Task 3: `ocr_lens_viewer.py` — heat search, mass-ranked list, no smoothing

**Files:**
- Modify: `docs/examples/ocr_lens_viewer.py` (module docstring 1-16; imports 19, 38; `main` docstring; `main` body from "Scene, probe chart, smoothing slider" to end)

- [ ] **Step 1: Imports and docstrings**

- delete `import threading`
- `from collab_splats.semantics.lifting import lift_features`
- add `from typing import Optional` (stdlib group)
- module docstring: summary "Probe a scene's mesh for the OCR lens's words as a continuous heat overlay."; drop the smoothing bullet; query bullet "query: comma-separated words; summed probability draws as heat, faces under min p hidden"; add "label list: words ranked by probability mass (expected vertex count); a click queries that word"
- `main` docstring: "Serve the mesh with a mass-ranked word list, a word query drawn as heat and a click probe."

- [ ] **Step 2: Delete `seen = vertices[observed]`** (only `transfer_features` used it)

- [ ] **Step 3: Replace the GUI block through the end of `main`**

```python
    # Scene, probe chart and query box
    viewer = Viewer(port=args.port)
    viewer.server.gui.configure_theme(control_layout="fixed")
    viewer.add_mesh("mesh", vertices, faces, colors, textured=textured)
    panel = viewer.server.gui.add_html("")
    query = viewer.server.gui.add_text("Query", initial_value="")
    min_prob = viewer.server.gui.add_slider("Query min p", min=0.0, max=1.0, step=0.01, initial_value=0.3)
    search_button = viewer.server.gui.add_button("Search")
    query_note = viewer.server.gui.add_markdown("")
    marker_radius = 0.005 * float(np.linalg.norm(np.ptp(vertices, axis=0)))

    def search(_=None) -> None:
        """
        Draw the summed raw probability of the query words as heat; faces under min p hidden.
        """
        words = [word.strip().lower() for word in query.value.split(",") if word.strip()]
        known = [word for word in words if word in row_of]
        unknown = [word for word in words if word not in row_of]
        query_note.content = f"not in vocabulary: {', '.join(unknown)}" if unknown else ""

        # Empty query clears; unobserved rows are zero so never reach min p
        score = None

        if known:
            columns = [row_of[word] for word in known]
            score = full[:, columns].astype(np.float32).sum(axis=1)

        viewer.show_heat("mesh", score, min_prob.value)

    def select(word: Optional[str]) -> None:
        """
        Put a label-list word in the query box (Clear empties it) and search.
        """
        query.value = word or ""
        search()

    def show(vertex: int) -> None:
        """
        Mark the clicked vertex and chart its top-10 word probabilities in the top left.
        """
        viewer.server.scene.add_icosphere(
            "/probe", radius=marker_radius, color=(255, 0, 255), position=vertices[vertex]
        )

        if not observed[vertex]:
            panel.content = _chart(f"vertex {vertex}: unobserved", [], np.zeros(0))
            return

        values = term_probs[observed_row[vertex]]
        top = np.argsort(-values)[:10]
        panel.content = _chart(f"vertex {vertex}", list(terms[top]), values[top])

    # Words ranked by probability mass: expected vertex count, the total of a click's heat
    mass = full.sum(axis=0, dtype=np.float64)
    viewer.add_label_list("mesh", vocab.words, mass, select)

    # Wire the query and clicks; block
    search_button.on_click(search)
    viewer.on_click("mesh", show)
    viewer.serve_forever()
```

- [ ] **Step 4: Static checks**

Run: `/opt/venv/reconstruction/bin/python -m pyflakes docs/examples/ocr_lens_viewer.py; /opt/venv/reconstruction/bin/python -m isort --check-only --diff docs/examples/ocr_lens_viewer.py collab_splats/viewer.py tests/test_viewer.py; /opt/venv/reconstruction/bin/python -m black --check docs/examples/ocr_lens_viewer.py collab_splats/viewer.py tests/test_viewer.py`
Expected: no unused names, no diffs (pyproject sets line-length 120; never run repo-wide black)

- [ ] **Step 5: Commit**

```bash
git commit --only docs/examples/ocr_lens_viewer.py -m "feat(semantics): ocr_lens_viewer queries draw heat; label list ranked by probability mass"
```

---

### Task 4: Live check on GH010229, defaults, close out

- [ ] **Step 1: Run the real viewer** (stop any prototype on 8080 first)

```bash
cd /workspace/collab-splats && HF_HOME=/workspace/models HF_HUB_OFFLINE=1 /opt/venv/reconstruction/bin/python docs/examples/ocr_lens_viewer.py /workspace/outputs/ocr_viewer/GH010229/vggt_omega --port 8080 --textured
```

Expected log: `viser scene viewer on port 8080`, no traceback. Label list shows `grass (~28.7k)` first.

- [ ] **Step 2: User browser check** — typed `tree` vs label click `tree` identical; opacity honored; z-fighting at offset 0. Rules:
  - alpha ignored → drop `opacity` from `show_heat` (+ its test) and send RGB
  - z-fighting → set `offset` default to smallest value that removes it (try 0.25, 0.5)

- [ ] **Step 3: Full viewer + contract tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_viewer.py tests/test_docstring_contract.py tests/test_import_style.py -p no:randomly -q`
Expected: all pass (compare against `docs/known-test-failures.md` for pre-existing failures)

- [ ] **Step 4: Close out** — append entry to `docs/superpowers/CHANGELOG.md`, remove the `mesh-query-heat` In-Flight line from `CLAUDE.md`, move to Recently Completed (five newest), run `graphify update .`, commit:

```bash
git add -f docs/superpowers/CHANGELOG.md docs/superpowers/plans/2026-10-07-mesh-query-heat.md
git commit --only CLAUDE.md docs/superpowers/CHANGELOG.md docs/superpowers/plans/2026-10-07-mesh-query-heat.md -m "docs(changelog): mesh-query-heat done"
```
