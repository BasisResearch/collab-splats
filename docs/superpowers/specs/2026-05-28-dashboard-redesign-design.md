# Dashboard Redesign — Design Spec

**Date:** 2026-05-28  
**Status:** Approved for planning  
**Scope:** Full redesign of `collab_splats/dashboard/` — replaces existing `semantics.py` dashboard with a unified pipeline-integrated app

---

## Overview

Replace the current single-mode `collab-dashboard semantics` with a unified 5-tab pipeline app (`collab-dashboard app`) that covers the full reconstruction workflow: video preprocessing → semantics exploration → reconstruction → visualization/comparison → localization.

**Tech stack:** Panel + param (unchanged). `panel-pyvista` for interactive 3D viewers. Subprocess for reconstruction execution.

---

## Architecture

### File Structure

```
collab_splats/dashboard/
  __init__.py
  __main__.py          ← adds "app" mode; "semantics" becomes alias → "app"
  app.py               ← App class: composes panes into MaterialTemplate, owns AppState
  state.py             ← AppState (shared data bus between panes)
  panes/
    __init__.py
    preprocess.py      ← PreprocessPane
    semantics.py       ← SemanticsPane
    reconstruct.py     ← ReconstructPane
    visualize.py       ← VisualizePane
    localize.py        ← LocalizePane
```

Old `semantics.py`, `config_panel.py`, `video_discovery.py` deleted. Old `ConfigPanel` (Splatter YAML format) dropped — the new pipeline uses `run_config.yaml`.

### AppState

```python
@dataclass
class AppState(param.Parameterized):
    output_dir: Path | None = None          # set by preprocess or "load existing"
    video_path: Path | None = None          # set by preprocess (None if loaded existing)
    frames: list[np.ndarray] = field(default_factory=list)
    feedforward_result: FeedforwardResult | None = None
    feature_maps_path: Path | None = None   # path to <output>/features/<extractor>/<extractor>.zarr
    lifted_features_path: Path | None = None  # path to <output>/<backend>/semantics/<extractor>/features.zarr
```

Panes observe `AppState` via `param.watch`. Downstream panes auto-enable when upstream data arrives (e.g., ReconstructPane enables when `state.output_dir` is set).

### Session Modes

Two ways to start a session — controlled from a persistent sidebar section:
- **New from video:** user provides video path, pipeline runs from scratch
- **Load existing results:** user browses `/workspace/outputs/` for a prior `run_config.yaml`-containing directory; `AppState` populated from loaded zarr + config

### Global Progress/Log Strip

Persistent collapsible panel at the bottom of the main area (not per-tab). Every long-running operation across all panes writes to it:
- Operation name + progress bar (0–100%)
- Last 20 log lines (scrollable)
- Status indicator: idle / running / error

All background threads and subprocesses route their progress/log output through a shared `OperationLog` helper (single queue, Panel periodic callback polls it).

---

## Launch

```bash
collab-dashboard app --base-dir /workspace/outputs --port 7860
# "semantics" still accepted as alias → redirects to app mode
```

---

## Tab 1: Preprocess

**Purpose:** Extract keyframes from video; inspect frame quality metrics; confirm frame set before reconstruction.

**Layout:** Video player left / controls right (side-by-side), stacked metrics below, scrollable frame strip at bottom.

### Video Player
- Load from path or file upload widget
- Play / pause / seek scrubber
- Current timestamp display

### Frame Selection Controls (right column)
- Method dropdown: `fps` | `optical_flow`
- `n_frames` integer slider
- Window range: start % → end % (float sliders, default 0–100%)
- Min disparity slider (visible only when `optical_flow` selected)
- **Extract Frames** button → runs `sample_frames_fps` or `sample_frames_optical_flow` in background thread

### Stacked Metrics Panel
Four time-series plots sharing x-axis (frame index), computed alongside frame extraction:
- **Optical flow magnitude** — inter-frame pixel displacement
- **Rotation (degrees)** — estimated inter-frame camera rotation (homography)
- **Translation** — inter-frame camera translation magnitude
- **Histogram similarity** — color histogram overlap [0–1]

Selected frames shown as vertical green bands across all four plots. Hovering a band highlights the corresponding thumbnail in the frame strip.

### Frame Strip
Horizontally scrollable thumbnails of selected frames. Clicking a thumbnail seeks the video player to that frame.

### AppState writes
`state.frames`, `state.video_path`, `state.output_dir` (set to configured output path)

---

## Tab 2: Semantics

**Purpose:** Extract 2D feature maps from a selected frame using multiple extractors simultaneously; compare outputs side-by-side; text-query queryable extractors.

**Layout:** 4-column grid — original frame always leftmost, up to 3 extractor output columns. Each output column independently selectable.

### Frame Selector
Integer slider + prev/next buttons to select the current frame from `state.frames`.

### Extractor Columns (up to 3)
Each column:
- Method dropdown populated from `BaseFeatureExtractor._registry`
- **Run** button → runs extractor in background thread → displays PCA-reduced feature map as RGB overlay
- Empty columns show a **+** add button

### Text Query Bar
Appears below the grid when any active column uses a `BaseQueryableExtractor` (talk2dino, maskclip).
- Text input for query prompt
- **Query** button → calls `.query(text)` on active queryable extractors → updates their column output
- Result: similarity heatmap overlay on the frame

### AppState writes
`state.feature_maps_path` (path to cached zarr — not loaded into RAM)

---

## Tab 3: Reconstruct

**Purpose:** Configure and run the reconstruction pipeline (`docs/examples/reconstruct.py`) for the current session's video/frames.

**Layout:** Config column left / log stream right (split panel). Progress bar full-width at top.

### Config Column
- **Backend** dropdown: `vggt_omega` | `vggtx` | `mapanything`
- **Bundle Adjustment** toggle + collapsible accordion:
  - iterations (int)
  - damping (float)
- **Loop Closure** toggle + collapsible accordion:
  - similarity threshold (float)
  - max submap size (int)
- **Clean** section:
  - outlier removal toggle
  - voxel size (float, null = adaptive)
  - confidence threshold (float, null = disabled)
- **Run Reconstruction** button

### Execution
Spawns `docs/examples/reconstruct.py` as subprocess with config written to a temp YAML. Streams stdout/stderr to log panel (right column) and global progress strip. Kill button appears while running.

### Log Stream (right column)
Scrollable textarea, colour-coded: green = success lines, white = info, red = errors.

### AppState writes
`state.feedforward_result` (loaded after subprocess completes), `state.output_dir`

---

## Tab 4: Visualize

**Purpose:** Side-by-side interactive 3D comparison of two independently selected scenes/backends. Each scene independently controls its representation (PCD / Mesh / Similarity).

**Layout:** Two scene panels (Scene A, Scene B) side by side. Shared query bar appears at top when both scenes are in Similarity mode.

### Per-Scene Panel (A and B identical structure)
- Dataset dropdown (discovered from `/workspace/outputs/`) + backend dropdown. Scene A auto-populates from `state.output_dir` + `state.feedforward_result` when a session is active — user can override.
- **Load** button — does not load anything by default (even with auto-suggest, explicit Load required)
- View type toggle: **PCD | Mesh | Similarity**
  - **Mesh** greyed (badge: "no mesh.ply") until `<output_dir>/<backend>/mesh/mesh.ply` exists
- `panel-pyvista` interactive viewer (rotate/zoom/pan in browser)
- Reset camera + snapshot buttons

### PCD Mode
Renders `feedforward.zarr` points + colors. Camera frustums overlay (toggle). Point size slider.

### Mesh Mode
Loads `mesh.ply` into PyVista mesh renderer.

### Similarity Mode (single scene)
Inline query controls appear below viewer:
- Extractor dropdown (only extractors with lifted features available for this backend)
- Text query input
- **Query** button → colors pointcloud by cosine similarity score using **viridis** colormap

### Shared Query Bar (both scenes in Similarity mode)
Appears at top of Visualize tab when both Scene A and Scene B are in Similarity mode:
- One shared text query input
- Per-scene extractor dropdown (A and B may use different extractors)
- **Query Both** button → fires on both viewers simultaneously, both update to viridis similarity coloring

Viridis colormap is always used for similarity — never substituted.

---

## Tab 5: Localize

**Purpose:** Localize a query image within a known reconstruction. Two upload modes; visualize correspondences and estimated camera pose.

**Layout:** Upload panel left / result viewer right (PyVista scene + correspondence image).

### Upload Modes (toggle)
**Mode A — Out-of-sample frame (same video):**
- Frame index input or video seek to select a frame not included in the processed set
- Runs `CameraLocalizer` against the loaded reconstruction

**Mode B — Arbitrary RGB image:**
- File upload widget (jpg/png)
- Runs `CameraLocalizer` against the loaded reconstruction
- If inlier count below threshold → reject with explanation ("insufficient inliers: N found, threshold M") and display best candidate reference frames tried

### Results (both modes)
On success:
- Correspondence visualization: side-by-side query frame / best reference frame with inlier (green) / outlier (red) keypoint connection lines — uses `LocalizationResult.plot_correspondences()`
- PyVista viewer showing the full scene pointcloud + localized camera frustum (distinct color)
- Estimated pose stats: inlier count, reprojection error

On failure (Mode B only):
- Rejection message with inlier count
- Gallery of top-K candidate reference frames (by global retrieval score)

### AppState reads
`state.feedforward_result` (scene to localize within)

---

## Migration

Old `collab_splats/dashboard/semantics.py`, `config_panel.py`, `video_discovery.py` deleted. Old `Splatter` training tab dropped (uses deprecated pipeline). `collab-dashboard semantics` CLI arg accepted as deprecated alias → logs warning, launches `app` mode.

---

## Implementation Phases

Each phase is a separate spec → plan → implementation cycle:

| Phase | Scope | Depends on |
|---|---|---|
| 1 | App skeleton + AppState + PreprocessPane | — |
| 2 | SemanticsPane | Phase 1 |
| 3 | ReconstructPane | Phase 1 |
| 4 | VisualizePane (PCD + Mesh + Similarity) | Phase 1, Phase 3 |
| 5 | LocalizePane | Phase 1, Phase 3 |

Phase 1 ships the working skeleton with one functional pane. Remaining tabs show "coming soon" placeholders until their phase ships.

---

## Out of Scope

- Dataset YAML config editing (old `ConfigPanel`) — dropped
- Splatter / nerfstudio training tab — dropped
- `video_discovery.py` fieldwork path scanner — replaced by generic output dir browser
- Mesh quality evaluation — ongoing research, Mesh sub-tab exists but defers to TSDF/Poisson until better methods settled
