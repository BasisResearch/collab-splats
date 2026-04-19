# Semantics Dashboard Redesign

**Date:** 2026-04-16  
**Branch:** tlb-semantics-refactor  
**Status:** Approved for implementation

## Context

The current semantics dashboard (`collab_splats/dashboard/semantics.py`) is a 4-tab Gradio app for semantic exploration (PCA features, MobileSAM segmentation, Talk2DINO heatmaps). It has two problems:

1. **NumPy crash on launch** — `collab_splats/__init__.py` eagerly imports `RadegsModel → torch`, which triggers a NumPy 1.x/2.x ABI conflict at import time, crashing the dashboard before it starts.
2. **Gradio limitations** — the framework is designed for linear input→output demos. Adding stateful workflow (base dir browser → video picker → per-video config → training launcher) breaks `gr.State` and produces blocky layouts.

The team already uses Panel/HoloViz with `param.Parameterized` in `collab-data/collab_data/data_dashboard/` — a mature dashboard with sidebar navigation, YAML-driven config, and subprocess-based long-running operations. This redesign ports the semantics dashboard to Panel, following the same patterns.

## Goals

1. Fix the NumPy crash (decouple torch imports from dashboard startup)
2. Add base directory browser with auto-discovery of fieldwork videos
3. Add per-video YAML config editor (read/write `docs/splats/configs/datasets/`)
4. Add Splatter training launcher with live log output
5. Port existing semantic exploration (PCA, segmentation, Talk2DINO) to Panel tabs
6. Add dashboard tests

## Architecture

```
SemanticsDashboard(param.Parameterized)
├── params: base_dir, selected_species, selected_date, selected_video
├── Sidebar (MaterialTemplate, 400px)
│   ├── base_dir TextInput  (default: /workspace/fieldwork-data)
│   ├── species Select      ← populated on base_dir change
│   ├── date Select         ← populated on species change
│   ├── video Select        ← populated on date change
│   └── ConfigPanel         ← loaded on video change
│       ├── method Dropdown
│       ├── sfm_tool Dropdown
│       ├── frame_proportion FloatSlider
│       ├── min_frames IntSlider
│       ├── [Save YAML] button
│       └── [Reset to base] button
└── Main (MaterialTemplate)
    ├── status_pane (HTML)
    ├── loading_modal (CSS overlay, same pattern as collab-data)
    └── Tabs
        ├── "Training" tab
        │   ├── output_path TextInput
        │   ├── [Launch Splatter] button
        │   ├── training_log (TextAreaInput, live-updating via periodic callback)
        │   └── [Stop] button
        └── "Explore" tab
            ├── fps_slider + [Extract Frames] + frame_gallery + frame_slider
            ├── extractor_dd + device_dd + [Extract Features] → PCA overlay
            ├── seg_strategy_dd + seg_device_dd + [Segment] → mask overlay
            └── text_pairs + method_dd + temp_slider + [Generate Heatmaps]
```

## File Structure

```
collab_splats/dashboard/
├── __main__.py          ← update: add --base-dir, --port CLI args; use panel serve
├── __init__.py
├── semantics.py         ← REPLACE: Gradio → Panel SemanticsDashboard class
├── video_discovery.py   ← NEW: scan fieldwork-data/, return species/date/video tree
└── config_panel.py      ← NEW: ConfigPanel(param.Parameterized), YAML load/save

docs/splats/configs/
├── base.yaml            ← unchanged (source of defaults)
└── datasets/            ← per-video YAMLs written here on Save

tests/dashboard/
├── test_video_discovery.py   ← NEW
├── test_config_panel.py      ← NEW
└── test_semantics_smoke.py   ← NEW
```

## Components

### `video_discovery.py` — pure function, no Panel dependency

```python
def discover_videos(base_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """Scan {base_dir}/{species}/{date}/SplatsSD/*.MP4
    Returns {species: {date: [video_paths]}}
    """
```

No side effects, fully testable without Panel.

### `config_panel.py` — `ConfigPanel(param.Parameterized)`

```python
class ConfigPanel(param.Parameterized):
    method = param.Selector(objects=["rade-gs", "rade-features", "splatfacto", "feature-splatting"])
    sfm_tool = param.Selector(objects=["hloc", "colmap"])
    frame_proportion = param.Number(default=0.25, bounds=(0.01, 1.0))
    min_frames = param.Integer(default=100, bounds=(10, 1000))

    def load_from_yaml(self, yaml_path: Path | None) -> None: ...
    def save_to_yaml(self, yaml_path: Path) -> None: ...
    def to_splatter_config(self, file_path: Path, output_path: Path) -> SplatterConfig: ...
    def panel(self) -> pn.Column: ...  # returns widget layout
```

YAML write only stores dataset-level overrides (fields that differ from base). Full config is always read via `ConfigLoader` which merges base + dataset.

### `semantics.py` — `SemanticsDashboard(param.Parameterized)`

Watch chain:
```
base_dir        → _refresh_species()
selected_species → _refresh_dates()
selected_date    → _refresh_videos()
selected_video   → _load_video()   # resolves path, loads ConfigPanel, loads first frame
```

Training launch:
```python
# [Launch Splatter] click
proc = subprocess.Popen(["python", "-m", "collab_splats", ...], stdout=PIPE, stderr=STDOUT)
self._training_process = proc
self._poll_cb = pn.state.add_periodic_callback(self._poll_training_log, period=500)  # ms

def _poll_training_log(self):
    line = self._training_process.stdout.readline()
    if line:
        self.training_log.value += line.decode()
    if self._training_process.poll() is not None:
        self._poll_cb.stop()
        self._poll_cb = None
```

## Data Flow

```
base_dir change
  → discover_videos(base_dir) → populate species Select

species change
  → filter discover results → populate date Select

date change
  → filter for species/date → populate video Select (display: video_id stem)

video change
  → file_path = base_dir / species / date / SplatsSD / video_id
  → yaml_path = docs/splats/configs/datasets/{species}_date-{date}_video-{video_id}.yaml
  → ConfigPanel.load_from_yaml(yaml_path)   # merges base + dataset via ConfigLoader
  → load first frame into Explore tab state

[Save YAML] click
  → ConfigPanel.save_to_yaml(yaml_path)     # writes dataset-override only
  → status: "Saved config for {video_id}"

[Launch Splatter] click
  → ConfigPanel.to_splatter_config(file_path, output_path)
  → subprocess.Popen → poll stdout → training_log updates every 500ms

[Stop] click
  → self._training_process.terminate()
```

## NumPy Fix

Remove eager model imports from `collab_splats/__init__.py`:

```python
# BEFORE (causes NumPy ABI crash at dashboard import time):
from collab_splats.models.rade_gs_model import RadegsModel, RadegsModelConfig
from collab_splats.models.rade_features_model import RadegsFeaturesModel, ...

# AFTER (lazy — only import when actually training):
# Models imported directly by callers that need them
```

`collab_splats/__init__.py` keeps only `Splatter`, `SplatterConfig`. Callers that need model classes import them directly.

## Error Handling

All watch callbacks and button handlers wrapped in `try/except`. Errors → `status_pane` as `<p style='color:red'>...</p>`. Consistent with collab-data pattern.

| Failure | Behaviour |
|---|---|
| base_dir not found | status: warning, selects cleared |
| No videos found | status: "No videos found in {base_dir}" |
| YAML save fails | status: error, no file written |
| Training process exits non-zero | training_log shows stderr, status: ❌ |
| Frame extraction fails | status: error, gallery stays empty |
| NumPy/torch import | fixed at source — no runtime fallback needed |

## Launch

```bash
# Current (Gradio):
python -m collab_splats.dashboard semantics

# New (Panel):
panel serve collab_splats/dashboard/__main__.py \
  --dev \
  --args --base-dir /workspace/fieldwork-data --port 7860
```

Entry point in `pyproject.toml` updated to wrap `panel serve`.

## Tests

### `tests/dashboard/test_video_discovery.py`
- Create temp dir with `{species}/{date}/SplatsSD/*.MP4` structure
- Assert `discover_videos()` returns correct nested dict
- Assert empty base_dir returns `{}`
- Assert dirs without `SplatsSD/` are ignored

### `tests/dashboard/test_config_panel.py`
- Load existing dataset YAML → assert param values match file
- Load `None` (no YAML) → assert param values match base.yaml defaults
- Save config → assert written YAML contains only overridden fields
- Round-trip: load → mutate param → save → reload → assert values preserved
- `to_splatter_config()` → assert returned dict has correct keys

### `tests/dashboard/test_semantics_smoke.py`
- `SemanticsDashboard(base_dir=tmp_path)` → no exception
- `dashboard.create_layout()` → returns `MaterialTemplate` instance
- Trigger `_refresh_species()` with valid base_dir → species list non-empty
- Trigger `_load_video()` with known video → ConfigPanel populated

All tests run without CUDA, without browser, without panel serve. Add `tests/dashboard/` to `pytest` default paths in `pyproject.toml`.

## Verification Checklist

1. `python -c "from collab_splats.dashboard import semantics"` — no NumPy warning
2. `make test` passes (including new dashboard tests)
3. `mypy collab_splats/dashboard/` passes
4. `panel serve` launches app at localhost without crash
5. Change base dir → selects populate correctly
6. Select video with existing YAML → config widgets show file values
7. Select video without YAML → config widgets show base defaults
8. [Save YAML] → file appears in `docs/splats/configs/datasets/`
9. [Launch Splatter] → training_log streams output, [Stop] terminates process
10. Extract frames in Explore tab → gallery populates
11. PCA feature extraction → overlay renders
12. `ruff check` and `black` pass
