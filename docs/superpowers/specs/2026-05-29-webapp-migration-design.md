# collab-splats webapp migration

**Date:** 2026-05-29  
**Status:** approved  
**Branch:** `feat/webapp` off `refactor/cu121`

---

## Problem

The Panel dashboard has three concrete pain points:
1. **Video player** — Panel HTML pane overhead causes slow decoding and distorted display.
2. **Frame strip** — thumbnails are base64-encoded 120×120 PNGs sent over a Bokeh websocket; blurry and slow.
3. **Mesh/pointcloud visualization** — panel-pyvista has heavy Bokeh model overhead; not fluid.

Underlying cause: Panel's `dynamic=True` tabs destroy and recreate Bokeh models on every tab switch, leading to blank panes, state loss, and general fragility. The viewers built during diagnosis (Three.js PLY viewer, HTML frame gallery) loaded instantly and felt fluid by comparison.

---

## Approach

**FastAPI backend + vanilla HTML/JS frontend**, developed on a separate branch. Panel dashboard stays untouched on `refactor/cu121`. When the new webapp is stable it replaces the Panel entry point.

Key patterns:
- **Media as static files** — frames, video, PLY served via `StaticFiles`; browser caches natively. No base64, no websocket.
- **Long ops via SSE** — frame extraction, reconstruction, semantics use `EventSource` to stream logs and progress in real time. No polling.
- **3D via Three.js** — PLYLoader + OrbitControls (same stack as the diagnostic viewers). GPU-accelerated, no Python rendering.
- **Tab-specific sidebar** — sidebar sections mount/unmount on tab switch via JS; no page reload.

---

## Architecture

```
collab_splats/
  webapp/                    ← NEW module
    __init__.py
    __main__.py              → uvicorn entry point (port 7860)
    app.py                   → FastAPI app; mounts routers + StaticFiles
    state.py                 → SessionState dataclass (output_dir, creator, conf, etc.)
    routers/
      session.py             → POST /session/new, POST /session/load, GET /session/current
      preprocess.py          → GET /preprocess/info, GET /preprocess/extract (SSE)
      semantics.py           → GET /semantics/extract (SSE)
      reconstruct.py         → GET /reconstruct/run (SSE)
      visualize.py           → GET /visualize/pointcloud (redirects to static PLY)
      localize.py            → GET /localize/run (SSE)
    static/
      index.html             → shell: top tab nav + sidebar + main content area
      css/
        app.css              → dark theme (monospace, #0a0a0f bg, #2596be accent)
      js/
        state.js             → client-side session state, tab switch logic
        preprocess.js        → video player, frame strip, extraction SSE consumer
        semantics.js         → extractor selection, SSE log
        reconstruct.js       → creator config, SSE log consumer
        visualize.js         → Three.js PLYLoader + OrbitControls; mesh toggle
        localize.js          → method/extractor selection, SSE log

  dashboard/                 ← UNCHANGED (Panel app stays)
```

**Static file mounts:**
- `/static` → `collab_splats/webapp/static/`
- `/outputs` → `/workspace/outputs/` (frames, video, PLY served directly)

**Entry point:**
```bash
python -m collab_splats.webapp          # port 7860 (same as Panel, different runner)
python -m collab_splats.webapp --port 7861  # run alongside Panel during transition
```

---

## Sidebar sections per tab

| Tab | Sidebar sections (top → bottom) |
|-----|----------------------------------|
| Preprocess | Session · Extraction |
| Semantics | Session · Semantic methods |
| Reconstruct | Session · Pointcloud |
| Visualize | Session · Pointcloud · Semantic methods · View / Mesh |
| Localize | Session · Pointcloud · Localize method |

Session section is always visible. All other sections are hidden/shown via JS on tab switch.

---

## Data flow

```
Browser (HTML/JS)
  ↕  fetch() REST calls
  ↕  EventSource SSE for long ops
FastAPI (collab_splats/webapp/app.py)
  ↕  direct Python calls
Existing pipeline modules
  (frame_sampling, VGGTXCreator, MeshPipeline, DinoSaladExtractor, …)
  ↕  reads/writes
/workspace/outputs/
  (zarr, PLY, JPEGs, run_config.yaml)  ← served as /outputs/* static files
```

---

## UI components

### Shell (`index.html`)
- Top tab nav (Preprocess · Semantics · Reconstruct · Visualize · Localize)
- Left sidebar (240px, fixed, scrollable)
- Main content area (fills remaining space)
- Bottom status bar: SSE progress bar + last log line

### Preprocess tab
- Left: HTML5 `<video src="/video">` — served via `GET /session/video` (FileResponse from `state.video_path`, which may be outside `/workspace/outputs/`)
- Right: frame quality metrics (lightweight Vega-Lite or Chart.js charts)
- Bottom: frame strip — `<img src="/outputs/{dataset}/frames/frame_0100.jpg">` URLs, lazy-loaded, click to seek video
- Sidebar: Session + Extraction (method dropdown, max-frames slider, Extract button)

### Semantics tab
- Log panel (SSE stream)
- Sidebar: Session + Semantic methods (extractor dropdown, Run button)

### Reconstruct tab
- Log panel (SSE stream), status indicator
- Sidebar: Session + Pointcloud (creator dropdown, conf slider, BA/LC toggles, Run button)

### Visualize tab
- Full-height Three.js canvas: PLY pointcloud + optional mesh overlay
- Sidebar: Session · Pointcloud · Semantic methods · View/Mesh (voxel size, run mesh button, ground plane, frustums toggle)

### Localize tab
- Query image upload / path input
- Results panel: top-k matched frames with scores
- Sidebar: Session · Pointcloud · Localize (method dropdown, extractor dropdown, Run button)

---

## SSE protocol

All long-running operations stream newline-delimited JSON:

```json
{"type": "progress", "pct": 42, "msg": "Processing frame 84/200"}
{"type": "log", "msg": "Backend: vggtx  conf: 35.0"}
{"type": "done", "msg": "187 frames extracted"}
{"type": "error", "msg": "frames dir not found: /workspace/outputs/…"}
```

Client consumes via `EventSource`, updates progress bar and log textarea. On `done`/`error`, closes the source and re-enables the trigger button.

---

## Branch strategy

- Branch: `feat/webapp` cut from `refactor/cu121`
- Panel dashboard code (`collab_splats/dashboard/`) not modified
- New entry point: `collab_splats/webapp/__main__.py`
- Can run Panel (`port 7860`) and webapp (`port 7861`) simultaneously during development
- Webapp replaces Panel as default entry point once all 5 tabs are stable

---

## Implementation order

1. **Scaffold** — FastAPI app, static file mounts, shell HTML + CSS + tab nav
2. **Preprocess tab** — session load, video player, frame strip (most visible pain point)
3. **Reconstruct tab** — SSE log streaming, creator config
4. **Visualize tab** — Three.js PLY viewer (already prototyped)
5. **Semantics tab** — extractor selection, SSE log
6. **Localize tab** — method/extractor, results display

Each tab is independently shippable. Panel dashboard remains the fallback until all tabs are complete.

---

## Out of scope

- No frontend build step (no Vite, no Svelte, no npm) — vanilla JS only
- No authentication
- No multi-user sessions
- Panel dashboard is not deleted in this branch — that happens in a follow-up PR to `main`
