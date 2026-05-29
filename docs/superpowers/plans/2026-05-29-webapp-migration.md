# collab-splats webapp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Panel dashboard with a FastAPI + vanilla HTML/JS webapp on branch `feat/webapp`, serving media as static files and long ops via SSE.

**Architecture:** FastAPI app in `collab_splats/webapp/` serves static HTML/JS/CSS plus REST endpoints. Long-running operations (extraction, reconstruction, semantics) stream JSON events via SSE. Media (frames, video, PLY) served as static files from `/workspace/outputs/`. Three.js renders pointclouds and meshes. Panel dashboard untouched.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, Three.js (CDN), vanilla JS ES modules, PIL for JPEG writes.

---

## Branch setup

```bash
git checkout refactor/cu121
git checkout -b feat/webapp
```

---

## File map

**Create:**
```
collab_splats/webapp/__init__.py
collab_splats/webapp/__main__.py
collab_splats/webapp/app.py
collab_splats/webapp/state.py
collab_splats/webapp/routers/__init__.py
collab_splats/webapp/routers/session.py
collab_splats/webapp/routers/preprocess.py
collab_splats/webapp/routers/reconstruct.py
collab_splats/webapp/routers/visualize.py
collab_splats/webapp/routers/semantics.py
collab_splats/webapp/routers/localize.py
collab_splats/webapp/static/index.html
collab_splats/webapp/static/css/app.css
collab_splats/webapp/static/js/state.js
collab_splats/webapp/static/js/preprocess.js
collab_splats/webapp/static/js/reconstruct.js
collab_splats/webapp/static/js/visualize.js
collab_splats/webapp/static/js/semantics.js
collab_splats/webapp/static/js/localize.js
tests/webapp/__init__.py
tests/webapp/test_session.py
tests/webapp/test_preprocess.py
tests/webapp/test_reconstruct.py
tests/webapp/test_visualize.py
```

**Unmodified:** `collab_splats/dashboard/` (Panel app stays intact)

---

## Task 1: Install deps + scaffold module

**Files:**
- Create: `collab_splats/webapp/__init__.py`
- Create: `collab_splats/webapp/__main__.py`
- Create: `collab_splats/webapp/app.py`
- Create: `collab_splats/webapp/routers/__init__.py`
- Create: `tests/webapp/__init__.py`
- Create: `tests/webapp/test_session.py` (smoke only)

- [ ] **Step 1.1: Install FastAPI + uvicorn**

```bash
/opt/conda/envs/reconstruction/bin/pip install "fastapi>=0.111" "uvicorn[standard]>=0.29" httpx pytest-asyncio
```

Expected: successful install, no conflicts.

- [ ] **Step 1.2: Write scaffold test**

```python
# tests/webapp/test_session.py
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.mark.asyncio
async def test_root_serves_index():
    from collab_splats.webapp.app import create_app
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "collab-splats" in resp.text
```

- [ ] **Step 1.3: Run test — expect ImportError (module doesn't exist yet)**

```bash
cd /workspace/collab-splats
/opt/conda/envs/reconstruction/bin/python -m pytest tests/webapp/test_session.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'collab_splats.webapp'`

- [ ] **Step 1.4: Create `collab_splats/webapp/__init__.py`**

```python
```
(empty)

- [ ] **Step 1.5: Create `collab_splats/webapp/routers/__init__.py`**

```python
```
(empty)

- [ ] **Step 1.6: Create `collab_splats/webapp/app.py`**

```python
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

_STATIC_DIR = Path(__file__).parent / "static"
_OUTPUTS_DIR = Path("/workspace/outputs")


def create_app() -> FastAPI:
    """Instantiate the FastAPI application."""
    app = FastAPI(title="collab-splats webapp")

    # Serve static app files (HTML, CSS, JS)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Serve reconstruction outputs (frames, video, PLY) directly as files
    if _OUTPUTS_DIR.exists():
        app.mount("/outputs", StaticFiles(directory=str(_OUTPUTS_DIR)), name="outputs")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (_STATIC_DIR / "index.html").read_text()

    return app
```

- [ ] **Step 1.7: Create minimal `collab_splats/webapp/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>collab-splats</title>
<link rel="stylesheet" href="/static/css/app.css">
</head>
<body>
<nav id="tab-nav">
  <button class="tab-btn active" data-tab="preprocess">Preprocess</button>
  <button class="tab-btn" data-tab="semantics">Semantics</button>
  <button class="tab-btn" data-tab="reconstruct">Reconstruct</button>
  <button class="tab-btn" data-tab="visualize">Visualize</button>
  <button class="tab-btn" data-tab="localize">Localize</button>
  <span id="session-label"></span>
</nav>
<div id="app">
  <aside id="sidebar"></aside>
  <main id="main"></main>
</div>
<div id="statusbar">
  <div id="progress-track"><div id="progress-fill"></div></div>
  <span id="status-msg"></span>
</div>
<script type="module" src="/static/js/state.js"></script>
</body>
</html>
```

- [ ] **Step 1.8: Create `collab_splats/webapp/static/css/app.css`**

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0a0a0f;
  --bg2: #0d0d0d;
  --bg3: #111;
  --border: #1e1e2e;
  --accent: #2596be;
  --accent2: #50c050;
  --text: #ccc;
  --text2: #888;
  --danger: #e05050;
  --font: 'JetBrains Mono', 'Fira Mono', monospace;
}

body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 13px; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

#tab-nav { display: flex; align-items: center; gap: 0; background: var(--bg2); border-bottom: 1px solid var(--border); flex-shrink: 0; }
.tab-btn { background: none; border: none; border-bottom: 2px solid transparent; color: var(--text2); padding: 10px 18px; font-family: var(--font); font-size: 12px; cursor: pointer; }
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 700; }
#session-label { margin-left: auto; padding: 0 14px; font-size: 11px; color: var(--text2); }

#app { display: flex; flex: 1; overflow: hidden; }

#sidebar { width: 240px; background: var(--bg2); border-right: 1px solid var(--border); padding: 12px; overflow-y: auto; flex-shrink: 0; display: flex; flex-direction: column; gap: 16px; }

.sidebar-section { display: flex; flex-direction: column; gap: 6px; }
.sidebar-section h4 { font-size: 10px; font-weight: 700; color: var(--accent); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }

#main { flex: 1; overflow: hidden; display: flex; flex-direction: column; }

.tab-panel { display: none; flex: 1; overflow: hidden; }
.tab-panel.active { display: flex; flex-direction: column; }

#statusbar { height: 28px; background: var(--bg2); border-top: 1px solid var(--border); display: flex; align-items: center; padding: 0 12px; gap: 10px; flex-shrink: 0; }
#progress-track { width: 140px; height: 5px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
#progress-fill { height: 100%; width: 0; background: var(--accent); border-radius: 3px; transition: width 0.2s; }
#status-msg { font-size: 11px; color: var(--text2); }

/* Form controls */
select, input[type=range], input[type=text], input[type=number] { width: 100%; background: #161b22; border: 1px solid #30363d; color: var(--text); padding: 4px 6px; border-radius: 3px; font-family: var(--font); font-size: 11px; }
select:focus, input:focus { outline: 1px solid var(--accent); }
input[type=range] { padding: 0; accent-color: var(--accent); }
label { font-size: 11px; color: var(--text2); }

button.primary { background: var(--accent); border: none; color: #000; padding: 7px 12px; border-radius: 3px; font-family: var(--font); font-size: 12px; font-weight: 700; cursor: pointer; width: 100%; }
button.primary:hover { opacity: 0.9; }
button.primary:disabled { opacity: 0.4; cursor: not-allowed; }
button.danger { background: var(--danger); border: none; color: #fff; padding: 7px 12px; border-radius: 3px; font-family: var(--font); font-size: 12px; cursor: pointer; width: 100%; }

.status-ok { color: var(--accent2); font-size: 11px; }
.status-err { color: var(--danger); font-size: 11px; }
.status-info { color: var(--text2); font-size: 11px; }

/* Log area */
.log-area { background: var(--bg2); border: 1px solid var(--border); border-radius: 4px; padding: 10px; font-size: 11px; color: var(--text2); overflow-y: auto; flex: 1; white-space: pre-wrap; word-break: break-all; }
.log-area .ok { color: var(--accent2); }
.log-area .err { color: var(--danger); }
```

- [ ] **Step 1.9: Create `collab_splats/webapp/static/js/state.js`**

```javascript
// Central client-side state + tab switching
export const state = {
  outputDir: null,
  videoPath: null,
  creator: 'vggtx',
  conf: 35.0,
  extractor: 'dinov2',
  localizeMethod: null,
  localizeExtractor: 'DISK+LightGlue',
};

// Tab switching: show/hide panels, update sidebar
const TAB_MODULES = {};

export function registerTab(name, module) {
  TAB_MODULES[name] = module;
}

export function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
  const sidebar = document.getElementById('sidebar');
  sidebar.innerHTML = '';
  // Always render session section
  sidebar.appendChild(renderSessionSection());
  // Render tab-specific sections
  if (TAB_MODULES[name]?.renderSidebar) {
    const sections = TAB_MODULES[name].renderSidebar();
    sections.forEach(s => sidebar.appendChild(s));
  }
}

function renderSessionSection() {
  const sec = document.createElement('div');
  sec.className = 'sidebar-section';
  sec.id = 'section-session';
  sec.innerHTML = `
    <h4>Session</h4>
    <input type="text" id="output-dir-input" placeholder="/workspace/outputs/my_scene" value="${state.outputDir || ''}">
    <button class="primary" id="load-session-btn">Load session</button>
    <div id="session-status" class="status-info"></div>
  `;
  sec.querySelector('#load-session-btn').addEventListener('click', loadSession);
  return sec;
}

async function loadSession() {
  const dir = document.getElementById('output-dir-input').value.trim();
  if (!dir) return;
  const resp = await fetch('/api/session/load', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ output_dir: dir }),
  });
  const data = await resp.json();
  if (data.ok) {
    state.outputDir = data.output_dir;
    state.videoPath = data.video_path;
    document.getElementById('session-label').textContent = data.name;
    document.getElementById('session-status').textContent = '✓ ' + data.name;
    document.getElementById('session-status').className = 'status-ok';
  } else {
    document.getElementById('session-status').textContent = data.error;
    document.getElementById('session-status').className = 'status-err';
  }
}

// Progress bar helpers (used by all tabs)
export function setProgress(pct, msg) {
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('status-msg').textContent = msg || '';
}

// Wire tab buttons
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// Initialize with preprocess tab
// (individual tabs register themselves and trigger first render)
```

- [ ] **Step 1.10: Create `collab_splats/webapp/__main__.py`**

```python
import argparse
import uvicorn
from collab_splats.webapp.app import create_app

def main() -> None:
    parser = argparse.ArgumentParser(description="collab-splats webapp")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
```

- [ ] **Step 1.11: Run test — expect pass**

```bash
cd /workspace/collab-splats
/opt/conda/envs/reconstruction/bin/python -m pytest tests/webapp/test_session.py -v
```

Expected: `PASSED tests/webapp/test_session.py::test_root_serves_index`

- [ ] **Step 1.12: Commit**

```bash
git add collab_splats/webapp/ tests/webapp/ docs/superpowers/plans/2026-05-29-webapp-migration.md
git commit -m "feat(webapp): scaffold FastAPI app, static files, shell HTML+CSS+JS"
```

---

## Task 2: Session state + router

**Files:**
- Create: `collab_splats/webapp/state.py`
- Create: `collab_splats/webapp/routers/session.py`
- Modify: `collab_splats/webapp/app.py`
- Modify: `tests/webapp/test_session.py`

- [ ] **Step 2.1: Create `collab_splats/webapp/state.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SessionState:
    """Single-user server-side session state."""
    output_dir: Optional[Path] = None
    video_path: Optional[Path] = None
    creator: str = "vggtx"
    conf: float = 35.0
    extractor: str = "dinov2"
    localize_method: str = ""
    localize_extractor: str = "DISK+LightGlue"


# Module-level singleton — single-user local tool
_session = SessionState()


def get_session() -> SessionState:
    return _session
```

- [ ] **Step 2.2: Write session router tests**

```python
# tests/webapp/test_session.py  (replace contents)
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.fixture
def app():
    from collab_splats.webapp.app import create_app
    return create_app()

@pytest.mark.asyncio
async def test_root_serves_index(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert "collab-splats" in r.text

@pytest.mark.asyncio
async def test_session_load_bad_dir(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/session/load", json={"output_dir": "/no/such/path"})
    assert r.status_code == 200
    assert r.json()["ok"] is False

@pytest.mark.asyncio
async def test_session_current_empty(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/session/current")
    assert r.status_code == 200
    data = r.json()
    assert "output_dir" in data
```

- [ ] **Step 2.3: Run tests — expect FAIL (no /api routes yet)**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/webapp/test_session.py -v 2>&1 | tail -10
```

Expected: 2 tests fail with 404.

- [ ] **Step 2.4: Create `collab_splats/webapp/routers/session.py`**

```python
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from collab_splats.webapp.state import get_session

router = APIRouter(prefix="/api/session")


@router.post("/load")
async def load_session(body: dict[str, Any]) -> JSONResponse:
    """Load an existing output directory as the current session."""
    raw = body.get("output_dir", "")
    out_dir = Path(raw)
    if not out_dir.is_dir():
        return JSONResponse({"ok": False, "error": f"Not a directory: {raw}"})

    session = get_session()
    session.output_dir = out_dir

    # Try to read video_path from run_config.yaml
    config_file = out_dir / "run_config.yaml"
    video_path = None
    if config_file.exists():
        try:
            cfg = yaml.safe_load(config_file.read_text())
            raw_vp = cfg.get("video_path") or cfg.get("input_path")
            if raw_vp and Path(raw_vp).exists():
                video_path = Path(raw_vp)
        except Exception:
            pass
    session.video_path = video_path

    return JSONResponse({
        "ok": True,
        "output_dir": str(out_dir),
        "video_path": str(video_path) if video_path else None,
        "name": out_dir.name,
    })


@router.get("/current")
async def current_session() -> JSONResponse:
    """Return current session state (all serialisable fields)."""
    s = get_session()
    return JSONResponse({
        "output_dir": str(s.output_dir) if s.output_dir else None,
        "video_path": str(s.video_path) if s.video_path else None,
        "creator": s.creator,
        "conf": s.conf,
        "extractor": s.extractor,
    })


@router.get("/video")
async def serve_video() -> FileResponse:
    """Stream the session video file."""
    s = get_session()
    if s.video_path is None or not s.video_path.exists():
        return JSONResponse({"error": "No video loaded"}, status_code=404)
    return FileResponse(str(s.video_path), media_type="video/mp4")


@router.post("/update")
async def update_session(body: dict[str, Any]) -> JSONResponse:
    """Partial update of session config (creator, conf, extractor, etc.)."""
    s = get_session()
    for key in ("creator", "conf", "extractor", "localize_method", "localize_extractor"):
        if key in body:
            setattr(s, key, body[key])
    return JSONResponse({"ok": True})
```

- [ ] **Step 2.5: Mount session router in `app.py`**

```python
# collab_splats/webapp/app.py — replace create_app body:
from collab_splats.webapp.routers import session as session_router

def create_app() -> FastAPI:
    app = FastAPI(title="collab-splats webapp")
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    if _OUTPUTS_DIR.exists():
        app.mount("/outputs", StaticFiles(directory=str(_OUTPUTS_DIR)), name="outputs")
    app.include_router(session_router.router)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (_STATIC_DIR / "index.html").read_text()

    return app
```

- [ ] **Step 2.6: Run tests — expect all pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/webapp/test_session.py -v
```

Expected: 3 PASSED.

- [ ] **Step 2.7: Commit**

```bash
git add collab_splats/webapp/state.py collab_splats/webapp/routers/session.py collab_splats/webapp/app.py tests/webapp/test_session.py
git commit -m "feat(webapp): session state + load/current/video/update endpoints"
```

---

## Task 3: Preprocess tab

**Files:**
- Create: `collab_splats/webapp/routers/preprocess.py`
- Create: `collab_splats/webapp/static/js/preprocess.js`
- Modify: `collab_splats/webapp/static/index.html` (add tab panel)
- Create: `tests/webapp/test_preprocess.py`

- [ ] **Step 3.1: Write preprocess router tests**

```python
# tests/webapp/test_preprocess.py
import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport

@pytest.fixture
def app_with_session(tmp_path):
    (tmp_path / "run_config.yaml").write_text("video_path: /dev/null\n")
    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session
    s = get_session()
    s.output_dir = tmp_path
    s.video_path = None
    return create_app()

@pytest.mark.asyncio
async def test_video_info_no_session():
    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session
    get_session().output_dir = None
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/preprocess/info")
    assert r.status_code == 200
    assert r.json()["ok"] is False

@pytest.mark.asyncio
async def test_video_info_with_session(app_with_session):
    async with AsyncClient(transport=ASGITransport(app=app_with_session), base_url="http://test") as c:
        r = await c.get("/api/preprocess/info")
    assert r.status_code == 200
    # output_dir exists so ok=True even if video_path is None
    data = r.json()
    assert "output_dir" in data
```

- [ ] **Step 3.2: Run tests — expect FAIL**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/webapp/test_preprocess.py -v 2>&1 | tail -5
```

Expected: 404 (route not registered yet).

- [ ] **Step 3.3: Create `collab_splats/webapp/routers/preprocess.py`**

```python
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import AsyncIterator

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from collab_splats.utils.frame_sampling import get_video_info, sample_frames_fps, sample_frames_optical_flow
from collab_splats.webapp.state import get_session

router = APIRouter(prefix="/api/preprocess")


@router.get("/info")
async def video_info() -> JSONResponse:
    """Return video metadata and session state for the preprocess tab."""
    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "error": "No session loaded"})
    info: dict = {"ok": True, "output_dir": str(s.output_dir)}
    if s.video_path and s.video_path.exists():
        meta = get_video_info(str(s.video_path))
        info["video"] = meta
    # Check if frames already extracted
    frames_dir = s.output_dir / "frames"
    if frames_dir.is_dir():
        jpgs = sorted(frames_dir.glob("*.jpg"))
        info["frames_extracted"] = len(jpgs)
        info["frames_dir"] = str(frames_dir)
    return JSONResponse(info)


def _write_frames(frames: list[np.ndarray], output_dir: Path) -> Path:
    """Write RGB numpy frames as JPEGs to output_dir/frames/. Return frames dir."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        out = frames_dir / f"frame_{i:06d}.jpg"
        Image.fromarray(frame).save(str(out), format="JPEG", quality=95)
    return frames_dir


async def _extract_sse(method: str, max_frames: int, min_disparity: float) -> AsyncIterator[str]:
    """Async generator: run frame extraction in thread, yield SSE JSON lines."""
    s = get_session()
    if s.video_path is None or not s.video_path.exists():
        yield _sse({"type": "error", "msg": "No video loaded"})
        return
    if s.output_dir is None:
        yield _sse({"type": "error", "msg": "No output directory"})
        return

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def progress(current: int, total: int) -> None:
        pct = int(current / max(total, 1) * 100)
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "pct": pct, "msg": f"{current}/{total} frames"})

    def run() -> None:
        try:
            if method == "optical_flow":
                frames, _ = sample_frames_optical_flow(
                    str(s.video_path), max_frames=max_frames,
                    min_disparity=min_disparity, on_progress=progress, verbose=False,
                )
            else:
                from collab_splats.utils.frame_sampling import get_video_info as _gvi
                info = _gvi(str(s.video_path))
                dur = info.get("duration_s") or (info["total_frames"] / (info.get("fps") or 30.0))
                target_fps = max_frames / max(dur, 1.0)
                frames, _ = sample_frames_fps(
                    str(s.video_path), fps=target_fps, max_frames=max_frames,
                    on_progress=progress, verbose=False,
                )
            _write_frames(frames, s.output_dir)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "msg": f"{len(frames)} frames extracted", "count": len(frames)})
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": str(exc)})

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    while True:
        event = await queue.get()
        yield _sse(event)
        if event["type"] in ("done", "error"):
            break


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.get("/extract")
async def extract_frames(method: str = "optical_flow", max_frames: int = 200, min_disparity: float = 50.0):
    """SSE endpoint: extract frames from session video."""
    return StreamingResponse(
        _extract_sse(method, max_frames, min_disparity),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 3.4: Mount preprocess router in `app.py`**

Add to `create_app()` alongside session router:

```python
from collab_splats.webapp.routers import preprocess as preprocess_router
# inside create_app():
app.include_router(preprocess_router.router)
```

- [ ] **Step 3.5: Run tests — expect pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/webapp/test_preprocess.py -v
```

Expected: 2 PASSED.

- [ ] **Step 3.6: Add preprocess panel to `index.html`**

Inside `<main id="main">`, add after any existing content:

```html
<!-- Preprocess tab panel -->
<section class="tab-panel active" id="tab-preprocess">
  <div style="display:flex;flex:1;overflow:hidden">
    <div id="video-container" style="flex:1;background:#000;display:flex;align-items:center;justify-content:center">
      <video id="main-video" controls style="max-width:100%;max-height:100%;object-fit:contain" preload="metadata"></video>
    </div>
    <div id="metrics-panel" style="width:260px;padding:10px;background:#0d0d0d;border-left:1px solid #1e1e2e;overflow-y:auto;font-size:11px;color:#888">
      <div id="video-meta"></div>
    </div>
  </div>
  <div id="frame-strip" style="height:88px;background:#0d0d0d;border-top:1px solid #1e1e2e;display:flex;gap:3px;padding:4px 8px;overflow-x:auto;flex-shrink:0;align-items:center"></div>
</section>

<!-- Other tab panels (initially hidden) -->
<section class="tab-panel" id="tab-semantics"><div class="log-area" id="log-semantics"></div></section>
<section class="tab-panel" id="tab-reconstruct">
  <div class="log-area" id="log-reconstruct"></div>
</section>
<section class="tab-panel" id="tab-visualize">
  <canvas id="three-canvas" style="width:100%;height:100%"></canvas>
</section>
<section class="tab-panel" id="tab-localize">
  <div class="log-area" id="log-localize"></div>
</section>
```

Also add preprocess module import before closing `</body>`:

```html
<script type="module" src="/static/js/preprocess.js"></script>
```

- [ ] **Step 3.7: Create `collab_splats/webapp/static/js/preprocess.js`**

```javascript
import { state, registerTab, setProgress } from './state.js';

// ── Sidebar section ──────────────────────────────────────────────
function renderSidebar() {
  const sec = document.createElement('div');
  sec.className = 'sidebar-section';
  sec.innerHTML = `
    <h4>Extraction</h4>
    <label>Method</label>
    <select id="extract-method">
      <option value="optical_flow">Optical flow</option>
      <option value="balanced">Balanced FPS</option>
    </select>
    <label>Max frames: <span id="max-frames-val">200</span></label>
    <input type="range" id="max-frames" min="20" max="500" value="200">
    <label>Min disparity: <span id="min-disp-val">50</span></label>
    <input type="range" id="min-disp" min="5" max="200" value="50">
    <button class="primary" id="extract-btn">▶ Extract frames</button>
    <div id="extract-status" class="status-info"></div>
  `;
  sec.querySelector('#max-frames').addEventListener('input', e => {
    sec.querySelector('#max-frames-val').textContent = e.target.value;
  });
  sec.querySelector('#min-disp').addEventListener('input', e => {
    sec.querySelector('#min-disp-val').textContent = e.target.value;
  });
  sec.querySelector('#extract-btn').addEventListener('click', startExtraction);
  return [sec];
}

// ── Load video into <video> element ─────────────────────────────
async function loadVideoInfo() {
  const resp = await fetch('/api/preprocess/info');
  const data = await resp.json();
  if (!data.ok) return;

  const video = document.getElementById('main-video');
  video.src = '/api/session/video';

  if (data.video) {
    const m = data.video;
    document.getElementById('video-meta').innerHTML = `
      <div style="color:#2596be;font-weight:bold;margin-bottom:6px">VIDEO INFO</div>
      <div>${m.width}×${m.height} · ${(m.fps||0).toFixed(1)} fps</div>
      <div>${m.total_frames} frames · ${(m.duration_s||0).toFixed(1)}s</div>
    `;
  }

  if (data.frames_extracted > 0) {
    renderFrameStrip(data.frames_extracted, data.frames_dir);
    document.getElementById('extract-status').textContent = `✓ ${data.frames_extracted} frames`;
    document.getElementById('extract-status').className = 'status-ok';
  }
}

// ── Frame strip ──────────────────────────────────────────────────
function renderFrameStrip(count, framesDir) {
  const strip = document.getElementById('frame-strip');
  strip.innerHTML = '';
  // Compute path relative to /outputs mount: strip /workspace/outputs prefix
  const relDir = framesDir.replace('/workspace/outputs/', '');
  for (let i = 0; i < count; i++) {
    const img = document.createElement('img');
    const padded = String(i).padStart(6, '0');
    img.src = `/outputs/${relDir}/frame_${padded}.jpg`;
    img.loading = 'lazy';
    img.style.cssText = 'height:72px;width:auto;border:2px solid #222;border-radius:2px;cursor:pointer;flex-shrink:0';
    img.addEventListener('click', () => seekVideo(i, count));
    img.addEventListener('mouseenter', () => img.style.borderColor = '#2596be');
    img.addEventListener('mouseleave', () => img.style.borderColor = '#222');
    strip.appendChild(img);
  }
}

function seekVideo(frameIdx, totalFrames) {
  const video = document.getElementById('main-video');
  if (!video.duration) return;
  video.currentTime = (frameIdx / totalFrames) * video.duration;
}

// ── SSE extraction ────────────────────────────────────────────────
function startExtraction() {
  if (!state.outputDir) {
    alert('Load a session first');
    return;
  }
  const method = document.getElementById('extract-method')?.value || 'optical_flow';
  const maxFrames = document.getElementById('max-frames')?.value || 200;
  const minDisp = document.getElementById('min-disp')?.value || 50;

  const btn = document.getElementById('extract-btn');
  btn.disabled = true;
  document.getElementById('extract-status').textContent = 'Extracting…';
  document.getElementById('extract-status').className = 'status-info';

  const url = `/api/preprocess/extract?method=${method}&max_frames=${maxFrames}&min_disparity=${minDisp}`;
  const es = new EventSource(url);

  es.onmessage = e => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'progress') {
      setProgress(ev.pct, ev.msg);
    } else if (ev.type === 'done') {
      setProgress(100, ev.msg);
      document.getElementById('extract-status').textContent = `✓ ${ev.msg}`;
      document.getElementById('extract-status').className = 'status-ok';
      btn.disabled = false;
      es.close();
      loadVideoInfo(); // reload strip
    } else if (ev.type === 'error') {
      setProgress(0, '');
      document.getElementById('extract-status').textContent = ev.msg;
      document.getElementById('extract-status').className = 'status-err';
      btn.disabled = false;
      es.close();
    }
  };
  es.onerror = () => { btn.disabled = false; es.close(); };
}

// ── Register + init ───────────────────────────────────────────────
registerTab('preprocess', { renderSidebar });
// Trigger initial sidebar render (preprocess is the default active tab)
import('./state.js').then(m => m.switchTab('preprocess'));
loadVideoInfo();
```

- [ ] **Step 3.8: Verify manually — start server and open browser**

```bash
cd /workspace/collab-splats
/opt/conda/envs/reconstruction/bin/python -m collab_splats.webapp --port 7861 &
```

Open http://localhost:7861. Load session `/workspace/outputs/birds_gh010164`. Video should play, frame strip should appear.

- [ ] **Step 3.9: Commit**

```bash
git add collab_splats/webapp/routers/preprocess.py collab_splats/webapp/static/ tests/webapp/test_preprocess.py
git commit -m "feat(webapp): preprocess tab — video player, frame strip, SSE extraction"
```

---

## Task 4: Reconstruct tab

**Files:**
- Create: `collab_splats/webapp/routers/reconstruct.py`
- Create: `collab_splats/webapp/static/js/reconstruct.js`
- Create: `tests/webapp/test_reconstruct.py`
- Modify: `collab_splats/webapp/app.py`

- [ ] **Step 4.1: Write reconstruct router test**

```python
# tests/webapp/test_reconstruct.py
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.mark.asyncio
async def test_reconstruct_no_session():
    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session
    get_session().output_dir = None
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/reconstruct/status")
    assert r.status_code == 200
    assert r.json()["ok"] is False

@pytest.mark.asyncio
async def test_reconstruct_status_with_session(tmp_path):
    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session
    get_session().output_dir = tmp_path
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/reconstruct/status")
    data = r.json()
    assert "has_frames" in data
```

- [ ] **Step 4.2: Create `collab_splats/webapp/routers/reconstruct.py`**

```python
from __future__ import annotations

import asyncio
import json
import threading
import traceback
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from collab_splats.webapp.state import get_session

router = APIRouter(prefix="/api/reconstruct")


@router.get("/status")
async def status() -> JSONResponse:
    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "error": "No session loaded"})
    frames_dir = s.output_dir / "frames"
    existing = sorted(frames_dir.glob("*.jpg")) if frames_dir.is_dir() else []
    backend_dir = s.output_dir / s.creator
    zarr_exists = (backend_dir / "feedforward.zarr").exists()
    return JSONResponse({
        "ok": True,
        "has_frames": len(existing) > 0,
        "frame_count": len(existing),
        "zarr_exists": zarr_exists,
        "creator": s.creator,
        "conf": s.conf,
    })


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _run_sse() -> AsyncIterator[str]:
    s = get_session()
    if s.output_dir is None:
        yield _sse({"type": "error", "msg": "No session loaded"}); return
    frames_dir = s.output_dir / "frames"
    if not frames_dir.is_dir() or not list(frames_dir.glob("*.jpg")):
        yield _sse({"type": "error", "msg": "No frames found — run Preprocess first"}); return

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    creator_name = s.creator
    conf = s.conf
    output_dir = s.output_dir

    def run() -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": f"Backend: {creator_name}  conf: {conf}"})
            if creator_name == "vggtx":
                from collab_splats.pointcloud.feedforward import VGGTXCreator
                creator = VGGTXCreator(conf_threshold=conf)
            elif creator_name == "mapanything":
                from collab_splats.pointcloud.feedforward import MapAnythingCreator
                creator = MapAnythingCreator(confidence_percentile=conf)
            elif creator_name == "vggt_omega":
                from collab_splats.pointcloud.feedforward import VGGTOmegaCreator
                creator = VGGTOmegaCreator(conf_threshold=conf)
            else:
                raise ValueError(f"Unknown creator: {creator_name!r}")

            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": f"Starting {creator_name} inference…"})
            backend_dir = output_dir / creator_name
            creator.reconstruct(frames_dir, backend_dir)
            ff = creator.outputs
            if ff is None:
                raise RuntimeError("Creator produced no outputs")

            zarr_path = backend_dir / "feedforward.zarr"
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": f"Saving {zarr_path.name}…"})
            ff.save_zarr(zarr_path)
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "done", "msg": f"Done. {len(ff.points):,} points.",
                "zarr_path": str(zarr_path),
            })
        except Exception as exc:
            tb = traceback.format_exc()
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": f"{exc}\n{tb}"})

    threading.Thread(target=run, daemon=True).start()

    while True:
        event = await queue.get()
        yield _sse(event)
        if event["type"] in ("done", "error"):
            break


@router.get("/run")
async def run_reconstruction():
    """SSE endpoint: run feedforward reconstruction."""
    return StreamingResponse(
        _run_sse(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 4.3: Mount router in `app.py`**

```python
from collab_splats.webapp.routers import reconstruct as reconstruct_router
# inside create_app():
app.include_router(reconstruct_router.router)
```

- [ ] **Step 4.4: Run tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/webapp/test_reconstruct.py -v
```

Expected: 2 PASSED.

- [ ] **Step 4.5: Create `collab_splats/webapp/static/js/reconstruct.js`**

```javascript
import { state, registerTab, setProgress } from './state.js';

function renderSidebar() {
  const sec = document.createElement('div');
  sec.className = 'sidebar-section';
  sec.innerHTML = `
    <h4>Pointcloud</h4>
    <label>Creator</label>
    <select id="rc-creator">
      <option value="vggtx">vggtx</option>
      <option value="mapanything">mapanything</option>
      <option value="vggt_omega">vggt_omega</option>
    </select>
    <label>Conf threshold: <span id="rc-conf-val">35</span></label>
    <input type="range" id="rc-conf" min="0" max="100" value="35">
    <div style="display:flex;gap:8px;margin-top:4px">
      <label style="display:flex;align-items:center;gap:4px"><input type="checkbox" id="rc-ba" disabled> BA</label>
      <label style="display:flex;align-items:center;gap:4px"><input type="checkbox" id="rc-lc" disabled> LC</label>
    </div>
    <button class="primary" id="rc-run-btn" style="margin-top:8px">▶ Run reconstruction</button>
    <div id="rc-status" class="status-info"></div>
  `;
  sec.querySelector('#rc-creator').addEventListener('change', e => {
    state.creator = e.target.value;
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({creator: e.target.value}) });
  });
  sec.querySelector('#rc-conf').addEventListener('input', e => {
    sec.querySelector('#rc-conf-val').textContent = e.target.value;
    state.conf = parseFloat(e.target.value);
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({conf: parseFloat(e.target.value)}) });
  });
  sec.querySelector('#rc-run-btn').addEventListener('click', runReconstruction);
  return [sec];
}

function runReconstruction() {
  const log = document.getElementById('log-reconstruct');
  const btn = document.getElementById('rc-run-btn');
  if (!state.outputDir) { alert('Load a session first'); return; }
  log.textContent = '';
  btn.disabled = true;
  document.getElementById('rc-status').textContent = 'Running…';
  document.getElementById('rc-status').className = 'status-info';

  const es = new EventSource('/api/reconstruct/run');
  es.onmessage = e => {
    const ev = JSON.parse(e.data);
    const line = document.createElement('div');
    line.textContent = ev.msg;
    if (ev.type === 'done') line.className = 'ok';
    if (ev.type === 'error') line.className = 'err';
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
    if (ev.type === 'progress') setProgress(ev.pct || 0, ev.msg);
    if (ev.type === 'done') {
      setProgress(100, 'Done');
      document.getElementById('rc-status').textContent = '✓ ' + ev.msg;
      document.getElementById('rc-status').className = 'status-ok';
      btn.disabled = false; es.close();
    }
    if (ev.type === 'error') {
      document.getElementById('rc-status').textContent = ev.msg.split('\n')[0];
      document.getElementById('rc-status').className = 'status-err';
      btn.disabled = false; es.close();
    }
  };
}

registerTab('reconstruct', { renderSidebar });
```

Add to `index.html` before `</body>`:
```html
<script type="module" src="/static/js/reconstruct.js"></script>
```

- [ ] **Step 4.6: Commit**

```bash
git add collab_splats/webapp/routers/reconstruct.py collab_splats/webapp/static/js/reconstruct.js tests/webapp/test_reconstruct.py collab_splats/webapp/app.py collab_splats/webapp/static/index.html
git commit -m "feat(webapp): reconstruct tab — creator config, SSE log streaming"
```

---

## Task 5: Visualize tab (Three.js)

**Files:**
- Create: `collab_splats/webapp/routers/visualize.py`
- Create: `collab_splats/webapp/static/js/visualize.js`
- Create: `tests/webapp/test_visualize.py`
- Modify: `collab_splats/webapp/app.py`

- [ ] **Step 5.1: Write visualize test**

```python
# tests/webapp/test_visualize.py
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.mark.asyncio
async def test_visualize_status_no_session():
    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session
    get_session().output_dir = None
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/visualize/status")
    assert r.json()["ok"] is False

@pytest.mark.asyncio
async def test_visualize_status_with_ply(tmp_path):
    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session
    s = get_session()
    s.output_dir = tmp_path
    s.creator = "vggtx"
    ply_path = tmp_path / "vggtx" / "sparse_pc.ply"
    ply_path.parent.mkdir()
    ply_path.write_text("ply\n")
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/visualize/status")
    data = r.json()
    assert data["ok"] is True
    assert data["ply_url"] is not None
```

- [ ] **Step 5.2: Create `collab_splats/webapp/routers/visualize.py`**

```python
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from collab_splats.webapp.state import get_session

router = APIRouter(prefix="/api/visualize")


@router.get("/status")
async def status() -> JSONResponse:
    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "error": "No session loaded"})
    backend_dir = s.output_dir / s.creator
    ply = backend_dir / "sparse_pc.ply"
    mesh = backend_dir / "mesh" / "mesh.ply"
    # Build /outputs-relative URLs for the browser to fetch
    base = str(s.output_dir).replace("/workspace/outputs", "")
    return JSONResponse({
        "ok": True,
        "ply_url": f"/outputs{base}/{s.creator}/sparse_pc.ply" if ply.exists() else None,
        "mesh_url": f"/outputs{base}/{s.creator}/mesh/mesh.ply" if mesh.exists() else None,
        "creator": s.creator,
    })
```

- [ ] **Step 5.3: Mount router + run tests**

Add to `app.py`:
```python
from collab_splats.webapp.routers import visualize as visualize_router
app.include_router(visualize_router.router)
```

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/webapp/test_visualize.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5.4: Create `collab_splats/webapp/static/js/visualize.js`**

```javascript
import { state, registerTab, setProgress } from './state.js';
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from 'https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/loaders/PLYLoader.js';

let renderer, scene, camera, controls, currentPoints;

function initThree() {
  const canvas = document.getElementById('three-canvas');
  if (!canvas || renderer) return;
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x0a0a0f);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.0001, 1000);
  camera.position.set(0, 0, 2.5);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  (function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); })();
}

function loadPLY(url, pointSize = 0.004) {
  if (!renderer) initThree();
  setProgress(10, 'Loading pointcloud…');
  new PLYLoader().load(url, geo => {
    if (currentPoints) { scene.remove(currentPoints); currentPoints.geometry.dispose(); }
    geo.computeBoundingBox();
    const center = new THREE.Vector3();
    geo.boundingBox.getCenter(center);
    geo.translate(-center.x, -center.y, -center.z);
    const size = new THREE.Vector3();
    geo.boundingBox.getSize(size);
    const scale = 2 / Math.max(size.x, size.y, size.z);
    geo.scale(scale, scale, scale);
    const mat = new THREE.PointsMaterial({ size: pointSize, vertexColors: !!geo.attributes.color, sizeAttenuation: true });
    if (!geo.attributes.color) mat.color.set(0x2596be);
    currentPoints = new THREE.Points(geo, mat);
    scene.add(currentPoints);
    setProgress(100, `${geo.attributes.position.count.toLocaleString()} points`);
  }, xhr => setProgress(Math.round(xhr.loaded / xhr.total * 90), 'Loading…'));
}

async function loadScene() {
  if (!state.outputDir) return;
  const resp = await fetch('/api/visualize/status');
  const data = await resp.json();
  if (data.ok && data.ply_url) {
    loadPLY(data.ply_url);
  }
}

function renderSidebar() {
  const sec = document.createElement('div');
  sec.className = 'sidebar-section';
  sec.innerHTML = `
    <h4>Pointcloud</h4>
    <label>Creator</label>
    <select id="viz-creator">
      <option value="vggtx">vggtx</option>
      <option value="mapanything">mapanything</option>
      <option value="vggt_omega">vggt_omega</option>
    </select>
    <button class="primary" id="viz-load-btn" style="margin-top:4px">Load scene</button>
    <h4 style="margin-top:10px">Semantic methods</h4>
    <select id="viz-extractor">
      <option value="dinov2">DINOv2</option>
      <option value="sam">SAM</option>
    </select>
    <h4 style="margin-top:10px">View / Mesh</h4>
    <label style="display:flex;align-items:center;gap:6px"><input type="checkbox" id="viz-frustums"> Show frustums</label>
    <label>Point size: <span id="viz-pt-val">4</span></label>
    <input type="range" id="viz-pt-size" min="1" max="20" value="4">
  `;
  sec.querySelector('#viz-creator').addEventListener('change', e => {
    state.creator = e.target.value;
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({creator: e.target.value}) });
  });
  sec.querySelector('#viz-load-btn').addEventListener('click', loadScene);
  sec.querySelector('#viz-pt-size').addEventListener('input', e => {
    sec.querySelector('#viz-pt-val').textContent = e.target.value;
    if (currentPoints) currentPoints.material.size = parseFloat(e.target.value) * 0.001;
  });
  return [sec];
}

// Initialize Three.js when this tab becomes active
registerTab('visualize', {
  renderSidebar,
  onActivate() { initThree(); loadScene(); },
});
```

Update `state.js` `switchTab` to call `onActivate` if defined:

In `state.js`, inside `switchTab`:
```javascript
// after sidebar rendering:
if (TAB_MODULES[name]?.onActivate) TAB_MODULES[name].onActivate();
```

Add to `index.html` before `</body>`:
```html
<script type="module" src="/static/js/visualize.js"></script>
```

- [ ] **Step 5.5: Commit**

```bash
git add collab_splats/webapp/routers/visualize.py collab_splats/webapp/static/js/visualize.js collab_splats/webapp/static/js/state.js tests/webapp/test_visualize.py collab_splats/webapp/app.py collab_splats/webapp/static/index.html
git commit -m "feat(webapp): visualize tab — Three.js PLY pointcloud/mesh viewer"
```

---

## Task 6: Semantics tab

**Files:**
- Create: `collab_splats/webapp/routers/semantics.py`
- Create: `collab_splats/webapp/static/js/semantics.js`
- Modify: `collab_splats/webapp/app.py`

- [ ] **Step 6.1: Create `collab_splats/webapp/routers/semantics.py`**

```python
from __future__ import annotations

import asyncio
import json
import threading
import traceback
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from collab_splats.webapp.state import get_session

router = APIRouter(prefix="/api/semantics")


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _run_sse() -> AsyncIterator[str]:
    s = get_session()
    if s.output_dir is None:
        yield _sse({"type": "error", "msg": "No session loaded"}); return

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    extractor_name = s.extractor
    output_dir = s.output_dir
    backend_dir = output_dir / s.creator

    def run() -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": f"Extractor: {extractor_name}"})
            from collab_splats.semantics.features import BaseFeatureExtractor
            extractor = BaseFeatureExtractor.from_registry(extractor_name)
            zarr_path = backend_dir / "feedforward.zarr"
            if not zarr_path.exists():
                raise FileNotFoundError(f"feedforward.zarr not found at {zarr_path}")
            from collab_splats.pointcloud.feedforward.base import FeedforwardResult
            ff = FeedforwardResult.load_zarr(zarr_path, load_images=True)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": "Running feature extraction…"})
            features_path = output_dir / "features" / extractor_name
            extractor.extract(ff, features_path)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "msg": f"Features saved to {features_path.name}"})
        except Exception as exc:
            tb = traceback.format_exc()
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": f"{exc}\n{tb}"})

    threading.Thread(target=run, daemon=True).start()
    while True:
        event = await queue.get()
        yield _sse(event)
        if event["type"] in ("done", "error"):
            break


@router.get("/run")
async def run_semantics():
    return StreamingResponse(
        _run_sse(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 6.2: Create `collab_splats/webapp/static/js/semantics.js`**

```javascript
import { state, registerTab, setProgress } from './state.js';

function renderSidebar() {
  const sec = document.createElement('div');
  sec.className = 'sidebar-section';
  sec.innerHTML = `
    <h4>Semantic methods</h4>
    <label>Extractor</label>
    <select id="sem-extractor">
      <option value="dinov2">DINOv2</option>
      <option value="sam">SAM</option>
    </select>
    <button class="primary" id="sem-run-btn" style="margin-top:8px">▶ Extract features</button>
    <div id="sem-status" class="status-info"></div>
  `;
  sec.querySelector('#sem-extractor').addEventListener('change', e => {
    state.extractor = e.target.value;
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({extractor: e.target.value}) });
  });
  sec.querySelector('#sem-run-btn').addEventListener('click', runSemantics);
  return [sec];
}

function runSemantics() {
  const log = document.getElementById('log-semantics');
  const btn = document.getElementById('sem-run-btn');
  if (!state.outputDir) { alert('Load a session first'); return; }
  log.textContent = '';
  btn.disabled = true;

  const es = new EventSource('/api/semantics/run');
  es.onmessage = e => {
    const ev = JSON.parse(e.data);
    const line = document.createElement('div');
    line.textContent = ev.msg;
    if (ev.type === 'done') line.className = 'ok';
    if (ev.type === 'error') line.className = 'err';
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
    if (ev.type === 'done' || ev.type === 'error') { btn.disabled = false; es.close(); }
  };
}

registerTab('semantics', { renderSidebar });
```

- [ ] **Step 6.3: Mount router and add script tag**

In `app.py`:
```python
from collab_splats.webapp.routers import semantics as semantics_router
app.include_router(semantics_router.router)
```

In `index.html` before `</body>`:
```html
<script type="module" src="/static/js/semantics.js"></script>
```

- [ ] **Step 6.4: Commit**

```bash
git add collab_splats/webapp/routers/semantics.py collab_splats/webapp/static/js/semantics.js collab_splats/webapp/app.py collab_splats/webapp/static/index.html
git commit -m "feat(webapp): semantics tab — extractor selection, SSE feature extraction"
```

---

## Task 7: Localize tab

**Files:**
- Create: `collab_splats/webapp/routers/localize.py`
- Create: `collab_splats/webapp/static/js/localize.js`
- Modify: `collab_splats/webapp/app.py`

- [ ] **Step 7.1: Create `collab_splats/webapp/routers/localize.py`**

```python
from __future__ import annotations

import asyncio
import json
import threading
import traceback
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from collab_splats.webapp.state import get_session

router = APIRouter(prefix="/api/localize")


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.get("/methods")
async def list_methods() -> JSONResponse:
    """Return available localization methods (zarr dirs with feedforward.zarr)."""
    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "methods": []})
    methods = [
        p.name for p in s.output_dir.iterdir()
        if p.is_dir() and (p / "feedforward.zarr").exists()
    ]
    return JSONResponse({"ok": True, "methods": sorted(methods)})


async def _run_sse(query_path: str) -> AsyncIterator[str]:
    s = get_session()
    if s.output_dir is None:
        yield _sse({"type": "error", "msg": "No session loaded"}); return

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    method = s.localize_method
    extractor = s.localize_extractor
    output_dir = s.output_dir

    def run() -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": f"Method: {method}  extractor: {extractor}"})
            from collab_splats.pointcloud.localization import DinoSaladExtractor
            loc = DinoSaladExtractor()
            zarr_path = output_dir / method / "feedforward.zarr"
            index_dir = output_dir / method / "localization_index" / extractor
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": "Building index…"})
            loc.build_index(zarr_path, index_dir)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": "Querying…"})
            results = loc.query(Path(query_path), index_dir, top_k=5)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "msg": "Localization complete", "results": results})
        except Exception as exc:
            tb = traceback.format_exc()
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": f"{exc}\n{tb}"})

    threading.Thread(target=run, daemon=True).start()
    while True:
        event = await queue.get()
        yield _sse(event)
        if event["type"] in ("done", "error"):
            break


@router.get("/run")
async def run_localize(query_path: str = ""):
    return StreamingResponse(
        _run_sse(query_path), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 7.2: Create `collab_splats/webapp/static/js/localize.js`**

```javascript
import { state, registerTab, setProgress } from './state.js';

function renderSidebar() {
  const pc = document.createElement('div');
  pc.className = 'sidebar-section';
  pc.innerHTML = `
    <h4>Pointcloud</h4>
    <select id="loc-method"><option value="">— select method —</option></select>
  `;

  const loc = document.createElement('div');
  loc.className = 'sidebar-section';
  loc.innerHTML = `
    <h4>Localize method</h4>
    <label>Extractor</label>
    <select id="loc-extractor">
      <option value="DISK+LightGlue">DISK+LightGlue</option>
      <option value="XFeat+MNN">XFeat+MNN</option>
    </select>
    <label style="margin-top:8px">Query image path</label>
    <input type="text" id="loc-query" placeholder="/path/to/query.jpg">
    <button class="primary" id="loc-run-btn" style="margin-top:8px">▶ Localize</button>
    <div id="loc-status" class="status-info"></div>
  `;

  fetch('/api/localize/methods').then(r => r.json()).then(data => {
    const sel = pc.querySelector('#loc-method');
    (data.methods || []).forEach(m => {
      const o = document.createElement('option'); o.value = m; o.textContent = m; sel.appendChild(o);
    });
    if (state.creator) sel.value = state.creator;
  });

  pc.querySelector('#loc-method').addEventListener('change', e => {
    state.creator = e.target.value;
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({creator: e.target.value}) });
  });

  loc.querySelector('#loc-extractor').addEventListener('change', e => {
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({localize_extractor: e.target.value}) });
  });

  loc.querySelector('#loc-run-btn').addEventListener('click', () => {
    const query = document.getElementById('loc-query').value.trim();
    runLocalize(query);
  });

  return [pc, loc];
}

function runLocalize(queryPath) {
  const log = document.getElementById('log-localize');
  const btn = document.getElementById('loc-run-btn');
  log.textContent = '';
  btn.disabled = true;

  const es = new EventSource(`/api/localize/run?query_path=${encodeURIComponent(queryPath)}`);
  es.onmessage = e => {
    const ev = JSON.parse(e.data);
    const line = document.createElement('div');
    if (ev.type === 'done' && ev.results) {
      line.innerHTML = `<span class="ok">${ev.msg}</span><br>${JSON.stringify(ev.results, null, 2)}`;
    } else {
      line.textContent = ev.msg;
    }
    if (ev.type === 'error') line.className = 'err';
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
    if (ev.type === 'done' || ev.type === 'error') { btn.disabled = false; es.close(); }
  };
}

registerTab('localize', { renderSidebar });
```

- [ ] **Step 7.3: Mount router and add script tag**

In `app.py`:
```python
from collab_splats.webapp.routers import localize as localize_router
app.include_router(localize_router.router)
```

In `index.html` before `</body>`:
```html
<script type="module" src="/static/js/localize.js"></script>
```

- [ ] **Step 7.4: Commit**

```bash
git add collab_splats/webapp/routers/localize.py collab_splats/webapp/static/js/localize.js collab_splats/webapp/app.py collab_splats/webapp/static/index.html
git commit -m "feat(webapp): localize tab — method/extractor selection, SSE localization"
```

---

## Task 8: Wire-up, branch push, test run

**Files:**
- Modify: `collab_splats/webapp/static/js/state.js` (fix session load → update global state + re-render sidebar)

- [ ] **Step 8.1: Fix `state.js` — persist outputDir on session load**

In `loadSession()` inside `state.js`, after `state.outputDir = data.output_dir;`:

```javascript
// Re-render current tab sidebar so tab-specific sections see the new session
const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
if (activeTab) switchTab(activeTab);
```

- [ ] **Step 8.2: Run full test suite**

```bash
cd /workspace/collab-splats
/opt/conda/envs/reconstruction/bin/python -m pytest tests/webapp/ -v
```

Expected: all tests pass.

- [ ] **Step 8.3: Smoke test the full app**

```bash
/opt/conda/envs/reconstruction/bin/python -m collab_splats.webapp --port 7861 &
```

1. Open http://localhost:7861
2. Load session `/workspace/outputs/birds_gh010164`
3. Preprocess tab: video should play, frame strip should show
4. Switch to Visualize tab: load scene → Three.js PLY viewer
5. Switch to Reconstruct tab: sidebar shows creator/conf, no pointcloud section on Preprocess

- [ ] **Step 8.4: Final commit + push**

```bash
git add -A
git commit -m "feat(webapp): wire-up session state re-render on load; all tabs complete"
git push -u origin feat/webapp
```

---

## Self-review checklist (completed inline)

- **Spec coverage:** scaffold ✓, session ✓, preprocess (video+strip+SSE) ✓, reconstruct (SSE) ✓, visualize (Three.js) ✓, semantics ✓, localize ✓, branch strategy ✓, tab-specific sidebars ✓
- **Placeholder scan:** no TBD/TODO present
- **Type consistency:** `_sse()` helper defined locally in each router (same signature); `state.creator` used consistently across JS modules; `get_session()` returns same `SessionState` instance throughout
- **SSE pattern:** identical async generator + threading pattern used in preprocess, reconstruct, semantics, localize — consistent
- **Static file paths:** `/outputs/` mount serves `/workspace/outputs/`; `visualize.py` builds URLs via string replace of `/workspace/outputs` prefix — consistent with mount
