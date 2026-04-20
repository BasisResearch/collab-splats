# PR2 — Dashboard Complete

**Branch:** `refactor/dashboard-complete` → `refactor/core-modules`

## Title
feat(dashboard): add Panel SemanticsDashboard with optical flow and CUDA auto-detect

## Body
Replaces previous dashboard with Panel-based SemanticsDashboard wired to the updated collab_splats module layout.

- Adds `collab_splats/dashboard/` — SemanticsDashboard (Panel/MaterialTemplate), ConfigPanel, video_discovery
- Wires optical flow frame sampling into dashboard UI (FPS and optical flow modes)
- Auto-detects CUDA, defaults all device dropdowns accordingly (`device_dd`, `seg_device_dd`, `query_device_dd`)
- Imports updated to canonical `collab_splats.utils.frame_sampling` path
- Exports `SemanticsDashboard`, `build_app`, `run_app` from package `__init__`

MapAnything integration deferred to Phase 3 (full UI design pending).

Tests: 31 passed, 2 pre-existing failures (no GPU/browser required).
