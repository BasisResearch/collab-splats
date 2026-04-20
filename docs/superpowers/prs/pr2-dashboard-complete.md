# PR2 — Dashboard Complete

**Branch:** `refactor/dashboard-complete` → `refactor/core-modules`

## Title
feat(dashboard): Panel SemanticsDashboard with optical flow, progress bar, and frame extraction fixes

## Body
Replaces previous dashboard with Panel-based SemanticsDashboard wired to the updated collab_splats module layout.

**Dashboard core:**
- Adds `collab_splats/dashboard/` — SemanticsDashboard (Panel/MaterialTemplate), ConfigPanel, video_discovery
- Wires optical flow frame sampling into dashboard UI (FPS and optical flow modes)
- Auto-detects CUDA, defaults all device dropdowns accordingly (`device_dd`, `seg_device_dd`, `query_device_dd`)
- Imports updated to canonical `collab_splats.utils.frame_sampling` path
- Exports `SemanticsDashboard`, `build_app`, `run_app` from package `__init__`

**Frame extraction fixes (session 2026-04-20):**
- Fix image orientation: re-enable OpenCV auto-rotation (`CAP_PROP_ORIENTATION_AUTO`), remove dead `_rotation_map`/`_apply_rotation` helpers
- Fix frame slider overflow: move `frame_slider` below image row with `sizing_mode="stretch_width"`
- Add determinate progress bar (`pn.widgets.Progress`) with threaded extraction and 200ms polling
- Seek-based FPS extraction (`CAP_PROP_POS_FRAMES`) — skips decoding of discarded frames
- Low-resolution OF analysis (max 480px wide) — selected frames kept full-res; ~4–9x speedup on 4K
- Thread-safe extraction state with re-entrant click guard

**Cleanup:**
- Deleted `dashboard` and `refactor/dashboard-optical-flow` branches (content absorbed via cherry-pick)

MapAnything integration deferred to Phase 3 (full UI design pending).

Tests: 43 passed (dashboard + frame_sampling), 2 pre-existing failures (no GPU/browser required).
