# Splats Dashboard Refinement — Design

**Date:** 2026-06-04
**Branch:** `refactor/cu121-uv-migration`
**Predecessor:** [splats-dashboard design](2026-06-03-splats-dashboard-design.md) · [handoff](../handoffs/2026-06-04-splats-dashboard-handoff.md)

## Context

The single-page splats dashboard works but only unit-tested (heavy work mocked).
A live smoke run against real `vggt_omega` + `talk2dino` + rclone surfaced UX gaps
and two runtime bugs. This spec refines six items. Guiding principle: **reuse the
existing collab-splats API; do not reinvent.** Every item below maps to an existing
function — no novel algorithms.

## Items

### 1. Conditional sampling widget

`min_disparity` is only consumed by `sample_frames_optical_flow` (frame_sampling.py:599),
never by `sample_frames_fps`. Today it shows unconditionally in the Frame-sampling card.

**Audit result:** sampling is the *only* pane with unselected-irrelevant knobs.
`sample_frames_fps` takes no special knob beyond shared `max_frames`. The env-model
`conf` widget maps to both creators' confidence arg (`conf_threshold` /
`confidence_percentile`) — one widget, no gating. Mesh always runs — no gating.

**Change (app.py):**
- Add a reusable helper `_bind_visibility(widget, selector, predicate)` that sets
  `widget.visible = predicate(selector.value)` immediately and on every
  `selector.param.watch(..., "value")`.
- Gate `min_disparity` visible only when `sampling.value == "optical_flow"`.
- Expose **only** `min_disparity` (not `motion_weight` / `coverage_weight` — YAGNI).
  The helper is reusable for any future per-attribute gating.

### 2. Per-step timings in the log

Progress is coarse (`update_progress` at 5–20/25/60/80/95). The user wants explicit
per-step timing — model processing time, mesh creation time — written out, since each
step already has internal tqdm/logging.

**Change (pipeline.py):**
- Wrap each step in `time.perf_counter()` and, on completion, append a log line via
  `op_log.update_progress(pct, msg)` of the form `"pointcloud (vggt_omega): 42.3s"`.
- Steps timed: sample, pointcloud, mesh, semantics, push.
- No capture of internal tqdm (heavy, brittle). Per-step wall-clock durations plus the
  existing coarse messages are sufficient. The log area already renders `log_lines`
  (operation_log.py:80).

### 3. Mesh defaults

Change defaults in **both** config.py (`RunConfig`) and app.py (widget construction):

| Param | Old | New |
|---|---|---|
| `mesh_voxel_size` | 0.01 | **0.005** |
| `mesh_sdf_trunc` | 0.04 | **0.02** |
| `mesh_depth_trunc` | 10.0 | **1.0** |
| `mesh_clean_repair` | True | **False** |

### 4. Semantics query — reuse `score_queries`

`viewer.query` (viewer.py:115-127) hand-rolls L2-norm + cosine + viridis. The semantics
API already provides exactly this: `BaseQueryableExtractor.score_queries(features,
positive, negative, temperature=0.05, reduction="max")` (features/base.py:463) —
contrastive softmax (Talk2DINO paper convention), accepts `(P, D)` point arrays, returns
`[0, 1]`, takes `List[str]` (native multi-term).

**Change (viewer.py):**
- Replace the hand-rolled math in `query()` with
  `extractor.score_queries(features=lifted, positive=pos, negative=neg)` → `apply_viridis`.
  `lifted` is the cached `_lifted_normed` (P, D). New signature:
  `query(positive: list[str], negative: list[str], extractor_name: str)`.
- `reduction="max"`, `temperature=0.05` fixed (not surfaced — YAGNI).
- Blank negative → pass `None` so the API applies its recommended default `["object"]`.

**Change (app.py):**
- Replace single `query` TextInput + value-watch with: `pos_query` TextInput,
  `neg_query` TextInput, and a **Run query** Button. Wire `run_query_btn.on_click`
  → `viewer.query(...)`. Remove the per-value `_on_query` watch.
- Comma-split each box into a `list[str]` (strip blanks) before calling `query`.

**Change (config.py):** replace `query: str` with `query_positive: str` and
`query_negative: str` (provenance only; written to `run_config.yaml`).

### 5. Push robustness — background dir-copy (live bug)

**Bug:** `RcloneClient.copy_local_to_remote` runs `rclone copyto` (single file→file
semantics) with a hardcoded `timeout=120`, but `SessionSource.push_outputs`
(sources.py:90) hands it a **directory**. Wrong verb + too-short cap → the observed
`Command '[... copyto ... PXL_...]' timed out after 120 seconds`.

**Change (sources.py):**
- `push_outputs` invokes `rclone copy <local_dir> collab-data:<bucket>/<path>` directly
  (correct recursive directory verb) with `--gcs-bucket-policy-only --transfers 8` and a
  generous `timeout=600`. Keep using `RcloneClient` for list/fetch/pull; push owns the
  dir-correct invocation. Reuse the client's `_cmd`/`remote_name` if accessible, else a
  small local subprocess call mirroring the client's pattern.

**Change (pipeline.py / app.py):** push runs **in a detached background thread** —
local outputs load into the viewer immediately when ready; the thread logs success or
failure to `op_log`. Push failure is non-fatal (outputs already on disk locally). The
push timer/log line (item 2) reports on completion of that thread.

### 6. Suppress meta-tensor warnings

**Cause:** talk2dino's outer `from_pretrained(low_cpu_mem_usage=False)` already avoids
its own meta-tensor no-op, but the warnings the user sees name `visual.transformer.*` /
`visual.ln_post.*` — talk2dino's **internal CLIP load** (remote HF code we don't control,
cannot pass `assign=True`). Harmless: talk2dino uses DINO as visual backbone and CLIP for
text only; the CLIP visual tower is unused, so the no-op drops nothing needed.

**Change (talk2dino.py):** scope a targeted
`warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")`
around / at extractor init, with a one-line comment explaining it is a harmless no-op on
CLIP's unused visual tower. Not a blanket filter.

## Out of scope

- Surfacing `reduction` / `temperature` / OF `motion_weight` / `coverage_weight` sliders.
- BA/LC toggles, camera localization, multi-video runs (deferred in predecessor spec).
- Patching collab-data's `RcloneClient` upstream (we route around it from our side).

## Testing

- **Item 1:** unit-test `_bind_visibility` toggles `min_disparity.visible` on sampling change.
- **Item 2:** assert each step appends a `"<step>: <Ns>"` line to `op_log.log_lines`
  (mock `perf_counter`).
- **Item 3:** assert `RunConfig()` defaults equal the new values; widget defaults match.
- **Item 4:** mock extractor; assert `viewer.query` calls `score_queries` with parsed
  positive/negative lists and feeds the result to `apply_viridis`. Blank neg → `None`.
- **Item 5:** mock subprocess; assert `push_outputs` builds an `rclone copy` (not
  `copyto`) command with `timeout=600`; assert push runs off the main load path and a
  failure does not raise into the caller.
- **Item 6:** assert no `UserWarning` matching the meta-tensor message escapes talk2dino init.
- Live smoke re-run (tmux, GPU) after implementation to confirm push succeeds and the log
  shows per-step timings.
