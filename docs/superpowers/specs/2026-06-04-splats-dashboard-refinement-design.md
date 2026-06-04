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

### 2. Granular progress + per-step timings in the log

Progress is coarse (`update_progress` at 5–20/25/60/80/95). The user wants **exact
progress**: image counts (`XX/XX`), which inference/processing step is running, AND the
same granularity when running a query — plus per-step wall-clock durations.

**Surface investigation.** The heavy steps expose **no progress callbacks** —
`BaseFeedforwardCreator.reconstruct(image_dir, output_dir)` (feedforward/base.py:723) and
`extract_and_cache` (features/base.py:84) take no hook. But the modules already **log**
the right detail per CLAUDE.md's "logging not print" rule: `extract_and_cache` emits
`"extract_and_cache: %d/%d frames written"` (base.py:151), the sampler already calls
`on_progress(done, total)`, and creators/mesh log step names. So granular progress lives
in Python `logging`, not in callbacks. We surface it without touching any core API.

**Change (operation_log.py) — logging bridge:**
- Add a `logging.Handler` subclass (`_OpLogHandler`) that forwards each emitted record's
  message into `op_log.log_lines` (thread-safe, reusing the existing lock/cap).
- `OperationLog` gains `attach_logging(*logger_names)` / `detach_logging()` context-manager
  helpers. `run_pipeline` attaches it to the `collab_splats` logger at `INFO` for the
  duration of the run, detaches in `finally`. Scoped to `collab_splats` only (no
  third-party noise), `INFO` level. This captures the real `XX/XX` frame counts, mesh
  steps, and creator step logs live.
- tqdm bars (model inference, `lift_features` `trange`) write to stderr, not `logging`,
  so they are not captured — accepted. The textual step/count logs are what the user asked
  for; tqdm bars stay in the tmux console.

**Change (pipeline.py) — per-step timings:**
- Wrap each step in `time.perf_counter()`; on completion append a summary line via
  `op_log.update_progress(pct, msg)` of the form `"pointcloud (vggt_omega): 42.3s"`.
- Steps timed: sample, pointcloud, mesh, semantics, push.

**Change (viewer.py / app.py) — query progress:**
- `viewer.query` is our own code, so it emits explicit op_log stage lines directly:
  `"query: encoding text"` → `"query: scoring P points"` → `"query: recolour done (Xs)"`.
  `query` takes an optional `op_log` (or a `progress` callback); `app` passes
  `self._op_log`. This shows the query's stages on every **Run query** press.

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

### 5. Push robustness — streamed background dir-copy (live bug)

**Bug:** `RcloneClient.copy_local_to_remote` runs `rclone copyto` (single file→file
semantics) with a hardcoded `timeout=120`, but `SessionSource.push_outputs`
(sources.py:90) hands it a **directory**. Wrong verb + too-short cap → the observed
`Command '[... copyto ... PXL_...]' timed out after 120 seconds`.

**Why collab-data's API can't help (debate).** Every `RcloneClient` transfer method is
**single-file, blocking, short-timeout** subprocess: `copyto` (`copy_local_to_remote`
120s, `copy_file` 60s), `cat`/`rcat` for read/write of one file. The only non-blocking,
long-lived method is `start_http_serve`, which `Popen`-streams `rclone serve http`. This
shape exists because the data dashboard's job is to **read/write small metadata files and
serve buckets for read-only browsing** — it never bulk-uploads a directory tree. So there
is no `copy`/`sync` method to reuse, and bumping the timeout to 600s would only paper over
the wrong verb with another arbitrary wall-clock cap that a slow link or larger tree
breaks again. The right tool is `rclone copy` (recursive, **idempotent** — reruns skip
already-uploaded objects), which collab-data simply never needed.

**Change (sources.py) — own the dir-copy, mirror their `serve` pattern:**
- `push_outputs` runs `rclone copy <local_dir> collab-data:<bucket>/<path>` via
  `subprocess.Popen` (streaming), with:
  - `--gcs-bucket-policy-only --transfers 8` (parallelism),
  - `--stats 2s --stats-one-line` (live progress lines on stdout),
  - rclone-**native** robustness instead of a python wall-clock kill:
    `--retries 3 --timeout 300 --contimeout 60` (per-operation/network, not whole-job).
- Stream stdout line-by-line; forward each `--stats-one-line` line to `op_log` (feeds
  item 2's live progress). Reuse `RcloneClient._cmd`/`remote_name` if accessible, else a
  small local subprocess mirroring the client's pattern.
- Keep using `RcloneClient` for list/fetch/pull unchanged.

**Change (pipeline.py / app.py) — background, non-fatal:** push runs in a **detached
background thread**. Local outputs load into the viewer immediately when ready; the thread
streams push progress and a final success/failure line to `op_log`. Push failure is
non-fatal — outputs are already on local disk, and `rclone copy`'s idempotence makes a
later manual retry cheap.

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
- **Item 2:** (a) logging bridge — emit a record on the `collab_splats` logger while
  attached and assert it lands in `op_log.log_lines`; assert detach stops capture.
  (b) per-step timing — assert each step appends a `"<step>: <Ns>"` line (mock
  `perf_counter`). (c) query — assert `viewer.query` pushes its stage lines to `op_log`.
- **Item 3:** assert `RunConfig()` defaults equal the new values; widget defaults match.
- **Item 4:** mock extractor; assert `viewer.query` calls `score_queries` with parsed
  positive/negative lists and feeds the result to `apply_viridis`. Blank neg → `None`.
- **Item 5:** mock `Popen`; assert `push_outputs` builds an `rclone copy` (not `copyto`)
  command carrying `--transfers`, `--stats-one-line`, `--retries 3`; assert streamed
  stdout lines reach `op_log`; assert push runs off the main load path and a non-zero exit
  logs a failure line without raising into the caller.
- **Item 6:** assert no `UserWarning` matching the meta-tensor message escapes talk2dino init.
- Live smoke re-run (tmux, GPU) after implementation to confirm push succeeds and the log
  shows per-step timings.
