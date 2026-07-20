# Minimal Pipeline Runner — Design

**Date:** 2026-07-20
**Status:** Approved (pending spec review)
**Scope:** A clean, minimal interface for running the reconstruction pipeline on one or
more scene videos, plus wiring the localization-database stage that the existing
`Reconstructor` never exposed.

## Goal

Give the user a one-command way to run the minimal pipeline
(keyframe extraction → VGGT-Omega → talk2dino semantics + autoencoder compression →
localization database) over a **video file** or a **set of video files**, without hand-authoring
a per-scene dataset YAML. Reuse the existing `Reconstructor` end to end — no duplicated
config, caching, or stage logic.

## Background / current state

Most of the requested pipeline already exists and must be reused, not rebuilt:

- `collab_splats.wrapper.Reconstructor` (`reconstructor.py`) — 4 wired stages:
  `preprocess` (keyframe extraction), `pointcloud` (default backend `vggt_omega`),
  `semantics` (talk2dino extraction + `FeatureAutoencoder` compression, triggered by
  `semantics.n_components`), `mesh`.
- `collab_splats.wrapper.config.ConfigLoader` — merges `base.yaml` + `datasets/<name>.yaml`
  + dotted `KEY=VALUE` overrides via `mergedeep`. Plain YAML, no hydra/OmegaConf.
- `docs/examples/reconstruct.py` — argparse CLI: `--dataset NAME | --config PATH`,
  `--stages`, `--overwrite`, positional overrides. Dumps `run_config.yaml` for reproducibility.
- `docs/examples/run_all_datasets.sh` — loops **named datasets** sequentially.
- `configs/base.yaml` + `configs/datasets/*.yaml`.

### Two gaps this design closes

1. **No scene-by-video / multi-path entry.** The CLI only accepts a named dataset or a
   config path — you cannot point it at raw video files. `run_all_datasets.sh` iterates
   named yamls, not paths.
2. **Localization database creation is not a pipeline stage.** `Reconstructor.localize()`
   is a `NotImplementedError` stub; `run_pipeline`/`_STAGE_ORDER` have no localization stage;
   the `localization.*` config block is dead.

### Key architectural finding (localization DB)

`CameraLocalizer` performs **exhaustive local matching — there is no global retrieval index**
in localization. The DinoSalad retrieval extractor is used only by loop closure. Therefore the
"localization database" is exactly **one artifact**: the per-frame local-feature cache written
into `feedforward.zarr` at group `local_features/{extractor}/reconstruction`. Building it warm is
a single call:

```python
CameraLocalizer.from_feedforward(
    ff_result,
    extractor=BaseLocalExtractor.get(name)(),
    extractor_name=name,
    zarr_path=feedforward_zarr,
    radius=radius,
)
```

`from_feedforward` is cache-first: if the group already exists it loads (no GPU); on miss it runs
per-frame GPU extraction and calls `save_index`. This satisfies the requested
"index + local-feature cache" — warm, query-ready — because the local-feature cache *is* the DB.

## Components

### 1. New launcher — `docs/examples/run_scenes.py`

A thin script (sits beside `reconstruct.py`) that maps each input video to a `Reconstructor` run.

```
python docs/examples/run_scenes.py \
    --output-root /workspace/outputs \
    [--config configs/minimal.yaml] \
    [--stages preprocess,pointcloud,semantics,localize] \
    [--overwrite] \
    SCENE.MP4 [SCENE2.MP4 ...]
```

Behavior, per scene:

1. Load `base.yaml` (via `ConfigLoader`), optionally deep-merged with a shared `--config` YAML
   and dotted `KEY=VALUE` overrides. This is a **template**, not a per-scene dataset file.
2. Set `input_path = <video>` and `output_path = <output-root>/<video-stem>`.
3. Construct `Reconstructor(config)`, dump `run_config.yaml` to the scene's output dir
   (same audit behavior as `reconstruct.py`).
4. `run_pipeline(stages, overwrite)`.
5. **Continue-on-failure:** a scene that raises is logged and recorded; the loop proceeds to the
   next scene. Exit non-zero if any scene failed, printing a per-scene OK/FAIL summary at the end
   (mirrors `run_all_datasets.sh` resilience, but in Python for the multi-path case).

Args:

- `--output-root PATH` (required) — parent dir; each scene lands in `<root>/<video-stem>/`.
- `--config PATH` (optional) — shared override YAML merged over `base.yaml`.
- `--stages STAGE[,STAGE,...]` (optional) — forwarded to `run_pipeline`; default = config-enabled.
- `--overwrite` (optional) — forwarded.
- Positional: one or more video paths (shell globs expand naturally).
- Trailing `KEY=VALUE` overrides are **not** mixed with positional video paths to avoid ambiguity;
  shared overrides go through `--config`. (Simpler contract than `reconstruct.py`'s positional
  overrides; revisit only if needed.)

Scene input is a **video file** (`.mp4/.mov/.avi`). Directory-of-images and session-dir inputs are
explicitly out of scope for this iteration.

### 2. New stage — `Reconstructor.build_localization_db`

```python
def build_localization_db(self, result=None, overwrite=False) -> Path:
    """Build/refresh the per-frame local-feature localization cache in feedforward.zarr."""
```

- Config: `localization.extractor` (local matcher registry key), `localization.radius` (float).
- Reads `self.backend_dir / "feedforward.zarr"`; raises `FileNotFoundError` if absent
  (same guard style as `mesh`).
- Skip-if-exists: if group `local_features/{extractor}/reconstruction` present and not `overwrite`,
  log + return early.
- Loads `FeedforwardResult.load_zarr(feedforward_zarr)`, instantiates
  `BaseLocalExtractor.get(extractor)()`, calls `CameraLocalizer.from_feedforward(...)` as above.
- Returns the `feedforward.zarr` path (the DB lives inside it).
- A module-level helper `_build_localization_db(...)` holds the heavy-import body, matching the
  existing `_run_feedforward` / `_lift_and_save` / `_run_tsdf_mesh` pattern (imports inline in the
  helper so the module loads without GPU deps).

**Wiring:**

- `_STAGE_ORDER = ["preprocess", "pointcloud", "semantics", "mesh", "localize"]`
- `_STAGE_DEPS["localize"] = ["pointcloud"]` (needs `feedforward.zarr`; independent of semantics/mesh)
- `run_pipeline`: when `stages is None`, append `"localize"` if `config.localization.enabled`.
  Add a `localize` dispatch branch calling `build_localization_db`.
- The dead `localize(self, image)` stub is removed (query-time localization is out of scope;
  `CameraLocalizer.localize` remains the query API and is unaffected).

### 3. Config — `configs/base.yaml`

Fix and extend the `localization` block:

```yaml
localization:
  enabled: false          # opt-in stage
  extractor: loma          # local matcher (was 'dinosalad' — invalid: that is a RETRIEVAL extractor)
  radius: 8.0              # CameraLocalizer search radius
```

Default extractor is `loma`. `enabled: false` keeps existing dataset runs unchanged.

### 4. Existing example scripts — keep current

- `reconstruct.py`: add `localize` to the `--stages` help text and docstring; add one usage example
  with `localization.enabled=true`. No behavioral change beyond exposing the new stage name.
- `run_all_datasets.sh`: verify it still runs correctly with the new stage present (it passes no
  `--stages`, so behavior is governed by config `enabled` flags — unchanged while
  `localization.enabled` defaults false). **Open item:** the script hardcodes
  `/opt/conda/envs/reconstruction/bin/python`, while `CLAUDE.md` cites
  `/opt/venv/reconstruction/bin/python`. Reconcile to a single correct interpreter during
  implementation (confirm which venv is live before editing).

## Data flow

```
video.mp4
  └─ run_scenes.py: base.yaml (+--config) → config{input_path=video, output_path=root/stem}
       └─ Reconstructor.run_pipeline(stages)
            preprocess  → output/images/frame_*.jpg
            pointcloud  → output/vggt_omega/{colmap, transforms.json, feedforward.zarr}
            semantics   → output/vggt_omega/semantics/talk2dino/features.zarr (+ compressor.pt)
            localize    → output/vggt_omega/feedforward.zarr :: local_features/loma/reconstruction
```

## Error handling

- **Per-scene isolation** (`run_scenes.py`): one scene's exception never aborts the batch; failures
  collected and reported; process exits non-zero if any failed.
- **Missing `feedforward.zarr`** in `build_localization_db`: `FileNotFoundError` with a message
  directing the user to run the pointcloud stage first (matches `mesh` guard).
- **Invalid extractor name**: `BaseLocalExtractor.get(name)` raises via the registry — surfaced as-is.
- **Stage dependency violation**: existing `run_pipeline` validation raises `ValueError` if
  `localize` is requested without `pointcloud`.

## Testing

`tests/wrapper/test_reconstructor.py` (flat functions; mock GPU-bound `CameraLocalizer.from_feedforward`):

- `localize` present in `_STAGE_ORDER` and `_STAGE_DEPS`.
- `run_pipeline(stages=None)` includes `localize` iff `localization.enabled`.
- Requesting `["localize"]` without `pointcloud` raises `ValueError`.
- `build_localization_db` returns early when the `reconstruction` group already exists and
  `overwrite=False`; re-runs when `overwrite=True`.
- Missing `feedforward.zarr` raises `FileNotFoundError`.

`tests/examples/test_run_scenes.py` (new; mock `Reconstructor`):

- video path `foo/bar.mp4` + `--output-root /out` → `output_path == /out/bar`, `input_path == foo/bar.mp4`.
- Multiple video paths → one `run_pipeline` call each.
- A scene raising → batch continues; process exit code non-zero; summary lists FAIL for it.
- `--config` override YAML is deep-merged over `base.yaml`.

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py tests/examples/test_run_scenes.py`

## Out of scope

- Query-time localization (solving a pose for a new image) — the DB build only.
- Directory-of-images / session-dir scene inputs.
- Retrieval-index construction (does not exist for localization).
- Bundle adjustment / loop closure at the `Reconstructor` level (already flagged unimplemented).
- Parallel/concurrent multi-scene execution (sequential only; avoids GPU OOM per CLAUDE.md).

## Files touched

- `docs/examples/run_scenes.py` (new)
- `collab_splats/wrapper/reconstructor.py` (add stage + wiring; remove dead `localize` stub)
- `configs/base.yaml` (fix + extend `localization`)
- `docs/examples/reconstruct.py` (docs/help refresh)
- `docs/examples/run_all_datasets.sh` (verify + interpreter reconcile)
- `tests/wrapper/test_reconstructor.py` (extend)
- `tests/examples/test_run_scenes.py` (new)
