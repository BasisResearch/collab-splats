# Reconstructor cleanup — readable config + one source of defaults

**Date:** 2026-07-20
**Status:** Design — approved, pre-plan
**Scope:** `collab_splats/wrapper/reconstructor.py`, `configs/base.yaml`, `docs/examples/run_pipeline.py`

## Problem

`Reconstructor` is not overengineered as a *class* — it's a thin 5-stage orchestrator.
The pain the user hit ("hard to read / understand how to specify a pipeline") lives in the
**config interface**, and in three concrete ways:

1. **Defaults duplicated and drifting.** Every knob's default exists twice: once in
   `configs/base.yaml`, once inline as `cfg.get("key", default)` scattered across eight
   methods. They have already drifted:
   - `min_frames`: base.yaml `150` vs code `pre_cfg.get("min_frames", 300)`
   - `backend`: base.yaml `vggt_omega`, `validate_config` default `vggtx`, `backend_dir`
     property default `nerfstudio` — a three-way disagreement
   - `frame_selection: fps` — a label the code never honours (`_extract_frames` only
     branches on `optical_flow`, else uniform)
   There is no single source of truth, so YAML and code disagree about what the pipeline does.

2. **Dead / aspirational knobs the reader must puzzle over.** The config surface advertises
   options that silently do nothing or raise deep in a stage:
   - `pointcloud.method: sfm` → `NotImplementedError`
   - `pointcloud.bundle_adjustment` → logs a warning, does nothing
   - `mesh.mesher: poisson` → validated as legal, but only tsdf is wired
   - `clean.confidence_threshold` → present in YAML, never read

3. **Dead code.** `Reconstructor.from_config_file` has zero callers and calls
   `ConfigLoader.load(dataset=...)` against the now-deleted `configs/datasets/` directory —
   it is both unused and broken.

## Goal

The whole config → execution path reads as one coherent thing, for **both** readers:
- **User** authoring a run: one commented `base.yaml` lists every knob; override by
  supplying only the keys you change.
- **Developer** reading the class: methods read top-to-bottom as labeled logical blocks with
  no scattered defaults and no dead branches.

## Approach

Keep the config a **plain dict** (full YAML flexibility — any override key, deep-merged).
Make `base.yaml` the single source of defaults *and* the human-readable schema. No typed
config layer: a `TypedDict`/dataclass schema is closed by default, so every new knob would
need a schema edit and free-form overrides would read as type errors — that fights the
YAML-flexible design. Readability for the developer comes from `base.yaml` + clean code,
not from a type wrapper.

### Config override model (documented, mostly unchanged)

Three layers, deep-merged, highest precedence wins:

1. `configs/base.yaml` — every default + documented schema. **Sole** source of defaults.
2. `--config my.yaml` — user YAML with only the keys they change, merged over base.
3. programmatic `overrides` dict — entrypoint sets `input_path`/`output_path` per video;
   notebooks may pass any override.

Each run dumps the merged config to `run_config.yaml` for reproducibility (unchanged).

### One populate path

`Reconstructor.__init__(config, config_dir=DEFAULT_CONFIG_DIR)`:
- `DEFAULT_CONFIG_DIR = Path(__file__).parents[2] / "configs"` (repo-relative, matches
  `run_pipeline.py`'s `_REPO_ROOT` derivation).
- Load `config_dir/base.yaml`, deep-merge the passed `config` over it (`mergedeep.merge`).
- Result: every key is guaranteed present, so **all** `cfg.get("k", default)` calls become
  flat `cfg["k"]` reads. Genuinely-optional-null values (e.g. `max_frames`, `voxel_size`)
  stay as `cfg["k"]` returning the `null` from base.yaml.

`run_pipeline.build_scene_config` drops its manual `merge({}, loader.base_config)` — it just
passes `{**override_config, input_path, output_path}` and lets `__init__` merge base.

### Not-yet-implemented → loud, documented stubs

Silent no-ops become explicit failures the caller can see:
- reconstructor-level `bundle_adjustment=True` → raise `NotImplementedError` with the same
  guidance the current warning gives (pass BA to the creator config directly).
- `mesh.mesher: poisson` → `mesh()` raises `NotImplementedError` (only tsdf wired).
- `pointcloud.method: sfm` → keep the existing `NotImplementedError` in `_run_sfm`.
- `clean.confidence_threshold` → mark as not-yet-read in base.yaml.

`base.yaml` groups these under a `# ── NOT YET IMPLEMENTED ──` banner so status is visible
at the point of configuration.

`validate_config` stays **shape-only**: required `input_path`/`output_path`, method ∈ valid
set, backend ↔ method compatibility, mesher ∈ valid set. The "not implemented yet" raises
live in the stage methods, not in validation.

### Honest label

`frame_selection: fps` → `uniform` in `base.yaml` and `_extract_frames`. Valid set becomes
`{uniform, optical_flow}` (uniform is the else-branch today; the name now tells the truth).

### Delete dead code

Remove `Reconstructor.from_config_file` (no callers, broken against deleted `datasets/`).

### Readability pass (all stage methods)

Apply the house style (CLAUDE.md) consistently so each method reads as labeled logical blocks:
- `########` dividers between constants / helpers / class.
- One short block comment per logical step (not per line).
- Flat `cfg["k"]` reads instead of defensive `.get(k, default)` noise.
- Keep only the lazy heavy imports that must stay inline; remove clutter that obscures flow.
- One-line docstrings, no padding.

Target: `build_pointcloud`, `run_pipeline`, `extract_semantics`, `mesh`, `preprocess`,
`build_localization_db` each readable top-to-bottom without cross-referencing.

## Out of scope (explicitly not touched)

- Structure of the module-level helper functions (`_extract_frames`, `_run_feedforward`,
  `_lift_and_save`, `_run_tsdf_mesh`, `_build_localization_db`) — they keep the class thin.
- `Splatter` (separate, nerfstudio-coupled).
- `ConfigLoader` internals (add nothing beyond what `__init__` needs to load base).
- No new config schema type (TypedDict / dataclass / pydantic).

## Implementation principles

- **Reuse, don't add.** Use existing `mergedeep.merge` and `ConfigLoader`; add no new
  config machinery.
- **Delete what this obsoletes.** `from_config_file` goes; every inline default goes.
- **No new abstraction.** Config stays a dict. No wrapper types.
- **Loud over silent.** Unimplemented paths raise `NotImplementedError`, never no-op.

## Testing

- `validate_config`: required-field and method/backend/mesher-compat raises (existing tests
  kept; add cases for merged-defaults presence).
- `__init__` merge: passing a partial config yields a fully-populated `self.config` with
  base.yaml defaults filled; a user override wins over the base value.
- Drift fixes: assert `min_frames`, `backend`, `frame_selection` resolve to the base.yaml
  value (no stale code default).
- Not-implemented stubs: `mesh(mesher=poisson)`, reconstructor-level `bundle_adjustment=True`
  raise `NotImplementedError`.
- Run the suite: `/opt/venv/reconstruction/bin/python -m pytest tests/`.

## Success criteria

- Zero `cfg.get(key, default)` with an inline default in `reconstructor.py`.
- `base.yaml` is the only place a default value appears.
- No default disagreement between `base.yaml` and code.
- Every advertised knob either works or raises `NotImplementedError` with a clear message.
- `from_config_file` removed; suite green.
