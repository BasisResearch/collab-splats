# Pointcloud unify — design

- **Status:** draft for user review, 2026-09-27
- **Branch:** `clean/pointcloud-release`, worktree `.worktrees/pointcloud-release`, tip `c3d9dbf3`
  (code unchanged since `f5ae80f9`; merge-base with `clean/final` is `93cacd75`)
- **Supersedes:** `2026-09-27-pointcloud-reorg-design.md` and `plans/2026-09-27-pointcloud-reorg.md`
  (reference only — both predate the contract and contradict it in 19 places,
  `/workspace/scratch/pc-release/reorg_contract_audit.md`)

## Goals

One result type, one creator path, one implementation per operation. Net-negative in lines.

1. **One result type.** `PointcloudResult` absorbs `FeedforwardResult`; the pycolmap-wrapping
   `PointcloudResult` is deleted. COLMAP is an export, never read back into a second type.
2. **One depth module.** `vda.py` + `depth_align.py` → `pointcloud/depth.py`.
3. **One creator path.** Feedforward and sfm creators both implement `create(images_dir, out_dir) -> PointcloudResult`;
   the sfm skeleton copied across instantsfm/colmap/hloc (and `Reconstructor._run_sfm`, `eval.py`) lives once.
4. **Listed dedups**, each its own row below: `geometry/projection.py`, `lift_features` move,
   clean/subsample order, SIFT DB reuse rewrite, InstantSfM in-memory conversion.

## Contract (binding — grep it before adding, renaming or deleting a name)

- landed decisions `93cacd75` (consistency) and `f612e69d` (geometry round3); ADR 017, 018, 019;
  consistency phase-2 deferred table (`plans/2026-09-26-consistency-phase2.md` "Deferred"); CHANGELOG; CLAUDE.md
- K model → original is two lines, each commented, never a wrapper:
  `K = rescale_intrinsics(K, model_hw, crop_hw)` then `K = shift_intrinsics(K, box[..., :2])`
  - written once, in `PointcloudResult.__post_init__` when `intrinsics is None` (**amended**, user
    2026-09-28); no consumer converts
  - why: LC `_assemble_result` and the eval BA branch never reach base `postprocess`
  - callers pass `model_intrinsics=K, intrinsics=None`; sfm `align_depth` and `load_zarr` pass both
  - full-res `intrinsics` is float32, as today; `to_colmap` upcasts, old-vs-new check at float32
- crop boxes (**amended here**, user 2026-09-27): one shared `center_crop_coords(orig_wh, resized_wh, crop_wh, scale)`
  in `feedforward/base.py`; each backend keeps only its resized-size rule (VGGT-X ×14 rounding, Omega
  aspect band, MapAnything floored cover-resize); `full_frame_coords` stays for LoGeR
  - bit-exact vs today's three functions: 3,007 sizes × (VGGT-X, Omega, 4 MapAnything grids), 0 mismatches
- ADR 018: colmap/hloc `min_registered_frac` floor, instantsfm strict; hloc stays an editable path source;
  `SFM_CREATORS` stays a `{name: Class}` dict with top-level imports
  - **amended here:** the provenance attrs (`pycolmap_version`, `hloc_commit`) and every `provenance()`
    are removed — written to zarr attrs, read by nothing but their own tests; ADR 018 gets a superseding note
  - `HLOC_PIN` goes with them (`setup/hloc.sh` carries its own `COMMIT` pin)
- ADR 017: tunables are kwargs with defaults (depth keeps `depth_width`, `device`, `fp32`);
  a public helper needs ≥ 2 real callers; decode with `read_image` / `read_frames`, never `cv2.imread`
- LC wrapper form review is deferred: `geometry/loop_closure/*` gets call-site edits only
- code style: no `f(g(x))`, imports at top, US spelling, comment runs are header + `- ` bullets
- no notebooks, no merge, no push

## Why: the circularity today

| Path | Flow |
|---|---|
| ff | postprocess → `FeedforwardResult` → `write_colmap` → `PointcloudResult(recon)`; wrapper also `save_zarr`s the ff result |
| sfm | mapper → pycolmap model → `result_from_reconstruction` → `FeedforwardResult` → zarr; also returns `PointcloudResult(recon)` |
| reload | `PointcloudResult.from_colmap(sparse/0)`, frame order taken from the zarr attrs |

- consumers of the pycolmap wrapper need only w2c poses, original-res K, sparse points + colors:
  build-stage SOR clean, mesh (`_run_tsdf_mesh`), splats, refine reload, LC wrapper return annotation
- `.intrinsics` means model-res K on one type and original-res K on the other; nothing asserts which
  - fixed by naming: `intrinsics` always full-res, `model_intrinsics` always model grid (§1)
- **latent bug:** `build_pointcloud` SOR-cleans the in-memory model and rewrites only the PLY;
  `colmap/sparse/0` on disk keeps the uncleaned points, so an in-process splats run seeds from the
  cleaned set and a `--stages splats` rerun (via `from_colmap`) from the uncleaned one

## Design

### 1. `PointcloudResult` — `pointcloud/base.py`

`FeedforwardResult` moved here and renamed. Fields as today (`points`, `colors`, `extrinsics`,
`image_paths`, `original_coords`, `model_width`, `model_height`, optional dense and `mv_*` arrays) except K,
which is stored twice (user decision 2026-09-27):

| Field | K on | Set by | Used by |
|---|---|---|---|
| `intrinsics` | full-res frame (the old `PointcloudResult` meaning) | `PointcloudResult.__post_init__`, once, via the two-line rescale + shift; sfm `align_depth` directly | mesh, splats, `to_colmap`, localization, dashboard |
| `model_intrinsics` | model grid, matching `depth` / `world_points` / `pixel_indices` | the model (ff); `align_depth` rescale to depth res (sfm) | refine/BA, `reproject`, multiview, lifting |

- resolution-free outputs (poses, world points, colors) need no mapping
- dense per-pixel arrays stay on the model grid: full-res is ~20x storage (300 × 1080p ≈ 2.4 GB per float
  channel); mesh and splats lift depth through the crop box at use, as today
- the rename is deliberate: any site still reading model-grid K as `intrinsics` must be touched, never
  silently picks the wrong K
- zarr schema clean break (P7): `load_zarr` raises a stale-schema error naming the pointcloud-stage rerun
  when `model_intrinsics` is missing — an old zarr's `intrinsics` is model-grid K and must never load as full-res
- methods: `save_zarr`, `load_zarr`, `reproject` (unchanged), plus
  `to_colmap(camera_model) -> pycolmap.Reconstruction`, the one place a result becomes a COLMAP model
  - absorbs `build_pycolmap_reconstruction` and `write_colmap`; reads `intrinsics` directly, so
    `_rescale_reconstruction_to_original_dimensions` is deleted
  - written at original resolution, as today
- `save_zarr` codec → `utils.io.LZ4` (owed deferred row)
- `write_ply(path)` kept as a method, rewritten on `points` / `colors` (no pycolmap `export_PLY`);
  the PLY is a points + colors viewer/seed file, not a COLMAP model
- deleted: old `PointcloudResult`, `from_colmap`

Wrapper flow after the change:

```
creator.create(images_dir, backend_dir, colmap_model_dir) -> result   # ff and sfm; cleans, writes the COLMAP model
result.save_zarr(pointcloud_zarr, extra_attrs=...)
result.write_ply(backend_dir / "sparse_pc.ply")
stage 2+: PointcloudResult.load_zarr(pointcloud_zarr)                   # never from COLMAP
```

- `colmap_model_dir` is a wrapper path property beside `pointcloud_zarr`: the one place `colmap/sparse/0` is spelled
- SOR clean moves from `build_pointcloud` into `create` (P2 order: conf mask → clean → subsample), so the zarr,
  the COLMAP model, the PLY and the splat seed hold one cleaned point set (fixes the latent bug)
  - ff: writes `result.to_colmap(...)` after its clean
  - sfm: drops the cleaned-out `points3D` from the mapper model, then writes it (SIMPLE_RADIAL + tracks kept)
- LC wrapper `reconstruct` → return annotation and call-site edits only

### 2. `pointcloud/depth.py`

`vda.py` + `depth_align.py` merged; plain functions, no class.

| Function | From |
|---|---|
| `_load_vda_model(device)` | `vda._load_vda_model` |
| `estimate_depth(frames, out_dir, names, *, depth_width=518, device="cuda", fp32=False) -> np.ndarray` | `generate_vda_depth`; folds in `vda_depth_complete` (a cache miss wipes `depth_vda/` then regenerates) |
| `align_depth(recon, depths, images, names, *, min_obs=20) -> tuple[PointcloudResult, dict]` | `result_from_reconstruction` + its three private helpers; sets `intrinsics` (COLMAP K) and `model_intrinsics` (rescaled to depth res) |

- `depth_vda/images/npy/` layout kept (InstantSfM's reader and existing caches depend on it)
- `depth_scale*` attrs kept
- numpy homogeneous transforms → `transforms.transform_points` (round3d follow-up)
- callers: `sfm/base.py`, `sfm/instantsfm.py` docstrings, `evals/scripts/eval.py`

### 3. One creator path

```python
class BasePointcloudCreator(ABC):
    def create(self, images_dir: Path, out_dir: Path, model_dir: Path) -> PointcloudResult: ...
```

- `out_dir`: backend scratch (SIFT DB, `depth_vda/`, mapper output); `model_dir`: where the COLMAP model goes

**Feedforward** — `BaseFeedforwardCreator.create` = today's `reconstruct` minus the PLY (moves to the wrapper, §1);
its COLMAP write becomes `write_colmap_reconstruction(self.result.to_colmap(...), model_dir)`. The five-step template stays.

**SfM** — new `sfm/base.py`:

```python
@dataclass
class BaseSfmCreator(BasePointcloudCreator):
    def create(self, images_dir, out_dir, model_dir) -> PointcloudResult:
        # 1 names from preproc.frames; VDA depth, cached
        # 2 recon = self._map(images_dir, out_dir, names)      # backend-specific, abstract
        # 3 registered subset: self._registered_rows(recon, names)
        # 4 SOR clean; drop cleaned-out points3D; write_colmap_reconstruction(recon, model_dir)
        # 5 result, attrs = align_depth(recon, depths[rows], keyframes, names)
        # 6 self.attrs = {"method": "sfm", **subset_attrs, **attrs}
```

- backends implement `_map` only
- registered subset from the in-memory model (`recon.reg_image_ids()`), not a disk re-read
- `_registered_rows`: base = strict (instantsfm); `_IncrementalSfmCreator` (private, colmap + hloc)
  adds the `min_registered_frac` field, `__post_init__` range check, floor + warning — moved verbatim
  from the wrapper
- `sfm/common.py` deleted: `prepare_sfm_dirs`, largest-model pick and `write_sfm_model` fold into the base;
  `sfm_image_dir` inlined; `rename_images_to_stems` into `write_colmap_reconstruction`
- `Reconstructor._run_sfm` → `creator.create()` + `save_zarr`; `_SFM_BLOCK_KEYS` special-casing of
  `min_registered_frac` goes
- `evals/scripts/eval.py` `_run_instantsfm` → `creator.create()`; its PIL decode → `read_image`
- `SFM_CREATORS` dict unchanged (ADR 018)

**Feedforward boilerplate** (net-negative only; else dropped):

- synthetic `image_paths` ×4, `extract_intermediate_features` near-twins (vggtx / omega)
- 4 numpy `transform_points` sites in `feedforward/base.py` (round3d follow-up)
- owed rows: `_decode_dir_to_frames` → `read_image`; `get_device` / `pytorch_gc` sites (phase-2 deferred)

### 4. Listed dedups

| Item | Change | Kind |
|---|---|---|
| COLMAP IO | `utils/colmap.py`: `write_colmap_reconstruction(recon, model_dir)` (lane A's atomic `write_model`, renamed; writes exactly `model_dir`, image names as stems) and `read_colmap_reconstruction(model_dir)` (kept by user decision 2026-09-27 despite no production caller: the inverse of the writer, for external models and tests — an explicit ADR 017 two-caller exception). No layout constant, no `sparse_dir`, no `is_complete`: the wrapper owns the layout as a path property `colmap_model_dir = backend_dir / "colmap" / "sparse" / "0"` beside `pointcloud_zarr`, and passes it to `create`; the atomic swap means the dir exists only as a whole model, so the done check is `pointcloud_zarr.exists() and colmap_model_dir.exists()` (pre-unify partial models: re-run the stage) | refactor |
| `geometry/projection.py` | lane B: `unproject`, `project` (torch). Lands in the same step as its callers: `_raw_to_world_points`, `_verify_geometry`, multiview loop, `_frustum_world_aabbs`, `depth.align_depth`, `reproject_pixels` (deleted; `reproject` uses `project` with `model_intrinsics`), `lift_features`, `metrics` NCC, `reconstructor` refine, BA :674, `evals/scripts/{analyze_splats,refit_at_fixed_poses}.py`. Exempt: `mesh/texture.py`, `mesh/tsdf.py` | P1 |
| crop boxes | `center_crop_coords` shared by VGGT-X / Omega / MapAnything (§Contract); the old-vs-new equality test over the size sweep is committed with it | refactor, bit-exact |
| PGSR removal | lane B `f338b732` (ADR 019) | refactor (splats) |
| `lift_features` | `pointcloud/utils.py` → `semantics/lifting.py`, with `_grid_sample_at_pixels` / `_sample_at_source_pixels` | refactor (AST-equal body) |
| subsample | one `subsample_points(mask, max_points, seed=0) -> mask` replaces `_limit_trues` + the array form | P3 |
| clean / subsample order | conf mask → SOR clean → subsample, inside `create` (today: creator subsamples, wrapper `build_pointcloud` cleans after) | P2 |
| cross-view depth test | `projection.depth_residual` shared by multiview confidence and lifting | P4 |
| SIFT DB reuse | reuse iff DB image names == `names` AND a params row in the DB == the call's params; `<db>.json` sidecar deleted | P8 |
| InstantSfM export | `_patch_instantsfm_colmap_write` + `WriteGlomapReconstruction` + read-back → private in-memory `_to_pycolmap(cameras, images, tracks)`; keeps cluster-0 selection (ADR 018); stale "verify stage" comment fixed | refactor, test first |
| ff COLMAP camera | `to_colmap` writes `PINHOLE` from stored K (drops the SIMPLE_PINHOLE mean path) | P6 |
| K stored twice | `intrinsics` (full-res) + `model_intrinsics` (model grid), §1; `load_zarr` stale-schema error | P7, with §1 |

## Risks and required handling

1. **K meaning.** Resolved by storing both Ks (§1): `intrinsics` keeps the old full-res meaning,
   model-grid K moves to `model_intrinsics`.
   - old-vs-new equality test of the full-res K each consumer receives (mesh, splats, `to_colmap`,
     localization, dashboard) on a **cropped** fixture (a full-frame box makes both K equal)
   - old zarrs refused by the P7 stale-schema error, never misread
2. **Loop closure bypasses `create`.** The LC wrapper runs `load_model → … → postprocess → build_colmap` itself.
   - SOR clean lives in `postprocess` (a template step), but LC's assembled path (frames ≥
     submap_size) skips base `postprocess` — Task 4 must cover it; not free
   - the COLMAP write is one creator method that `create` and the LC wrapper both call — call-site edit only
3. **Refine re-dirties the cloud.** `refine_poses` → `reproject()` rebuilds points from depth, uncleaned.
   - refine runs clean → subsample after `reproject()`, then rewrites zarr, COLMAP model and PLY
   - own commit (refine's output moves)
4. **P2 moves every ff point set** (clean before subsample: different survivors, counts).
   - moves: splat seed (splats run on ff backends too), lifted semantics
   - unchanged: TSDF mesh (fuses depth, not points), localization (`world_points`)
   - splat PSNR baselines from before this work are not comparable
   - processed scenes keep their old zarr: re-run the pointcloud stage (P7 refuses them anyway)
5. **Import breakage accepted.** ~17 prod files, ~45 tests updated in the commits that move names;
   6 tutorial notebooks (`02_pointcloud` ×3, `05_lifting`, `06_mesh`, `07_localization`) left broken,
   fixed after this lands (tutorial-rework). No shims.

## Out of scope

- LC wrapper form, moving `cross_frame_attention_ratio` / `mean_top_quarter` (deferred review)
- `read_frames` rework (later pass); VGGT-X mixed-orientation pad offsets (lane C2 `3a5532f2`, parked)
- `SFM_CREATORS` vs `get_creator` merge; hloc install change; SIFT DB half of unified COLMAP IO (vismatch-fork)
- zarr field renames, `depth_vda/` rename, notebooks

## Lanes

| Lane | Disposition |
|---|---|
| A `2ee04ba7` + `f7caf1a1` | gate `f7caf1a1` (unverified), then cherry-pick as §4 COLMAP IO, trimmed to the two functions and renamed |
| B `7a6ce23a` `f338b732` `c2202a71` | confirm the hand-formatted matrix in `transforms.py` survived; gate; land with the P1 caller migration. `c2202a71`'s torch-typed `transform_points` kept (user 2026-09-27): body unchanged, annotations only; `projection.py` and the P1 torch sites call it instead of hand-rolling `R @ p + t` |
| C `92e6fc1e` `a6fd4ed6` | rejected (the `intrinsics_to_original` wrapper); its crop helper signature is reused as `center_crop_coords`, re-applied fresh with the sweep test |
| C2 `d19abb0e` | lands (owed docstring row) |
| C2 `3a5532f2` | parked |
| dirty plan amendment in the worktree (+225/−458) | discarded 2026-09-27 (outdated by this spec); copy at `/workspace/scratch/pc-release/old-plan-amendment.md` |

## Gates

- per commit: G′ (`$SP/gprime.sh <wt> <log>` → "IDS IDENTICAL") + `tests/utils tests/splats tests/semantics`,
  one full gate at a time, from the worktree with the `collab_splats.__file__` proof line
- per commit: `$SP/parity.py --check` (synthetic, CPU, no weights, bit-exact); the script's imports
  follow the renames (`depth_align` → `depth`, `FeedforwardResult` → `PointcloudResult`,
  model-grid `intrinsics` → `model_intrinsics`)
- baseline re-recorded at `c3d9dbf3` before the first commit
- refactor commits: parity bit-exact, net-negative lines
- parity commits (P1–P4, P6–P8): unit test first on a fixture that can see the change (non-identity poses,
  cropped boxes); `--check` fails only on the fields expected to move, diff in the commit body;
  `--save` re-baseline only with user OK
- COLMAP export and crop/K are invisible to parity: `to_colmap` and the mesh/splats original-res K get
  an old-vs-new equality test on a fixture
- no pipeline reruns per change; one real smoke at the end — ff + sfm on C0043 through `Reconstructor`,
  every artifact written, no quality measurement
- one review per task covering spec, contract and quality; the reviewer gets the contract section

## Order

1. baseline G′ + parity at `c3d9dbf3`
2. COLMAP IO (lane A)
3. `PointcloudResult` single type + wrapper flow + K stored twice / P7 (§1)
4. `depth.py` (§2)
5. sfm skeleton + `_run_sfm` / eval shrink (§3)
6. InstantSfM in-memory export
7. feedforward boilerplate (§3)
8. `lift_features` move
9. lane B + P1 caller migration
10. P3, P2, P4, P6, P8 — one commit each
11. C2 docstring; real smoke; CHANGELOG entry

## Resulting layout

```
pointcloud/
  base.py        PointcloudResult, BasePointcloudCreator
  depth.py       estimate_depth, align_depth
  utils.py       clean_pointcloud, confidence_mask, subsample_points, LC attention helpers
  feedforward/   base.py (template + multiview), vggtx, vggt_omega, mapanything, loger
  sfm/           base.py (BaseSfmCreator, _IncrementalSfmCreator), sift_db, colmap, hloc, instantsfm
geometry/projection.py   unproject, project, depth_residual
semantics/lifting.py     lift_features
utils/colmap.py          write_colmap_reconstruction, read_colmap_reconstruction
```

Deleted: `pointcloud/vda.py`, `pointcloud/depth_align.py`, `sfm/common.py`, `splats/pgsr.py`.
