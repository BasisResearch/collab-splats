# Module consistency audit — design

- **Status:** audit run 2026-10-08; findings below await triage; fix plan written only after triage
- **Supersedes:** phases 3-7 of [2026-09-26-consistency-design.md](2026-09-26-consistency-design.md)
  - phases 1, 1b, 2 of that spec are landed on `clean/final` and stay as-is
  - its 2026-09-26 audit predates the geometry, pointcloud, mesh, splats, reconstructor,
    localization and dashboard releases; its findings are not carried over unverified
  - phase 4 `SceneLayout` / kind-tagged registry conflict with later rules (no new classes,
    no layout constants) and are dropped unless this audit re-derives them
- **Target:** `collab_splats/` on `clean/final` @ `0ed7dd09`
- **Out of scope:** `evals/`, `tests/`, notebooks, scripts — counted as callers only

## Goal

One implementation per job, one way of doing each job, across `collab_splats/` subpackages.

- no function implemented twice across modules
- copies that disagree on a convention found and fixed (highest bug risk)
- same job done the same way in every package
- one name per meaning in public APIs

## Dimensions

| Dimension | Finds | Fix shape |
|---|---|---|
| Redundant implementation | same logic in 2+ places | keep one, delete the rest |
| Convention divergence | copies disagreeing on pose frame, K resolution, image layout/dtype, NaN policy, units | test-first bug fix, then dedup |
| Pattern divergence | device selection, GC, logging, registry, return kind, error type, zarr codec, config plumbing | adopt the majority / existing shared helper |
| API naming | one meaning, several param or return names | rename to the most-used name |

## Process

Four steps; steps 1-3 are read-only.

### 1. Inventory

Eight read-only agents, one per slice, on the main checkout.

| Agent | Slice |
|---|---|
| 1 | `geometry/` |
| 2 | `semantics/` |
| 3 | `pointcloud/` |
| 4 | `splats/` |
| 5 | `preproc/` + `utils/` |
| 6 | `mesh/` + `localization/` |
| 7 | `dashboard/` + `viewer.py` |
| 8 | `reconstructor.py` + `__main__.py` + `remote.py` |

One JSON row per function or method, private included:

- `file:line`, `qualname`, signature
- `does`: one-line semantic from a shared verb vocabulary (`invert_pose`, `rescale_intrinsics`,
  `read_image`, `open_zarr`, `to_numpy`, `select_device`, ...); agents add verbs when none fit
- `conventions`: pose frame (w2c / c2w, 3x4 / 4x4), K resolution (model / full),
  image layout and dtype, device, NaN policy, units — only those that apply
- `patterns`: logging, device selection, GC, registry, return kind, error type, zarr codec, config plumbing
- `params`: names of semantically loaded params (`conf_*`, `*_dir`, `poses` / `extrinsics`, ...)
- `callers`: call count inside `collab_splats/`

### 2. Cross-join

Done by the orchestrating session over the merged inventory.

- verb synonyms merged first, so `inv_pose` and `invert_extrinsics` land in one group
- redundancy candidates: groups by `does` with more than one row
- convention candidates: rows in one group with conflicting `conventions`
- pattern tally: per package, which pattern each job uses
- naming table: param names grouped by meaning

### 3. Verify

One adversarial agent per batch of candidate groups, reading the code, not the inventory.

- `REAL`: same semantic; copies interchangeable or differ by accident
- `INTENTIONAL`: difference is load-bearing (e.g. PIL-exact preproc matching an upstream loader);
  reason recorded, no fix
- `BUG`: copies disagree on a convention and one is wrong; must name the input giving wrong output
- `DEAD`: copy with zero callers; `main` and `origin/db-*` grepped before the verdict
- unverified candidates are dropped, not reported

Each finding is tagged `conflicts_with` when it touches files changed on an unlanded branch:
`clean/localization`, `feat/rgbd-ba-cf`, `perf/inplace`, `perf/splats-speed`, `clean/rclone-api`, `cleanup/fc-*`.

### 4. Report and triage

- published as an artifact; the same findings table appended to this spec under `## Findings`
- columns: id, dimension, verdict, files, canonical, proposed fix, LOC delta, risk, `conflicts_with`, triage
- the user fills `triage` (keep / drop / defer); the plan covers `keep` rows only

## Canonical choice

For each `REAL` group, in order:

1. a copy already in `utils/` or `geometry/transforms.py` / `geometry/projection.py` wins
2. otherwise the copy with more callers, tests, and correct conventions
3. a new shared home only when 2+ packages need it — a plain function, never a class

## Fix constraints

- behavior-preserving, except `BUG` rows
- `BUG` rows are test-first on a fixture that can see the convention:
  non-identity poses, off-centre principal point, cropped or non-square images;
  a test that passes on the unfixed code is rejected
- losing copies deleted outright: no shims, no re-exports, no wrappers
- tunables stay function keyword defaults; one call per line; absolute imports; docstring contract
- renames take the name with the most call sites
  - a rename that changes a checkpoint key, zarr attr, or config key is flagged and not done blind

## Fix phase

Only after triage; its plan goes in `docs/superpowers/plans/`.

- **Branch:** `clean/consistency-audit`, worktree `.worktrees/consistency-audit`,
  forked from the `clean/final` tip at plan time
- **Commits:** one group per theme, independently revertible
  - `BUG` fixes are `fix(scope):` commits ahead of any dedup touching the same code
- **Deferred:** `conflicts_with` rows wait for their branch to land; listed in the plan, not done
- **Tests:**
  - `cd .worktrees/consistency-audit && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest ...`
  - gated on a printed `collab_splats.__file__`; `third_party/*` symlinked in
  - control run of the same per-package gates on the fork point, recorded first
  - pass/fail counts match control except new `BUG` tests (fail on fork, pass on branch)
  - exit codes checked directly, never through a pipe
- **Behavior proof for dedups:** existing tests, plus a bit-identical array comparison on the
  tutorial scene for any pipeline stage whose code path changed
- **Landing:** squash per theme onto `clean/final` immediately; backup ref under
  `refs/backup/consistency-audit/`; push is the user's call
- **Bookkeeping:** CHANGELOG entry; CLAUDE.md `consistency` line points here, phases 3-7 marked superseded

## Findings

- audit run 2026-10-08 on `clean/final` @ `0ed7dd09`; 724 inventory rows, 94 verified candidates
- report: https://claude.ai/artifact/4u6GaAtqFPDj2vCPQ3krBF
- `perf/inplace` dropped from `conflicts_with`: branch merged; its 12 surviving uncommitted ports are rows P1-P12
  ([handoff](../handoffs/2026-10-08-perf-inplace-port-handoff.md))
- `triage`: outcome as of 2026-10-09; "held" = committed on an `audit/*` branch or saved as a diff, waits on the other session's WIP in the same file

| id | dimension | verdict | files | canonical | proposed fix | LOC | risk | conflicts_with | triage |
|---|---|---|---|---|---|---|---|---|---|
| A2 | convention | BUG | `splats/trainer.py:252`<br>`splats/checkpoint.py:94`<br>`splats/checkpoint.py:198` | geometry/transforms.py:shift_intrinsics | render_tsdf_inputs returns the center K (shift_intrinsics(K, (-0.5,-0.5))) so both sources hand mesh/ a center K; create_tsdf_mesh shifts +0.5 once before passing K to Open3D (floor indexing). | 4 | med | feat/rgbd-ba-cf, perf/splats-speed, cleanup/fc-integrate, cleanup/fc-lint2, clean/localization | landed 0c64634a, 1ed70437 |
| A3 | convention | BUG | `localization/localizer.py:62`<br>`localization/localizer.py:978`<br>`geometry/projection.py:174` | localization/localizer.py:_crop_to_model_grid | shifted = px - box[:2] + 0.5; return shifted * scale - 0.5. | 2 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed f8b09477 |
| A3n | convention | BUG | `localization/localizer.py:45`<br>`localization/localizer.py:76`<br>`localization/localizer.py:965` | geometry/transforms.py:shift_intrinsics | Pick one convention at the localize() boundary: shift pts by +0.5 into pycolmap and K back by -0.5 on return (package center K), and map pts2d with (p+0.5)*s-0.5; document query_intrinsics as pixel-center. | 6 | med | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed 2949d642 |
| A4n | convention | BUG | `utils/visualization.py:497`<br>`utils/visualization.py:425` | geometry/transforms.py:rescale_intrinsics | Once A3n makes query_intrinsics pixel-center, wrap the rescale in shift +0.5 / -0.5 as pointcloud/base.py does. | 2 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed dc6a0b06 |
| C5 | convention | BUG | `geometry/tracks.py:98`<br>`geometry/tracks.py:100`<br>`utils/io.py:163` | utils/io.py:open_valid | Replace the hand-rolled block with store = open_valid(cache_path, {'cache_key': key}) and return the arrays on a hit; on a miss the mode='w' rewrite already replaces the old store, so the rmtree goes. | -5 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | held: tracks.py WIP |
| D5n | convention | BUG | `localization/retrieval.py:110-115`<br>`localization/retrieval.py:84-90` | localization/retrieval.py:MegaLocExtractor.forward (T.functional.resize antialias=True) | Test-first: tensor vs PIL path preprocessing parity on a non-square 518-wide frame; then replace the interpolate call with T.functional.resize(imgs, [224,224], antialias=True). | -4 | med | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed 6f5dcad2 |
| F9n | convention | BUG | `tests/reconstructor/test_validate_config.py:110`<br>`reconstructor.py:339`<br>`CLAUDE.md (Architecture: 'both feedforward only — sfm refuses them')` | reconstructor.py:Reconstructor.validate_config | Rename the test to test_sfm_refuses_loop_closure and test LC only. | 2 | low | feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed 789bc6cc |
| A8n | convention | PLAUSIBLE_BUG | `geometry/bundle_adjustment.py:354`<br>`pointcloud/feedforward/loger.py:233` | — | Proof test: run refine on a LoGeR scene twice, once with sigmoid conf and once with logit-to-expp1 conf (1+exp(logit)) passed to BA, and compare ATE against SfM pGT. | 0 | med | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | not a bug: expp1 conf worse on chess + office |
| C3 | pattern | PLAUSIBLE_BUG | `reconstructor.py:476`<br>`pointcloud/base.py:127`<br>`reconstructor.py:1134` | utils/io.py:write_json (tmp + os.replace pattern) | Not proven; if confirmed, stamp a validity attr last on pointcloud.zarr and check it in done(), or write to .tmp and rename like write_point_features. | 8 | med | clean/localization, feat/rgbd-ba-cf, perf/splats-speed, cleanup/fc-integrate, cleanup/fc-lint2 | landed 7ac13765 (proof confirmed) |
| F7 | convention | PLAUSIBLE_BUG | `dashboard/viewer.py:482`<br>`dashboard/app.py:642`<br>`viewer.py:646` | viewer.py:_text_mode | SplitViewer._get_extractor reads extractor and extractor_kwargs from the lifted store attrs, keying the cache by name. | 3 | low | cleanup/fc-integrate, cleanup/fc-lint2 | landed c73336c8 |
| G4 | convention | PLAUSIBLE_BUG | `pointcloud/feedforward/mapanything.py:309`<br>`pointcloud/feedforward/vggtx.py:182`<br>`pointcloud/feedforward/base.py:232` | pointcloud/feedforward/mapanything.py:MapAnythingCreator._stack_predictions | Proof test: on a real MapAnything pair, median \|pts3d - unproject_frames(depth_z, w2c, K)\| / depth > 1% confirms. | -10 | med | clean/localization, feat/rgbd-ba-cf | not a bug (proof) |
| B1 | redundancy | REAL | `viewer.py:144`<br>`utils/visualization.py:355` | geometry/transforms.py:invert_poses | Replace np.linalg.inv(pose) with invert_poses(pose) at the two numpy sites (visualization.py already imports from geometry.transforms; viewer.py adds the import). | 0 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed 335ccc45, 8544348a |
| B2n | redundancy | REAL | `splats/scaffold.py:482-489`<br>`geometry/projection.py:107` | geometry/projection.py:project | pixels, points_cam = project(anchors, world_to_cam, intrinsics[0], min_depth=1e-3); depth = points_cam[:, 2]; keep the margin test. | -3 | low | perf/splats-speed, cleanup/fc-integrate, cleanup/fc-lint2 | landed 5def2c1e |
| B7 | redundancy | REAL | `geometry/tracks.py:348`<br>`geometry/transforms.py:96` | geometry/transforms.py:extract_intrinsics | params = extract_intrinsics(intrinsics[i]) at tracks.py:348; add the import. | 0 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | held: tracks.py WIP |
| C2 | convention | REAL | `semantics/store.py:116`<br>`semantics/store.py:177`<br>`semantics/store.py:180` | utils/io.py:LZ4 | Pass compressors=LZ4 at the semantics/store.py write sites; switch store[key]=arr to create_array(..., compressors=LZ4). | -2 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | partial cf400c86; rest held (WIP) |
| C6 | redundancy | REAL | `reconstructor.py:499`<br>`reconstructor.py:1003`<br>`viewer.py:527` | utils/io.py:file_sha256 (new, plain function; reconstructor, viewer, pointcloud/sfm need it) | Optional: add file_sha256(path) to utils/io.py and call it at all four sites. | 8 | low | clean/localization, feat/rgbd-ba-cf, perf/splats-speed, cleanup/fc-integrate, cleanup/fc-lint2 | held 25899d53 (reconstructor WIP) |
| C8 | convention | REAL | `pointcloud/depth.py:356`<br>`pointcloud/depth.py:368`<br>`splats/checkpoint.py:171` | utils/torch_utils.py:load_hf_weights | depth._load_vda_model: ckpt = load_hf_weights(...) and torch.load(..., weights_only=True); splats load_checkpoint: weights_only=True. | -1 | low | feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed 01196202 |
| D1 | convention | REAL | `utils/io.py:110`<br>`semantics/segmentation/insid3.py:136`<br>`semantics/features/base.py:205` | utils/io.py:to_uint8_hwc | Replace insid3._tensor_to_pil body with to_uint8_hwc(t.cpu().float().numpy(), channels_first=True); drop the unreachable >1.0 branch in LC wrapper dense_colors (keep upstream truncation). | -4 | low | feat/rgbd-ba-cf, clean/localization, cleanup/fc-integrate, cleanup/fc-lint2 | held d24b5d0b (WIP) |
| D2 | redundancy | REAL | `semantics/features/base.py:187`<br>`utils/visualization.py:87` | utils/visualization.py:pca_to_rgb | Hoist the torch-SVD projection (no sklearn) into utils/visualization as the patch-grid PCA image; pca_to_rgb resizes+blends it; delete BaseFeatureExtractor.features_to_rgb and point get_bias_visualization + tests at the utils function. | -15 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed 4786bfe9 |
| D3 | redundancy | REAL | `dashboard/viewer.py:52`<br>`viewer.py:294-297`<br>`utils/visualization.py:78-81` | dashboard/viewer.py:apply_viridis (move to utils/visualization.py) | Move apply_viridis into utils/visualization (both dashboard/viewer.py and viewer.py are separate packages) and have show_heat call it; delete the inline copy. | -4 | low | cleanup/fc-integrate, cleanup/fc-lint2, clean/localization, feat/rgbd-ba-cf | landed 8544348a |
| D6 | pattern | REAL | `semantics/features/base.py:150-153`<br>`semantics/features/ocr_lens.py:643-646`<br>`semantics/features/maskclip.py:105` | torch.nn.functional.normalize for L2; crop: no new helper (4 lines, same package) | Switch maskclip.encode_text to F.normalize(embed, dim=-1). | 0 | low | cleanup/fc-integrate, cleanup/fc-lint2 | landed fa023dff |
| D7 | redundancy | REAL | `utils/visualization.py:266-284`<br>`dashboard/viewer.py:350`<br>`utils/visualization.py:299` | utils/visualization.py (new apply_view(plotter, viz_kwargs) from _apply_view body) | Lift _apply_view's body into utils/visualization as a plotter function; dashboard and visualize_splat call it. | -10 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed 5da06461 |
| E1 | redundancy | REAL | `geometry/bundle_adjustment.py:456`<br>`geometry/loop_closure/wrapper.py:626-627`<br>`mesh/texture.py:157` | utils/torch_utils.py:pytorch_gc | Replace the three direct calls with pytorch_gc() (drop wrapper's inline is_available guard); BA per-scale call is optional — keep as-is if the per-scale gc.collect cost is unwanted. | -2 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | held 9bdd23a8 (WIP) |
| E2n1 | convention | REAL | `semantics/segmentation/mobile_sam.py:29`<br>`semantics/segmentation/mobile_sam.py:90`<br>`semantics/features/ocr_lens.py:98` | semantics/features/dino.py:DINOFeatureExtractor.__init__ (device=None -> get_device()) | MobileSAM: device: str \| None = None -> get_device(); OCRLens.__init__ gains device=None with the same fallback. | 4 | low | cleanup/fc-integrate, cleanup/fc-lint2 | landed 61a4382b |
| E2n2 | convention | REAL | `preproc/undistort.py:64`<br>`pointcloud/sfm/sift_db.py:105` | pointcloud/sfm/sift_db.py (get_device() form) | Spell undistort's gate as `get_device() == "cuda"` like sift_db and drop its `import torch`; no shared helper (one-liner). | -1 | low | cleanup/fc-integrate, cleanup/fc-lint2 | held: undistort.py WIP |
| E6 | convention | REAL | `utils/torch_utils.py:160`<br>`splats/cameras.py:166`<br>`splats/rendering.py:90` | — | Convert the 4 asserts to `if ...: raise ValueError(...)`; update test_batch_iterator_mismatched_raises to expect ValueError. | 6 | low | clean/localization, feat/rgbd-ba-cf, perf/splats-speed, cleanup/fc-integrate, cleanup/fc-lint2 | landed 0de1f39d, 4ca3eba7 |
| F1 | pattern | REAL | `pointcloud/sfm/__init__.py:13`<br>`pointcloud/__init__.py:16`<br>`reconstructor.py:126` | utils/torch_utils.py:RegistryMixin | Optional and low value. | 0 | low | cleanup/fc-integrate, cleanup/fc-lint2, clean/localization, feat/rgbd-ba-cf, perf/splats-speed | held (optional) |
| F10 | redundancy | REAL | `reconstructor.py:1088`<br>`mesh/tsdf.py:155`<br>`reconstructor.py:1081` | mesh/tsdf.py:compute_tsdf_voxel_size | Low priority. | 2 | low | feat/rgbd-ba-cf, clean/localization, perf/splats-speed, cleanup/fc-integrate, cleanup/fc-lint2 | held ee8391fd (WIP) |
| F11 | redundancy | REAL | `reconstructor.py:967`<br>`reconstructor.py:969`<br>`semantics/features/ocr_lens.py:91` | semantics/features/ocr_lens.py:load_decoder | Give load_processor the same model_id default. | 0 | low | clean/localization, feat/rgbd-ba-cf, perf/splats-speed, cleanup/fc-integrate, cleanup/fc-lint2 | held 16666e8d (WIP) |
| F4 | redundancy | REAL | `reconstructor.py:207`<br>`reconstructor.py:840`<br>`reconstructor.py:1056` | — | No change recommended. | 0 | low | clean/localization, feat/rgbd-ba-cf, perf/splats-speed, cleanup/fc-integrate, cleanup/fc-lint2 | no change |
| F5 | convention | REAL | `semantics/segmentation/sky.py:191`<br>`reconstructor.py:1073`<br>`viewer.py:527` | reconstructor.py:Reconstructor.mesh | Make sky_masks' cache_dir a required argument and have Reconstructor.mesh pass `self.images_dir.parent / "sky"` (or Path(output_path)/'sky'). | 0 | low | clean/localization, feat/rgbd-ba-cf, perf/splats-speed, cleanup/fc-integrate, cleanup/fc-lint2, clean/rclone-api | held: reconstructor WIP |
| F6 | redundancy | REAL | `dashboard/app.py:50`<br>`viewer.py:547`<br>`__main__.py:353` | — | No change recommended. | 0 | low | cleanup/fc-integrate, cleanup/fc-lint2 | no change |
| G1 | redundancy | REAL | `pointcloud/feedforward/vggtx.py:149`<br>`pointcloud/feedforward/vggt_omega.py:243`<br>`pointcloud/feedforward/vggtx.py:128` | pointcloud/feedforward/base.py:BaseFeedforwardCreator.extract_intermediate_features | Base extract_intermediate_features wraps `self._forward(self.model, frames)` in capture_qk on a per-backend `_lc_attn(layer_index)` and owns the shared unproject tail; VGGT-X LC forward gains autocast (tiny numeric change, recalibration check on verify_match_ratio advisable). | -40 | med | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed ebb48c03 |
| G2 | pattern | REAL | `pointcloud/feedforward/vggtx.py:87`<br>`pointcloud/feedforward/vggt_omega.py:153`<br>`pointcloud/feedforward/mapanything.py:116` | pointcloud/feedforward/base.py:BaseFeedforwardCreator | Optional, low value: hoist the skeleton into a concrete base _preprocess calling subclass _preprocess_files/_preprocess_arrays/_crop_boxes(sizes, views); keep all upstream-mirroring helpers per backend. | -15 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | partial e4822498 |
| G4n | redundancy | REAL | `pointcloud/feedforward/base.py:232`<br>`pointcloud/feedforward/vggtx.py:182`<br>`pointcloud/feedforward/vggt_omega.py:275` | geometry/projection.py:unproject_frames | Replace the three blocks with `unproject_frames(depth, extrinsic, intrinsics)` (one site after G1); drop the now-unused `unproject` imports. | -10 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed 12cec945 |
| G7n | convention | REAL | `semantics/segmentation/mobile_sam.py:50`<br>`semantics/segmentation/sam3.py:55` | semantics/segmentation/sam3.py:SAM3Segmentation.segment | mobile_sam._stack_masks returns bool; update BaseSegmentation.segment docstring. | 0 | low | cleanup/fc-integrate, cleanup/fc-lint2 | landed 32bb1dad, d7910b02 |
| H1 | naming | REAL | `pointcloud/feedforward/base.py:61`<br>`pointcloud/feedforward/base.py:240`<br>`pointcloud/feedforward/vggtx.py:59` | geometry/loop_closure/wrapper.py:LoopClosureConfig.conf_percentile (name) | Rename BaseFeedforwardCreator.conf_threshold to conf_percentile, give MapAnythingCreator a conf_percentile=35.0 override, and delete its confidence_percentile field (see H1n). | -2 | med | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed 4e175d13 |
| H1n | naming | REAL | `pointcloud/feedforward/mapanything.py:91`<br>`pointcloud/feedforward/base.py:61`<br>`pointcloud/feedforward/base.py:237` | — | Fold into H1: one conf_percentile field, with MapAnything overriding the default to 35.0. | -1 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed 4e175d13 |
| H4 | naming | REAL | `geometry/projection.py:29`<br>`geometry/projection.py:66`<br>`geometry/projection.py:107` | extrinsics (most call sites; also the zarr array name) | Rename only function params: world_to_cam becomes extrinsics in geometry/projection.py and photometric.py, and w2c becomes extrinsics in mesh/texture privates. | 0 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | held 928e5818 (WIP) |
| H5 | naming | REAL | `geometry/tracks.py:147`<br>`geometry/tracks.py:323`<br>`semantics/lifting.py:35` | rel_thresh for relative depth tolerance (most defs; geometry/projection.py) | Rename build_tracks depth_tol to a px name (e.g. | 0 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | held: tracks.py WIP |
| H7 | naming | REAL | `viewer.py`<br>`evals/eval.py:388`<br>`__main__.py` | dash style (13 of 15 multi-word flags) | Rename to --texture-size and --dry-run with no alias, and update README.md, CLAUDE.md and evals/README.md. | 0 | low | cleanup/fc-integrate, cleanup/fc-lint2 | held 894185c9 (WIP) |
| I2 | dead | REAL | `remote/__pycache__/`<br>`dashboard/__pycache__/` | — | rm -rf collab_splats/remote and the orphan dashboard .pyc files. | 0 | low | — | done (dir removed) |
| I3 | redundancy | REAL | `utils/progress.py:11`<br>`localization/localizer.py:404-433`<br>`localization/localizer.py:707-709` | utils/progress.py:progress | Delete the unused logger. | -8 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed 34dd41dc |
| I4 | pattern | REAL | `tests/test_docstring_contract.py:22`<br>`utils/torch_utils.py:9`<br>`utils/visualization.py:1` | — | Bring utils/ to the contract and add 'utils' to PACKAGES. | 30 | low | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed c42e4612 |
| B3n | dead | DEAD | `geometry/transforms.py:455` | — | None by default: user decision keeps it as public API. | 0 | low | feat/rgbd-ba-cf, clean/localization, cleanup/fc-integrate, cleanup/fc-lint2 | kept (user) |
| F5n | dead | DEAD | `dashboard/__main__.py:92`<br>`dashboard/serve.py:204`<br>`dashboard/app.py:159` | dashboard/__main__.py:main | Drop the default values from serve.run_app(base_dir) and SplatsApp.__init__(base_dir) so they are required. | 0 | low | cleanup/fc-integrate, cleanup/fc-lint2 | landed 5ff88b29 |
| F9n2 | dead | DEAD | `reconstructor.py:322`<br>`reconstructor.py:324`<br>`geometry/bundle_adjustment.py:101` | — | Remove 'tracks_cache_dir' from the set and the parenthetical from the comment. | 0 | low | clean/localization, feat/rgbd-ba-cf, perf/splats-speed, cleanup/fc-integrate, cleanup/fc-lint2 | held: reconstructor WIP |
| I1b | dead | DEAD | `localization/localizer.py:491`<br>`localization/localizer.py:661`<br>`localization/localizer.py:676` | — | Once clean/localization lands, delete update_index, add_localized_frame, clear_localized_frames, frame_sources, the localized-group load and the _localized_* state, plus their tests. | -170 | med | clean/localization, feat/rgbd-ba-cf, cleanup/fc-integrate, cleanup/fc-lint2 | landed 34dd41dc |
| I1c | dead | DEAD | `preproc/viz.py:66` | — | Delete plot_frame_scores and its two smoke tests. | -36 | low | cleanup/fc-integrate, cleanup/fc-lint2 | landed d29ca3dc |
| I1d | dead | DEAD | `utils/colmap.py:45` | — | Delete it; tests call pycolmap.Reconstruction(str(path)) directly. | -16 | low | cleanup/fc-integrate, cleanup/fc-lint2 | landed 65acc2fa |
| P1 | perf | PORT | `splats/checkpoint.py::render_tsdf_inputs` | — | device uint8 + mask, 5.1×; after A2, keep max>1.5 guard | — | low | — | landed 66a3688c |
| P2 | perf | PORT | `semantics/features/base.py` | — | _resize + device _normalize_batch, 4×; canonical uint8→float01 on device | — | low | — | landed 12c4cecc |
| P3 | perf | PORT | `semantics/segmentation/sky.py` | — | threaded PNG read 2.07×; grayscale mode on frames.read_frames; with F5 | — | low | — | landed bab863ab |
| P4 | perf | PORT | `pointcloud/depth.py::align_depth` | — | vectorized points3D gather 1.65×; same file as C8 | — | low | — | landed 9fa17513 |
| P5 | perf | PORT | `pointcloud/sfm/base.py` | — | keyframes via read_frames; with C1 side note | — | low | — | landed a834642e |
| P6 | perf | PORT | `utils/image.py::_guided_upsample_depth` | — | cv2-exact nearest gather on device 1.8×; stays private | — | low | — | landed 4de975c0 |
| P7 | perf | PORT | `splats/rendering.py` | — | inv_ex, no host sync; with E6 | — | low | — | landed fcd08a94 |
| P8 | perf | PORT | `semantics/compression.py::fit` | — | device-side loss accumulation ~1.7-2× | — | low | — | landed in ab512103 |
| P9 | perf | PORT | `pointcloud/feedforward/mapanything.py` | — | stack per key on device; after G4 proof | — | low | — | skipped (G4 not a bug) |
| P10 | perf | PORT | `preproc/qa.py` | — | shared analysis gray, 4% | — | low | — | dropped (user) |
| P11 | perf | PORT | `preproc/sampling.py` | — | optical-flow hist cache | — | low | — | landed 36d61928 |
| P12 | perf | PORT | `semantics utils + segmentation` | — | vectorized clustering/masks up to 223×; with G7n | — | low | — | landed 32bb1dad |

Intentional (no fix): A5 (K_sub /= stride is exact for strided subsample, not a resize), A10 (COLMAP export trims points3D by exact float32 xyz match), A7 (iter_frames BGR vs extract_frame RGB is documented and honored), A8 (LoGeR sigmoid conf matches upstream; percentile gates are rank-invariant), B2 (BA and photometric projections must stay bae map_transform-shaped), B3 (BA drift-undo and LC frame scale are different estimators than Umeyama), B4 (Two O(N^2) cross-view loops produce different outputs over shared primitives), B5 (Scene-scale variants match different upstreams; denormalize acts on different params), B8 (Submap percentile threshold is a VGGT-SLAM port, not confidence_mask), C1 (Image readers differ by contract, not by accident), C7 (Each mesh reader feeds its own renderer; greys mean different things), D4 (Resize helpers differ in contract, type and upstream match), D8 (Display decimation indices vs seeded random keep-mask differ in contract), E2 (Hardcoded cuda / backend-specific device checks are library-forced), E3 (.cpu().float().numpy() is not to_numpy: forces float32), E4 (Lazy _chunked and read_frames_chunked differ from sized batch_iterator), E5 (inference_mode only where outputs leave torch; no_grad elsewhere), E7 (read_localization_db KeyError is the rebuild signal), E8 (loss_weight and _decay_lambda share one line of math only), F12 (Dashboard error-hint substring, smoke prints, collab_data import), F2 (Unknown config keys refused only for splats/LC/BA/scaffold), F3 (Reconstructor build-and-run in CLI, dashboard, eval differ by source), F9 (BA/LC/sfm rule checked at config, default stages, named refine), G3 (Underscore helpers imported within their own package by design), G5 (calibrate_camera SIFT differs from sift_db: OPENCV model, subset, scratch DB), G6 (Three retrieval rankings answer different questions; L2 on unit = cosine), G7 (Segmentation/feature backend contract differences are documented and upstream-driven), H2 (rescale_intrinsics (h,w) vs shift_intrinsics (x,y) axis order), H3 (Directory params name distinct things, not one meaning), H6 (max_hole_perimeter_ratio 0.014 vs 3.9: same unit, different stage), I1a (Grouping mask helpers: zero callers here, kept for main's grouping.py), I5 (scaffold uses gsplat private _update_param_with_optimizer; pinned rev), I6 (write_textured_obj in utils/io with one mesh caller)

Rejected: A1, A4, A6, A9, B6, C4, D5, F8, G8, H8, I1
