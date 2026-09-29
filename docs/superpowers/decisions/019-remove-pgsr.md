# 019 — Remove PGSR from splats

Date: 2026-09-27 · Status: accepted · Branch: `reorg-lane-b` (pointcloud reorg, task T2b)

## Context

- `splats/pgsr.py` reimplemented PGSR's planar losses (yanxian-ll/GS-SR @ 566359be):
  `pgsr_normal`, `pgsr_multiview`, the `render_plane` path in `render_gaussians`, neighbor
  selection and `render_neighbor` in the trainer
- never validated end-to-end: no scene run, no mesh A/B, only unit tests and a parity config
- known open bug: `pgsr_multiview` + `pose_opt: true` drives pose deltas NaN within 50 steps
  (`docs/known-test-failures.md`, 2026-09-06), never root-caused
- only consumer of `geometry/projection.pixel_rays` and its `pixel_offset=0.5` pixel-center
  convention; every other caller uses integer pixel coordinates
- default config never enabled it; `configs/base.yaml` carried it as a commented block

## Decision

- delete `splats/pgsr.py` and `tests/splats/test_pgsr.py`
- drop `pgsr_normal` / `pgsr_multiview` from `OPTIONAL_LOSSES` and `LOSS_SPEC_KEYS`,
  plus `neighbor_selection`
- drop `render_plane` from `render_gaussians`, `Gaussians.render`, `Scaffold.render`;
  `gaussian_normals_in_camera_frame` returns normals only
- fold `pixel_rays` into `unproject`; no `pixel_offset` parameter — integer pixel coords only;
  `project` unchanged

## Consequences

- 3dgs has no planar-geometry losses; `normal_consistency` is the remaining geometry prior
- a config naming `pgsr_normal` / `pgsr_multiview` now fails schedule validation as an unknown loss
- the NaN bug is resolved by removal, not fixed
- recovery: the full implementation and its tests live at `66006a79`
  (`git show 66006a79:collab_splats/splats/pgsr.py`); a restore needs the pixel-center ray
  (`pixel_offset=0.5`) back in `geometry/projection.py`
