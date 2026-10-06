# Pointcloud Module

`collab_splats.pointcloud` builds a `PointcloudResult` (points, poses, depth, per-pixel
provenance) from raw video or images, via one of two backend families: feedforward
(VGGT-X, VGGT-Omega, MapAnything, LoGeR) or sfm (InstantSfM, COLMAP or hloc, + VDA depth).

---

## Depth alignment (sfm path)

`depth.py` supplies the sfm backend's metric depth: `estimate_depth` runs Video-Depth-Anything
per keyframe (cached under `depth_vda/`), and `align_depth` fits a per-frame scale from COLMAP
track observations to bring that depth to the reconstruction's world scale.

`estimate_depth` loads **Metric-Video-Depth-Anything-Large**, licensed **cc-by-nc-4.0**
(non-commercial); Metric-Small (Apache-2.0) is the commercial-safe alternative, not taken here.

## Bundle adjustment with loop closure

With `pointcloud.loop_closure` and `pointcloud.bundle_adjustment.enabled` both on, BA runs
inside each loop-closure window, right after the window's forward pass and before its submap
enters the pose graph; the `refine` stage is skipped. The first refined window sets the focal
and later windows hold it fixed. A window whose solve fails keeps its feedforward poses. Each
window's outcome (start frame, focal, loss, time, peak GPU memory) is stored in the
`pointcloud.zarr` attrs under `window_ba`.
