"""Notebook display helpers — visualization and debug utilities for tutorial notebooks.

These functions are intentionally print()-based and pyvista/matplotlib-dependent.
Do not import this module from core library code.

TODO(reorganize): review final home — may warrant a dedicated collab_splats/notebooks/ package
once the set of helpers stabilizes. See worklog/notes/2026-05-23-notebook-abstraction-opportunities.md.
"""
from __future__ import annotations

import numpy as np


########################################################
########## Pointcloud helpers ##########################
########################################################


def clean_and_extract_result(
    result,
    name: str = "",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Clean a FeedforwardResult and return (pts3d, colors_uint8, conf_mean, conf_std).

    Runs Open3D statistical outlier removal + voxel downsampling, computes confidence
    statistics, and prints a raw→filtered + conf summary line.

    Args:
        result:     FeedforwardResult to clean.
        name:       Label prefix for print output.
        verbose: When True, print point counts after each individual cleaning step.
    """
    import open3d as o3d
    from collab_splats.pointcloud.utils import (
        clean_pointcloud,
        voxel_downsample,
        filter_distance,
        _DEFAULT_DOWNSAMPLE_KWARGS,
        _DEFAULT_OUTLIER_KWARGS,
        _DEFAULT_DISTANCE_KWARGS,
    )

    label = f"[{name}] " if name else ""

    # Build Open3D point cloud from result arrays
    _pcd = o3d.geometry.PointCloud()
    _pcd.points = o3d.utility.Vector3dVector(result.pts3d)
    _pcd.colors = o3d.utility.Vector3dVector(result.colors.astype(np.float64) / 255.0)

    if verbose:
        # Run each cleaning step manually to expose per-step counts
        n = len(_pcd.points)
        print(f"{label}Diagnostic cleaning ({n:,} raw):")

        _pcd, _ = voxel_downsample(_pcd, **_DEFAULT_DOWNSAMPLE_KWARGS)
        print(f"{label}  voxel downsample  → {len(_pcd.points):,}  (removed {n - len(_pcd.points):,})")
        n = len(_pcd.points)

        _pcd, _ = _pcd.remove_statistical_outlier(**_DEFAULT_OUTLIER_KWARGS)
        print(f"{label}  stat outlier      → {len(_pcd.points):,}  (removed {n - len(_pcd.points):,})")
        n = len(_pcd.points)

        _pcd, _ = filter_distance(_pcd, return_mask=True, **_DEFAULT_DISTANCE_KWARGS)
        print(f"{label}  distance filter   → {len(_pcd.points):,}  (removed {n - len(_pcd.points):,})")

        _cleaned = _pcd
    else:
        _cleaned, _ = clean_pointcloud(_pcd)

    pts3d = np.asarray(_cleaned.points, dtype=np.float32)
    colors = (np.asarray(_cleaned.colors) * 255).astype(np.uint8)

    # Confidence stats with nan guard for results without confidence maps
    conf_mean = result.conf.cpu().float().mean().item() if result.conf is not None else float("nan")
    conf_std  = result.conf.cpu().float().std().item()  if result.conf is not None else float("nan")

    print(f"{label}Points: {len(result.pts3d):,} raw → {len(pts3d):,} filtered")
    print(f"{label}Conf:   mean={conf_mean:.3f}  std={conf_std:.3f}")
    return pts3d, colors, conf_mean, conf_std


def add_camera_frustums(
    plotter,
    extrinsics: np.ndarray,
    color: str = "cornflowerblue",
    scale: float = 0.05,
    line_width: int = 2,
) -> None:
    """Add camera frustum meshes for all extrinsic matrices to a PyVista plotter."""
    from collab_splats.utils.visualization import create_camera_frustum_pyvista

    for ext in extrinsics:
        frustum = create_camera_frustum_pyvista(np.linalg.inv(ext), scale=scale)
        plotter.add_mesh(frustum, color=color, line_width=line_width)


########################################################
########## Semantics / feature helpers #################
########################################################


def feature_viz_row(
    axes,
    frame: np.ndarray,
    features,
    sim_map: np.ndarray,
    title_prefix: str = "",
    query_label: str = "",
) -> None:
    """Render PCA→RGB, similarity heatmap, and masked image into a row of 3 matplotlib axes."""
    from collab_splats.utils.visualization import pca_to_rgb, compute_heatmap, compute_masked_image

    sim_title = (
        f"{title_prefix}Similarity: {query_label}" if query_label
        else f"{title_prefix}Similarity"
    )
    axes[0].imshow(pca_to_rgb(features, frame))
    axes[0].set_title(f"{title_prefix}PCA → RGB")
    axes[1].imshow(compute_heatmap(frame, sim_map))
    axes[1].set_title(sim_title)
    axes[2].imshow(compute_masked_image(frame, sim_map))
    axes[2].set_title(f"{title_prefix}Masked Image")
    for ax in axes:
        ax.axis("off")
