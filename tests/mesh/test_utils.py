from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from collab_splats.mesh.tsdf import Open3DTSDFFusion
from collab_splats.mesh.utils import optimize_color_map, pointcloud_to_mesh


def test_find_depth_edges_shape():
    from collab_splats.mesh.utils import find_depth_edges

    depth = np.random.rand(64, 64).astype(np.float32)
    edges = find_depth_edges(depth, threshold=0.01, dilation_itr=1)
    assert edges.shape == (64, 64)
    assert edges.dtype == bool


def test_find_depth_edges_constant_depth_no_edges():
    from collab_splats.mesh.utils import find_depth_edges

    depth = np.ones((32, 32), dtype=np.float32)
    edges = find_depth_edges(depth, threshold=0.01, dilation_itr=0)
    assert not edges.any()


def test_normals2vertex_output_shape():
    from collab_splats.mesh.utils import normals2vertex

    rng = np.random.default_rng(0)
    mesh_vertices = rng.random((50, 3)).astype(np.float32)
    points = rng.random((200, 3)).astype(np.float32)
    normals = rng.random((200, 3)).astype(np.float32)
    result = normals2vertex(mesh_vertices, points, normals, k=5)
    assert result.shape == (50, 3)


def test_features2vertex_output_shape():
    from collab_splats.mesh.utils import features2vertex

    rng = np.random.default_rng(0)
    mesh_vertices = rng.random((50, 3)).astype(np.float32)
    points = rng.random((200, 3)).astype(np.float32)
    features = rng.random((200, 16)).astype(np.float32)
    result = features2vertex(mesh_vertices, points, features, k=5)
    assert result.shape == (50, 16)


def _features2vertex_numpy_reference(mesh_vertices, points, features, k=5, sdf_trunc=0.03):
    """Frozen copy of the original CPU implementation — parity oracle for the GPU rewrite."""
    from scipy.spatial import cKDTree

    vertices = np.asarray(mesh_vertices)
    tree = cKDTree(vertices)
    distances, indices = tree.query(points, k=k)
    valid_mask = distances[:, 0] <= sdf_trunc
    if not np.any(valid_mask):
        return np.zeros((len(vertices), features.shape[1]), dtype=features.dtype)
    distances = distances[valid_mask]
    indices = indices[valid_mask]
    features = features[valid_mask]
    sigma = np.mean(distances)
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)
    out = np.zeros((len(vertices), features.shape[1]), dtype=features.dtype)
    wsum = np.zeros((len(vertices), 1), dtype=features.dtype)
    for i in range(k):
        np.add.at(out, indices[:, i], features * weights[:, i : i + 1])
        np.add.at(wsum, indices[:, i], weights[:, i : i + 1])
    nz = wsum.squeeze() > 0
    out[nz] /= wsum[nz]
    return out


def test_features2vertex_matches_numpy_reference():
    from collab_splats.mesh.utils import features2vertex

    rng = np.random.default_rng(0)
    vertices = rng.random((200, 3)).astype(np.float64)
    points = rng.random((1000, 3)).astype(np.float64)
    features = rng.random((1000, 8)).astype(np.float32)

    got = features2vertex(vertices, points, features, k=5, sdf_trunc=0.1)
    want = _features2vertex_numpy_reference(vertices, points, features, k=5, sdf_trunc=0.1)

    assert got.shape == (200, 8)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_features2vertex_all_far_returns_zeros():
    from collab_splats.mesh.utils import features2vertex

    vertices = np.zeros((10, 3), dtype=np.float64)
    points = np.full((20, 3), 100.0, dtype=np.float64)  # all far beyond sdf_trunc
    features = np.ones((20, 4), dtype=np.float32)

    out = features2vertex(vertices, points, features, k=3, sdf_trunc=0.03)
    assert out.shape == (10, 4)
    assert np.all(out == 0.0)


def test_features2vertex_dtype_preserved():
    from collab_splats.mesh.utils import features2vertex

    rng = np.random.default_rng(1)
    vertices = rng.random((50, 3))
    points = rng.random((100, 3))
    features = rng.random((100, 6)).astype(np.float32)

    out = features2vertex(vertices, points, features, k=4, sdf_trunc=0.2)
    assert out.dtype == np.float32
    assert out.shape == (50, 6)


def test_meshresult_has_vertex_features_field():
    from pathlib import Path

    from collab_splats.mesh.base import MeshResult

    r = MeshResult(mesh_path=Path("/tmp/m.ply"))
    assert r.vertex_features is None  # default

    r2 = MeshResult(mesh_path=Path("/tmp/m.ply"), vertex_features=np.zeros((3, 2)))
    assert r2.vertex_features.shape == (3, 2)


def test_persist_mesh_vertex_features(tmp_path):
    import open3d as o3d

    from collab_splats.mesh.utils import persist_mesh_vertex_features

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    mesh_path = tmp_path / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    points = verts.copy()
    feats = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)

    out = persist_mesh_vertex_features(mesh_path, points, feats, k=1, sdf_trunc=0.5)

    assert out.shape == (3, 2)
    saved = np.load(mesh_path.parent / "vertex_features.npy")
    np.testing.assert_allclose(saved, out)


########
# clean_repair_mesh — component filtering + hole filling (meshlib)
########


def _holed_sphere_with_strays(path, radius=1.0, resolution=20, color=None):
    """Sphere missing a cap, plus one stray blob inside its bbox and one far outside."""
    import open3d as o3d

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    tris = np.asarray(sphere.triangles)
    sphere.triangles = o3d.utility.Vector3iVector(tris[:-12])  # punch a hole
    sphere.remove_unreferenced_vertices()

    inside = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=6)
    inside.translate((0.2, 0.0, 0.0))
    outside = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=6)
    outside.translate((radius * 9, 0.0, 0.0))

    combined = sphere + inside + outside
    if color is not None:  # colored variant for the color-preservation test
        combined.paint_uniform_color(color)
    o3d.io.write_triangle_mesh(str(path), combined)
    return path


def test_clean_repair_mesh_drops_out_of_bounds_components_and_fills_holes(tmp_path):
    """The two jobs of the cleanup, on a mesh built to need both.

    A TSDF scene comes out with floating specks from stray depth and small holes where coverage
    thinned. Detached geometry *inside* the room (furniture) must survive — that is why the
    bounding-box test exists instead of a plain keep-the-largest.
    """
    import open3d as o3d

    from collab_splats.mesh.utils import clean_repair_mesh

    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply")
    before = o3d.io.read_triangle_mesh(str(mesh_path))
    assert len(before.cluster_connected_triangles()[2]) == 3
    assert not before.is_watertight()

    out = clean_repair_mesh(mesh_path, max_hole_size=3.0)

    assert out == mesh_path  # rewritten in place, not to a new name
    after = o3d.io.read_triangle_mesh(str(mesh_path))
    # The far blob is dropped, the one inside the bbox is kept, and the sphere's hole is closed.
    assert len(after.cluster_connected_triangles()[2]) == 2
    assert after.is_watertight()


def test_clean_repair_mesh_leaves_large_holes_alone(tmp_path):
    """A hole bigger than max_hole_size is a real opening (unscanned wall), not a defect."""
    import open3d as o3d

    from collab_splats.mesh.utils import clean_repair_mesh

    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply")
    clean_repair_mesh(mesh_path, max_hole_size=1e-6)  # below any real perimeter → fill nothing

    after = o3d.io.read_triangle_mesh(str(mesh_path))
    assert not after.is_watertight()  # the hole survived
    assert len(after.get_non_manifold_edges(allow_boundary_edges=False)) == 14  # same boundary
    assert len(after.cluster_connected_triangles()[2]) == 2  # component filtering still ran


def test_clean_repair_mesh_use_largest_keeps_only_the_main_component(tmp_path):
    """use_largest=True is the aggressive mode: everything but the biggest body is discarded."""
    import open3d as o3d

    from collab_splats.mesh.utils import clean_repair_mesh

    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply")
    clean_repair_mesh(mesh_path, use_largest=True)

    after = o3d.io.read_triangle_mesh(str(mesh_path))
    assert len(after.cluster_connected_triangles()[2]) == 1  # the in-bbox blob went too
    assert after.is_watertight()


def test_clean_repair_mesh_preserves_vertex_colors(tmp_path):
    """Vertex colors survive the rewrite — the meshlib file round-trip used to strip them."""
    import open3d as o3d

    from collab_splats.mesh.utils import clean_repair_mesh

    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply", color=(0.2, 0.6, 0.9))
    clean_repair_mesh(mesh_path, max_hole_size=3.0)

    after = o3d.io.read_triangle_mesh(str(mesh_path))
    assert after.has_vertex_colors()
    # Every vertex — original and hole-patch alike — carries the painted color
    # (atol covers the uint8 PLY quantisation).
    assert np.allclose(np.asarray(after.vertex_colors), (0.2, 0.6, 0.9), atol=0.02)


########
# guided_upsample_depth
########


def _step_scene(factor=4):
    """Model-res depth with a vertical step edge + RGB guide whose edge aligns with it."""
    h, w = 32, 32
    depth = np.full((h, w), 1.0, dtype=np.float32)
    depth[:, w // 2 :] = 2.0
    H, W = h * factor, w * factor
    rgb = np.full((H, W, 3), 40, dtype=np.uint8)
    rgb[:, W // 2 :] = 200
    return depth, rgb


def test_guided_upsample_depth_places_crop():
    """Output canvas is zero outside the crop box and populated inside it."""
    from collab_splats.mesh.utils import guided_upsample_depth

    depth, rgb = _step_scene(factor=2)
    canvas_hw = (100, 120)  # bigger than the 64x64 crop
    full_rgb = np.zeros((*canvas_hw, 3), dtype=np.uint8)
    full_rgb[10:74, 20:84] = rgb
    out = guided_upsample_depth(depth, full_rgb, crop_box=(20, 10, 84, 74), out_hw=canvas_hw)

    assert out.shape == canvas_hw
    assert out.dtype == np.float32
    assert np.all(out[:10] == 0) and np.all(out[74:] == 0)
    assert np.all(out[:, :20] == 0) and np.all(out[:, 84:] == 0)
    assert (out[10:74, 20:84] > 0).mean() > 0.99


def test_guided_upsample_depth_masked_pixels_stay_zero():
    """Depth==0 (masked / no observation) must never be resurrected by the filter."""
    from collab_splats.mesh.utils import guided_upsample_depth

    depth, rgb = _step_scene(factor=4)
    depth[8:16, 8:16] = 0.0  # masked block
    H, W = rgb.shape[:2]
    out = guided_upsample_depth(depth, rgb, crop_box=(0, 0, W, H), out_hw=(H, W))

    assert np.all(out[32:64, 32:64] == 0)  # the masked block, upsampled 4x
    valid = out[out > 0]
    assert valid.min() >= 1.0 - 1e-3 and valid.max() <= 2.0 + 1e-3  # no overshoot


def test_guided_upsample_depth_step_edge_stays_sharp():
    """The anti-bilinear property: an aligned guide edge keeps the depth step sharp.

    Bilinear at 4x smears intermediates across the whole kernel; the guided filter with a
    matching guide edge confines them to a thin transition band.
    """
    from collab_splats.mesh.utils import guided_upsample_depth

    depth, rgb = _step_scene(factor=4)
    H, W = rgb.shape[:2]
    out = guided_upsample_depth(depth, rgb, crop_box=(0, 0, W, H), out_hw=(H, W))

    interior = out[:, np.r_[0 : W // 2 - 8, W // 2 + 8 : W]]  # away from the edge band
    fabricated = (interior > 1.1) & (interior < 1.9)
    assert fabricated.mean() < 0.01


########
# _feedforward_to_tsdf_inputs — confidence masking + native-resolution adapter
########


def _tiny_ff_result(with_confidence=True):
    """Minimal FeedforwardResult for adapter tests: 2 frames, 8x8 model res, 16x16 original."""
    import torch

    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    n, h, w = 2, 8, 8
    rng = np.random.default_rng(0)
    depth = rng.uniform(1.0, 2.0, (n, h, w)).astype(np.float32)
    conf = np.zeros((n, h, w), dtype=np.float32)
    conf[:, :, : w // 2] = 1.0  # right half low-confidence
    ext = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    K = np.tile(np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]], np.float32), (n, 1, 1))
    return FeedforwardResult(
        points=np.zeros((1, 3), np.float32),
        colors=np.zeros((1, 3), np.uint8),
        extrinsics=ext,
        intrinsics=K,
        image_paths=[Path(f"f{i}.jpg") for i in range(n)],
        original_coords=np.tile(np.array([0, 0, 16, 16, 16, 16], np.float32), (n, 1)),
        model_width=w,
        model_height=h,
        images=torch.rand(n, 3, h, w),
        confidence=torch.from_numpy(conf) if with_confidence else None,
        depth=depth,
    )


def test_tsdf_inputs_conf_percentile_zeroes_low_confidence_depth():
    from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs

    ff = _tiny_ff_result()
    depths, _, _, _ = _feedforward_to_tsdf_inputs(ff, conf_percentile=50.0)
    assert np.all(depths[:, :, 4:] == 0)  # low-confidence half masked
    assert np.all(depths[:, :, :4] > 0)  # high-confidence half untouched


def test_tsdf_inputs_conf_percentile_without_confidence_raises():
    from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs

    ff = _tiny_ff_result(with_confidence=False)
    with pytest.raises(ValueError, match="confidence"):
        _feedforward_to_tsdf_inputs(ff, conf_percentile=50.0)


def test_tsdf_inputs_defaults_unchanged():
    """Options off → byte-identical to the pre-change adapter output."""
    from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs

    ff = _tiny_ff_result()
    depths, rgbs, c2w, K = _feedforward_to_tsdf_inputs(ff)
    np.testing.assert_array_equal(depths, ff.depth)
    assert rgbs.shape == (2, 8, 8, 3) and rgbs.dtype == np.float32
    np.testing.assert_array_equal(K, ff.intrinsics)


def test_tsdf_inputs_native_resolution_uses_store_rgb_and_upsampled_depth():
    from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs

    class FakeStore:  # FrameStore duck-type: len + images()
        def __len__(self):
            return 2

        def images(self):
            return np.full((2, 16, 16, 3), 128, dtype=np.uint8)

    ff = _tiny_ff_result()
    native_K = ff.intrinsics * np.array([[2, 1, 2], [1, 2, 2], [1, 1, 1]], np.float32)
    depths, rgbs, _, K = _feedforward_to_tsdf_inputs(
        ff, frame_store=FakeStore(), native_intrinsics=native_K
    )
    assert depths.shape == (2, 16, 16)
    assert rgbs.dtype == np.uint8 and rgbs.shape == (2, 16, 16, 3)
    np.testing.assert_array_equal(K, native_K)
    assert (depths > 0).all()  # full-frame crop, no masking → fully populated


def test_tsdf_inputs_native_resolution_frame_count_mismatch_raises():
    from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs

    class ShortStore:
        def __len__(self):
            return 1

        def images(self):
            return np.zeros((1, 16, 16, 3), dtype=np.uint8)

    ff = _tiny_ff_result()
    with pytest.raises(ValueError, match="[Ff]rame"):
        _feedforward_to_tsdf_inputs(ff, frame_store=ShortStore(), native_intrinsics=ff.intrinsics)


def test_tsdf_inputs_native_resolution_wrong_store_resolution_raises():
    """Same frame count but different resolution than original_coords → loud failure."""
    from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs

    class WrongResStore:
        def __len__(self):
            return 2

        def images(self):
            return np.zeros((2, 32, 32, 3), dtype=np.uint8)  # original_coords say 16x16

    ff = _tiny_ff_result()
    with pytest.raises(ValueError, match="resolution"):
        _feedforward_to_tsdf_inputs(
            ff, frame_store=WrongResStore(), native_intrinsics=ff.intrinsics
        )


########
# optimize_color_map — rigid Zhou-Koltun color map optimization
########


def test_optimize_color_map_runs_and_recolors(tmp_path):
    """Rigid optimizer runs on a tiny synthetic scene and leaves a valid colored mesh in place."""
    # Constant-depth plane seen by 3 slightly-translated cameras; left half bright so the
    # optimizer has an image gradient to work with. 48px min: Open3D's default
    # image_boundary_margin=10 marks every vertex invisible on a 32px frame (all-black mesh).
    n, h, w = 3, 48, 48
    depths = np.full((n, h, w), 1.0, np.float32)
    rgbs = np.zeros((n, h, w, 3), np.uint8)
    rgbs[:, :, : w // 2] = 200
    K = np.array([[48.0, 0.0, 24.0], [0.0, 48.0, 24.0], [0.0, 0.0, 1.0]], np.float32)
    intrinsics = np.tile(K, (n, 1, 1))
    c2w = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    c2w[:, 0, 3] = np.linspace(-0.02, 0.02, n)

    fusion = Open3DTSDFFusion(
        output_dir=tmp_path, voxel_size=0.05, sdf_trunc=0.15, depth_trunc=5.0
    )
    result = fusion.create(depths, rgbs, c2w, intrinsics)

    optimize_color_map(
        result.mesh_path, depths, rgbs, c2w, intrinsics, iterations=5, depth_trunc=5.0
    )

    mesh = o3d.io.read_triangle_mesh(str(result.mesh_path))
    assert len(mesh.vertices) > 0
    assert len(mesh.vertex_colors) == len(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    assert colors.max() > 0.0  # optimizer reassigned real colors, not a zeroed mesh


def test_optimize_color_map_float_rgb_and_empty_mesh_guard(tmp_path):
    """Float [0,1] RGB converts at the boundary; a missing mesh fails loudly."""
    n, h, w = 2, 16, 16
    depths = np.full((n, h, w), 1.0, np.float32)
    rgbs = np.full((n, h, w, 3), 0.5, np.float32)  # float path
    K = np.array([[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]], np.float32)
    intrinsics = np.tile(K, (n, 1, 1))
    c2w = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    c2w[:, 0, 3] = np.linspace(-0.01, 0.01, n)

    fusion = Open3DTSDFFusion(
        output_dir=tmp_path, voxel_size=0.05, sdf_trunc=0.15, depth_trunc=5.0
    )
    result = fusion.create(depths, rgbs, c2w, intrinsics)
    optimize_color_map(
        result.mesh_path, depths, rgbs, c2w, intrinsics, iterations=2, depth_trunc=5.0
    )
    assert o3d.io.read_triangle_mesh(str(result.mesh_path)).has_vertices()

    with pytest.raises(ValueError, match="missing or empty"):
        optimize_color_map(
            tmp_path / "nope.ply", depths, rgbs, c2w, intrinsics, iterations=2, depth_trunc=5.0
        )


########
# pointcloud_to_mesh — color_map_iterations hook
########


def test_pointcloud_to_mesh_default_skips_color_map(tmp_path, monkeypatch):
    """color_map_iterations=0 (default) never touches the optimizer — shipping path unchanged."""
    called = []
    monkeypatch.setattr(
        "collab_splats.mesh.utils.optimize_color_map",
        lambda *a, **k: called.append(1),
    )
    result = _tiny_ff_result()
    pointcloud_to_mesh(
        result, tmp_path, voxel_size=0.05, sdf_trunc=0.15, depth_trunc=10.0
    )
    assert called == []


def test_pointcloud_to_mesh_color_map_requires_tsdf(tmp_path):
    """A non-TSDF method with color_map_iterations>0 fails loudly before any work."""
    result = _tiny_ff_result()
    with pytest.raises(ValueError, match="color_map_iterations"):
        pointcloud_to_mesh(
            result, tmp_path, method="depth_normal_poisson", color_map_iterations=10
        )


def test_pointcloud_to_mesh_color_map_called_with_fusion_arrays(tmp_path, monkeypatch):
    """color_map_iterations>0 calls the optimizer with the mesh path and the fusion's arrays."""
    calls = {}

    def fake_optimize(mesh_path, depths, rgbs, c2w, intrinsics, iterations, depth_trunc):
        calls.update(
            mesh_path=mesh_path,
            n_frames=depths.shape[0],
            iterations=iterations,
            depth_trunc=depth_trunc,
        )

    monkeypatch.setattr("collab_splats.mesh.utils.optimize_color_map", fake_optimize)
    result = _tiny_ff_result()
    mesh_result = pointcloud_to_mesh(
        result,
        tmp_path,
        voxel_size=0.05,
        sdf_trunc=0.15,
        depth_trunc=10.0,
        color_map_iterations=7,
    )
    assert calls["mesh_path"] == mesh_result.mesh_path
    assert calls["n_frames"] == result.depth.shape[0]
    assert calls["iterations"] == 7
    assert calls["depth_trunc"] == 10.0
