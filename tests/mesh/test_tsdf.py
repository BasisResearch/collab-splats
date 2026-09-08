import numpy as np
import open3d as o3d
import pytest

from collab_splats.mesh.tsdf import check_bands, fuse_tsdf, fuse_tsdf_bands


def _views(n=3, h=32, w=32):
    """n cameras looking down +z at a plane 1 unit away, shifted 0.05 along x per view."""
    depths = np.ones((n, h, w), dtype=np.float32)
    rgbs = np.full((n, h, w, 3), 128, dtype=np.uint8)
    c2w = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c2w[:, 0, 3] = 0.05 * np.arange(n)
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64), (n, 1, 1))
    return depths, rgbs, c2w, K


def test_fuse_tsdf_writes_mesh_ply(tmp_path):
    depths, rgbs, c2w, K = _views()
    out = fuse_tsdf(depths, rgbs, c2w, K, tmp_path / "a" / "b", voxel_size=0.02, depth_trunc=2.0)
    assert out == tmp_path / "a" / "b" / "mesh.ply" and out.exists()
    mesh = o3d.io.read_triangle_mesh(str(out))
    assert len(mesh.vertices) > 0 and len(mesh.triangles) > 0
    assert mesh.has_vertex_colors()
    assert np.allclose(np.asarray(mesh.vertex_colors), 128 / 255, atol=0.05)


def test_fuse_tsdf_rejects_float_rgb(tmp_path):
    depths, rgbs, c2w, K = _views()
    with pytest.raises(ValueError, match="uint8"):
        fuse_tsdf(depths, rgbs.astype(np.float32) / 255, c2w, K, tmp_path, voxel_size=0.02, depth_trunc=2.0)


def test_fuse_tsdf_rejects_principal_point_outside_grid(tmp_path):
    depths, rgbs, c2w, K = _views()
    K[:, 0, 2] = 64  # cx beyond the 32-wide grid: K is at a different resolution than depth
    with pytest.raises(ValueError, match="Principal point"):
        fuse_tsdf(depths, rgbs, c2w, K, tmp_path, voxel_size=0.02, depth_trunc=2.0)


def test_fuse_tsdf_rejects_frame_count_mismatch(tmp_path):
    depths, rgbs, c2w, K = _views()
    with pytest.raises(ValueError, match="views"):
        fuse_tsdf(depths, rgbs[:2], c2w, K, tmp_path, voxel_size=0.02, depth_trunc=2.0)


def test_fuse_tsdf_sdf_trunc_defaults_to_four_voxels(tmp_path, monkeypatch):
    seen = []

    class _Recorder:
        def __init__(self, voxel_length, sdf_trunc, color_type):
            seen.append((voxel_length, sdf_trunc, color_type))

        def integrate(self, *args, **kwargs):
            pass

        def extract_triangle_mesh(self):
            return o3d.geometry.TriangleMesh.create_sphere(0.1)

    monkeypatch.setattr(o3d.pipelines.integration, "ScalableTSDFVolume", _Recorder)
    depths, rgbs, c2w, K = _views(n=1)
    fuse_tsdf(depths, rgbs, c2w, K, tmp_path, voxel_size=0.01, depth_trunc=2.0)
    assert len(seen) == 1
    assert seen[0][0] == 0.01 and seen[0][1] == pytest.approx(0.04)
    assert seen[0][2] == o3d.pipelines.integration.TSDFVolumeColorType.RGB8


########################################
# Banded fusion
########################################


def _two_plane_views(n=3, h=32, w=32, near=1.0, far=3.0):
    """n cameras looking down +z at two planes: left half of the frame near, right half far."""
    depths, rgbs, c2w, K = _views(n=n, h=h, w=w)
    depths[:, :, : w // 2] = near
    depths[:, :, w // 2 :] = far
    return depths, rgbs, c2w, K


def test_fuse_tsdf_depth_min_drops_the_near_surface(tmp_path):
    depths, rgbs, c2w, K = _two_plane_views()

    both = o3d.io.read_triangle_mesh(
        str(fuse_tsdf(depths, rgbs, c2w, K, tmp_path / "both", voxel_size=0.04, depth_trunc=5.0))
    )
    assert np.asarray(both.vertices)[:, 2].min() < 1.5

    far_only = o3d.io.read_triangle_mesh(
        str(fuse_tsdf(depths, rgbs, c2w, K, tmp_path / "far", voxel_size=0.04, depth_trunc=5.0, depth_min=2.0))
    )
    assert len(far_only.vertices) > 0
    assert np.asarray(far_only.vertices)[:, 2].min() > 2.0


def test_fuse_tsdf_bands_gives_each_band_its_own_voxel_and_truncation(tmp_path, monkeypatch):
    seen = []

    class _Recorder:
        def __init__(self, voxel_length, sdf_trunc, color_type):
            seen.append((voxel_length, sdf_trunc))

        def integrate(self, *args, **kwargs):
            pass

        def extract_triangle_mesh(self):
            return o3d.geometry.TriangleMesh()

    monkeypatch.setattr(o3d.pipelines.integration, "ScalableTSDFVolume", _Recorder)
    depths, rgbs, c2w, K = _views(n=1)
    bands = [
        {"depth_min": 0.0, "depth_trunc": 2.0, "voxel_size": 0.01},
        {"depth_min": 2.0, "depth_trunc": 8.0, "voxel_size": 0.04},
    ]
    fuse_tsdf_bands(depths, rgbs, c2w, K, tmp_path, bands=bands, sdf_trunc_mult=1.5)

    assert seen == [(0.01, pytest.approx(0.015)), (0.04, pytest.approx(0.06))]


def test_fuse_tsdf_bands_refines_the_near_band_only(tmp_path):
    depths, rgbs, c2w, K = _two_plane_views()
    coarse = 0.08

    uniform = o3d.io.read_triangle_mesh(
        str(fuse_tsdf(depths, rgbs, c2w, K, tmp_path / "uniform", voxel_size=coarse, depth_trunc=5.0))
    )
    banded = o3d.io.read_triangle_mesh(
        str(
            fuse_tsdf_bands(
                depths,
                rgbs,
                c2w,
                K,
                tmp_path / "banded",
                bands=[
                    {"depth_min": 0.0, "depth_trunc": 2.0, "voxel_size": 0.02},
                    {"depth_min": 2.0, "depth_trunc": 5.0, "voxel_size": coarse},
                ],
            )
        )
    )

    def split(mesh):
        z = np.asarray(mesh.vertices)[:, 2]
        return int((z < 2.0).sum()), int((z >= 2.0).sum())

    uniform_near, uniform_far = split(uniform)
    banded_near, banded_far = split(banded)

    # The near band is 4x finer, so it carries many more vertices; the far band is fused at the
    # same voxel as the uniform pass and must come out at the same density
    assert banded_near > 3 * uniform_near
    assert banded_far == pytest.approx(uniform_far, rel=0.15)


def test_fuse_tsdf_bands_at_one_voxel_size_does_not_double_the_surface(tmp_path):
    depths, rgbs, c2w, K = _two_plane_views()
    voxel = 0.04

    uniform = o3d.io.read_triangle_mesh(
        str(fuse_tsdf(depths, rgbs, c2w, K, tmp_path / "uniform", voxel_size=voxel, depth_trunc=5.0))
    )
    banded = o3d.io.read_triangle_mesh(
        str(
            fuse_tsdf_bands(
                depths,
                rgbs,
                c2w,
                K,
                tmp_path / "banded",
                bands=[
                    {"depth_min": 0.0, "depth_trunc": 2.0, "voxel_size": voxel},
                    {"depth_min": 2.0, "depth_trunc": 5.0, "voxel_size": voxel},
                ],
            )
        )
    )

    # Splitting a scene into bands at one voxel size reproduces the single volume: neither a
    # second copy of a surface nor a band's worth of geometry missing
    assert len(banded.vertices) == pytest.approx(len(uniform.vertices), rel=0.05)


def test_fuse_tsdf_bands_keeps_geometry_only_a_coarse_band_saw(tmp_path):
    depths, rgbs, c2w, K = _two_plane_views()

    # The far plane is outside every near band, so only the coarse band ever fuses it. Merging
    # must not drop it — that is the hole a geometric partition by camera distance would open
    banded = o3d.io.read_triangle_mesh(
        str(
            fuse_tsdf_bands(
                depths,
                rgbs,
                c2w,
                K,
                tmp_path,
                bands=[
                    {"depth_min": 0.0, "depth_trunc": 2.0, "voxel_size": 0.02},
                    {"depth_min": 2.0, "depth_trunc": 5.0, "voxel_size": 0.08},
                ],
            )
        )
    )
    z = np.asarray(banded.vertices)[:, 2]
    assert (z >= 2.0).sum() > 100
    assert (z < 2.0).sum() > 100


def test_fuse_tsdf_bands_keeps_one_copy_of_a_surface_two_bands_saw(tmp_path):
    h = w = 32
    voxel = 0.04

    # One plane at z=3, filmed from two ranges: two cameras at z=0 see it at depth 3, two at
    # z=2 see it at depth 1 through a 3x wider lens, so both cover the same patch of it
    depths = np.concatenate([np.full((2, h, w), 3.0), np.full((2, h, w), 1.0)]).astype(np.float32)
    rgbs = np.full((4, h, w, 3), 128, dtype=np.uint8)
    c2w = np.tile(np.eye(4, dtype=np.float64), (4, 1, 1))
    c2w[2:, 2, 3] = 2.0
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64), (4, 1, 1))
    K[2:, 0, 0] = K[2:, 1, 1] = w / 3

    near_band = {"depth_min": 0.0, "depth_trunc": 2.0, "voxel_size": voxel}
    far_band = {"depth_min": 2.0, "depth_trunc": 5.0, "voxel_size": voxel}
    one_band = o3d.io.read_triangle_mesh(str(fuse_tsdf_bands(depths, rgbs, c2w, K, tmp_path / "one", bands=[far_band])))
    both = o3d.io.read_triangle_mesh(
        str(fuse_tsdf_bands(depths, rgbs, c2w, K, tmp_path / "both", bands=[near_band, far_band]))
    )

    # Every band saw the whole plane, so the merge is one plane's worth of surface, not two
    assert len(one_band.vertices) > 100
    assert len(both.vertices) == pytest.approx(len(one_band.vertices), rel=0.2)


########################################
# Band validation
########################################


def test_check_bands_accepts_a_contiguous_ascending_list():
    checked = check_bands(
        [
            {"depth_min": 0, "depth_trunc": 2, "voxel_size": 0.01},
            {"depth_min": 2, "depth_trunc": 8, "voxel_size": 0.04},
        ]
    )
    assert checked == [
        {"depth_min": 0.0, "depth_trunc": 2.0, "voxel_size": 0.01},
        {"depth_min": 2.0, "depth_trunc": 8.0, "voxel_size": 0.04},
    ]
    assert all(isinstance(v, float) for band in checked for v in band.values())


@pytest.mark.parametrize(
    "bands, match",
    [
        (None, "non-empty list"),
        ([], "non-empty list"),
        ({"depth_min": 0, "depth_trunc": 2, "voxel_size": 0.01}, "non-empty list"),
        ([{"depth_min": 0, "depth_trunc": 2}], "keys depth_min, depth_trunc, voxel_size"),
        ([{"depth_min": 0, "depth_trunc": 2, "voxel_size": 0.01, "extra": 1}], "keys depth_min"),
        ([{"depth_min": 0, "depth_trunc": "2", "voxel_size": 0.01}], "depth_trunc must be a number"),
        ([{"depth_min": True, "depth_trunc": 2, "voxel_size": 0.01}], "depth_min must be a number"),
        ([{"depth_min": -1, "depth_trunc": 2, "voxel_size": 0.01}], "0 <= depth_min < depth_trunc"),
        ([{"depth_min": 2, "depth_trunc": 2, "voxel_size": 0.01}], "0 <= depth_min < depth_trunc"),
        ([{"depth_min": 0, "depth_trunc": 2, "voxel_size": 0}], "voxel_size must be > 0"),
        # A gap between bands loses the geometry inside it, an overlap double-surfaces it
        (
            [
                {"depth_min": 0, "depth_trunc": 2, "voxel_size": 0.01},
                {"depth_min": 3, "depth_trunc": 8, "voxel_size": 0.04},
            ],
            "must be contiguous",
        ),
        (
            [
                {"depth_min": 0, "depth_trunc": 4, "voxel_size": 0.01},
                {"depth_min": 2, "depth_trunc": 8, "voxel_size": 0.04},
            ],
            "must be contiguous",
        ),
    ],
)
def test_check_bands_rejects_malformed_bands(bands, match):
    with pytest.raises(ValueError, match=match):
        check_bands(bands)
