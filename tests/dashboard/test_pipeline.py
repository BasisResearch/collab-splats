# tests/dashboard/test_pipeline.py
import inspect
import logging
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
import torch
import yaml
import zarr

from collab_splats.dashboard import pipeline as pl
from collab_splats.dashboard.config import RunConfig
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.utils import ae_path, write_point_features


def _fake_frames(n=3):
    frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(n)]
    records = [{"frame_idx": i, "blur_score": 200.0} for i in range(n)]
    return frames, records


def _write_semantics(semantics_dir, n_points=32, latent=8, input_dim=32, extractor="talk2dino"):
    """Write the semantics artifact pair a scene dir is expected to carry.

    latent MUST stay != input_dim: with equal widths latent codes and decoded features have
    the same shape and a reader that skips the decode is indistinguishable from a correct one.
    """
    assert latent != input_dim, "fixture must not hide a latent/decoded mixup behind equal widths"
    torch.manual_seed(0)
    codes = np.random.default_rng(0).random((n_points, latent), dtype=np.float32)
    write_point_features(semantics_dir, extractor, codes, FeatureAutoencoder(input_dim=input_dim, latent_dim=latent))


class _InlineThread:
    """Stand-in for threading.Thread that runs the target synchronously on start()."""

    def __init__(self, target=None, daemon=None, **kw):
        self._target = target

    def start(self):
        if self._target is not None:
            self._target()


def test_run_pipeline_orders_steps_and_pushes(tmp_path):
    op_log = MagicMock()
    source = MagicMock()
    cfg = RunConfig(env_model="vggt_omega", semantic_extractor="talk2dino")
    video = tmp_path / "clip_03.mp4"
    video.write_bytes(b"x")

    fake_result = MagicMock()
    fake_result.points = list(range(10))  # len() used in the pointcloud step log line
    creator = MagicMock()
    # Pipeline decomposes run() into load_model→setup_inference→run_inference→postprocess;
    # result comes from creator.outputs after postprocess.
    creator.outputs = fake_result

    with (
        patch.object(pl, "sample_fps", return_value=_fake_frames()),
        patch.object(pl, "load_video_quality", return_value={"available": True, "frames": {}}),
        patch.object(pl, "_write_frames_zarr") as wz,
        patch.object(pl, "_build_creator", return_value=creator),
        patch.object(pl, "pointcloud_to_mesh") as mesh,
        patch.object(pl, "_extract_semantics") as sem,
        patch.object(pl, "_lift_and_compress") as liftc,
        patch.object(pl.threading, "Thread", _InlineThread),
    ):
        out = pl.run_pipeline(
            video_path=video,
            scene="2026_05_07-birds-clip_03",
            config=cfg,
            op_log=op_log,
            source=source,
            base_dir=tmp_path / "outputs",
        )

    assert out == tmp_path / "outputs" / "2026_05_07-birds-clip_03"
    wz.assert_called_once()
    creator.load_model.assert_called_once()
    creator.setup_inference.assert_called_once()
    creator.run_inference.assert_called_once()
    creator.postprocess.assert_called_once()
    fake_result.save_zarr.assert_called_once()
    mesh.assert_called_once()
    liftc.assert_called_once()
    sem.assert_called_once()
    cfg_path = out / "run_config.yaml"
    assert cfg_path.exists()
    loaded = RunConfig.from_yaml(cfg_path)
    assert loaded.frame_indices == [0, 1, 2]
    # video_ref is permanent provenance (run_config.yaml -> zarr local_features attrs):
    # "<scene>/<filename>", no bucket/session prefix and never a bare scene id.
    assert loaded.video_ref == "2026_05_07-birds-clip_03/clip_03.mp4"
    source.push_outputs.assert_called_with(out, "2026_05_07-birds-clip_03", on_line=ANY)
    op_log.finish_op.assert_called_once()


def test_run_pipeline_does_not_push_on_failure(tmp_path):
    op_log = MagicMock()
    source = MagicMock()
    cfg = RunConfig()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")

    with (
        patch.object(pl, "sample_fps", return_value=_fake_frames()),
        patch.object(pl, "load_video_quality", return_value={"available": True, "frames": {}}),
        patch.object(pl, "_write_frames_zarr"),
        patch.object(pl, "_build_creator", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError):
            pl.run_pipeline(
                video_path=video,
                scene="2026_05_07-birds-clip",
                config=cfg,
                op_log=op_log,
                source=source,
                base_dir=tmp_path / "o",
            )

    source.push_outputs.assert_not_called()
    op_log.error_op.assert_called_once()


def test_build_creator_maps_models():
    with patch.object(pl, "VGGTOmegaCreator") as omega:
        pl._build_creator("vggt_omega", 50.0)
        omega.assert_called_with(conf_threshold=50.0)
    with patch.object(pl, "VGGTXCreator") as vx:
        pl._build_creator("vggtx", 35.0)
        vx.assert_called_with(conf_threshold=35.0)
    with patch.object(pl, "MapAnythingCreator") as ma:
        pl._build_creator("mapanything", 35.0)
        ma.assert_called_with(confidence_percentile=35.0)


def test_transfer_mesh_features_writes_decoded_features(tmp_path):
    """vertex_features.npy must hold DECODED features, not the latent codes on disk.

    Widths differ (2-D codes, 5-D decoded) and the values are checked, so reading latent
    instead of decoded fails on both counts — its only reader (viewer.load_mesh_vertex_features)
    scores against full-dim text embeddings and cannot decode.
    """
    import open3d as o3d

    from collab_splats.dashboard import pipeline

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))
    o3d.io.write_triangle_mesh(str(tmp_path / "mesh.ply"), mesh)

    sem_dir = tmp_path / "semantics"
    _write_semantics(sem_dir, n_points=3, latent=2, input_dim=5)

    # Points offset off the vertices: k=1 with coincident points gives sigma=0 -> NaN weights.
    # One point per vertex at equal distance => each vertex gets exactly its own point feature.
    class _Result:
        points = verts + 0.01

    pipeline._transfer_mesh_features(_Result(), tmp_path, k=1, sdf_trunc=0.5)

    out = np.load(tmp_path / "vertex_features.npy")
    assert out.shape == (3, 5)  # decoded input_dim, NOT the 2-D latent width
    np.testing.assert_allclose(out, pipeline.load_point_features(sem_dir), rtol=1e-5, atol=1e-6)


########
# Per-point artifact pair: what _lift_and_compress writes, what load_point_features accepts
########


def test_lift_and_compress_caches_latent_codes_not_decoded(tmp_path):
    """The lifted store gets the LATENT codes (+ weights + dim attrs), never the decoded features.

    Input width 80 vs latent 64 so the two are distinguishable: writing decoded features (or
    lifting the raw maps instead of the encoded ones) lands 80 columns and fails here.
    """
    sem_dir = tmp_path / "semantics"
    sem_dir.mkdir()
    # The 2D cache store is where the write path recovers the extractor name from — it is handed
    # the directory, never the extractor, so both halves of the pair are named after this stem.
    zarr.open(str(sem_dir / "talk2dino.zarr"), mode="w")
    maps = [torch.randn(80, 2, 2) for _ in range(2)]  # (D, H_p, W_p)

    # Fake lift returns the maps it was handed, flattened to points — so the stored width
    # is exactly the width of whatever _lift_and_compress chose to lift.
    def fake_lift(feature_maps, result):
        return torch.cat([m.flatten(1).T for m in feature_maps])

    with (
        patch.object(pl, "load_feature_maps", return_value=maps),
        patch.object(pl, "lift_features", fake_lift),
    ):
        pl._lift_and_compress(object(), sem_dir, MagicMock())

    latent = pl.semantics_ae_policy().latent_dim
    store = zarr.open(str(sem_dir / "talk2dino_lifted.zarr"), mode="r")
    assert np.asarray(store["features"]).shape == (8, latent)
    assert dict(store.attrs) == {"input_dim": 80, "latent_dim": latent}
    assert (sem_dir / "talk2dino_ae.pt").is_file()
    # Round-trip: the cached pair decodes back to the full input width
    assert pl.load_point_features(sem_dir).shape == (8, 80)


def test_load_point_features_full_dim_pair_needs_no_weights(tmp_path):
    """semantics.n_components: null writes full-dim codes and no autoencoder — still readable."""
    feats = np.random.default_rng(0).random((6, 5), dtype=np.float32)
    write_point_features(tmp_path, "talk2dino", feats)  # ae=None -> uncompressed

    assert not (tmp_path / "talk2dino_ae.pt").exists()
    assert pl.point_features_cached(tmp_path)
    out = pl.load_point_features(tmp_path)
    # Returned as-is apart from the L2 normalization every consumer expects
    np.testing.assert_allclose(out, feats / np.linalg.norm(feats, axis=1, keepdims=True), rtol=1e-6)


def test_load_point_features_rejects_latent_codes_without_weights(tmp_path):
    """Half-written pair (codes, no weights): raise, never hand back undecoded codes."""
    _write_semantics(tmp_path, n_points=4, latent=2, input_dim=5)
    (tmp_path / "talk2dino_ae.pt").unlink()  # crash between the two writes

    assert not pl.point_features_cached(tmp_path)  # -> caller re-lifts instead of getting stuck
    with pytest.raises(FileNotFoundError, match="re-lift"):
        pl.load_point_features(tmp_path)


def test_load_point_features_rejects_legacy_codes_without_weights(tmp_path):
    """Pre-attrs scenes with no weights are indistinguishable from orphans — raise, don't guess."""
    store = zarr.open(str(tmp_path / "talk2dino_lifted.zarr"), mode="w")
    store["features"] = np.zeros((4, 2), dtype=np.float32)  # no dim attrs

    assert not pl.point_features_cached(tmp_path)
    with pytest.raises(FileNotFoundError, match="re-lift"):
        pl.load_point_features(tmp_path)


########
# Autoencoder policy: ONE gate, sourced from configs/base.yaml, shared by both fit paths
########


def test_ae_policy_is_sourced_from_base_yaml():
    """The dashboard must read the same semantics gate the Reconstructor does."""
    cfg_path = Path(__file__).parents[2] / "configs" / "base.yaml"
    semantics = yaml.safe_load(cfg_path.read_text())["semantics"]
    policy = pl.semantics_ae_policy()
    assert policy.latent_dim == semantics["n_components"]
    assert policy.target_cosine == semantics["target_cosine"]
    assert policy.max_epochs == semantics["max_epochs"]


def test_no_second_or_third_ae_policy_survives():
    """The per-path constants are gone — a third variant would silently diverge again."""
    from collab_splats.dashboard import viewer as viewer_mod

    for attr in ("_AE_EPOCHS", "_AE_LATENT_DIM"):
        assert not hasattr(pl, attr), f"pipeline.{attr} is a duplicate autoencoder policy"
    for attr in ("UPGRADE_MAX_EPOCHS", "UPGRADE_TARGET_COSINE"):
        assert not hasattr(viewer_mod, attr), f"viewer.{attr} is a duplicate autoencoder policy"


def test_no_fourth_ae_policy_hides_in_a_parameter_default():
    """A default IS a policy. reconstructor._lift_and_save carried max_epochs=10 against base.yaml's
    100 — invisible because its only caller passes both, and wrong for any caller that forgets."""
    from collab_splats.wrapper import reconstructor as recon_mod

    params = inspect.signature(recon_mod._lift_and_save).parameters
    for name in ("target_cosine", "max_epochs"):
        assert params[name].default is inspect.Parameter.empty, (
            f"reconstructor._lift_and_save.{name} defaults to {params[name].default!r} — "
            "a fourth autoencoder policy a caller can inherit by omission"
        )


def test_lift_and_compress_fit_uses_the_shared_policy(tmp_path):
    """The fresh-reconstruction fit is gated on the config's fidelity target, not just epochs."""
    sem_dir = tmp_path / "semantics"
    sem_dir.mkdir()
    zarr.open(str(sem_dir / "talk2dino.zarr"), mode="w")  # the stem both written halves take their name from
    maps = [torch.randn(80, 2, 2) for _ in range(2)]
    seen = {}
    real_fit = FeatureAutoencoder.fit

    def spy_fit(self, features, **kwargs):
        seen.update(kwargs)
        return real_fit(self, features, **{k: v for k, v in kwargs.items() if k != "on_epoch"})

    def fake_lift(feature_maps, result):
        return torch.cat([m.flatten(1).T for m in feature_maps])

    policy = pl.semantics_ae_policy()
    with (
        patch.object(pl, "load_feature_maps", return_value=maps),
        patch.object(pl, "lift_features", fake_lift),
        patch.object(FeatureAutoencoder, "fit", spy_fit),
    ):
        pl._lift_and_compress(object(), sem_dir, MagicMock())

    assert seen["epochs"] == policy.max_epochs
    assert seen["target_cosine"] == policy.target_cosine


########
# Identity-width latent clamp (item 10): a real guard, but compression becomes a no-op
########


def test_resolve_latent_dim_warns_when_the_clamp_binds(caplog):
    """input_dim <= latent_dim -> no compression AND a lossy round-trip; say so out loud."""
    with caplog.at_level(logging.WARNING, logger="collab_splats.dashboard.pipeline"):
        assert pl.resolve_latent_dim(4, 64) == 4
    assert "no-op" in caplog.text


def test_resolve_latent_dim_is_quiet_when_compression_is_real(caplog):
    with caplog.at_level(logging.WARNING, logger="collab_splats.dashboard.pipeline"):
        assert pl.resolve_latent_dim(768, 64) == 64
    assert caplog.text == ""


def test_resolve_latent_dim_treats_none_as_no_compression(caplog):
    """semantics.n_components: null -> full-width codes; still the clamp case, still warned."""
    with caplog.at_level(logging.WARNING, logger="collab_splats.dashboard.pipeline"):
        assert pl.resolve_latent_dim(768, None) == 768
    assert "no-op" in caplog.text


########
# Batched decode (item 11): a full (P, input_dim) one-shot decode is GBs of resident CPU RAM
########


def test_load_point_features_decodes_in_batches_matching_the_unbatched_result(tmp_path, monkeypatch):
    """Batched decode must be numerically identical to the one-shot decode it replaces."""
    sem_dir = tmp_path / "semantics"
    _write_semantics(sem_dir, n_points=37, latent=8, input_dim=32)

    # Reference: decode every code in ONE call through the same weights.
    codes = torch.from_numpy(np.asarray(zarr.open(str(sem_dir / "talk2dino_lifted.zarr"), mode="r")["features"]))
    ae = FeatureAutoencoder.load(ae_path(sem_dir, "talk2dino"))
    with torch.no_grad():
        expected = torch.nn.functional.normalize(ae.per_point_decode(codes), dim=1).numpy()

    # Shrink the batch so 37 points genuinely span several calls, and count them.
    sizes = []
    real_decode = FeatureAutoencoder.per_point_decode

    def counting_decode(self, x):
        sizes.append(len(x))
        return real_decode(self, x)

    monkeypatch.setattr(FeatureAutoencoder, "per_point_decode", counting_decode)
    out = pl.load_point_features(sem_dir, batch_size=5)

    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    assert len(sizes) == 8 and max(sizes) <= 5  # 37 = 7*5 + 2; never the whole array at once


########
# Ignored return value (item 13)
########


def test_extract_semantics_returns_nothing(tmp_path, monkeypatch):
    """_extract_semantics drops the cache path — every consumer re-resolves it by glob."""
    calls = []
    monkeypatch.setattr(pl, "extract_feature_cache", lambda extractor, frames, out: calls.append((frames, out)))
    monkeypatch.setattr(pl.BaseFeatureExtractor, "get", staticmethod(lambda name: lambda: object()))
    assert pl._extract_semantics("talk2dino", tmp_path / "frames.zarr", tmp_path) is None
    assert len(calls) == 1


def _make_localized_zarr(tmp_path, extractor="loma", n=2):
    """Minimal pointcloud.zarr with a localized/ group for one extractor."""
    import zarr
    from zarr.codecs import BloscCodec

    zpath = tmp_path / "pointcloud.zarr"
    store = zarr.open(str(zpath), mode="a")
    group = store.require_group(f"local_features/{extractor}/localized")
    lz4 = BloscCodec(cname="lz4")
    ext = np.stack([np.eye(4, dtype=np.float32) * (i + 1) for i in range(n)])
    group.create_array("extrinsics", data=ext, chunks=(1, 4, 4), compressors=lz4)
    group.attrs["image_paths"] = [f"/builder/machine/localized_frames/f{i}.jpg" for i in range(n)]
    return zpath, ext


def test_read_localized_group_returns_poses_and_local_paths(tmp_path):
    zpath, ext = _make_localized_zarr(tmp_path, extractor="loma", n=2)
    poses, paths = pl.read_localized_group(zpath, "loma", tmp_path)
    assert poses.shape == (2, 4, 4)
    np.testing.assert_allclose(poses, ext)
    # Paths remapped to this machine's localized_frames/ by basename
    assert paths == [tmp_path / "localized_frames" / "f0.jpg", tmp_path / "localized_frames" / "f1.jpg"]


def test_read_localized_group_missing_group_is_empty(tmp_path):
    import zarr

    zpath = tmp_path / "pointcloud.zarr"
    zarr.open(str(zpath), mode="a")  # store exists, no localized group
    poses, paths = pl.read_localized_group(zpath, "disk", tmp_path)
    assert poses.shape == (0, 4, 4)
    assert paths == []


def test_load_browse_data_composes_result_and_zarr(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from collab_splats.dashboard.operation_log import OperationLog

    out_dir = tmp_path / "2026_05_07-birds-vid"
    out_dir.mkdir(parents=True)
    zpath, _ = _make_localized_zarr(out_dir, extractor="loma", n=1)
    ref_ext = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    monkeypatch.setattr(
        "collab_splats.dashboard.pipeline._load_feedforward_result",
        lambda d: SimpleNamespace(extrinsics=ref_ext),
    )
    source = MagicMock()
    data = pl.load_browse_data(
        scene="2026_05_07-birds-vid",
        extractor="loma",
        source=source,
        base_dir=tmp_path,
        op_log=OperationLog(),
    )
    source.pull_processed.assert_not_called()  # zarr already local -> no pull
    assert data.extractor == "loma"
    assert data.ref_extrinsics.shape == (3, 4, 4)
    assert data.localized_extrinsics.shape == (1, 4, 4)
    assert data.mesh_path == out_dir / "mesh.ply"


def test_load_browse_data_pulls_scene_into_its_local_dir(tmp_path, monkeypatch):
    """pull_processed(scene, dest) — untyped and mirrored against push_outputs(local_dir, scene),
    so a silent arg swap here would pull into a directory named after the scene id."""
    from types import SimpleNamespace

    from collab_splats.dashboard.operation_log import OperationLog

    scene = "2026_05_07-birds-vid"
    out_dir = tmp_path / scene
    ref_ext = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    monkeypatch.setattr(
        "collab_splats.dashboard.pipeline._load_feedforward_result",
        lambda d: SimpleNamespace(extrinsics=ref_ext),
    )
    source = MagicMock()
    # pointcloud.zarr absent -> the pull fires; create it so the read path proceeds.
    source.pull_processed.side_effect = lambda *a, **k: _make_localized_zarr(out_dir, extractor="loma", n=1)
    pl.load_browse_data(
        scene=scene,
        extractor="loma",
        source=source,
        base_dir=tmp_path,
        op_log=OperationLog(),
    )
    source.pull_processed.assert_called_once_with(scene, out_dir, excludes=pl.PULL_EXCLUDES)
