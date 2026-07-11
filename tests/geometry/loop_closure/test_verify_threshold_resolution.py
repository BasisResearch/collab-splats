"""Per-model verify_match_ratio resolution at LoopClosure init (GPU-free)."""

from collab_splats.geometry.loop_closure import LoopClosureConfig
from collab_splats.geometry.loop_closure.wrapper import LoopClosure


class _CreatorWithDefault:
    """Stand-in creator exposing a per-model calibrated threshold."""

    default_verify_match_ratio = 0.72


class _CreatorWithoutDefault:
    """Stand-in creator with no per-model threshold attribute."""


def test_config_verify_match_ratio_defaults_to_none():
    # None means "resolve per-model default at wrapper init"
    assert LoopClosureConfig().verify_match_ratio is None


def test_explicit_config_none_resolves_to_model_default():
    # The plumbing-gap case: explicit config (eval_gt.py path) must still pick
    # up the creator's calibrated threshold when verify_match_ratio is unset.
    lc = LoopClosure(_CreatorWithDefault(), config=LoopClosureConfig(submap_size=16))
    assert lc.config.verify_match_ratio == 0.72
    assert lc.config.submap_size == 16


def test_explicit_config_none_falls_back_to_085_without_attr():
    lc = LoopClosure(_CreatorWithoutDefault(), config=LoopClosureConfig())
    assert lc.config.verify_match_ratio == 0.85


def test_explicit_float_wins_over_model_default():
    cfg = LoopClosureConfig(verify_match_ratio=0.6)
    lc = LoopClosure(_CreatorWithDefault(), config=cfg)
    assert lc.config.verify_match_ratio == 0.6


def test_no_config_resolves_to_model_default():
    lc = LoopClosure(_CreatorWithDefault())
    assert lc.config.verify_match_ratio == 0.72


def test_no_config_falls_back_to_085_without_attr():
    lc = LoopClosure(_CreatorWithoutDefault())
    assert lc.config.verify_match_ratio == 0.85


def test_vggtx_has_calibrated_classvar():
    # Chess d5 layer-sweep calibration (2026-07-09): layer 10, threshold 1.17.
    # Import inside test: vggt is an optional heavy dep in some environments.
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

    assert VGGTXCreator.default_verify_match_ratio == 1.17
    assert VGGTXCreator._lc_layer_index == 10


def test_omega_has_calibrated_classvars():
    # Chess d5 clean-negative sweep (2026-07-10): layer 13, threshold 1.55.
    from collab_splats.pointcloud.feedforward.vggt_omega import VGGTOmegaCreator

    assert VGGTOmegaCreator._lc_layer_index == 13
    assert VGGTOmegaCreator.default_verify_match_ratio == 1.55


def test_mapanything_has_calibrated_classvars():
    # Chess d5 clean-negative sweep (2026-07-10): layer 4 confirmed, threshold 1.46.
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    assert MapAnythingCreator._lc_layer_index == 4
    assert MapAnythingCreator.default_verify_match_ratio == 1.46


def test_spark_overrides_vggtx_calibration():
    # Spark subclasses VGGTXCreator but verifies via NATIVE similarity scores
    # (own _verify_loop_candidate, ~1.02-1.05 on accepts) — it must define its
    # own threshold rather than inherit vggtx's hook-ratio calibration.
    from collab_splats.pointcloud.feedforward.vggt_spark_creator import VGGTSPARKCreator

    assert "default_verify_match_ratio" in vars(VGGTSPARKCreator)
    assert VGGTSPARKCreator.default_verify_match_ratio == 0.95
