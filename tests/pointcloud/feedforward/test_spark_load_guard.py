import pytest

from collab_splats.pointcloud.feedforward.vggt_spark_creator import (
    _VGGT_SPARK_ROOT,
    _assert_loaded_from_spark,
)


def test_accepts_file_under_spark_root():
    # A path inside the SPARK tree passes silently.
    _assert_loaded_from_spark(f"{_VGGT_SPARK_ROOT}/vggt/models/vggt.py")


def test_rejects_file_outside_spark_root():
    # An installed VGGT-X path (site-packages) must raise, not warn.
    with pytest.raises(RuntimeError, match="VGGT-X"):
        _assert_loaded_from_spark(
            "/opt/conda/envs/reconstruction/lib/python3.11/site-packages/vggt/models/vggt.py"
        )
