"""VGGTSPARKCreator inherits the mv path. Fresh process: a cached vggt package shadows SPARK."""

import subprocess
import sys

CHECK = """
from collab_splats.pointcloud.feedforward.vggt_spark_creator import VGGTSPARKCreator
c = VGGTSPARKCreator()
assert isinstance(c.min_views, int), "min_views not inherited"
assert isinstance(c.mv_conf_rel_thresh, float), "mv_conf_rel_thresh not inherited"
assert not hasattr(c, "mv_conf_threshold"), "stale mv_conf_threshold survived"
assert hasattr(type(c), "_postprocess"), "no _postprocess to carry the mv path"
from collab_splats.pointcloud.feedforward import vggtx
# SPARK must not override _postprocess away from the VGGT-X implementation that calls mv
assert type(c)._postprocess is vggtx.VGGTXCreator._postprocess, "SPARK overrides the mv path"
print("SPARK_MV_OK")
"""


def test_spark_inherits_mv_path():
    proc = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, timeout=600)
    assert "SPARK_MV_OK" in proc.stdout, f"stdout={proc.stdout}\nstderr={proc.stderr}"
