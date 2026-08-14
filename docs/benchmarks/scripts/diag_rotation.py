"""Localise where the rotation disagreement lives: the poses, or my hand-eye fit.

The angle of a relative rotation is invariant under conjugation, so it is invariant to the
unknown camera-convention offset AND to the unknown world alignment.  If the estimated and
reference relative-rotation ANGLES agree but the full relative rotations do not, the poses
are fine and the axis fit is at fault.  If the angles themselves disagree, the problem is
upstream of any alignment -- in the poses or in the reference stream.

ARCHIVED DIAGNOSTIC -- see docs/benchmarks/scripts/README.md.  Its conclusion is now carried
by `evals/gopro_telemetry.py` (the ORIENTATION_MODES docstring) and re-measured on every run
by `evals/scripts/eval_gopro_reference.py --orientation auto`.

Run from the repo root:
    python docs/benchmarks/scripts/diag_rotation.py <telemetry.parquet> <colmap/sparse/0>
"""

import sys
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "evals"))
from gopro_telemetry import sample_reference_at_frames as reference_at_frames

TEL, SPARSE = sys.argv[1], sys.argv[2]

rec = pycolmap.Reconstruction(SPARSE)
rows = sorted(((int(im.name.split("_")[1]), im.cam_from_world()) for im in rec.images.values()),
              key=lambda r: r[0])
fidx = np.array([r[0] for r in rows])
R_est = np.stack([r[1].rotation.matrix().T for r in rows])


def rel_angles(R):
    """Angle in degrees of each consecutive relative rotation."""
    rel = np.einsum("nij,njk->nik", R[:-1].transpose(0, 2, 1), R[1:])
    return np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))


a_est = rel_angles(R_est)
print(f"recon inter-frame rotation: median {np.median(a_est):.2f}d  "
      f"p95 {np.percentile(a_est, 95):.2f}d  max {a_est.max():.2f}d")

for mode in ["cori", "cori_iori", "cori_iori_conj", "iori_cori", "iori_conj_cori"]:
    _, _, q = reference_at_frames(TEL, fidx, mode)
    R_ref = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    a_ref = rel_angles(R_ref)
    # The invariant comparison: same magnitude of turn between the same two frames?
    d = np.abs(a_est - a_ref)
    corr = np.corrcoef(a_est, a_ref)[0, 1]
    print(f"{mode:<16} ref median {np.median(a_ref):6.2f}d | "
          f"|angle diff| median {np.median(d):5.2f}d  p95 {np.percentile(d, 95):5.2f}d | "
          f"corr {corr:.3f}")

# A conjugated reference stream (opposite handedness convention) is the other likely cause.
_, _, q = reference_at_frames(TEL, fidx, "cori")
R_ref = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
a_conj = rel_angles(np.transpose(R_ref, (0, 2, 1)))
print(f"\ncori transposed (w2c vs c2w): |angle diff| median "
      f"{np.median(np.abs(a_est - a_conj)):.2f}d  corr {np.corrcoef(a_est, a_conj)[0, 1]:.3f}")
