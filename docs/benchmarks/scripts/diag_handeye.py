"""Why the axis-Kabsch hand-eye fit misses the 0.29 deg floor.

Two candidate causes, both tested here rather than argued:
  1. Axis degeneracy -- walking is mostly yaw about gravity, so the relative-rotation axes
     may span barely more than one direction.  Kabsch on a rank-deficient cloud leaves the
     offset unidentified in the remaining degrees of freedom.
  2. Wrong model -- the reference stream may be world-from-body where the reconstruction is
     body-from-world, so the conjugation relation Kabsch assumes never held.

Cause 2 is settled by enumerating the four transpose combinations and optimising the offset
DIRECTLY against the residual, instead of via the axis proxy.  Direct optimisation is
immune to cause 1: a degenerate direction simply means many equally good minima, and any
of them drives the residual down.

ARCHIVED DIAGNOSTIC -- see docs/benchmarks/scripts/README.md.  Both conclusions now live in
shipped code: the direct multi-start optimisation is `evals/rotation_alignment.py`
(`fit_camera_offset`, whose docstring records the rejected axis-Kabsch numbers), and the
winning transpose is `evals/gopro_telemetry.py` (`reference_rotations_c2w`).

Run from the repo root:
    python docs/benchmarks/scripts/diag_handeye.py <telemetry.parquet> <colmap/sparse/0>
"""

import sys
from pathlib import Path

import numpy as np
import pycolmap
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "evals"))
from gopro_telemetry import sample_reference_at_frames as reference_at_frames

rec = pycolmap.Reconstruction(sys.argv[2])
rows = sorted(((int(im.name.split("_")[1]), im.cam_from_world()) for im in rec.images.values()),
              key=lambda r: r[0])
fidx = np.array([r[0] for r in rows])
R_est = np.stack([r[1].rotation.matrix().T for r in rows])
_, _, q = reference_at_frames(sys.argv[1], fidx, "iori_cori")
R_ref = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def rel(R):
    return np.einsum("nij,njk->nik", R[:-1].transpose(0, 2, 1), R[1:])


# --- Cause 1: how much of SO(3) do the rotation axes actually span? ---
for label, R in [("recon", R_est), ("ref", R_ref)]:
    rv = Rotation.from_matrix(rel(R)).as_rotvec()
    axes = rv / np.clip(np.linalg.norm(rv, axis=1, keepdims=True), 1e-9, None)
    sv = np.linalg.svd(axes, compute_uv=False)
    print(f"{label} axis-cloud singular values {sv.round(2)}  "
          f"(ratio smallest/largest {sv[-1] / sv[0]:.3f})")

# --- Cause 2 + robust fit: optimise the offset directly against the residual ---
def residual(rotvec, A, B):
    """Median relative-rotation disagreement in degrees for offset exp(rotvec)."""
    X = Rotation.from_rotvec(rotvec).as_matrix()
    dA = np.einsum("ij,njk,kl->nil", X.T, rel(A), X)
    err = np.einsum("nij,njk->nik", dA.transpose(0, 2, 1), rel(B))
    return np.degrees(np.median(np.linalg.norm(Rotation.from_matrix(err).as_rotvec(), axis=1)))


print("\ndirect optimisation of the offset, all four transpose conventions:")
best = None
for est_t in (False, True):
    for ref_t in (False, True):
        A = np.transpose(R_est, (0, 2, 1)) if est_t else R_est
        B = np.transpose(R_ref, (0, 2, 1)) if ref_t else R_ref
        # Multi-start: the residual is non-convex on SO(3), so seed from several rotations.
        seeds = [np.zeros(3), [np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi],
                 [np.pi / 2, 0, 0], [0, np.pi / 2, 0], [0, 0, np.pi / 2]]
        r = min((minimize(residual, s, args=(A, B), method="Nelder-Mead",
                          options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
                 for s in seeds), key=lambda o: o.fun)
        tag = f"est{'^T' if est_t else '  '} ref{'^T' if ref_t else '  '}"
        print(f"  {tag}  median residual {r.fun:.3f} deg")
        if best is None or r.fun < best[0]:
            best = (r.fun, tag, r.x)
print(f"\nbest: {best[1]} at {best[0]:.3f} deg (invariant floor is 0.288 deg)")
