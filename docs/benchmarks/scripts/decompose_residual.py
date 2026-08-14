"""Decompose the 0.272% pinhole residual: is it LoGeR, or is it us?

The parity test measures |ours - native| and attributes all of it to LoGeR's `xy` ray
field being non-pinhole. That attribution is untested. This splits the residual into:

  (A) NON-AFFINE  — LoGeR's xy really is not a pinhole grid. Irreducible; not our bug.
  (B) K-FIT       — our estimate_intrinsics_from_points is worse than the optimal
                    least-squares K for this ray field. Our bug if large.
  (C) K-SHARING   — one K across all frames when LoGeR's per-frame rays differ.
                    Our design choice; measurable.

Decisive check: if xy is EXACTLY affine in (u, v), a pinhole K reproduces it perfectly
and the entire residual is ours.
"""

import numpy as np
import torch

from tests.pointcloud.test_loger_creator import _tutorial_frames
from collab_splats.pointcloud.feedforward.loger import (
    LOGER_CONF_THRESHOLD,
    LoGeRCreator,
    _compute_target_size,
)

creator = LoGeRCreator()
frames = _tutorial_frames()
n, orig_h, orig_w = frames.shape[:3]
w, h = _compute_target_size(orig_w, orig_h, creator.pixel_limit)
model = creator._load_model("cuda")
views, _, _ = creator._preprocess(frames, list(range(n)))

# One forward; take local_points (camera frame) and the confidence, straight off the model.
with torch.no_grad():
    preds = model(views.to("cuda")[None], **creator._forward_kwargs())
local = preds["local_points"].squeeze(0).cpu().float().numpy()      # (N,H,W,3)
conf = torch.sigmoid(preds["conf"]).squeeze(0).cpu().float().numpy()
conf = conf.reshape(n, h, w)
mask = conf > LOGER_CONF_THRESHOLD
print(f"frames={n} model={w}x{h} local={local.shape} mask={mask.mean():.3%}")

# LoGeR's normalised rays. For an ideal pinhole: x = (u - cx)/fx exactly, with NO v term.
z = local[..., 2]
ray_ok = mask & (np.abs(z) > 1e-6)
xn = local[..., 0] / np.where(np.abs(z) > 1e-6, z, 1.0)
yn = local[..., 1] / np.where(np.abs(z) > 1e-6, z, 1.0)

uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))

print("\n=== (A) is the ray field affine in pixel coords? ===")
print("per-frame least-squares fit of xn,yn ~ [u, v, 1]; a true pinhole has zero cross term")
per_frame_K = []
for i in range(n):
    m = ray_ok[i]
    A = np.stack([uu[m], vv[m], np.ones(m.sum())], axis=1)
    cx_sol, *_ = np.linalg.lstsq(A, xn[i][m], rcond=None)
    cy_sol, *_ = np.linalg.lstsq(A, yn[i][m], rcond=None)
    # Residual of the BEST affine model = the part no pinhole K can ever represent.
    rx = xn[i][m] - A @ cx_sol
    ry = yn[i][m] - A @ cy_sol
    # Convert the optimal affine fit to a K: xn = (u - cx)/fx  =>  slope = 1/fx
    fx, fy = 1.0 / cx_sol[0], 1.0 / cy_sol[1]
    per_frame_K.append((fx, fy, -cx_sol[2] * fx, -cy_sol[2] * fy))
    if i < 3 or i == n - 1:
        print(f"  f{i}: cross terms du/dv={cx_sol[1]:+.3e} dv/du={cy_sol[0]:+.3e} | "
              f"non-affine resid rms=({rx.std():.3e}, {ry.std():.3e}) | "
              f"optimal fx={fx:.2f} fy={fy:.2f} cx={-cx_sol[2]*fx:.2f} cy={-cy_sol[2]*fy:.2f}")

per_frame_K = np.array(per_frame_K)
# Scale-free measure: non-affine residual relative to the ray field's own spread.
all_rx, all_ry = [], []
for i in range(n):
    m = ray_ok[i]
    A = np.stack([uu[m], vv[m], np.ones(m.sum())], axis=1)
    sx, *_ = np.linalg.lstsq(A, xn[i][m], rcond=None)
    sy, *_ = np.linalg.lstsq(A, yn[i][m], rcond=None)
    all_rx.append(np.abs(xn[i][m] - A @ sx))
    all_ry.append(np.abs(yn[i][m] - A @ sy))
all_rx, all_ry = np.concatenate(all_rx), np.concatenate(all_ry)
spread = np.percentile(np.abs(np.concatenate([xn[ray_ok], yn[ray_ok]])), 95)
print(f"  ALL FRAMES non-affine |resid|: median {np.median(np.r_[all_rx, all_ry]):.3e}  "
      f"p99 {np.percentile(np.r_[all_rx, all_ry], 99):.3e}  (ray spread p95 {spread:.4f})")
print(f"  -> non-affine share of ray magnitude: median "
      f"{np.median(np.r_[all_rx, all_ry]) / spread:.3%}")

print("\n=== (C) do the per-frame optimal K agree with each other? ===")
print(f"  fx  min {per_frame_K[:,0].min():.2f}  max {per_frame_K[:,0].max():.2f}  "
      f"spread {np.ptp(per_frame_K[:,0]) / per_frame_K[:,0].mean():.3%}")
print(f"  fy  min {per_frame_K[:,1].min():.2f}  max {per_frame_K[:,1].max():.2f}  "
      f"spread {np.ptp(per_frame_K[:,1]) / per_frame_K[:,1].mean():.3%}")
print(f"  cx  min {per_frame_K[:,2].min():.2f}  max {per_frame_K[:,2].max():.2f}")
print(f"  cy  min {per_frame_K[:,3].min():.2f}  max {per_frame_K[:,3].max():.2f}")

print("\n=== (B) our fitted K vs the optimal least-squares K ===")
raw = creator._forward(model, views)
ours_K = raw["intrinsics"][0]
print(f"  ours     fx={ours_K[0,0]:.2f} fy={ours_K[1,1]:.2f} cx={ours_K[0,2]:.2f} cy={ours_K[1,2]:.2f}")
print(f"  optimal  fx={per_frame_K[:,0].mean():.2f} fy={per_frame_K[:,1].mean():.2f} "
      f"cx={per_frame_K[:,2].mean():.2f} cy={per_frame_K[:,3].mean():.2f}  (mean of per-frame)")
for j, name in enumerate(["fx", "fy", "cx", "cy"]):
    o = [ours_K[0,0], ours_K[1,1], ours_K[0,2], ours_K[1,2]][j]
    print(f"  {name}: ours/optimal = {o / per_frame_K[:,j].mean():.5f}")

# Direct camera-frame residual: rebuild local points from OUR K + LoGeR's z, no poses involved.
print("\n=== camera-frame residual, poses excluded entirely ===")
fx, fy, cx, cy = ours_K[0,0], ours_K[1,1], ours_K[0,2], ours_K[1,2]
rebuilt = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], axis=-1)
err = np.linalg.norm(rebuilt[mask] - local[mask], axis=-1)
scale = float(np.percentile(np.linalg.norm(local[mask], axis=-1), 95))
print(f"  ours-K vs LoGeR local_points: median {np.median(err)/scale:.4%} "
      f"p95 {np.percentile(err,95)/scale:.4%} p99 {np.percentile(err,99)/scale:.4%}")

# Same, but with each frame's OWN optimal K — removes (B) and (C), leaving only (A).
errs = []
for i in range(n):
    f_x, f_y, c_x, c_y = per_frame_K[i]
    rb = np.stack([(uu - c_x) / f_x * z[i], (vv - c_y) / f_y * z[i], z[i]], axis=-1)
    errs.append(np.linalg.norm(rb[mask[i]] - local[i][mask[i]], axis=-1))
errs = np.concatenate(errs)
print(f"  per-frame optimal K:          median {np.median(errs)/scale:.4%} "
      f"p95 {np.percentile(errs,95)/scale:.4%} p99 {np.percentile(errs,99)/scale:.4%}")
print("\n  If the two lines above are close, the residual is LoGeR's ray field (A).")
print("  If per-frame optimal is MUCH lower, it is our K fit (B) or K sharing (C) — our bug.")
