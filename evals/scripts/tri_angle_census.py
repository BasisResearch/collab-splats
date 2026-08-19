"""D5 — triangulation-angle census on the vggt_omega chess/seq-01 track cache.

Read-only: camera centres + existing landmark positions. No triangulation, no BA.
"""
import numpy as np
import zarr

CACHE = "evals/results/ba_convergence_chess/cache/vggt_omega/tracks.zarr"
POSES = "evals/results/ba_convergence_chess/vggt_omega/trajectories.npz"
VIS_THRESH = 0.2

# Load tracks + poses; pred_ba[0] is near-identity so poses are in the recon frame
z = zarr.open(CACHE, mode="r")
vis = np.asarray(z["vis_scores"])          # (N, P)
pts = np.asarray(z["pts3d_tracks"])        # (P, 3)
T = np.load(POSES)["pred_ba"].astype(np.float64)  # (N, 4, 4)
N, P = vis.shape

# Resolve the pose convention by cheirality: whichever gives more positive depths wins
R, t = T[:, :3, :3], T[:, :3, 3]
C_c2w = t                                   # camera-to-world: centre is the translation
C_w2c = -np.einsum("nji,nj->ni", R, t)      # world-to-camera: C = -R^T t
V = vis > VIS_THRESH


def cheirality_frac(C, w2c: bool) -> float:
    """Fraction of visible observations with the point in front of the camera."""
    sub = np.linspace(0, P - 1, 4000).astype(int)
    p = pts[sub]                                              # (S, 3)
    if w2c:
        z_cam = np.einsum("nij,sj->nsi", R, p)[..., 2] + t[:, None, 2]
    else:
        d = p[None] - C[:, None]                              # (N, S, 3)
        z_cam = np.einsum("nji,nsj->nsi", R, d)[..., 2]
    m = V[:, sub]
    return float((z_cam[m] > 0).mean())


f_c2w = cheirality_frac(C_c2w, w2c=False)
f_w2c = cheirality_frac(C_w2c, w2c=True)
C = C_c2w if f_c2w >= f_w2c else C_w2c
print(f"cheirality frac  c2w={f_c2w:.4f}  w2c={f_w2c:.4f}  -> using {'c2w' if f_c2w >= f_w2c else 'w2c'}")

extent = np.linalg.norm(C[:, None] - C[None], axis=-1).max()
print(f"camera-centre extent: {extent:.4f}   median landmark depth: {np.median(np.linalg.norm(pts - C.mean(0), axis=1)):.4f}")

# Per-landmark max pairwise ray angle, chunked over points
n_obs_per_pt = V.sum(0)
max_ang = np.zeros(P, dtype=np.float64)
CH = 2000
for s in range(0, P, CH):
    e = min(s + CH, P)
    d = pts[s:e][None].astype(np.float64) - C[:, None]        # (N, B, 3)
    r = d / np.linalg.norm(d, axis=-1, keepdims=True)
    m = V[:, s:e]                                             # (N, B)
    r = np.where(m[..., None], r, 0.0)
    dot = np.einsum("ibk,jbk->ijb", r, r)                     # (N, N, B)
    pair = m[:, None, :] & m[None, :, :]
    dot = np.where(pair, np.clip(dot, -1.0, 1.0), 1.0)        # unobserved pairs -> 0 deg
    max_ang[s:e] = np.degrees(np.arccos(dot.min(axis=(0, 1))))

valid = n_obs_per_pt >= 2
print(f"\nlandmarks: {P} total, {int(valid.sum())} with >=2 observations at vis>{VIS_THRESH}")
qs = [1, 5, 25, 50, 75, 95, 99]
print("angle percentiles (deg): " + ", ".join(f"p{q}={np.percentile(max_ang[valid], q):.2f}" for q in qs))

print(f"\n{'gate':>8} {'pts dropped':>14} {'%pts':>7} {'obs dropped':>14} {'%obs':>7} {'pts kept':>10}")
tot_obs = int(n_obs_per_pt[valid].sum())
for g in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 16.0]:
    drop = valid & (max_ang < g)
    dp, do = int(drop.sum()), int(n_obs_per_pt[drop].sum())
    print(f"{g:>7.1f}° {dp:>14d} {100*dp/valid.sum():>6.1f}% {do:>14d} {100*do/tot_obs:>6.1f}% {int(valid.sum())-dp:>10d}")

# Per-frame observation count after each gate vs min_inliers_per_frame=64
print("\nper-frame surviving observations (min_inliers_per_frame=64):")
for g in [1.5, 3.0, 6.0]:
    keep = valid & (max_ang >= g)
    per_frame = V[:, keep].sum(1)
    print(f"  gate {g:>4.1f}°  min={per_frame.min():>6d}  median={int(np.median(per_frame)):>6d}  frames<64: {int((per_frame < 64).sum())}")
