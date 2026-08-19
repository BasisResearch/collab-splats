"""Loss at fixed poses, after the structure is refitted to those poses.

Closes the one confound in the A/B start-at-GT experiment: run B started from GT poses but
from landmarks seeded off the model's `world_points`, so its 4.66e7 starting loss is "truth
with mismatched structure", not "loss at truth". Here every pose set gets its own best-fit
structure, so the comparison isolates the poses.

Inputs are BA's own arrays, dumped by ba_start_at_gt.py — same tracks, same model-resolution
K, same visibility gate, same loss definition (sum of squared reprojection residuals, shared
SIMPLE_PINHOLE focal, per-camera principal points).

Correctness gate: the loss at the seeded pts3d_tracks computed here is byte-identical to the
loss computed through the package's own `_reproject_shared` (3.609295e+07 at the model poses).
Note the LM history's first entry (1.948468e+06) is the loss *after* step 1, not the initial
loss — LM's first step already pulls 4.28 px RMS down to ~1 px.

Refits are initialised twice, from the seeded landmark and from the ray midpoint, and the
cheaper of the two is kept per landmark, so a refit can never score worse than its seed.
"""
from __future__ import annotations

import numpy as np
import torch

NPZ = "evals/results/ba_start_at_gt/ba_inputs.npz"
VIS_THRESH = 0.2
BLOCK = 4096  # landmarks per chunk; (N, BLOCK, 2, 3) Jacobian is the memory driver

d = np.load(NPZ)
tracks = d["tracks"].astype(np.float64)          # (N, P, 2) model-res pixels
vis = d["vis_scores"] > VIS_THRESH               # (N, P)
K = d["intrinsics_model"].astype(np.float64)     # (N, 3, 3)
pts_seed = d["pts3d_tracks"].astype(np.float64)  # (P, 3)

# Same observation set as the LM solve: visibility gate, then landmarks with >=2 views.
# max_reproj_error was None in that run, so no reprojection filter and no frame drops.
keep = vis.sum(0) >= 2
tracks, vis, pts_seed = tracks[:, keep], vis[:, keep], pts_seed[keep]
N, P, _ = tracks.shape
n_obs = int(vis.sum())
print(f"observation set: {P:,} landmarks, {n_obs:,} observations across {N} frames")
assert (P, n_obs) == (37257, 1966262), "observation set does not match the LM solve"

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_seed = torch.from_numpy(pts_seed).to(dev)
obs_all = torch.from_numpy(tracks).to(dev)
mask_all = torch.from_numpy(vis).to(dev)
ctr = torch.from_numpy(K[:, :2, 2]).to(dev)                     # (N, 2) per-camera
FOCAL = float(((K[:, 0, 0] + K[:, 1, 1]) / 2.0).mean())         # shared_camera=True
print(f"shared focal {FOCAL:.4f}, principal point spread "
      f"x {ctr[:, 0].min():.1f}-{ctr[:, 0].max():.1f} y {ctr[:, 1].min():.1f}-{ctr[:, 1].max():.1f}")


def residuals(X, R, t, obs, m, f):
    """(N,B) squared residual sum per landmark, plus the projection intermediates."""
    Xc = torch.einsum("nij,bj->nbi", R, X) + t[:, None]
    z = Xc[..., 2]
    zs = z.clamp(min=1e-6)
    proj = torch.stack([f * Xc[..., 0] / zs + ctr[:, None, 0],
                        f * Xc[..., 1] / zs + ctr[:, None, 1]], -1)
    r = (proj - obs) * m[..., None]
    # A landmark behind the camera is infeasible, not cheap — never let it score well
    r = torch.where(((z <= 1e-6) & m)[..., None], torch.full_like(r, 1e4), r)
    return r, Xc, zs, (r ** 2).sum((0, 2))


def refit(poses, f, iters=60):
    """Per-landmark LM triangulation at fixed poses; returns points and total squared loss."""
    T = torch.from_numpy(poses).to(dev)
    R, t = T[:, :3, :3], T[:, :3, 3]
    C = -torch.einsum("nji,nj->ni", R, t)
    eye = torch.eye(3, dtype=torch.float64, device=dev)
    out, total = torch.empty(P, 3, dtype=torch.float64, device=dev), 0.0

    for s in range(0, P, BLOCK):
        e = min(s + BLOCK, P)
        obs, m = obs_all[:, s:e], mask_all[:, s:e]
        B = e - s

        # Midpoint init: minimise summed squared point-to-ray distance over the seeing views
        rays = torch.stack([(obs[..., 0] - ctr[:, None, 0]) / f,
                            (obs[..., 1] - ctr[:, None, 1]) / f,
                            torch.ones_like(obs[..., 0])], -1)
        rays = rays / rays.norm(dim=-1, keepdim=True)
        dirs = torch.einsum("nji,nbj->nbi", R, rays)
        A = (eye - dirs.unsqueeze(-1) * dirs.unsqueeze(-2)) * m[..., None, None]
        An = A.sum(0)
        bn = torch.einsum("nbij,nj->nbi", A, C).sum(0)
        ridge = 1e-9 * An.diagonal(dim1=-2, dim2=-1).abs().sum(-1).clamp(min=1e-12)
        X_mid = torch.linalg.solve(An + ridge[:, None, None] * eye, bn.unsqueeze(-1)).squeeze(-1)

        # Take whichever start is already cheaper — the seed keeps the refit monotone against
        # the seeded structure; the midpoint rescues landmarks the seed placed badly
        X_seed_blk = X_seed[s:e]
        _, _, _, c_mid = residuals(X_mid, R, t, obs, m, f)
        _, _, _, c_seed = residuals(X_seed_blk, R, t, obs, m, f)
        X = torch.where((c_mid < c_seed)[:, None], X_mid, X_seed_blk)

        r, Xc, zs, c = residuals(X, R, t, obs, m, f)
        lam = torch.full((B,), 1e-4, dtype=torch.float64, device=dev)
        for _ in range(iters):
            # Gauss-Newton normal equations with LM damping, solved independently per landmark
            J = torch.zeros(N, B, 2, 3, dtype=torch.float64, device=dev)
            J[..., 0, 0] = f / zs
            J[..., 1, 1] = f / zs
            J[..., 0, 2] = -f * Xc[..., 0] / zs ** 2
            J[..., 1, 2] = -f * Xc[..., 1] / zs ** 2
            J = torch.einsum("nbac,ncd->nbad", J, R) * m[..., None, None]
            H = torch.einsum("nbac,nbad->bcd", J, J)
            g = torch.einsum("nbac,nba->bc", J, r)
            diag = H.diagonal(dim1=-2, dim2=-1).abs().clamp(min=1e-12)
            step = torch.linalg.solve(H + lam[:, None, None] * torch.diag_embed(diag),
                                      g.unsqueeze(-1)).squeeze(-1)
            rn, Xcn, zsn, cn = residuals(X - step, R, t, obs, m, f)
            # Accept per landmark, only where the cost actually fell
            acc = cn < c
            X = torch.where(acc[:, None], X - step, X)
            r = torch.where(acc[None, :, None], rn, r)
            Xc = torch.where(acc[None, :, None], Xcn, Xc)
            zs = torch.where(acc[None, :], zsn, zs)
            c = torch.where(acc, cn, c)
            lam = torch.where(acc, (lam * 0.3).clamp(min=1e-10), (lam * 10).clamp(max=1e8))

        out[s:e] = X
        total += float(c.sum())
    return out, total


def loss_at(poses, X, f):
    """Total squared reprojection loss for fixed poses and fixed structure."""
    T = torch.from_numpy(poses).to(dev)
    R, t = T[:, :3, :3], T[:, :3, 3]
    tot = 0.0
    for s in range(0, P, BLOCK):
        e = min(s + BLOCK, P)
        _, _, _, c = residuals(X[s:e], R, t, obs_all[:, s:e], mask_all[:, s:e], f)
        tot += float(c.sum())
    return tot


pose_sets = (("model poses", d["poses_model"]), ("GT poses", d["poses_gt_recon"]))

print("\n--- structure held at the seeded pts3d_tracks (initial state, before any LM step) ---")
for name, poses in pose_sets:
    tot = loss_at(poses, X_seed, FOCAL)
    print(f"  {name:12s} loss {tot:.6e}   rms {np.sqrt(tot / n_obs):.4f} px")

print("\n--- structure refitted to each pose set (poses fixed, shared focal fixed) ---")
for name, poses in pose_sets:
    X, tot = refit(poses, FOCAL)
    print(f"  {name:12s} loss {tot:.6e}   rms {np.sqrt(tot / n_obs):.4f} px")
