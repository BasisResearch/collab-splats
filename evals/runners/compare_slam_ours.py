"""
Systematic step-by-step numerical comparison: our LC pipeline vs VGGT-SLAM logic.

Both pipelines receive IDENTICAL VGGT model outputs (same frames, same model).
We trace through every step and save intermediate values to numpy, then diff them.

Steps compared:
  1. VGGT outputs (depth, extrinsics, intrinsics, depth_conf) per submap
  2. Point clouds stored per submap (SLAM: camera-local; ours: cam0-frame)
  3. Scale estimation inputs (t1, t2) and scale value
  4. H_w at each boundary
  5. Initial H values (pre-PGO) for all nodes
  6. H values after PGO
  7. Extracted (R, t) per frame
"""
import sys, os
sys.path.insert(0, "/workspace/collab-splats")
sys.path.insert(0, "/workspace/collab-splats/third_party/VGGT-SLAM")
sys.path.insert(0, "/workspace/collab-splats/third_party/vggt_spark")

import numpy as np
from pathlib import Path

OUT = Path("/tmp/compare_slam_ours")
OUT.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
KF_FILE = Path("evals/baselines/disparity_sweep/slam_d10/selected_frames.txt")
SEQ_DIR = Path("evals/data/7scenes/chess/chess/seq-01")
SUBMAP_SIZE = 16
SUBMAP_OVERLAP = 1

# ── Step 1: Run our LC pipeline, capture raw VGGT outputs per submap ────────
import torch
from collab_splats.pointcloud.feedforward.vggt_spark_creator import VGGTSPARKCreator
from collab_splats.pointcloud.wrappers import LoopClosure
from collab_splats.pointcloud.loop_closure import LoopClosureConfig, Submap
from collab_splats.pointcloud.loop_closure.closure import run_pose_graph_optimization
from collab_splats.pointcloud.loop_closure.graph import (
    estimate_scale_pairwise, decompose_camera, PoseGraph
)
from collab_splats.utils.geometry import extrinsics_to_homogeneous, invert_poses
from collab_splats.pointcloud.feedforward.base import _raw_to_world_points
from collab_splats.pointcloud.loop_closure.submap import assert_world_to_cam
from collab_splats.localization import BaseRetrievalExtractor

# Load keyframes
keyframes = [l.strip() for l in KF_FILE.read_text().splitlines() if l.strip()]
print(f"Keyframes: {len(keyframes)}")

# Load model
creator = VGGTSPARKCreator()
creator.load_model()
creator.setup_inference(SEQ_DIR)
# Filter to keyframes
kf_set = set(keyframes)
filtered = [(i, p) for i, p in enumerate(creator.image_paths) if str(p) in kf_set or p.name in kf_set]
indices = [i for i, _ in filtered]
creator.image_paths = [p for _, p in filtered]
creator.views = creator.views[indices]
views = creator.views
N = len(creator.image_paths)
print(f"Filtered to {N} frames")

# DINO-SALAD
device = str(next(creator.model.parameters()).device)
retrieval_cls = BaseRetrievalExtractor.get("dino-salad")
retrieval_extractor = retrieval_cls(device=device)

K_size = SUBMAP_SIZE
O = SUBMAP_OVERLAP
step = K_size

# ── Collect submaps ──────────────────────────────────────────────────────────
submaps_raw = []  # raw VGGT outputs per submap
submaps_our = []  # our Submap objects

for wi, start in enumerate(range(0, N, step)):
    end = min(start + K_size + O, N)
    window = views[start:end]
    k = window.shape[0]
    with torch.no_grad():
        raw = creator._forward(creator.model, window)
    torch.cuda.empty_cache()

    ext_3x4 = raw["extrinsic"]
    poses_4x4 = extrinsics_to_homogeneous(ext_3x4)
    intr_key = "intrinsics" if "intrinsics" in raw else "intrinsic"
    intrinsics = raw.get(intr_key, np.tile(np.eye(3), (k, 1, 1)).astype(np.float32))

    frames_cpu = window.cpu()
    ret_vecs = retrieval_extractor(frames_cpu)
    wp, wp_conf = _raw_to_world_points(raw)

    submap = Submap(
        submap_id=wi,
        frames=frames_cpu,
        poses=poses_4x4,
        intrinsics=intrinsics,
        retrieval_vectors=ret_vecs,
        image_paths=list(creator.image_paths[start:end]),
        raw_outputs=raw,
        frame_start=start,
        world_points=wp,
        world_points_conf=wp_conf,
    )
    submaps_our.append(submap)
    submaps_raw.append({
        "depth": raw["depth"],
        "extrinsic": ext_3x4,
        "intrinsic": intrinsics,
        "depth_conf": raw.get("depth_conf"),
        "start": start, "end": end, "k": k,
    })
    print(f"Submap {wi}: frames {start}-{end-1}, k={k}, poses[0]≈I: {np.allclose(poses_4x4[0], np.eye(4), atol=0.01)}")
    if end >= N:
        break

n_submaps = len(submaps_our)
print(f"\nTotal submaps: {n_submaps}")

# ── Step 2: Save depth/pointclouds per submap (camera-local vs world-frame) ──
from vggt.utils.geometry import depth_to_cam_coords_points  # SLAM-style camera-local

print("\n=== STEP 2: Point clouds ===")
for i, (sm, raw) in enumerate(zip(submaps_our, submaps_raw)):
    depth = raw["depth"].squeeze(-1) if raw["depth"].ndim == 4 else raw["depth"]  # (k, H, W)
    intr = raw["intrinsic"]
    ext_3x4 = raw["extrinsic"]
    k = raw["k"]

    # SLAM style: camera-local per frame (depth_to_cam_coords_points)
    slam_pts = []
    for fi in range(k):
        cam_pts = depth_to_cam_coords_points(depth[fi], intr[fi])  # (H, W, 3)
        slam_pts.append(cam_pts.reshape(-1, 3))

    # Our style: world-frame (cam0-frame) via _raw_to_world_points
    # sm.world_points is (k, P, 3) subsampled; use full depth for fair comparison
    # Recompute full-res world_pts
    c2w = invert_poses(extrinsics_to_homogeneous(ext_3x4).astype(np.float64)).astype(np.float32)
    ours_pts = []
    for fi in range(k):
        cam_pts = depth_to_cam_coords_points(depth[fi], intr[fi]).reshape(-1, 3)  # camera-local
        hom = np.hstack([cam_pts, np.ones((len(cam_pts), 1))])
        world_pts = (c2w[fi] @ hom.T).T[:, :3]
        ours_pts.append(world_pts)

    np.save(OUT / f"s{i}_slam_cam_pts_frame0.npy", slam_pts[0])
    np.save(OUT / f"s{i}_slam_cam_pts_lastframe.npy", slam_pts[-1])
    np.save(OUT / f"s{i}_ours_world_pts_frame0.npy", ours_pts[0])
    np.save(OUT / f"s{i}_ours_world_pts_lastframe.npy", ours_pts[-1])
    np.save(OUT / f"s{i}_slam_cam_pts_all.npy", np.stack(slam_pts))
    np.save(OUT / f"s{i}_ours_world_pts_all.npy", np.stack(ours_pts))

    print(f"  Submap {i} frame0 | SLAM cam-local |Y|: {np.linalg.norm(slam_pts[0], axis=1).mean():.4f}")
    print(f"  Submap {i} frame0 |  ours world    |Y|: {np.linalg.norm(ours_pts[0], axis=1).mean():.4f}")
    print(f"  Submap {i} last   | SLAM cam-local |Y|: {np.linalg.norm(slam_pts[-1], axis=1).mean():.4f}")
    print(f"  Submap {i} last   |  ours world    |Y|: {np.linalg.norm(ours_pts[-1], axis=1).mean():.4f}")

# ── Step 3: Scale estimation comparison (only for submap 1+) ────────────────
print("\n=== STEP 3: Scale estimation ===")
conf_threshold = 25.0  # our config default

for i in range(1, n_submaps):
    prev_sm = submaps_our[i - 1]
    curr_sm = submaps_our[i]
    prev_raw = submaps_raw[i - 1]
    curr_raw = submaps_raw[i]

    # --- SLAM's approach: camera-local points ---
    depth_prev = prev_raw["depth"].squeeze(-1) if prev_raw["depth"].ndim == 4 else prev_raw["depth"]
    depth_curr = curr_raw["depth"].squeeze(-1) if curr_raw["depth"].ndim == 4 else curr_raw["depth"]
    intr_prev = prev_raw["intrinsic"]
    intr_curr = curr_raw["intrinsic"]

    # SLAM uses frame K-1 of prev (last non-LC frame) and frame 0 of curr
    slam_t2_full = depth_to_cam_coords_points(depth_prev[-1], intr_prev[-1]).reshape(-1, 3)
    slam_t1_full = depth_to_cam_coords_points(depth_curr[0], intr_curr[0]).reshape(-1, 3)

    # SLAM confidence mask: prior_conf > percentile(conf, 25)
    conf_prev = prev_raw["depth_conf"]
    conf_curr = curr_raw["depth_conf"]
    if conf_prev is not None and conf_curr is not None:
        slam_thresh_prev = float(np.percentile(conf_prev, 25)) + 1e-6
        conf_prev_last = conf_prev[-1].reshape(-1)
        conf_curr_first = conf_curr[0].reshape(-1)
        slam_good_mask = (conf_prev_last > slam_thresh_prev) & (conf_curr_first > slam_thresh_prev)
        if slam_good_mask.sum() < 100:
            slam_good_mask = conf_prev_last > slam_thresh_prev
        if slam_good_mask.sum() < 100:
            slam_good_mask = conf_prev_last > 0
    else:
        slam_good_mask = np.ones(len(slam_t2_full), dtype=bool)

    # P_temp = inv(K_prev) @ K_curr (for same camera = I)
    K_prev_4x4 = np.tile(np.eye(4), (1, 1, 1)).squeeze()
    K_prev_4x4[:3, :3] = intr_prev[-1]
    K_curr_4x4 = np.tile(np.eye(4), (1, 1, 1)).squeeze()
    K_curr_4x4[:3, :3] = intr_curr[0]
    P_temp = np.linalg.inv(K_prev_4x4) @ K_curr_4x4  # should be I for same K

    t1_slam = (P_temp[:3, :3] @ slam_t1_full[slam_good_mask].T).T
    t2_slam = slam_t2_full[slam_good_mask]
    slam_scale, _ = estimate_scale_pairwise(t1_slam, t2_slam), None
    slam_scale = float(np.median(np.linalg.norm(t2_slam, axis=1) / np.linalg.norm(t1_slam, axis=1)))

    # --- Our approach: world-frame prev_pts ---
    O = SUBMAP_OVERLAP
    ours_curr_pts = curr_sm.world_points[:O].reshape(-1, 3).astype(np.float64)
    ours_prev_wps = prev_sm.world_points[-O:]
    P_per = ours_prev_wps.shape[1]
    prev_cam_list = []
    for oi in range(O):
        W2C = prev_sm.poses[-O + oi].astype(np.float64)
        wh = np.hstack([ours_prev_wps[oi], np.ones((P_per, 1))])
        prev_cam_list.append((W2C @ wh.T).T[:, :3])
    ours_prev_pts = np.concatenate(prev_cam_list, axis=0)

    # Our confidence mask
    if curr_sm.world_points_conf is not None and prev_sm.world_points_conf is not None:
        curr_conf = curr_sm.world_points_conf[:O].reshape(-1)
        prev_conf = prev_sm.world_points_conf[-O:].reshape(-1)
        joint_mask = (curr_conf > conf_threshold) & (prev_conf > conf_threshold)
        if joint_mask.sum() >= 100:
            ours_mask = joint_mask
        else:
            prior_only = prev_conf > conf_threshold
            ours_mask = prior_only if prior_only.sum() >= 100 else np.ones(len(curr_conf), dtype=bool)
    else:
        ours_mask = np.ones(len(ours_curr_pts), dtype=bool)

    ours_curr_in = ours_curr_pts  # rotation_only with T=I for same camera
    ours_scale = float(np.median(np.linalg.norm(ours_prev_pts[ours_mask], axis=1) /
                                  np.linalg.norm(ours_curr_in[ours_mask], axis=1)))

    print(f"\n  Boundary s{i-1}→s{i}:")
    print(f"    SLAM conf_thresh_prev (percentile-25): {slam_thresh_prev:.4f}" if conf_prev is not None else "    no conf")
    print(f"    SLAM good_mask points: {slam_good_mask.sum()} / {len(slam_good_mask)}")
    print(f"    SLAM t1 (curr cam-local) norms mean: {np.linalg.norm(t1_slam, axis=1).mean():.4f}")
    print(f"    SLAM t2 (prev cam-local) norms mean: {np.linalg.norm(t2_slam, axis=1).mean():.4f}")
    print(f"    SLAM scale: {slam_scale:.6f}")
    print(f"    Ours conf_threshold: {conf_threshold}, mask points: {ours_mask.sum()} / {len(ours_mask)}")
    print(f"    Ours curr_pts (cam-local after fix) norms mean: {np.linalg.norm(ours_curr_in[ours_mask], axis=1).mean():.4f}")
    print(f"    Ours prev_pts (cam-local via W2C)  norms mean: {np.linalg.norm(ours_prev_pts[ours_mask], axis=1).mean():.4f}")
    print(f"    Ours scale: {ours_scale:.6f}")
    print(f"    Scale diff SLAM vs ours: {abs(slam_scale - ours_scale):.6f}")

    np.savez(OUT / f"scale_boundary_{i}.npz",
             slam_t1=t1_slam, slam_t2=t2_slam, slam_scale=np.array(slam_scale),
             ours_curr=ours_curr_in, ours_prev=ours_prev_pts, ours_scale=np.array(ours_scale),
             slam_good_mask=slam_good_mask, ours_mask=ours_mask)

# ── Step 4: Run our full PGO and save H values ────────────────────────────────
print("\n=== STEP 4+5+6: PGO trace ===")

# Capture initial and final H via patched PGO
import collab_splats.pointcloud.loop_closure.closure as closure_mod
_original_pgo = closure_mod.run_pose_graph_optimization

captured = {"debug_out": [], "initial_H": {}, "final_H": {}, "poses_out": None}

def _patched_pgo(submaps, lc_submaps, total_frames, overlap_frames,
                 manifold="sl4", conf_threshold=25.0, scale_method="rotation_only",
                 debug_out=None):
    from collab_splats.pointcloud.loop_closure.graph import PoseGraph as _PG, decompose_camera as _dc
    import numpy as _np

    # Rebuild PGO manually with capture
    pg = _PG(manifold=manifold)
    global_node_id = 0
    frame_to_node = {}
    submap_node_ids = {}
    initial_H_vals = {}

    for s_idx, submap in enumerate(submaps):
        k = submap.poses.shape[0]
        node_ids_this = list(range(global_node_id, global_node_id + k))
        K_4x4 = _np.tile(_np.eye(4, dtype=_np.float64), (k, 1, 1))
        K_4x4[:, :3, :3] = submap.intrinsics.astype(_np.float64)

        if s_idx == 0:
            pg.add_node(node_ids_this[0], _np.eye(4))
            pg.add_prior(node_ids_this[0], _np.eye(4))
            initial_H_vals[node_ids_this[0]] = _np.eye(4)
            for li in range(1, k):
                H_inner = (submap.poses[li-1].astype(_np.float64)
                           @ _np.linalg.inv(submap.poses[li].astype(_np.float64)))
                prev_H = pg.get_homography(node_ids_this[li - 1])
                H_new = prev_H @ H_inner
                pg.add_node(node_ids_this[li], H_new)
                pg.add_sequential_edge(node_ids_this[li-1], node_ids_this[li], H_inner)
                initial_H_vals[node_ids_this[li]] = H_new
        else:
            prev_submap = submaps[s_idx - 1]
            prev_K_4x4 = _np.tile(_np.eye(4, dtype=_np.float64), (len(prev_submap.poses), 1, 1))
            prev_K_4x4[:, :3, :3] = prev_submap.intrinsics.astype(_np.float64)
            O2 = min(overlap_frames, k, len(prev_submap.poses))
            T = _np.linalg.inv(prev_K_4x4[-1]) @ K_4x4[0]

            scale = 1.0
            if submap.world_points is not None and prev_submap.world_points is not None and O2 > 0:
                curr_pts = submap.world_points[:O2].reshape(-1, 3).astype(_np.float64)
                prev_wps = prev_submap.world_points[-O2:]
                P_per = prev_wps.shape[1]
                prev_cam_list2 = []
                for oi in range(O2):
                    W2C = prev_submap.poses[-O2 + oi].astype(_np.float64)
                    wh = _np.hstack([prev_wps[oi], _np.ones((P_per, 1))])
                    prev_cam_list2.append((W2C @ wh.T).T[:, :3])
                prev_pts = _np.concatenate(prev_cam_list2, axis=0)
                n = curr_pts.shape[0]
                mask = _np.ones(n, dtype=bool)
                if submap.world_points_conf is not None and prev_submap.world_points_conf is not None:
                    curr_conf = submap.world_points_conf[:O2].reshape(-1)
                    prev_conf = prev_submap.world_points_conf[-O2:].reshape(-1)
                    jm = (curr_conf > conf_threshold) & (prev_conf > conf_threshold)
                    if jm.sum() >= 100:
                        mask = jm
                    else:
                        po = prev_conf > conf_threshold
                        if po.sum() >= 100:
                            mask = po
                curr_in_prev = (T[:3, :3] @ curr_pts.T).T  # rotation_only
                from collab_splats.pointcloud.loop_closure.graph import estimate_scale_pairwise as _esp
                scale = _esp(curr_in_prev[mask], prev_pts[mask])

            H_scale = _np.diag([scale, scale, scale, 1.0])
            prev_last_nid = submap_node_ids[prev_submap.submap_id][-1]
            H_overlap = pg.get_homography(prev_last_nid)
            H_w = H_overlap @ T @ H_scale

            pg.add_node(node_ids_this[0], H_w)
            H_rel_inter = _np.linalg.inv(H_overlap) @ H_w
            pg.add_sequential_edge(prev_last_nid, node_ids_this[0], H_rel_inter)
            initial_H_vals[node_ids_this[0]] = H_w

            print(f"  Boundary s{s_idx-1}→s{s_idx}: scale={scale:.6f}")
            print(f"    H_overlap diag: {_np.diag(H_overlap)}")
            print(f"    H_w diag: {_np.diag(H_w)}")
            print(f"    det(H_overlap)={_np.linalg.det(H_overlap):.6f}  det(H_w)={_np.linalg.det(H_w):.6f}")

            for li in range(1, k):
                H_inner = (submap.poses[li-1].astype(_np.float64)
                           @ _np.linalg.inv(submap.poses[li].astype(_np.float64)))
                prev_H = pg.get_homography(node_ids_this[li - 1])
                H_new = prev_H @ H_inner
                pg.add_node(node_ids_this[li], H_new)
                pg.add_sequential_edge(node_ids_this[li-1], node_ids_this[li], H_inner)
                initial_H_vals[node_ids_this[li]] = H_new

        for li, nid in enumerate(node_ids_this):
            frame_to_node[(submap.submap_id, li)] = nid
        submap_node_ids[submap.submap_id] = node_ids_this
        global_node_id += k

    # Save initial H values
    for nid, H in initial_H_vals.items():
        captured["initial_H"][nid] = H.copy()
        _np.save(OUT / f"initial_H_node{nid:03d}.npy", H)

    pg.optimize()

    # Save final H values
    all_node_ids = sorted([nid for sid in submap_node_ids.values() for nid in sid])
    for nid in all_node_ids:
        H_opt = pg.get_homography(nid)
        captured["final_H"][nid] = H_opt.copy()
        _np.save(OUT / f"final_H_node{nid:03d}.npy", H_opt)

    # Extract poses
    corrected_per_submap = {}
    all_poses_out = {}
    for submap in submaps:
        node_ids = submap_node_ids[submap.submap_id]
        k = len(node_ids)
        s_K = _np.tile(_np.eye(4, dtype=_np.float64), (k, 1, 1))
        s_K[:, :3, :3] = submap.intrinsics.astype(_np.float64)
        poses_out = _np.zeros((k, 4, 4), dtype=_np.float32)
        for li, nid in enumerate(node_ids):
            H_opt = pg.get_homography(nid)
            corrected = s_K[li] @ _np.linalg.inv(H_opt)
            _, R, t, _ = _dc(corrected)
            mat = _np.eye(4, dtype=_np.float32)
            mat[:3, :3] = R; mat[:3, 3] = t
            poses_out[li] = mat
            all_poses_out[submap.frame_start + li] = {"R": R, "t": t,
                "submap": submap.submap_id, "local_i": li}
        corrected_per_submap[submap.submap_id] = poses_out

    # Save extracted poses
    for g, info in sorted(all_poses_out.items()):
        _np.savez(OUT / f"pose_global{g:03d}.npz", R=info["R"], t=info["t"],
                  submap=_np.array(info["submap"]), local_i=_np.array(info["local_i"]))

    print(f"\n  Extracted poses saved for {len(all_poses_out)} frames")
    captured["poses_out"] = all_poses_out

    from collab_splats.pointcloud.loop_closure.closure import dedup_overlap
    return dedup_overlap(
        submap_ids=[s.submap_id for s in submaps],
        submap_starts=[s.frame_start for s in submaps],
        corrected=corrected_per_submap,
        total_frames=total_frames,
    )

closure_mod.run_pose_graph_optimization = _patched_pgo

result = _patched_pgo(
    submaps_our, [], total_frames=N,
    overlap_frames=SUBMAP_OVERLAP,
    scale_method="rotation_only",
    conf_threshold=25.0,
)

closure_mod.run_pose_graph_optimization = _original_pgo

# ── Step 7: Summary comparison ──────────────────────────────────────────────
print("\n=== STEP 7: Extracted trajectory ===")
print(f"{'Frame':>5} {'Submap':>6} {'t_x':>8} {'t_y':>8} {'t_z':>8} {'R_tr':>6}")
for g in sorted(captured["poses_out"].keys()):
    info = captured["poses_out"][g]
    t = info["t"]; R = info["R"]
    print(f"  {g:3d}   s{info['submap']}  {t[0]:8.4f} {t[1]:8.4f} {t[2]:8.4f}  {np.trace(R):.4f}")

# Save full trajectory
traj = np.array([captured["poses_out"][g]["t"] for g in sorted(captured["poses_out"].keys())])
np.save(OUT / "trajectory_ours.npy", traj)
print(f"\nAll outputs saved to {OUT}")

# ── Step 8: Compare initial H pre-PGO for overlap node ─────────────────────
print("\n=== STEP 8: H values at boundary node ===")
n_sub0 = submaps_our[0].poses.shape[0]
boundary_nid = n_sub0 - 1  # last node of submap 0
if n_submaps > 1:
    boundary_s1_nid = n_sub0  # first node of submap 1 (= boundary_nid + 1... NO)
    # Actually submap 1's first node comes after submap 0's k frames
    # With k=17 for submap 0: s1 frame0 = node 17
    s1_frame0_nid = submaps_our[0].poses.shape[0]
    print(f"  Submap 0 last node ({boundary_nid}) initial H diag: {np.diag(captured['initial_H'].get(boundary_nid, np.eye(4)))}")
    print(f"  Submap 1 frame0 node ({s1_frame0_nid}) initial H diag: {np.diag(captured['initial_H'].get(s1_frame0_nid, np.eye(4)))}")
    print(f"  det(H_s0_last_initial): {np.linalg.det(captured['initial_H'].get(boundary_nid, np.eye(4))):.6f}")
    print(f"  det(H_s1_frame0_initial): {np.linalg.det(captured['initial_H'].get(s1_frame0_nid, np.eye(4))):.6f}")
    print(f"  Submap 0 last node ({boundary_nid}) FINAL H diag: {np.diag(captured['final_H'].get(boundary_nid, np.eye(4)))}")
    print(f"  Submap 1 frame0 node ({s1_frame0_nid}) FINAL H diag: {np.diag(captured['final_H'].get(s1_frame0_nid, np.eye(4)))}")
    print(f"  det(H_s0_last_final): {np.linalg.det(captured['final_H'].get(boundary_nid, np.eye(4))):.6f}")
    print(f"  det(H_s1_frame0_final): {np.linalg.det(captured['final_H'].get(s1_frame0_nid, np.eye(4))):.6f}")
