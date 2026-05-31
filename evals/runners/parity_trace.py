"""Stage-by-stage parity trace: our vggt_spark LC pipeline vs VGGT-SLAM.

Runs each pipeline as REAL code with monkey-patched dump hooks (no source edits,
no re-derived math), writes per-stage numpy arrays, then diffs them and prints a
PASS/DIVERGE report per stage, stopping at the first divergence.

Memory-safe: one model at a time. Run the three sides sequentially:

    python evals/runners/parity_trace.py --side slam --min_disparity 50
    python evals/runners/parity_trace.py --side ours --min_disparity 50
    python evals/runners/parity_trace.py --side diff --min_disparity 50

Stages (single-submap levels d50/d30/d20 exercise 1-5; d10 adds 6-8):
  1 preprocess image tensor      4 pose extraction (R,t / TUM)
  2 VGGT forward (ext/intr/depth/conf)  5 ATE vs 7-Scenes GT
  3 point cloud                  6-7 boundary scale + PGO (debug_out)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "VGGT-SLAM"))
sys.path.insert(0, str(REPO / "third_party" / "vggt_spark"))

DEFAULT_SEQ = REPO / "evals/data/7scenes/chess/chess/seq-01"
SWEEP_DIR = REPO / "evals/baselines/disparity_sweep"
SUBMAP_SIZE = 16
SUBMAP_OVERLAP = 1
CONF_THRESHOLD = 25.0


########################################################################
########## Helpers #####################################################
########################################################################


def _out_dir(level: int, side: str) -> Path:
    """Per-level, per-side dump directory under /tmp."""
    d = Path(f"/tmp/parity_d{level}/{side}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _frame_id(path: str) -> int:
    """Parse the integer frame number from a 7-Scenes filename (frame-000051.color.png → 51)."""
    import re

    m = re.search(r"(\d+)", Path(path).name)
    return int(m.group(1)) if m else -1


def _selected_frames(level: int) -> list[str]:
    """Read the level's selected_frames.txt (the shared keyframe list)."""
    kf = SWEEP_DIR / f"slam_d{level}" / "selected_frames.txt"
    return [l.strip() for l in kf.read_text().splitlines() if l.strip()]


def _dump(out: Path, name: str, arr) -> None:
    """Save a numpy array (or scalar) to out/name.npy."""
    np.save(out / f"{name}.npy", np.asarray(arr))


########################################################################
########## SLAM side ###################################################
########################################################################


def run_slam(level: int, seq_dir: Path) -> None:
    """Run the real VGGT-SLAM pipeline with dump hooks on solver internals."""
    out = _out_dir(level, "slam")
    # Load the runner by file path — avoids name collision with the installed `evals` package.
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "run_vggt_slam_lc", str(REPO / "evals/runners/run_vggt_slam_lc.py")
    )
    runner = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(runner)
    from vggt_slam.solver import Solver

    state = {"submap": 0}

    # Patch the preprocessing fn in the solver's namespace → dump stage-1 tensor.
    _orig_load = runner.load_and_preprocess_images if hasattr(runner, "load_and_preprocess_images") else None
    import vggt_slam.solver as solver_mod

    _orig_pp = solver_mod.load_and_preprocess_images

    def _pp(image_names):
        imgs = _orig_pp(image_names)
        si = state["submap"]
        _dump(out, f"stage1_preprocess_s{si}", imgs.detach().cpu().float().numpy())
        with open(out / f"stage1_frames_s{si}.json", "w") as f:
            json.dump([str(p) for p in image_names], f)
        return imgs

    solver_mod.load_and_preprocess_images = _pp

    # Patch run_predictions → dump stage-2 forward outputs.
    _orig_rp = Solver.run_predictions

    def _rp(self, image_names, model, max_loops, clip_model, clip_preprocess):
        pred = _orig_rp(self, image_names, model, max_loops, clip_model, clip_preprocess)
        si = state["submap"]
        for key in ("extrinsic", "intrinsic", "depth", "depth_conf"):
            if pred.get(key) is not None:
                v = pred[key]
                v = v.detach().cpu().float().numpy() if hasattr(v, "detach") else np.asarray(v)
                _dump(out, f"stage2_{key}_s{si}", v)
        _dump(out, f"stage2_frameids_s{si}", [_frame_id(p) for p in image_names])
        state["submap"] += 1
        return pred

    Solver.run_predictions = _rp

    # Patch Map.write_poses_to_file → dump per-node final homography (post-optimize),
    # normalized by H[3,3], for node-by-node comparison against our graph.
    from vggt_slam.map import GraphMap as _Map

    _orig_wpf = _Map.write_poses_to_file

    def _wpf(self, file_name, graph, give_camera_mat=False, kitti_format=False):
        Hs, ids = [], []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            for i in range(len(submap.get_all_poses())):
                nid = submap.get_id() + i
                H = np.asarray(graph.get_homography(nid), dtype=np.float64)
                Hs.append(H / H[-1, -1])
                ids.append(int(nid))
        _dump(out, "stageH_slam_nodes", np.stack(Hs))
        _dump(out, "stageH_slam_ids", np.array(ids))
        return _orig_wpf(self, file_name, graph, give_camera_mat, kitti_format)

    _Map.write_poses_to_file = _wpf

    out_tum = out / "traj.tum"
    runner.run_vggt_slam_lc(
        seq_dir=seq_dir,
        out_tum=out_tum,
        max_frames=200,
        submap_size=SUBMAP_SIZE,
        overlapping_window_size=SUBMAP_OVERLAP,
        conf_threshold=CONF_THRESHOLD,
        max_loops=0,
        min_disparity=float(level),
    )

    # Restore
    solver_mod.load_and_preprocess_images = _orig_pp
    Solver.run_predictions = _orig_rp
    _Map.write_poses_to_file = _orig_wpf

    # Stage 4/5: parse final TUM (camera-to-world), dedup duplicate frame ids (first wins).
    _dump_tum(out, out_tum)
    print(f"[slam d{level}] dumped {state['submap']} submaps → {out}")


def _dump_tum(out: Path, tum: Path) -> None:
    """Parse a TUM file → dump (frame_id, tx,ty,tz, qx,qy,qz,qw), first-occurrence dedup."""
    rows, seen = [], set()
    dup = 0
    for line in tum.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        vals = [float(x) for x in line.split()]
        fid = int(vals[0])
        if fid in seen:
            dup += 1
            continue
        seen.add(fid)
        rows.append(vals)
    arr = np.array(sorted(rows, key=lambda r: r[0]))
    _dump(out, "stage4_tum", arr)
    _dump(out, "stage4_dupcount", dup)


########################################################################
########## Ours side ###################################################
########################################################################


def run_ours(level: int, seq_dir: Path, scale_method: str = "rotation_only") -> None:
    """Run the real LoopClosure(VGGTSPARKCreator) pipeline with dump hooks."""
    out = _out_dir(level, "ours")
    import collab_splats.pointcloud.wrappers as wrappers_mod
    from collab_splats.pointcloud.feedforward.vggt_spark_creator import VGGTSPARKCreator
    from collab_splats.pointcloud.loop_closure import LoopClosureConfig
    from collab_splats.pointcloud.wrappers import LoopClosure

    frames = _selected_frames(level)
    # Symlink keyframes into a temp dir (eval_gt convention) so creator processes exactly these.
    tmp = Path(tempfile.mkdtemp(prefix=f"parity_d{level}_"))
    for i, src in enumerate(frames):
        (tmp / f"{i:06d}.png").symlink_to(Path(src).resolve())
    print(f"[ours d{level}] {len(frames)} keyframes → {tmp}")

    creator = VGGTSPARKCreator()  # lc.run() loads the model itself
    cfg = LoopClosureConfig(
        submap_size=SUBMAP_SIZE,
        submap_overlap=SUBMAP_OVERLAP,
        conf_threshold=CONF_THRESHOLD,
        scale_method=scale_method,
    )

    state = {"submap": 0}

    # Patch creator._forward → dump stage-1 (input views) + stage-2 (raw outputs).
    _orig_forward = creator._forward

    def _forward(model, views, **kw):
        si = state["submap"]
        v = views.detach().cpu().float().numpy() if hasattr(views, "detach") else np.asarray(views)
        _dump(out, f"stage1_preprocess_s{si}", v)
        raw = _orig_forward(model, views, **kw)
        for key in ("extrinsic", "intrinsics", "depth", "depth_conf"):
            if raw.get(key) is not None:
                _dump(out, f"stage2_{key}_s{si}", np.asarray(raw[key]))
        state["submap"] += 1
        return raw

    creator._forward = _forward

    # Patch _raw_to_world_points → dump stage-3 (world points), keyed by call order.
    _orig_wp = wrappers_mod._raw_to_world_points
    wp_state = {"i": 0}

    def _wp(raw, subsample: int = 8):
        wp, conf = _orig_wp(raw, subsample=subsample)
        si = wp_state["i"]
        if wp is not None:
            _dump(out, f"stage3_world_points_s{si}", np.asarray(wp))
        wp_state["i"] += 1
        return wp, conf

    wrappers_mod._raw_to_world_points = _wp

    # Patch run_pose_graph_optimization → capture boundary scale/H_w via debug_out (stage 6-7).
    _orig_pgo = wrappers_mod.run_pose_graph_optimization

    def _pgo(*a, **kw):
        dbg: list = []
        kw["debug_out"] = dbg
        res = _orig_pgo(*a, **kw)
        for j, entry in enumerate(dbg):
            for k2, v2 in entry.items():
                if v2 is not None:
                    _dump(out, f"stage6_boundary{j}_{k2}", np.asarray(v2))
        return res

    wrappers_mod.run_pose_graph_optimization = _pgo

    # Run real pipeline.
    lc = LoopClosure(creator, cfg)
    result = lc.run(tmp)

    # Restore
    creator._forward = _orig_forward
    wrappers_mod._raw_to_world_points = _orig_wp
    wrappers_mod.run_pose_graph_optimization = _orig_pgo

    # Stage 4: final extrinsics (world-to-cam, N×4×4) for trajectory/ATE compare.
    poses = np.asarray(result.extrinsics)
    _dump(out, "stage4_poses_w2c", poses)
    _dump(out, "stage4_frameids", [_frame_id(p) for p in frames])
    print(f"[ours d{level}] dumped {state['submap']} submaps, {len(poses)} poses → {out}")


########################################################################
########## Diff ########################################################
########################################################################


def _load(out: Path, name: str):
    p = out / f"{name}.npy"
    return np.load(p) if p.exists() else None


def diff(level: int) -> None:
    """Diff slam vs ours dumps, print PASS/DIVERGE per stage, write diff_report.json."""
    slam = _out_dir(level, "slam")
    ours = _out_dir(level, "ours")
    report: list[dict] = []
    diverged_at = None

    def row(stage, status, maxd, tol, note=""):
        nonlocal diverged_at
        report.append({"stage": stage, "status": status, "max_abs_delta": maxd, "tol": tol, "note": note})
        flag = "PASS" if status == "PASS" else "DIVERGE"
        md = f"{maxd:.3e}" if maxd is not None else "  n/a "
        print(f"  stage {stage:<22} {flag:8} max|Δ|={md}  tol={tol:<8} {note}")
        if status != "PASS" and diverged_at is None:
            diverged_at = stage

    print(f"\n=== Parity diff d{level} ===")

    # Stage 1: preprocess (submap 0 only; identical fn/mode expected → tol 0).
    s_pp, o_pp = _load(slam, "stage1_preprocess_s0"), _load(ours, "stage1_preprocess_s0")
    if s_pp is not None and o_pp is not None:
        n = min(s_pp.shape[0], o_pp.shape[0])
        d = float(np.max(np.abs(s_pp[:n] - o_pp[:n]))) if s_pp.shape[1:] == o_pp.shape[1:] else None
        row("1 preprocess", "PASS" if (d is not None and d <= 1e-4) else "DIVERGE", d, "1e-4",
            f"shapes slam={s_pp.shape} ours={o_pp.shape}")
    else:
        row("1 preprocess", "DIVERGE", None, "1e-4", "missing dump")

    # Stage 2: forward outputs (bf16 tol). intrinsic key differs (slam 'intrinsic' vs ours 'intrinsics').
    for skey, okey, tol in [("extrinsic", "extrinsic", 1e-2), ("intrinsic", "intrinsics", 1e-1),
                            ("depth", "depth", 5e-2), ("depth_conf", "depth_conf", 5e-2)]:
        s, o = _load(slam, f"stage2_{skey}_s0"), _load(ours, f"stage2_{okey}_s0")
        if s is None or o is None:
            row(f"2 {skey}", "DIVERGE", None, str(tol), "missing dump")
            continue
        s2, o2 = np.squeeze(s), np.squeeze(o)
        n = min(s2.shape[0], o2.shape[0])
        if s2.shape[1:] != o2.shape[1:]:
            row(f"2 {skey}", "DIVERGE", None, str(tol), f"shape slam={s2.shape} ours={o2.shape}")
            continue
        d = float(np.max(np.abs(s2[:n] - o2[:n])))
        row(f"2 {skey}", "PASS" if d <= tol else "DIVERGE", d, str(tol), f"shape={s2.shape}")

    # Stage 4: final trajectory translations (align by frame id, ATE-style not yet — raw compare).
    s_tum = _load(slam, "stage4_tum")
    o_poses = _load(ours, "stage4_poses_w2c")
    o_fids = _load(ours, "stage4_frameids")
    dup = _load(slam, "stage4_dupcount")
    note = f"slam_dup_frames={int(dup)}" if dup is not None else ""
    if s_tum is not None and o_poses is not None and o_fids is not None:
        # ours w2c → camera position in world: -R^T t
        R = o_poses[:, :3, :3]
        t = o_poses[:, :3, 3]
        o_pos = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)
        # slam TUM tx,ty,tz already camera-to-world translation
        s_map = {int(r[0]): r[1:4] for r in s_tum}
        common = [i for i, f in enumerate(o_fids) if int(f) in s_map]
        if common:
            o_sel = o_pos[common]
            s_sel = np.array([s_map[int(o_fids[i])] for i in common])
            d = float(np.max(np.abs(o_sel - s_sel)))
            # Tolerance 5e-3: SLAM stores homographies in bf16 (1/256 quantized);
            # over a multi-submap chain that accumulates ~1.5mm vs our float64.
            tol = "5e-3"
            row("4 trajectory(raw)", "PASS" if d <= 5e-3 else "DIVERGE", d, tol,
                f"{len(common)} common frames; {note} (pre-align)")
        else:
            row("4 trajectory(raw)", "DIVERGE", None, "1e-3", "no common frames")
    else:
        row("4 trajectory(raw)", "DIVERGE", None, "1e-3", "missing dump")

    (Path(f"/tmp/parity_d{level}") / "diff_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n-> first divergence: {diverged_at or 'none'}")
    print(f"-> report: /tmp/parity_d{level}/diff_report.json")


########################################################################
########## CLI #########################################################
########################################################################


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", required=True, choices=["slam", "ours", "diff"])
    ap.add_argument("--min_disparity", type=int, required=True, help="disparity level (50/30/20/10)")
    ap.add_argument("--seq_dir", type=Path, default=DEFAULT_SEQ)
    ap.add_argument("--scale_method", default="rotation_only")
    args = ap.parse_args()

    if args.side == "slam":
        run_slam(args.min_disparity, args.seq_dir)
    elif args.side == "ours":
        run_ours(args.min_disparity, args.seq_dir, scale_method=args.scale_method)
    else:
        diff(args.min_disparity)


if __name__ == "__main__":
    main()
