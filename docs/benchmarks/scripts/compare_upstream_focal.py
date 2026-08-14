"""Fit K the UPSTREAM way and compare its residual to ours on the same fixture.

The decomposition already showed our K is not the residual source. This closes the other
half of the question: does the ORIGINAL code's own estimator do better than ours?

Upstream LoGeR fits a focal only in its eval scripts, never in the demo/viser path
(github.com/Junyi42/LoGeR @ 7685b7a - loger/utils/viser_utils.py:445-449 hardcodes a 60 deg
FOV and comments that the missing intrinsics are "a limitation"). The eval path is
`estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")` with
`pp = (W // 2, H // 2)` (eval/relpose/launch.py:528-534, same at :721-730 and
eval/video_depth/launch.py:530-537). That function comes from dust3r, which is NOT installed
here, so upstream's own code would take its fallback `torch.full((B,), max(H, W))`
(eval/relpose/launch.py:102-105) - a constant, not a fit. Both are measured below.

The Weiszfeld body is reimplemented from dust3r's published algorithm
(github.com/naver/dust3r - dust3r/post_process.py, estimate_focal_knowing_depth): closed-form
L2 init followed by 10 IRLS iterations reweighting by inverse reprojection distance. It is a
single shared focal (fx == fy) over ALL pixels, unweighted by confidence - both differ from
ours deliberately.

Three estimators, one fixture, one metric: rebuild local_points from each K and compare to
LoGeR's own local_points. Camera frame only, so poses cannot confound it.
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

with torch.no_grad():
    preds = model(views.to("cuda")[None], **creator._forward_kwargs())
local = preds["local_points"].squeeze(0).cpu().float().numpy()
conf = torch.sigmoid(preds["conf"]).squeeze(0).cpu().float().numpy().reshape(n, h, w)
mask = conf > LOGER_CONF_THRESHOLD

uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
z = local[..., 2]
scale = float(np.percentile(np.linalg.norm(local[mask], axis=-1), 95))


def weiszfeld_focal(pts, pp_x, pp_y, iters=10):
    """dust3r's estimate_focal_knowing_depth(focal_mode='weiszfeld'), one focal for fx and fy."""
    # Pixel offsets from the principal point, and the normalised ray field xy/z.
    px = np.stack([uu - pp_x, vv - pp_y], axis=-1).reshape(-1, 2)
    xy_over_z = np.nan_to_num(pts[..., :2] / pts[..., 2:3], posinf=0.0, neginf=0.0).reshape(-1, 2)
    dot_xy_px = (xy_over_z * px).sum(-1)
    dot_xy_xy = (xy_over_z ** 2).sum(-1)
    # Closed-form L2 init, then IRLS reweighting by inverse reprojection distance.
    focal = dot_xy_px.mean() / dot_xy_xy.mean()
    for _ in range(iters):
        dist = np.linalg.norm(px - focal * xy_over_z, axis=-1)
        wgt = 1.0 / np.clip(dist, 1e-8, None)
        focal = (wgt * dot_xy_px).mean() / (wgt * dot_xy_xy).mean()
    return float(focal)


def residual(fx, fy, cx, cy):
    """Rebuild local_points from this K and report the error against LoGeR's own, as % of scale."""
    rebuilt = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], axis=-1)
    err = np.linalg.norm(rebuilt[mask] - local[mask], axis=-1)
    return np.median(err) / scale, np.percentile(err, 95) / scale, np.percentile(err, 99) / scale


ours_K = creator._forward(model, views)["intrinsics"][0]
rows = [("ours (conf-weighted median, fx!=fy, pp=(W-1)/2)",
         float(ours_K[0, 0]), float(ours_K[1, 1]), float(ours_K[0, 2]), float(ours_K[1, 2]))]

# Upstream's eval estimator: ONE focal per frame, pp = (W//2, H//2), no confidence weighting.
up_focals = [weiszfeld_focal(local[i], w // 2, h // 2) for i in range(n)]
print(f"upstream weiszfeld per-frame focal: {['%.2f' % f for f in up_focals]}")
f_up = float(np.mean(up_focals))
rows.append((f"upstream dust3r weiszfeld (fx==fy={f_up:.2f}, pp=(W//2,H//2))",
             f_up, f_up, w // 2, h // 2))

# What upstream would ACTUALLY run here: dust3r absent -> constant focal = max(H, W).
f_fb = float(max(h, w))
rows.append((f"upstream fallback, dust3r absent (f=max(H,W)={f_fb:.0f})",
             f_fb, f_fb, w // 2, h // 2))

print(f"\n{'estimator':<52} {'median':>9} {'p95':>9} {'p99':>9}")
for name, fx, fy, cx, cy in rows:
    m, p95, p99 = residual(fx, fy, cx, cy)
    print(f"{name:<52} {m:>8.4%} {p95:>8.4%} {p99:>8.4%}")

# Isolate the two deliberate differences one at a time, so a gap can be attributed.
print("\nablations (our K, one upstream choice swapped in):")
for name, fx, fy, cx, cy in [
    ("ours but pp=(W//2,H//2)", ours_K[0, 0], ours_K[1, 1], w // 2, h // 2),
    ("ours but fx==fy (mean)", (ours_K[0, 0] + ours_K[1, 1]) / 2,
     (ours_K[0, 0] + ours_K[1, 1]) / 2, ours_K[0, 2], ours_K[1, 2]),
]:
    m, p95, p99 = residual(float(fx), float(fy), float(cx), float(cy))
    print(f"  {name:<48} {m:>8.4%} {p95:>8.4%} {p99:>8.4%}")
