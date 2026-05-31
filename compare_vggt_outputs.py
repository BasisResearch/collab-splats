"""Compare VGGT extrinsics: our preprocessing vs SLAM's load_and_preprocess_images.

Runs the SAME 17 frames through VGGT twice and diffs the extrinsics.
"""
import sys, numpy as np, torch
sys.path.insert(0, "/workspace/collab-splats")
sys.path.insert(0, "/workspace/collab-splats/third_party/vggt_spark")
sys.path.insert(0, "/workspace/collab-splats/third_party/VGGT-SLAM")

from pathlib import Path
KF = list(open("evals/baselines/disparity_sweep/slam_d10/selected_frames.txt").read().splitlines())
KF = [k.strip() for k in KF if k.strip()]

# Submap 0 = first 17 frames (indices 0-16)
SUB0 = KF[:17]
SUB1 = KF[16:26]   # last 10 of 26 = frames 16-25

# ── SLAM path: load_and_preprocess_images ───────────────────────────────────
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.models.vggt import VGGT
from vggt_slam.slam_utils import decompose_camera as slam_decompose

print("Loading model (SLAM path)...")
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
vggt = VGGT()
vggt.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
vggt = vggt.eval().to(dtype).to(device)
print("Model loaded")

def run_slam_path(frames):
    imgs = load_and_preprocess_images(frames).to(device)
    with torch.no_grad():
        pred = vggt(imgs.to(dtype))
    for k in pred:
        if isinstance(pred[k], torch.Tensor) and k != "target_tokens":
            pred[k] = pred[k].float().cpu().numpy().squeeze(0)
    ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
    return ext, intr, imgs.shape[-2:]

# ── Our path: creator preprocess ────────────────────────────────────────────
from collab_splats.pointcloud.feedforward.vggt_spark_creator import VGGTSPARKCreator
from pathlib import Path as P

# Reuse loaded model rather than loading twice
creator = VGGTSPARKCreator()
creator.model = vggt  # reuse same model weights

def run_our_path(frames):
    """Run our preprocessing + forward pass on given image paths."""
    from collab_splats.utils.geometry import extrinsics_to_homogeneous
    import torch
    # Use creator's preprocess but only for these frames
    from vggt.utils.load_fn import load_and_preprocess_images as lp
    # Temporarily use creator._forward with pre-loaded images
    imgs = lp(frames).to(device)  # same as SLAM path
    with torch.no_grad():
        raw = creator._forward(creator.model, imgs.to(dtype))
    ext = raw["extrinsic"]  # (k, 3, 4) W2C
    intr_k = "intrinsics" if "intrinsics" in raw else "intrinsic"
    intr = raw.get(intr_k)
    return ext, intr, imgs.shape[-2:]

print("\n=== Submap 0 (17 frames) ===")
slam_ext0, slam_intr0, slam_shape0 = run_slam_path(SUB0)
our_ext0, our_intr0, our_shape0 = run_our_path(SUB0)

print(f"SLAM image shape: {slam_shape0}  Ours: {our_shape0}")
print(f"SLAM ext shape: {slam_ext0.shape}  Ours: {our_ext0.shape}")

# Compare first few extrinsics
print("\nExtrinsic diff (SLAM vs ours) per frame:")
for i in range(min(slam_ext0.shape[0], our_ext0.shape[0])):
    diff = np.max(np.abs(slam_ext0[i] - our_ext0[i]))
    print(f"  frame {i}: max|diff|={diff:.6f}  SLAM_t={slam_ext0[i,:,3]}  ours_t={our_ext0[i,:,3]}")

np.save("/tmp/slam_ext0.npy", slam_ext0)
np.save("/tmp/ours_ext0.npy", our_ext0)

print("\n=== Submap 1 (10 frames) ===")
slam_ext1, slam_intr1, _ = run_slam_path(SUB1)
our_ext1, our_intr1, _ = run_our_path(SUB1)

print("Extrinsic diff (SLAM vs ours) per frame:")
for i in range(min(slam_ext1.shape[0], our_ext1.shape[0])):
    diff = np.max(np.abs(slam_ext1[i] - our_ext1[i]))
    print(f"  frame {i}: max|diff|={diff:.6f}  SLAM_t={slam_ext1[i,:,3]}  ours_t={our_ext1[i,:,3]}")

np.save("/tmp/slam_ext1.npy", slam_ext1)
np.save("/tmp/ours_ext1.npy", our_ext1)
