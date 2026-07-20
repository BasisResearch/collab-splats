"""Sidecar script: convert a Waymo Open Dataset .tfrecord to the flat layout
that ``evals/datasets.py:_load_waymo`` expects.

Run this in a Python env where the ``waymo-open-dataset`` pip wheel is
installed (it pins TensorFlow and conflicts with our torch/CUDA stack — keep
it isolated). Example::

    conda create -n waymo-export python=3.10 pip
    conda activate waymo-export
    pip install waymo-open-dataset-tf-2-12-0

    python evals/data/extract_waymo.py \\
        --tfrecord /data/waymo/segment-1234.tfrecord \\
        --output    evals/data/waymo/segment-1234 \\
        --camera    FRONT

Produces::

    <output>/images/000000.png, 000001.png, ...   (decoded camera frames)
    <output>/groundtruth.txt                       (TUM-format camera c2w trajectory)

The TUM trajectory is the camera-to-world transform: vehicle-to-world from
``frame.pose.transform`` composed with the camera-to-vehicle extrinsic from
``frame.context.camera_calibrations``.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

_CAMERA_NAMES = {
    "FRONT": 1,
    "FRONT_LEFT": 2,
    "FRONT_RIGHT": 3,
    "SIDE_LEFT": 4,
    "SIDE_RIGHT": 5,
}


def _require_waymo():
    try:
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2
    except ImportError as e:
        raise SystemExit(
            "extract_waymo.py needs waymo-open-dataset + tensorflow installed. "
            "Run this script in a sidecar env (see module docstring)."
        ) from e
    return tf, dataset_pb2


def _quat_xyzw_from_matrix(R_mat: np.ndarray) -> tuple[float, float, float, float]:
    from scipy.spatial.transform import Rotation

    return tuple(Rotation.from_matrix(R_mat).as_quat())  # [x, y, z, w]


def extract_segment(
    tfrecord_path: Path,
    output_dir: Path,
    camera: str = "FRONT",
) -> int:
    tf, dataset_pb2 = _require_waymo()

    cam_id = _CAMERA_NAMES[camera]
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    gt_path = output_dir / "groundtruth.txt"

    from PIL import Image

    cam_to_vehicle: np.ndarray | None = None
    n = 0
    with gt_path.open("w") as gt_f:
        ds = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
        for raw in ds:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(raw.numpy()))

            if cam_to_vehicle is None:
                for calib in frame.context.camera_calibrations:
                    if calib.name == cam_id:
                        cam_to_vehicle = np.array(calib.extrinsic.transform).reshape(4, 4)
                        break
                if cam_to_vehicle is None:
                    raise RuntimeError(f"Camera {camera} not found in tfrecord calibrations")

            image_proto = next(im for im in frame.images if im.name == cam_id)
            img = Image.open(io.BytesIO(image_proto.image))
            img.save(images_dir / f"{n:06d}.png")

            T_v2w = np.array(frame.pose.transform).reshape(4, 4)
            T_c2w = T_v2w @ cam_to_vehicle
            t = T_c2w[:3, 3]
            qx, qy, qz, qw = _quat_xyzw_from_matrix(T_c2w[:3, :3])
            ts = frame.timestamp_micros / 1e6
            gt_f.write(f"{ts:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} " f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tfrecord", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--camera", default="FRONT", choices=sorted(_CAMERA_NAMES))
    args = ap.parse_args()
    n = extract_segment(args.tfrecord, args.output, args.camera)
    print(f"Extracted {n} frames to {args.output}/")


if __name__ == "__main__":
    main()
