# import torch
# import numpy as np
# from PIL import Image

# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Optional, Type, Literal, List, Dict

# from nerfstudio.data.utils.dataloaders import ImageBatchStream, _undistort_image
# from nerfstudio.cameras.cameras import Cameras
# from nerfstudio.data.datamanagers.full_images_datamanager import (
#     FullImageDatamanager,
#     FullImageDatamanagerConfig,
# )
# from nerfstudio.utils.rich_utils import CONSOLE

# # Local imports
# from collab_splats.utils.segmentation import (
#     Segmentation,
#     create_composite_mask,
#     create_patch_mask,
#     convert_matched_mask,
#     mask_id_to_binary_mask,
# )

# @dataclass
# class GroupingDataManagerConfig(FullImageDatamanagerConfig):
#     """Configuration for the GroupingDataManager."""
#     _target: Type = field(default_factory=lambda: GroupingDataManager)

#     """Directory containing raw masks."""
#     associated_mask_dir: Optional[Path] = None

# @dataclass
# class GroupingDataManager(FullImageDatamanager):
#     config: GroupingDataManagerConfig

#     def __init__(self, config: GroupingDataManagerConfig, *args, **kwargs):
#         super().__init__(config, *args, **kwargs)

#         # Set up associated mask directory
#         if self.config.associated_mask_dir is None:
#             self.associated_mask_dir = self.config.dataparser.data / "masks" / "associated"
#             self.associated_mask_dir.mkdir(parents=True, exist_ok=True)
#         else:
#             self.associated_mask_dir = self.config.associated_mask_dir

#     def _load_images(
#         self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]
#     ) -> List[Dict[str, torch.Tensor]]:

#         if split == "train":
#             dataset = self.train_dataset
#         elif split == "eval":
#             dataset = self.eval_dataset
#         else:
#             assert_never(split)

#         def undistort_idx(idx: int) -> Dict[str, torch.Tensor]:
#             data = dataset.get_data(idx, image_type=self.config.cache_images_type)
#             camera = dataset.cameras[idx].reshape(())
#             assert data["image"].shape[1] == camera.width.item() and data["image"].shape[0] == camera.height.item(), (
#                 f"The size of image ({data['image'].shape[1]}, {data['image'].shape[0]}) loaded "
#                 f"does not match the camera parameters ({camera.width.item(), camera.height.item()})"
#             )

#             # Undistort RGB if needed
#             if camera.distortion_params is not None and not torch.all(camera.distortion_params == 0):
#                 K = camera.get_intrinsics_matrices().numpy()
#                 distortion_params = camera.distortion_params.numpy()
#                 image = data["image"].numpy()

#                 K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
#                 data["image"] = torch.from_numpy(image)
#                 if mask is not None:
#                     data["mask"] = mask

#                 dataset.cameras.fx[idx] = float(K[0, 0])
#                 dataset.cameras.fy[idx] = float(K[1, 1])
#                 dataset.cameras.cx[idx] = float(K[0, 2])
#                 dataset.cameras.cy[idx] = float(K[1, 2])
#                 dataset.cameras.width[idx] = image.shape[1]
#                 dataset.cameras.height[idx] = image.shape[0]

#             # Load segmentation if available
#             if self.config.associated_mask_dir is not None:
#                 seg_path = self.config.associated_mask_dir / dataset.image_filenames[idx].name
#                 if seg_path.exists():
#                     seg = Image.open(seg_path).convert("L")
#                     seg = np.array(seg, dtype=np.int64)

#                     # Apply undistortion to segmentation if RGB was undistorted
#                     if camera.distortion_params is not None and not torch.all(camera.distortion_params == 0):
#                         _, seg, _ = _undistort_image(camera, distortion_params, {"image": seg}, seg, K)

#                     data["segmentation"] = torch.from_numpy(seg)
#                 else:
#                     data["segmentation"] = None

#             return data

#         CONSOLE.log(f"Caching / undistorting {split} images (with associated masks)")
#         with ThreadPoolExecutor(max_workers=2) as executor:
#             undistorted_images = list(
#                 track(
#                     executor.map(
#                         undistort_idx,
#                         range(len(dataset)),
#                     ),
#                     description=f"Caching / undistorting {split} images",
#                     transient=True,
#                     total=len(dataset),
#                 )
#             )

#         # Move to device
#         for cache in undistorted_images:
#             cache["image"] = cache["image"].to(self.device) if cache_images_device == "gpu" else cache["image"].pin_memory()
#             if "mask" in cache and cache["mask"] is not None:
#                 cache["mask"] = cache["mask"].to(self.device) if cache_images_device == "gpu" else cache["mask"].pin_memory()
#             if "depth" in cache and cache["depth"] is not None:
#                 cache["depth"] = cache["depth"].to(self.device) if cache_images_device == "gpu" else cache["depth"].pin_memory()
#             if "segmentation" in cache and cache["segmentation"] is not None:
#                 cache["segmentation"] = cache["segmentation"].to(self.device) if cache_images_device == "gpu" else cache["segmentation"].pin_memory()

#         # Update cached cameras
#         if split == "train":
#             self.train_cameras = dataset.cameras.to(self.device)
#         else:
#             self.eval_cameras = dataset.cameras.to(self.device)

#         return undistorted_images