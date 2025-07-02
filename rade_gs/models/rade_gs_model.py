"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type, Dict, Union, List

import math
import torch 

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from gsplat.strategy import DefaultStrategy

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat  # for subclassing Nerfacto model

from gsplat.cuda._wrapper import fully_fused_projection
from rade_gs.utils.camera_utils import convert_to_colmap_camera

@dataclass
class RadegsModelConfig(SplatfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: RadegsModel)

    regularization_from_iter: int = 15000
    """Regularization starts from this iteration"""

    use_depth_normal_loss: bool = True
    """Whether to use depth normal loss"""

    # RaDeGS specific parameters
    depth_normal_lambda: float = 1.0
    """Weight for depth normal loss"""

    depth_ratio: float = 0.6
    """Ratio for depth normal loss"""


class RadegsModel(SplatfactoModel):
    """Template Model."""

    config: RadegsModelConfig

    def populate_modules(self):
        super().populate_modules()

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.
    # the following functions depths_double_to_points and depth_double_to_normal are adopted from https://github.com/hugoycj/2dgs-gaustudio/blob/main/utils/graphics_utils.py
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
        #     optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        # else:
        #     optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)

        # viewmat = get_viewmat(optimized_camera_to_world)
        # K = camera.get_intrinsics_matrices().cuda()

        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        # Get visible gaussian mask
        voxel_visible_mask = self._prefilter_voxel(camera)
        
        # camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        # Modified rasterization function from https://github.com/brian-xu/gsplat-rade/blob/main/gsplat/rendering.py
        # Enables returning depth and normal maps for computing of loss

        # Rendered contains the following:
        # - rgb: [N, 3]
        # - alphas: [N, 1]
        # - expected_depths: [N, 3]
        # - median_depths: [N, 1]
        # - expected_normals: [N, 1]
        # - meta (set to self.info)
        render, alpha, expected_depths, median_depths, expected_normals, self.info = self._render(
            camera, 
            means_crop,
            quats_crop,
            scales_crop,
            opacities_crop,
            colors_crop,
            render_mode,
            sh_degree_to_use,
            voxel_visible_mask
        )
        
        # if self.training:
        #     self.strategy.step_pre_backward(
        #         self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
        #     )
        
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)
        
        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        # Calculate depth_middepth_normal --> used for depth_normal_loss
        # Tensor shape: [2, H, W, 3]
        depth_middepth_normal = self.depth_double_to_normal(camera, expected_depths, median_depths)

        return {
            "rgb": rgb.squeeze(0),
            'depth': depth_im, # depth_im is typical depth map of rasterization
            'accumulation': alpha.squeeze(0),
            "expected_depth": expected_depths.squeeze(0), # these are the new depth maps
            "median_depth": median_depths.squeeze(0),
            "expected_normals": expected_normals.squeeze(0),
            "depth_middepth_normal": depth_middepth_normal,
            "background": background,
            "info": self.info,
        }

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        # This returns the following losses:
        # Always:
        # - loss_dict["rgb_loss"] = rgb loss (SSIM + L1)
        # During training:
        # - loss_dict["tv_loss"] = total variation loss (cameras)
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # If we want to use depth normal loss and we're past the regularization start iteration
        if self.config.use_depth_normal_loss and self.step >= self.config.regularization_from_iter:
            # Calculate normal error map based on expected normals and depth_middepth_normal
            normal_error_map = (1 - (outputs["expected_normals"].unsqueeze(0) * outputs["depth_middepth_normal"]).sum(dim=0))

            # Calculate depth_normal_loss
            depth_normal_loss = (1 - self.config.depth_ratio) * normal_error_map[0].mean() + self.config.depth_ratio * normal_error_map[1].mean()

            # Scale by lambda
            depth_normal_loss = self.config.depth_normal_lambda * depth_normal_loss

            # Add to loss dict
            loss_dict["depth_normal_loss"] = depth_normal_loss

        return loss_dict

    # def step_post_backward(self, step):
    #     assert step == self.step
    #     if isinstance(self.strategy, DefaultStrategy):
    #         self.strategy.step_post_backward(
    #             params=self.gauss_params,
    #             optimizers=self.optimizers,
    #             state=self.strategy_state,
    #             step=self.step,
    #             info=self.info,
    #             packed=False,
    #         )
    #     elif isinstance(self.strategy, MCMCStrategy):
    #         self.strategy.step_post_backward(
    #             params=self.gauss_params,
    #             optimizers=self.optimizers,
    #             state=self.strategy_state,
    #             step=step,
    #             info=self.info,
    #             lr=self.schedulers["means"].get_last_lr()[0],  # the learning rate for the "means" attribute of the GS
    #         )
    #     elif isinstance(self.strategy, RaDeGSStrategy):
    #         pass
    #     else:
    #         raise ValueError(f"Unknown strategy {self.strategy}")

    def _prefilter_voxel(self, camera: Cameras):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!

        Taken from https://github.com/brian-xu/scaffold-gs-nerfstudio
        """

        colmap_camera = convert_to_colmap_camera(camera)

        means = self.means
        scales = torch.exp(self.scales)
        quats = self.quats

        # Set up rasterization configuration
        tanfovx = math.tan(colmap_camera.FoVx * 0.5)
        tanfovy = math.tan(colmap_camera.FoVy * 0.5)
        focal_length_x = colmap_camera.image_width / (2 * tanfovx)
        focal_length_y = colmap_camera.image_height / (2 * tanfovy)

        Ks = torch.tensor(
            [
                [focal_length_x, 0, colmap_camera.image_width / 2.0],
                [0, focal_length_y, colmap_camera.image_height / 2.0],
                [0, 0, 1],
            ],
            device=self.device,
        )[None]
        viewmats = colmap_camera.world_view_transform.transpose(0, 1)[None]

        N = means.shape[0]
        C = viewmats.shape[0]
        
        assert means.shape == (N, 3), means.shape
        assert quats.shape == (N, 4), quats.shape
        assert scales.shape == (N, 3), scales.shape
        assert viewmats.shape == (C, 4, 4), viewmats.shape
        assert Ks.shape == (C, 3, 3), Ks.shape

        # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
        proj_results = fully_fused_projection(
            means,
            None,  # covars,
            quats,
            scales,
            viewmats,
            Ks,
            int(colmap_camera.image_width),
            int(colmap_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
            calc_compensations=False,
        )

        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations, ray_ts, ray_planes, normals = (
            proj_results
        )
        camera_ids, gaussian_ids = None, None

        return torch.sum(radii, dim=-1).squeeze() > 0

    def _render(
        self,
        camera: Cameras,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        color: torch.Tensor,
        render_mode: str,
        sh_degree_to_use: int,
        visible_mask: torch.Tensor,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        colmap_camera = convert_to_colmap_camera(camera)
        background = self._get_background_color()

        if visible_mask is not None:
            means = means[visible_mask]
            quats = quats[visible_mask]
            scales = scales[visible_mask]
            opacities = opacities[visible_mask]
            color = color[visible_mask]
        else:
            means = means
            quats = quats
            scales = scales
            opacities = opacities
            color = color

        # Set up rasterization configuration
        tanfovx = math.tan(colmap_camera.FoVx * 0.5)
        tanfovy = math.tan(colmap_camera.FoVy * 0.5)

        focal_length_x = colmap_camera.image_width / (2 * tanfovx)
        focal_length_y = colmap_camera.image_height / (2 * tanfovy)

        K = torch.tensor(
            [
                [focal_length_x, 0, colmap_camera.image_width / 2.0],
                [0, focal_length_y, colmap_camera.image_height / 2.0],
                [0, 0, 1],
            ],
            device="cuda",
        )

        viewmat = colmap_camera.world_view_transform.transpose(0, 1)  # [4, 4]
        
        # Items are:
        # - render_colors: [N, 3]
        # - render_alphas: [N, 1]
        # - expected_depths: [N, 3]
        # - median_depths: [N, 1]
        # - expected_normals: [N, 1]
        # - info: dict
        render, alpha, expected_depths, median_depths, expected_normals, meta = rasterization(
            means=means,  # [N, 3]
            quats=quats,  # [N, 4]
            scales=torch.exp(scales),  # [N, 3]
            opacities=torch.sigmoid(opacities.squeeze(-1)),  # [N,]
            colors=color,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=background[None],
            width=int(colmap_camera.image_width),
            height=int(colmap_camera.image_height),
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            radius_clip=3.0,
            # Output depth and normal maps
            return_depth_normal=True,
        )

        return render, alpha, expected_depths, median_depths, expected_normals, meta