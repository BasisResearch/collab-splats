"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type, Dict, Union, List, Optional

import torch 
from torch.nn import functional as F
from torch.nn import Parameter

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from gsplat.strategy import DefaultStrategy
from gsplat.cuda._wrapper import spherical_harmonics

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox

from collab_splats.utils import depth_double_to_normal
from collab_splats.models.rade_gs_model import RadegsModelConfig, RadegsModel
from collab_splats.utils.features import TwoLayerMLP, BaseFeatureExtractor

@dataclass
class RadegsFeaturesModelConfig(RadegsModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: RadegsFeaturesModel)

    features_loss_lambda: float = 1e-3
    """Lambda for the features loss."""

    features_regularization_lambda: float = 0.1
    """Lambda for the features regularization loss (computed for non-main features)."""

    # TODO(roger): this feat_dim has to add up depth/color to a number that can be rasterized without padding
    # https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/_wrapper.py#L431
    # gsplat's N-D implementation seems to have some bugs that cause padded tensors to have memory issues

    # So this problem has to do with dim_sh (spherical harmonics degree) + features_latent_dim being evenly divisible 

    # From gsplat docs:
    # **Support N-D Features**: If `sh_degree` is None,
    # the `colors` is expected to be with shape [..., N, D] or [..., C, N, D], in which D is the channel of
    # the features to be rendered. The computation is slow when D > 32 at the moment.
    # If `sh_degree` is set, the `colors` is expected to be the SH coefficients with
    # shape [..., N, K, 3] or [..., C, N, K, 3], where K is the number of SH bases. In this case, it is expected
    # that :math:`(\\textit{sh_degree} + 1) ^ 2 \\leq K`, where `sh_degree` controls the
    # activated bases in the SH coefficients.
    #
    # TLB we're gonna start with 13 (so that it adds to 16 SH coefficients, but lets see if we can reduce)
    features_latent_dim: int = 13 
    """Latent dimensionality of the learned feature space."""

    mlp_hidden_dim: int = 64
    """Size of the hidden layer in the decoder MLP."""

    #### Text query parameters ####
    positive_queries: Optional[List[str]] = field(default_factory=lambda: [""])
    """Positive text queries."""

    negative_queries: Optional[List[str]] = field(default_factory=lambda: ["object"])
    """Negative text queries. Default is 'object'."""

    similarity_method: str = "pairwise"
    """Method to use for computing similarity."""

class RadegsFeaturesModel(RadegsModel):
    """Template Model."""

    config: RadegsFeaturesModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Moving this from populate models to here because we need 
        # the device initialized before we can populate text encoder
        
        # Initialize per-Gaussian features
        distill_features = torch.nn.Parameter(torch.zeros((self.means.shape[0], self.config.features_latent_dim)))
        self.gauss_params["distill_features"] = distill_features

        # Store information about the main features
        self.main_features_name = self.kwargs["metadata"]["feature_type"]
        self.main_features_dims = self.kwargs["metadata"]["feature_dims"][self.main_features_name] # C, H, W

        # Initialize the decoder MLP
        self.decoder = TwoLayerMLP(
            input_dim=self.config.features_latent_dim, # Input are the latents
            hidden_dim=self.config.mlp_hidden_dim, # Size of the hidden layer
            features_dim_dict=self.kwargs["metadata"]["feature_dims"] # Size of each feature branch
        )

        # Populate text encoder and set default text queries
        self.populate_text_encoder()
        self.set_text_queries(self.config.positive_queries, self.config.negative_queries)

    ########################################################
    ############## Feature Properties #####################
    ########################################################

    def populate_text_encoder(self):
        # Populate text encoder if it's a CLIP model
        if "clip" in self.kwargs["metadata"]["feature_type"].lower():
            self.text_encoder = BaseFeatureExtractor.get(self.kwargs["metadata"]["feature_type"])(device=self.device)

            # Register it as a submodule and turn off the gradients -->
            # This handles moving to the correct device
            self.add_module("text_encoder", self.text_encoder)

            for param in self.text_encoder.parameters():
                param.requires_grad = False

            # Track the similarity function
            self.similarity_fx = self.text_encoder.compute_similarity
        else:
            self.similarity_fx = None

    @property
    def distill_features(self):
        return self.gauss_params["distill_features"]

    # @property
    # def similarity(self):
    #     # Grab the features and pass them through the decoder
    #     features = self.distill_features
    #     features = self.decoder.per_gaussian_forward(features)


    def decode_features(self, features: torch.Tensor, resize_factor: float = 1.0) -> torch.Tensor:
        """
        Decode features from latent space back to model dimensionality.

        Inputs:
            - features: [H, W, C]
        """
        # Put channels first for input to decoder
        features = features.permute(2, 0, 1)

        # Reshape and upsample for rendering
        features_shape = (int(self.main_features_dims[1] * resize_factor), int(self.main_features_dims[2] * resize_factor))
        rendered_features = F.interpolate(features.unsqueeze(0), size=features_shape, mode="bilinear", align_corners=False)

        # Decode the features --> one branch per model type (i.e., decoding both from same embedding)
        rendered_features_dict = self.decoder(rendered_features)

        # Interpolate the rest of the features
        for model_name, features_dim in self.kwargs["metadata"]["feature_dims"].items():
            if model_name != self.main_features_name:
                rendered_features_dict[model_name] = F.interpolate(rendered_features_dict[model_name], size=features_dim[1:], mode="bilinear", align_corners=False)
            rendered_features_dict[model_name] = rendered_features_dict[model_name].squeeze(0)

        return rendered_features_dict

    ########################################################
    ############## Typical functions #######################
    ########################################################

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
            distill_features_crop = self.distill_features[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            distill_features_crop = self.distill_features

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)

        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        # Get camera parameters of colmap camera for rasterization
        camera_params = self._get_camera_parameters(camera)

        # Get visible gaussian mask
        if self.config.prefilter_voxel:
            voxel_visible_mask = self._prefilter_voxel(
                camera_params=camera_params,
            )
        else:
            voxel_visible_mask = None

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
            means=means_crop,
            quats=quats_crop,
            scales=scales_crop,
            opacities=opacities_crop,
            colors=colors_crop,
            features=distill_features_crop,
            render_mode=render_mode,
            sh_degree_to_use=sh_degree_to_use,
            visible_mask=voxel_visible_mask,
            camera_params=camera_params,
        )

        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )

        # Calculate depth_middepth_normal --> used for depth_normal_loss
        # Tensor shape: [2, H, W, 3]
        if self.config.use_depth_normal_loss and self.step >= self.config.regularization_from_iter:
            depth_middepth_normal = depth_double_to_normal(camera, expected_depths, median_depths)

            # Sum over channels (keep views) then take the dot product with the normal map
            # results in an  angular error map per view (depth and middept)
            normal_error_map = 1 - (expected_normals.unsqueeze(0) * depth_middepth_normal).sum(dim=-1).squeeze(0)
        else:
            # Create zero tensor with shape [2, H, W] to match depth_middepth_normal structure
            normal_error_map = torch.zeros(2, *expected_normals.shape[:2], device=expected_normals.device)

        normals = (expected_normals + 1) / 2 # Convert normals to 0-1 range
        
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore    

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

        features = render[:, ..., 3:3 + self.config.features_latent_dim]

        # Threshold out by alpha values
        expected_depths = torch.where(alpha > 0, expected_depths, expected_depths.detach().max())
        median_depths = torch.where(alpha > 0, median_depths, median_depths.detach().max())
        normals = torch.where(alpha > 0, normals, normals.detach().max())
        
        return {
            "rgb": rgb.squeeze(0),
            'depth': expected_depths.squeeze(0), # depth_im is typical depth map of rasterization
            "median_depth": median_depths.squeeze(0),
            'depth_im': depth_im,
            'accumulation': alpha.squeeze(0),
            "normals": normals.squeeze(0),
            "depth_normal_error_map": normal_error_map[0, ...].unsqueeze(-1), # [H, W, 1]
            "middepth_normal_error_map": normal_error_map[1, ...].unsqueeze(-1), # [H, W, 1]
            "background": background,
            'features': features.squeeze(0),
        }
        
    def _render(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        features: torch.Tensor,
        render_mode: str,
        sh_degree_to_use: int,
        visible_mask: torch.Tensor,
        camera_params: Dict[str, torch.Tensor],
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        if visible_mask is not None:
            means = means[visible_mask]
            quats = quats[visible_mask]
            scales = scales[visible_mask]
            opacities = opacities[visible_mask]
            colors = colors[visible_mask]
            features = features[visible_mask]
        else:
            means = means
            quats = quats
            scales = scales
            opacities = opacities
            colors = colors
            features = features

        # We need a hack to get features into model gsplat for rendering
        # Convert the SH coefficients to RGB via gsplat
        # Found here: https://github.com/nerfstudio-project/gsplat/issues/529#issuecomment-2575128309
        if sh_degree_to_use is not None:
            dirs = means - camera_params["camera_center"] # directions of the gaussians
            
            colors = spherical_harmonics(
                degrees_to_use=sh_degree_to_use,
                dirs=dirs,
                coeffs=colors, # Current spherical harmonics coefficients
            )

            # Squeeze back just in case
            colors = colors.squeeze(1)
            colors = torch.clamp_min(colors + 0.5, 0.0)
        
        # Now fuse our features with the colors for rendering
        fused_features = torch.cat((colors, features), dim=-1)

        # Items are:
        # - render_colors: [N, 3 + feature_dim]
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
            colors=fused_features,
            viewmats=camera_params["viewmats"],  # [1, 4, 4]
            Ks=camera_params["Ks"],  # [1, 3, 3]
            width=int(camera_params["image_width"]),
            height=int(camera_params["image_height"]),
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=None, # We've computed this in advance --> hacking to use the features
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
            # # set some threshold to disregard small gaussians for faster rendering.
            # radius_clip=3.0,
            # Output depth and normal maps
            return_depth_normal=True,
        )

        return render, alpha, expected_depths, median_depths, expected_normals, meta

    ########################################################
    ########### Visualization functions ####################
    ########################################################

    def set_text_queries(self, positive_queries: List[str], negative_queries: List[str] = None):
        """
        Sets the text queries for the text encoder.
        """
        self.positive_queries = positive_queries
        self.negative_queries = negative_queries

    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle

        Not called during training, but used for visualization and rendering. Can be used to 
        add outputs not needed during training.
        """
        # Call the super method to get the base outputs
        outs = super().get_outputs_for_camera(camera, obb_box)

        # This resize factor affects the resolution of similarity map. Maybe we should use a fixed size?
        decoded_features_dict = self.decode_features(outs["features"], resize_factor=8.0)

        # If we have a text encoder, compute a similarity map if requested
        if self.similarity_fx is not None:
            similarity_map = self.similarity_fx(
                features=decoded_features_dict[self.main_features_name], 
                positive=self.positive_queries, 
                negative=self.negative_queries,
                method=self.config.similarity_method
            )

            # Upsample heatmap to match size of RGB image
            # It's a bit slow since we do it on full resolution; but interpolation seems to have aliasing issues
            assert similarity_map.shape[2] == 1
            if similarity_map.shape[:2] != outs["rgb"].shape[:2]:
                similarity_map = F.interpolate(
                        similarity_map.permute(2, 0, 1)[None],  # H,W,1 -> 1,1,H,W
                        size=outs["rgb"].shape[:2], 
                        mode="bilinear", 
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)  # 1,1,H,W -> H,W,1
                
                # Assign to outputs
                outs["similarity"] = similarity_map
        return outs

    ########################################################
    ############## Optimizer functions #####################
    ########################################################

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        # Get base losses: rgb (SSIM + L1) and tv (cameras) + depth_normal_loss
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Move features to device for loss computation
        for model_name in batch['features_dict']:
            batch['features_dict'][model_name] = batch['features_dict'][model_name].to(self.device)

        # Decoded features for each branch
        decoded_features_dict = self.decode_features(outputs["features"])
        features_loss = torch.tensor(0.0, device=self.device)

        # Computed weighted cosine loss for each model
        for model_name, pred in decoded_features_dict.items():
            # Set weight for the current model type (whether its a main model or not)
            weight = 1.0 if model_name == self.main_features_name else self.config.features_regularization_lambda

            # Compute cosine loss (cosine similarity between predicted and ground truth features)
            gt = batch['features_dict'][model_name]
            features_loss += (1 - F.cosine_similarity(pred, gt, dim=0)).mean() * weight       

        # Now scale the loss by the overall features loss lambda
        loss_dict["features_loss"] = features_loss * self.config.features_loss_lambda

        return loss_dict
    
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in self.gauss_params.keys()
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        # Gather Gaussian-related parameters
        # The distill_features parameter is added via the get_gaussian_param_groups method
        param_groups = super().get_param_groups()
        param_groups["decoder"] = list(self.decoder.parameters())
        return param_groups