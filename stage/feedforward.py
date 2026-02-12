"""
Reconstructor Wrapper for FeedForward 3D Reconstruction

This module builds on top of MapAnything to enable feedforward inference
of pointclouds, mapping to COLMAP format, and creation of meshes.

Plan here:
1. INFERENCE:
    - Use MapAnything interface for inference across many different models
    - Conversion to COLMAP / nerfstudio format
2. REFINEMENT:
    - Bundle adjustment via pycolmap? --> should improve the alignment
    - Create function for cleaning pointclouds --> background removal, confidence filtering
        - Probably will need to think more about this
3. FEATURE MAPPING:
    - Flexible module for mapping features to pointclouds
    - Transform features to mesh?
4. MESHING:
    - Create function for creating meshes from pointclouds
        - Probably will need to think more about this
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import cv2
import numpy as np
import torch

# MapAnything's built-in API
from mapanything.models import init_model_from_config
from mapanything.utils.image import load_images

# Optional: optical flow for frame selection
from optical_flow import OpticalFlowFrameSelector

def validate_input_views(views: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate input views structure for inference.

    Args:
        views: List of view dictionaries to validate

    Returns:
        Validated views (same as input if valid)

    Raises:
        ValueError: If views structure is invalid
    """
    if not isinstance(views, list):
        raise ValueError(f"views must be a list, got {type(views)}")

    if len(views) == 0:
        raise ValueError("views list is empty")

    for i, view in enumerate(views):
        if not isinstance(view, dict):
            raise ValueError(f"view {i} must be a dict, got {type(view)}")

        if "img" not in view:
            raise ValueError(f"view {i} missing required key 'img'")

    return views


def transfer_views_to_device(
    views: List[Dict[str, Any]], device: torch.device
) -> List[Dict[str, Any]]:
    """Transfer all view tensors to the specified device.

    Args:
        views: List of view dictionaries
        device: Target device

    Returns:
        Views with tensors transferred to device
    """
    ignore_keys = {"instance", "idx", "true_shape", "data_norm_type"}

    for view in views:
        for name, val in view.items():
            if name in ignore_keys:
                continue
            if name == "camera_poses" and isinstance(val, tuple):
                view[name] = tuple(x.to(device, non_blocking=True) for x in val)
            elif hasattr(val, "to"):
                view[name] = val.to(device, non_blocking=True)

    return views


def get_autocast_dtype(use_amp: bool, amp_dtype: str) -> torch.dtype:
    """Determine the mixed precision dtype for autocast.

    Args:
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Requested dtype ("fp16", "bf16", or "fp32")

    Returns:
        torch.dtype to use for autocast

    Raises:
        ValueError: If amp_dtype is invalid
    """
    if not use_amp:
        return torch.float32

    if amp_dtype == "fp16":
        return torch.float16
    elif amp_dtype == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            print("Warning: bf16 not supported, using fp16 instead")
            return torch.float16
    elif amp_dtype == "fp32":
        return torch.float32
    else:
        raise ValueError(
            f"Invalid amp_dtype: {amp_dtype}. Must be 'fp16', 'bf16', or 'fp32'"
        )


class ReconstructorConfig(TypedDict):
    """Configuration for the Reconstructor class.

    Required Keys:
        file_path: Path to directory containing images
        model_name: Name of model to use (e.g., "mapanything", "vggt", "dust3r")

    Optional Keys:
        output_path: Path for output data. Defaults to file_path.parent / "environment" / file_path.name
        max_images: Maximum number of images to process
        use_optical_flow: Whether to use optical flow for frame selection
    """

    file_path: Union[str, Path]
    model_name: str
    output_path: Optional[Union[str, Path]]
    max_images: Optional[int]
    use_optical_flow: Optional[bool]


class Reconstructor:
    """Wrapper for feedforward 3D reconstruction using MapAnything.

    Follows Splatter class patterns for consistency.

    Example:
        >>> config = {
        ...     "file_path": "data/video.mp4",
        ...     "model_name": "mapanything",
        ...     "max_images": 50
        ... }
        >>> rec = Reconstructor(config)
        >>> rec.preprocess(use_optical_flow=True)  # Extract frames only
        >>> rec.setup_inference()  # Load model + images
        >>> predictions = rec.infer()
    """

    def __init__(self, config: ReconstructorConfig):
        """Initialize Reconstructor with configuration.

        Args:
            config: Configuration dictionary
        """
        validated_config = self.validate_config(config)
        self.config: Dict[str, Any] = dict(validated_config)

        # State
        self.model = None
        self.views = None
        self.outputs = None

    @classmethod
    def from_config_file(
        cls,
        dataset: str,
        config_dir: Union[str, Path],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "Reconstructor":
        """Create Reconstructor instance from YAML configuration.

        Args:
            dataset: Dataset config name (from datasets/ subdirectory)
            config_dir: Directory containing config files (base.yaml and datasets/)
            overrides: Optional runtime overrides

        Returns:
            Configured Reconstructor instance with pipeline configs attached

        Example:
            >>> rec = Reconstructor.from_config_file(
            ...     dataset='ants_001',
            ...     config_dir='configs'
            ... )
            >>> rec.preprocess()  # Extract/select frames
            >>> rec.setup_inference()  # Load model + images
            >>> rec.infer()
        """
        from collab_splats.wrapper.config import ConfigLoader

        loader = ConfigLoader(config_dir)
        config = loader.load(dataset=dataset, overrides=overrides)

        # Store full config for later use
        full_config = config.copy()

        # Extract ReconstructorConfig fields
        reconstructor_fields: Dict[str, Any] = {
            "file_path": config["file_path"],
            "model_name": config.get("model_name", "mapanything"),
        }

        # Add optional fields if present
        if "output_path" in config:
            reconstructor_fields["output_path"] = config["output_path"]
        if "max_images" in config:
            reconstructor_fields["max_images"] = config["max_images"]
        if "use_optical_flow" in config:
            reconstructor_fields["use_optical_flow"] = config["use_optical_flow"]

        reconstructor_config: ReconstructorConfig = reconstructor_fields  # type: ignore
        instance = cls(reconstructor_config)

        # Attach configs for pipeline methods
        instance._preprocess_config = full_config.get("preprocess", {})
        instance._inference_config = full_config.get("inference", {})

        return instance

    @classmethod
    def validate_config(cls, config: ReconstructorConfig) -> ReconstructorConfig:
        """Validate configuration and set defaults.

        Args:
            config: Configuration to validate

        Returns:
            Validated configuration

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required fields
        required_fields = {"file_path", "model_name"}
        missing_fields = required_fields - set(config.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Convert to Path
        file_path = Path(config["file_path"])
        if not file_path.exists():
            raise ValueError(f"Path not found: {file_path}")

        # Detect input type based on file extension
        video_extensions = {".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV"}
        image_extensions = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"}

        if file_path.is_dir():
            input_type = "images"
        elif file_path.is_file():
            if file_path.suffix in video_extensions:
                input_type = "video"
            elif file_path.suffix in image_extensions:
                input_type = "image"
            else:
                raise ValueError(
                    f"Unsupported file type: {file_path.suffix}. "
                    f"Supported: {video_extensions | image_extensions}"
                )
        else:
            raise ValueError(f"file_path must be a file or directory: {file_path}")

        config["file_path"] = file_path
        config["input_type"] = input_type

        # Set default output path following Splatter pattern
        if config.get("output_path") is None:
            if input_type == "images":
                # For image directories: parent / environment / directory_name
                default_output_path = file_path.parent / "environment" / file_path.name
            else:
                # For video/image files: parent / environment / stem
                default_output_path = file_path.parent / "environment" / file_path.stem
            config["output_path"] = default_output_path

        config["output_path"] = Path(config["output_path"])

        # Set defaults
        config.setdefault("max_images", None)
        config.setdefault("use_optical_flow", False)

        return config

    @staticmethod
    def available_models() -> None:
        """Print available models from MapAnything."""
        from mapanything.models import get_available_models

        models = get_available_models()
        print("Available models:")
        print("  ", sorted(models))

    def _get_images(self, directory: Path) -> List[Path]:
        """Get sorted image paths from directory."""
        exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        paths = []
        for ext in exts:
            paths.extend(directory.glob(ext))
        return sorted(paths)

    def _select_subset(self, paths: List[Path], max_images: Optional[int]) -> List[Path]:
        """Select subset using uniform decimation."""
        if not max_images or max_images >= len(paths):
            return paths
        indices = np.linspace(0, len(paths) - 1, max_images, dtype=int)
        return [paths[i] for i in indices]

    def _process_video(
        self, video_path: Path, output_dir: Path, max_images: Optional[int], use_optical_flow: bool, optical_flow_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[Path]:
        """Extract and select frames from video."""
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting frames from video: {video_path}")

        optical_flow_kwargs = optical_flow_kwargs or {}
        selector = OpticalFlowFrameSelector(**optical_flow_kwargs)

        if use_optical_flow:
            # Use optical flow for frame selection
            max_msg = f"up to {max_images}" if max_images else "based on motion/coverage thresholds"
            print(f"Using optical flow to select frames ({max_msg})")
            selector.process_video(
                video_path=video_path,
                max_frames=max_images,  # max_frames is optional in optical flow
                save_selected_frames=True,
                output_dir=output_dir,
            )
        else:
            # Use uniform decimation
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            indices = self._select_subset(list(range(total_frames)), max_images)
            print(f"Extracting {len(indices)} frames from {total_frames} total")
            selector.export_frames_by_indices(video_path, indices, output_dir)

        return self._get_images(output_dir)

    def _process_images(
        self, image_dir: Path, max_images: Optional[int], use_optical_flow: bool, optical_flow_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[Path]:
        """Select images from directory."""
        paths = self._get_images(image_dir)
        if not paths:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Found {len(paths)} images in {image_dir}")

        if use_optical_flow:
            max_msg = f"up to {max_images}" if max_images else "based on motion/coverage thresholds"
            print(f"Using optical flow to select images ({max_msg})")
            optical_flow_kwargs = optical_flow_kwargs or {}
            selector = OpticalFlowFrameSelector(**optical_flow_kwargs)
            indices, _ = selector.process_image_directory(image_dir, max_images=max_images)
            return [paths[i] for i in indices]

        return self._select_subset(paths, max_images)

    def preprocess(
        self,
        use_optical_flow: Optional[bool] = None,
        max_images: Optional[int] = None,
        optical_flow_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract and select frames with optional optical flow decimation.

        This method only extracts/selects frames to disk. It does NOT load them into memory.
        Call setup_inference() to load the model and images before running inference.

        Args:
            use_optical_flow: Whether to use optical flow for frame selection
            max_images: Maximum number of images to process
            optical_flow_kwargs: Parameters for OpticalFlowFrameSelector (min_disparity, motion_weight, coverage_weight)
        """
        file_path = self.config["file_path"]
        input_type = self.config["input_type"]
        use_optical_flow = use_optical_flow if use_optical_flow is not None else self.config["use_optical_flow"]
        max_images = max_images or self.config.get("max_images")
        optical_flow_kwargs = optical_flow_kwargs or {}

        # Get image paths based on input type
        if input_type == "video":
            paths = self._process_video(
                file_path, self.config["output_path"] / "frames", max_images, use_optical_flow, optical_flow_kwargs
            )
        elif input_type == "image":
            print(f"Selected single image: {file_path}")
            paths = [file_path]
        else:  # input_type == "images"
            paths = self._process_images(file_path, max_images, use_optical_flow, optical_flow_kwargs)

        # Store paths for later loading
        self.config["image_paths"] = paths
        print(f"✓ Frame selection complete: {len(paths)} images ready")

    def setup_inference(self, device: str = "cuda", **load_images_kwargs) -> None:
        """Load model and images in preparation for inference.

        Both model loading and image loading are heavy operations that should
        happen right before inference to maximize efficiency.

        Args:
            device: Device to load model on
            **load_images_kwargs: Additional arguments passed to MapAnything's load_images()

        Raises:
            RuntimeError: If preprocess() hasn't been called yet
        """
        if "image_paths" not in self.config:
            raise RuntimeError("No image paths found. Call preprocess() first to select frames.")

        model_name = self.config["model_name"]
        paths = self.config["image_paths"]

        print("Setting up inference environment")
        print("="*70)

        # Load model
        print(f"Loading model: {model_name}")
        self.model = init_model_from_config(model_name, device=device)
        self.model.eval()
        print(f"✓ Model loaded")

        # Load images into memory
        print(f"Loading {len(paths)} images into memory")
        self.views = load_images([str(p) for p in paths], **load_images_kwargs)
        print(f"✓ Images loaded")

        print("="*70)
        print(f"✓ Ready for inference: {len(self.views)} images")
        print("="*70)

    def load_model(self, device: str = "cuda") -> Any:
        """Load model only (for backward compatibility).

        Consider using setup_inference() instead for the recommended workflow.

        Args:
            device: Device to load model on

        Returns:
            Loaded model
        """
        model_name = self.config["model_name"]
        print(f"Loading model: {model_name}")
        self.model = init_model_from_config(model_name, device=device)
        self.model.eval()
        print(f"✓ Model loaded: {model_name}")
        return self.model

    def infer(
        self,
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        device: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """Run inference using MapAnything's unified forward() interface.

        All model wrappers follow the same forward() interface and return
        outputs in a unified format with keys: pts3d, depth_z, conf, camera_poses, etc.

        Args:
            use_amp: Whether to use automatic mixed precision. Defaults to True.
            amp_dtype: The dtype for mixed precision ("fp16", "bf16", "fp32"). Defaults to "bf16".
            device: Device to run inference on. Defaults to model's device.
            **kwargs: Additional model-specific arguments passed to forward().
                Available kwargs depend on the model. Common MapAnything options include:
                - memory_efficient_inference: bool
                - minibatch_size: int

        Returns:
            List of prediction dictionaries

        Raises:
            RuntimeError: If model or views not loaded
            ValueError: If views are invalid or amp_dtype is invalid
        """
        # Validate state
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first")
        if self.views is None:
            raise RuntimeError("Images not loaded. Call preprocess() first")

        # Validate and prepare views
        validated_views = validate_input_views(self.views)

        print(f"Running inference on {len(validated_views)} images")

        # Determine device
        if device is None:
            device = next(self.model.parameters()).device
        else:
            device = torch.device(device)

        # Transfer views to device
        transfer_views_to_device(validated_views, device)

        # Determine mixed precision dtype
        dtype = get_autocast_dtype(use_amp, amp_dtype)

        # Run inference with MapAnything's unified forward() interface
        # All model wrappers (mapanything, vggt, dust3r, mast3r, etc.) use forward()
        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device.type, enabled=use_amp, dtype=dtype):
                self.outputs = self.model(validated_views, **kwargs)

        print(f"✓ Inference complete")
        return self.outputs
