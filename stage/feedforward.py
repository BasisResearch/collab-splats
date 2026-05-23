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
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm.auto import tqdm

# MapAnything's built-in API
from mapanything.models import init_model_from_config, get_available_models
from mapanything.utils.image import load_images
from mapanything.utils.inference import postprocess_model_outputs_for_inference
from mapanything.utils.geometry import depthmap_to_world_frame

# Optional: optical flow for frame selection
from optical_flow import OpticalFlowFrameSelector

from collab_splats.wrapper.config import ConfigLoader
from collab_splats.semantics.features import BaseFeatureExtractor, MaskCLIPExtractor, DINOFeatureExtractor

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
        ...     "model_name": "mapanything_v1",
        ...     "max_images": 50
        ... }
        >>> rec = Reconstructor(config)
        >>> rec.preprocess(use_optical_flow=True)  # Extract frames only
        >>> rec.setup_inference()  # Load model + images
        >>> predictions = rec.infer()
    """

    # Core MapAnything variants that require HuggingFace for pretrained weights.
    # External wrappers (vggt, dust3r, mast3r, da3, etc.) self-load weights in their __init__
    # and are handled by init_model_from_config.
    HF_MODELS = {
        "mapanything":    "facebook/map-anything",
        "mapanything_v1": "facebook/map-anything-v1",
    }

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
        self._loaded_image_paths = None  # Track which images are loaded

        # Feature extraction state
        self._feature_extractor: Optional[BaseFeatureExtractor] = None
        self._feature_extractor_name: Optional[str] = None
        self._pca: Optional[PCA] = None  # Fitted PCA for dimensionality reduction

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
        loader = ConfigLoader(config_dir)
        config = loader.load(dataset=dataset, overrides=overrides)

        # Store full config for later use
        full_config = config.copy()

        # Extract ReconstructorConfig fields
        reconstructor_fields: Dict[str, Any] = {
            "file_path": config["file_path"],
            "model_name": config.get("model_name", "mapanything_v1"),
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
                # For image directories: parent.parent / environment / directory_name
                default_output_path = file_path.parent.parent / "environment" / file_path.name
            else:
                # For video/image files: parent.parent / environment / stem
                default_output_path = file_path.parent.parent / "environment" / file_path.stem
            config["output_path"] = default_output_path

        config["output_path"] = Path(config["output_path"])

        # Set defaults
        config.setdefault("max_images", None)
        config.setdefault("use_optical_flow", False)

        return config

    @staticmethod
    def available_models() -> None:
        """Print available models from MapAnything."""
        models = get_available_models()
        print("Available models:")
        print("  ", sorted(models))

    #########################################################
    ############### Preprocessing Methods ###################
    #########################################################

    def preprocess(
        self,
        use_optical_flow: Optional[bool] = None,
        max_images: Optional[int] = None,
        optical_flow_kwargs: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """Extract and select frames with optional optical flow decimation.

        This method only extracts/selects frames to disk. It does NOT load them into memory.
        Call setup_inference() to load the model and images before running inference.

        Args:
            use_optical_flow: Whether to use optical flow for frame selection
            max_images: Maximum number of images to process
            optical_flow_kwargs: Parameters for OpticalFlowFrameSelector. Can include:
                - Selector init params: min_disparity, max_features, motion_weight, coverage_weight,
                  histogram_similarity_threshold, adaptive_threshold, rotation_threshold, verbose
                - Process params: selection_threshold, save_metadata_json
            overwrite: If False and output directory exists with images, skip processing and use existing frames

        Example:
            >>> recon.preprocess(
            ...     use_optical_flow=True,
            ...     optical_flow_kwargs={
            ...         'min_disparity': 60.0,
            ...         'motion_weight': 0.5,
            ...         'coverage_weight': 0.5,
            ...         'selection_threshold': 0.5,
            ...         'verbose': True
            ...     },
            ...     overwrite=False
            ... )
        """
        file_path = self.config["file_path"]
        input_type = self.config["input_type"]
        use_optical_flow = use_optical_flow if use_optical_flow is not None else self.config["use_optical_flow"]
        max_images = max_images or self.config.get("max_images")
        optical_flow_kwargs = optical_flow_kwargs or {}

        # Determine output directory based on input type
        if input_type == "video":
            output_dir = self.config["output_path"] / "preproc" / "images"
        elif input_type == "images":
            output_dir = file_path  # For image directories, use the source directory
        else:  # single image
            output_dir = None

        # Check if we can skip processing
        if not overwrite and output_dir is not None and output_dir.exists():
            existing_paths = self._get_images(output_dir)
            if existing_paths:
                print(f"⚠ Output directory already exists with {len(existing_paths)} images: {output_dir}")
                print(f"  Skipping preprocessing. Use overwrite=True to reprocess.")
                self.config["image_paths"] = existing_paths
                return

        # Get image paths based on input type
        if input_type == "video":
            paths = self._process_video(
                file_path, output_dir, max_images, use_optical_flow, optical_flow_kwargs, overwrite=overwrite
            )
        elif input_type == "image":
            print(f"Selected single image: {file_path}")
            paths = [file_path]
        else:  # input_type == "images"
            paths = self._process_images(file_path, max_images, use_optical_flow, optical_flow_kwargs)

        # Store paths for later loading
        self.config["image_paths"] = paths
        print(f"✓ Frame selection complete: {len(paths)} images ready")

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
        self,
        video_path: Path,
        output_dir: Path,
        max_images: Optional[int],
        use_optical_flow: bool,
        optical_flow_kwargs: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> List[Path]:
        """Extract and select frames from video.

        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            max_images: Maximum number of frames to extract
            use_optical_flow: Whether to use optical flow for selection
            optical_flow_kwargs: Parameters for optical flow selector (both init and process_video params)
            overwrite: If True, delete existing frames before processing

        Returns:
            List of paths to extracted frames
        """
        if overwrite and output_dir.exists():
            import shutil
            print(f"Removing existing frames: {output_dir}")
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting frames from video: {video_path}")

        optical_flow_kwargs = optical_flow_kwargs or {}

        # Separate parameters for __init__ and process_video()
        init_params = {
            'min_disparity', 'max_features', 'motion_weight', 'coverage_weight',
            'histogram_similarity_threshold', 'adaptive_threshold', 'rotation_threshold', 'verbose'
        }
        process_params = {
            'selection_threshold', 'save_metadata_json'
        }

        # Split kwargs
        selector_init_kwargs = {k: v for k, v in optical_flow_kwargs.items() if k in init_params}
        selector_process_kwargs = {k: v for k, v in optical_flow_kwargs.items() if k in process_params}

        # Create selector with init parameters
        selector = OpticalFlowFrameSelector(**selector_init_kwargs)

        if use_optical_flow:
            # Use optical flow for frame selection
            max_msg = f"up to {max_images}" if max_images else "based on motion/coverage thresholds"
            print(f"Using optical flow to select frames ({max_msg})")
            selector.process_video(
                video_path=video_path,
                max_frames=max_images,  # max_frames is optional in optical flow
                save_selected_frames=True,
                output_dir=output_dir,
                **selector_process_kwargs  # Pass process_video specific params
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
        self,
        image_dir: Path,
        max_images: Optional[int],
        use_optical_flow: bool,
        optical_flow_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[Path]:
        """Select images from directory.

        Args:
            image_dir: Directory containing images
            max_images: Maximum number of images to select
            use_optical_flow: Whether to use optical flow for selection
            optical_flow_kwargs: Parameters for optical flow selector

        Returns:
            List of selected image paths
        """
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

    #########################################################
    ############### Inference Methods #######################
    #########################################################

    def setup_inference(self, device: Optional[str] = None, force_reload: bool = False, **load_images_kwargs) -> None:
        """Load model and images in preparation for inference.

        Both model loading and image loading are heavy operations that should
        happen right before inference to maximize efficiency.

        Args:
            device: Device to load model on. If None, auto-detects ("cuda" if available, else "cpu")
            force_reload: If True, reload model and images even if already loaded
            **load_images_kwargs: Additional arguments passed to MapAnything's load_images()

        Raises:
            RuntimeError: If preprocess() hasn't been called yet
        """
        if "image_paths" not in self.config:
            raise RuntimeError("No image paths found. Call preprocess() first to select frames.")

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Auto-detected device: {device}")

        model_name = self.config["model_name"]
        paths = self.config["image_paths"]

        # Check if already loaded
        model_loaded = self.model is not None
        images_loaded = self.views is not None and len(self.views) == len(paths)
        images_match = (
            images_loaded and
            hasattr(self, '_loaded_image_paths') and
            self._loaded_image_paths == paths
        )

        if not force_reload and model_loaded and images_match:
            print("⚠ Model and images already loaded, skipping setup.")
            print(f"  Model: {model_name}")
            print(f"  Images: {len(self.views)}")
            print(f"  Use force_reload=True to reload anyway.")
            return

        print("Setting up inference environment")
        print("="*70)

        # Load model if needed
        if force_reload or not model_loaded:
            print(f"Loading model: {model_name}")
            self.model = self._load_model_weights(model_name, device)
            self.model.eval()
            print(f"✓ Model loaded")
        else:
            print(f"✓ Model already loaded: {model_name}")

        # Load images if needed
        if force_reload or not images_match:
            print(f"Loading {len(paths)} images into memory")
            self.views = load_images([str(p) for p in paths], **load_images_kwargs)
            self._loaded_image_paths = paths  # Track which images are loaded
            print(f"✓ Images loaded")
        else:
            print(f"✓ Images already loaded: {len(self.views)} images")

        print("="*70)
        print(f"✓ Ready for inference: {len(self.views)} images")
        print("="*70)

    def _load_model_weights(self, model_name: str, device: str) -> Any:
        """Load model with pretrained weights.

        Core MapAnything variants load weights from HuggingFace via from_pretrained().
        External wrappers (vggt, dust3r, mast3r, etc.) self-load weights in their __init__
        via init_model_from_config().
        """
        if model_name in self.HF_MODELS:
            from mapanything.models import MapAnything
            hf_id = self.HF_MODELS[model_name]
            print(f"  Loading from HuggingFace: {hf_id}")
            return MapAnything.from_pretrained(hf_id).to(device)
        return init_model_from_config(model_name, device=device)

    def load_model(self, device: Optional[str] = None, force_reload: bool = False) -> Any:
        """Load model only (for backward compatibility).

        Consider using setup_inference() instead for the recommended workflow.

        Args:
            device: Device to load model on. If None, auto-detects ("cuda" if available, else "cpu")
            force_reload: If True, reload model even if already loaded

        Returns:
            Loaded model
        """
        if not force_reload and self.model is not None:
            print(f"⚠ Model already loaded, skipping. Use force_reload=True to reload anyway.")
            return self.model

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Auto-detected device: {device}")

        model_name = self.config["model_name"]
        print(f"Loading model: {model_name}")
        self.model = self._load_model_weights(model_name, device)
        self.model.eval()
        print(f"✓ Model loaded: {model_name}")
        return self.model

    def infer(
        self,
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        device: Optional[str] = None,
        force_rerun: bool = False,
        verbose: bool = True,
        postprocess: bool = True,
        **kwargs
    ) -> List[Dict]:
        """Run inference using MapAnything's unified forward() interface.

        All model wrappers follow the same forward() interface and return
        outputs in a unified format with keys: pts3d, depth_z, conf, camera_poses, etc.

        Args:
            use_amp: Whether to use automatic mixed precision. Defaults to True.
            amp_dtype: The dtype for mixed precision ("fp16", "bf16", "fp32"). Defaults to "bf16".
            device: Device to run inference on. Defaults to model's device.
            force_rerun: If True, rerun inference even if outputs already exist
            verbose: Whether to print verbose output
            postprocess: Whether to apply post-processing to outputs. Defaults to True.
            **kwargs: Additional arguments for inference and post-processing.
                Model-specific inference kwargs (passed to forward()):
                - memory_efficient_inference: bool
                - minibatch_size: int

                Post-processing kwargs (used when postprocess=True):
                - apply_mask: bool (default: True) - Apply mask to outputs
                - mask_edges: bool (default: True) - Mask edges based on discontinuities
                - edge_normal_threshold: float (default: 5.0) - Surface normal threshold
                - edge_depth_threshold: float (default: 0.03) - Depth discontinuity threshold
                - apply_confidence_mask: bool (default: False) - Apply confidence-based masking
                - confidence_percentile: float (default: 10) - Percentile threshold for confidence
                - use_multiview_confidence: bool (default: False) - Compute multi-view depth consistency
                - multiview_conf_depth_abs_thresh: float (default: 0.02) - Absolute depth threshold
                - multiview_conf_depth_rel_thresh: float (default: 0.02) - Relative depth threshold

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

        # Check if inference already run
        if not force_rerun and self.outputs is not None:
            print("⚠ Inference outputs already exist, skipping rerun.")
            print(f"  Using cached outputs from {len(self.outputs)} images")
            print(f"  Use force_rerun=True to recompute.")
            return self.outputs

        # Validate and prepare views
        validated_views = validate_input_views(self.views)

        # Separate post-processing kwargs from model inference kwargs
        postprocess_params = {
            'apply_mask', 'mask_edges', 'edge_normal_threshold', 'edge_depth_threshold',
            'apply_confidence_mask', 'confidence_percentile', 'use_multiview_confidence',
            'multiview_conf_depth_abs_thresh', 'multiview_conf_depth_rel_thresh'
        }

        postprocess_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in postprocess_params}
        model_kwargs = kwargs  # Remaining kwargs are for the model

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

        # Track timing and memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.time()

        # Run inference with MapAnything's unified forward() interface
        # All model wrappers (mapanything, vggt, dust3r, mast3r, etc.) use forward()
        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device.type, enabled=use_amp, dtype=dtype):
                outputs = self.model(validated_views, **model_kwargs)

        # Synchronize and calculate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_time = time.time() - start_time

        # Set defaults for post-processing parameters
        pp_defaults = {
            'apply_mask': True,
            'mask_edges': True,
            'edge_normal_threshold': 5.0,
            'edge_depth_threshold': 0.03,
            'apply_confidence_mask': False,
            'confidence_percentile': 10,
            'use_multiview_confidence': False,
            'multiview_conf_depth_abs_thresh': 0.02,
            'multiview_conf_depth_rel_thresh': 0.02,
        }
        # Merge user-provided kwargs with defaults
        pp_config = {**pp_defaults, **postprocess_kwargs}

        # Apply post-processing if requested
        if postprocess:
            if verbose:
                print("Applying post-processing...")

            outputs = postprocess_model_outputs_for_inference(
                raw_outputs=outputs,
                input_views=validated_views,
                **pp_config
            )

        # # Aggregate and stack results
        # output_keys = sorted(set().union(*(pred.keys() for pred in outputs)))
        # results = {
        #     key: torch.stack([pred[key] for pred in outputs]).cpu().squeeze(1)
        #     for key in output_keys
        # }

        # Print unified summary
        n_frames = len(validated_views)
        sep = "=" * 70

        print(f"\n{sep}")
        print(f"✓ Inference Complete")
        print(sep)
        print(f"Performance:")
        print(f"  • Total time        : {inference_time:.2f}s")
        print(f"  • Time per frame    : {inference_time / n_frames:.3f}s")
        print(f"  • Throughput (FPS)  : {n_frames / inference_time:.2f}")

        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  • Peak GPU memory   : {peak_memory_gb:.2f} GB")
        
        print(f"{sep}\n")

        self.outputs = outputs
        return self.outputs

    #########################################################
    ############### Feature Extraction Methods ##############
    #########################################################

    def setup_feature_extractor(
        self,
        name: str,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Load and store a feature extractor for later use in pointcloud building.

        Args:
            name: Extractor name registered in BaseFeatureExtractor (e.g. "maskclip", "dinov2").
            device: Device to place the extractor on. Auto-detects if None.
            **kwargs: Extra arguments forwarded to the extractor constructor
                      (e.g. model_name, resolution).

        Example:
            >>> rec.setup_feature_extractor("maskclip", device="cuda")
            >>> rec.setup_feature_extractor("dinov2", device="cuda", resolution=640)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        extractor_cls = BaseFeatureExtractor.get(name)
        self._feature_extractor = extractor_cls(device=device, **kwargs)
        self._feature_extractor_name = name

        # Report feature dimensionality
        if isinstance(self._feature_extractor, MaskCLIPExtractor):
            feat_dim = self._feature_extractor.model.visual.output_dim
        elif isinstance(self._feature_extractor, DINOFeatureExtractor):
            feat_dim = self._feature_extractor.model.embed_dim
        else:
            feat_dim = "?"

        print(f"✓ Feature extractor ready: {name} (dim={feat_dim}, device={device})")

    def _extract_image_features(
        self,
        image_paths: List[Path],
        resolution: int = 1024,
    ) -> List[torch.Tensor]:
        """Extract dense pixel-aligned feature maps from a list of images.

        Handles both CLIP (maskclip) and DINO (dinov2) extractors.
        Features are upsampled from patch resolution back to full image resolution
        so they can be directly indexed at any pixel position.

        Args:
            image_paths: Ordered list of image file paths.
            resolution: Longest-edge resolution for feature extraction.

        Returns:
            List of (C, H, W) float32 tensors on CPU, one per image.
            C = feature dimension (e.g. 768 for ViT-L/14).
        """
        extractor = self._feature_extractor
        if extractor is None:
            raise RuntimeError(
                "No feature extractor loaded. Call setup_feature_extractor() first."
            )

        feat_maps: List[torch.Tensor] = []

        print(f"Extracting features from {len(image_paths)} images...")

        for i, path in enumerate(image_paths):
            if isinstance(extractor, MaskCLIPExtractor):
                # preprocess returns (C, H, W) on extractor device
                img_tensor = extractor.preprocess(path, resolution=resolution)
                img_batch = img_tensor.unsqueeze(0)  # (1, C, H, W)

                # forward returns (1, D, H_p, W_p)
                feat = extractor(img_batch).squeeze(0)  # (D, H_p, W_p)

                # Upsample to original preprocessed image resolution
                target_h, target_w = img_tensor.shape[-2], img_tensor.shape[-1]
                feat = F.interpolate(
                    feat.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # (D, H, W)

            elif isinstance(extractor, DINOFeatureExtractor):
                # preprocess returns ((1, C, H, W), target_H, target_W)
                img_batch, target_h, target_w = extractor.preprocess(path)

                # forward returns flat patch tokens (N_patches, D)
                tokens = extractor(img_batch)

                # reshape to (D, H_p, W_p)
                feat = extractor.reshape(tokens, target_h, target_w)  # (D, H_p, W_p)

                # Upsample to preprocessed image resolution
                feat = F.interpolate(
                    feat.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # (D, H, W)

            else:
                raise ValueError(
                    f"Unsupported extractor type: {type(extractor)}. "
                    "Expected MaskCLIPExtractor or DINOFeatureExtractor."
                )

            feat_maps.append(feat.cpu().float())
            print(f"  [{i+1}/{len(image_paths)}] {path.name}: features {tuple(feat.shape)}")

        print(f"✓ Feature extraction complete")
        return feat_maps

    #########################################################
    ############### Pointcloud Methods ######################
    #########################################################

    def prepare_outputs_for_export(self) -> Dict[str, np.ndarray]:
        """Prepare inference outputs for export to COLMAP, trimesh, etc.

        Converts the stacked tensor outputs to numpy arrays and computes world-space
        point clouds from depth maps, intrinsics, and camera poses.

        Returns:
            Dictionary with the following keys:
                - extrinsic: (S, 4, 4) camera extrinsic matrices (world-to-camera)
                - intrinsic: (S, 3, 3) camera intrinsic matrices
                - world_points: (S, H, W, 3) 3D world coordinates for each pixel
                - depth: (S, H, W) depth maps
                - images: (S, H, W, 3) RGB images (0-255 range)
                - final_mask: (S, H, W) boolean masks for valid pixels
                - conf: (S, H, W) confidence maps

        Raises:
            RuntimeError: If inference has not been run yet
        """

        if self.outputs is None:
            raise RuntimeError("No outputs found. Call infer() first to run inference.")

        # Aggregate and stack results
        output_keys = sorted(set().union(*(pred.keys() for pred in self.outputs)))
        outputs = {
            key: torch.stack([pred[key] for pred in self.outputs]).cpu().squeeze(1)
            for key in output_keys
        }

        # Check for required keys
        required_keys = ["depth_z", "intrinsics", "camera_poses"]
        missing_keys = [k for k in required_keys if k not in outputs]
        if missing_keys:
            raise RuntimeError(f"Missing required output keys: {missing_keys}")

        # Get number of views and dimensions
        S = outputs["depth_z"].shape[0]  # Number of views

        print(f"Preparing {S} views for export...")

        # Initialize lists to accumulate results
        extrinsic_list = []
        intrinsic_list = []
        world_points_list = []
        depth_maps_list = []
        images_list = []
        final_mask_list = []
        confidences_list = []

        # Process each view
        for i in range(S):
            # Extract data for this view
            depthmap_torch = outputs["depth_z"][i]  # (H, W) or (H, W, 1)
            intrinsics_torch = outputs["intrinsics"][i]  # (3, 3)
            camera_pose_torch = outputs["camera_poses"][i]  # (4, 4)

            # Squeeze depth if needed
            if depthmap_torch.dim() == 3:
                depthmap_torch = depthmap_torch.squeeze(-1)

            # Get confidence if available
            if "conf" in outputs:
                conf = outputs["conf"][i]
                if conf.dim() == 3:
                    conf = conf.squeeze(-1)
            else:
                conf = torch.ones_like(depthmap_torch)

            # Compute world points using depth, intrinsics, and camera pose
            pts3d_computed, valid_mask = depthmap_to_world_frame(
                depthmap_torch, intrinsics_torch, camera_pose_torch
            )

            # Get mask if available, otherwise use all valid depth points
            if "mask" in outputs:
                mask = outputs["mask"][i]
                if mask.dim() == 3:
                    mask = mask.squeeze(-1)
                mask = mask.cpu().numpy().astype(bool)
            else:
                # Default to all True
                mask = np.ones_like(depthmap_torch.cpu().numpy(), dtype=bool)

            # Combine with valid depth mask
            mask = mask & valid_mask.cpu().numpy()

            # Get image if available
            if "img_no_norm" in outputs:
                image = outputs["img_no_norm"][i].cpu().numpy()
            elif "img" in outputs:
                # If only normalized image is available, denormalize it
                # Assume ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                image = outputs["img"][i].cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
                image = (image * std + mean) * 255.0
                image = np.clip(image, 0, 255)
            else:
                # No image available, create placeholder
                H, W = depthmap_torch.shape
                image = np.zeros((H, W, 3), dtype=np.uint8)

            # Append to lists
            extrinsic_list.append(camera_pose_torch.cpu().numpy())
            intrinsic_list.append(intrinsics_torch.cpu().numpy())
            world_points_list.append(pts3d_computed.cpu().numpy())
            depth_maps_list.append(depthmap_torch.cpu().numpy())
            images_list.append(image)
            final_mask_list.append(mask)
            confidences_list.append(conf.cpu().numpy())

        # Stack all results
        predictions = {
            "extrinsic": np.stack(extrinsic_list, axis=0),  # (S, 4, 4)
            "intrinsic": np.stack(intrinsic_list, axis=0),  # (S, 3, 3)
            "world_points": np.stack(world_points_list, axis=0),  # (S, H, W, 3)
            "depth": np.stack(depth_maps_list, axis=0),  # (S, H, W)
            "images": np.stack(images_list, axis=0),  # (S, H, W, 3)
            "final_mask": np.stack(final_mask_list, axis=0),  # (S, H, W)
            "conf": np.stack(confidences_list, axis=0),  # (S, H, W)
        }

        print(f"✓ Export preparation complete")
        print(f"  • Views: {S}")
        print(f"  • Image shape: {predictions['images'].shape[1:3]}")
        print(f"  • Total points: {predictions['world_points'].shape[0] * predictions['world_points'].shape[1] * predictions['world_points'].shape[2]}")

        return predictions

    def feedforward_to_pointcloud(
        self,
        confidence_threshold: Optional[float] = None,
        subsample_factor: int = 1,
        feature_extractor: Optional[str] = None,
        n_components: int = 64,
        feature_resolution: int = 1024,
    ) -> Dict[str, np.ndarray]:
        """Create a merged pointcloud from all views, optionally with semantic features.

        Args:
            confidence_threshold: Optional threshold for filtering points by confidence.
                                If None, uses all valid masked points.
            subsample_factor: Subsample points by this factor (1 = use all points, 2 = use every other point, etc.)
            feature_extractor: Name of the feature extractor to use for semantic features
                               (e.g. "maskclip", "dinov2"). If None, no features are extracted.
                               Falls back to the extractor set via setup_feature_extractor() if available.
            n_components: Number of PCA components to reduce features to. Features are compressed
                          from their native dimension (e.g. 768 for CLIP ViT-L/14) down to this size.
                          The fitted PCA is stored on self._pca for use in query_pointcloud().
            feature_resolution: Longest-edge resolution at which to extract features. Higher values
                                 give finer spatial detail but require more VRAM. Defaults to 1024.

        Returns:
            Dictionary with:
                - points: (N, 3) array of 3D world coordinates
                - colors: (N, 3) array of RGB colors (0-255 range)
                - confidences: (N,) array of confidence values
                - view_indices: (N,) array indicating which view each point came from
                - features: (N, n_components) array of PCA-compressed semantic features
                            [only present when a feature_extractor is specified]
                - feature_extractor: str name of the extractor used [only when features present]
                - feature_dim: int, equals n_components [only when features present]

        Raises:
            RuntimeError: If inference has not been run yet, or if image_paths are missing
                          when features are requested.

        Example:
            >>> pcd = rec.feedforward_to_pointcloud(
            ...     confidence_threshold=0.8,
            ...     feature_extractor="maskclip",
            ...     n_components=64,
            ... )
            >>> pcd["features"].shape  # (N, 64)
        """
        # Resolve extractor name: explicit arg > previously set up extractor
        extractor_name = feature_extractor or self._feature_extractor_name

        # If features requested, ensure extractor is loaded and image paths are available
        extract_features = extractor_name is not None
        if extract_features:
            if "image_paths" not in self.config:
                raise RuntimeError(
                    "image_paths not found in config. Call preprocess() before requesting features."
                )
            if self._feature_extractor is None or self._feature_extractor_name != extractor_name:
                print(f"Setting up feature extractor: {extractor_name}")
                self.setup_feature_extractor(extractor_name)

            image_paths = self.config["image_paths"]

        # Get prepared data
        data = self.prepare_outputs_for_export()

        S = data["world_points"].shape[0]

        print(f"Creating pointcloud from {S} views...")
        if extract_features:
            print(f"Extracting features per-view (memory-efficient mode)...")

        # Collect points from all views
        all_points = []
        all_colors = []
        all_confidences = []
        all_view_indices = []
        all_raw_features: List[np.ndarray] = []  # Collected before PCA

        for view_idx in range(S):
            # Get data for this view
            world_pts = data["world_points"][view_idx]  # (H, W, 3)
            colors = data["images"][view_idx]  # (H, W, 3)
            mask = data["final_mask"][view_idx]  # (H, W)
            conf = data["conf"][view_idx]  # (H, W)

            H, W = mask.shape

            # Apply subsampling if requested
            if subsample_factor > 1:
                mask_subsampled = np.zeros_like(mask)
                mask_subsampled[::subsample_factor, ::subsample_factor] = mask[::subsample_factor, ::subsample_factor]
                mask = mask_subsampled

            # Apply confidence threshold if specified
            if confidence_threshold is not None:
                conf_mask = conf >= confidence_threshold
                mask = mask & conf_mask

            # Flatten and filter by mask
            valid_points = world_pts[mask]  # (N_valid, 3)
            valid_colors = colors[mask]  # (N_valid, 3)
            valid_conf = conf[mask]  # (N_valid,)

            # Track which view these points came from
            view_indices = np.full(len(valid_points), view_idx, dtype=np.int32)

            all_points.append(valid_points)
            all_colors.append(valid_colors)
            all_confidences.append(valid_conf)
            all_view_indices.append(view_indices)

            # Extract and map features for this view (memory-efficient: one at a time)
            if extract_features:
                # Extract features for just this image
                image_path = image_paths[view_idx]
                extractor = self._feature_extractor

                # Extract feature map for this single image
                if isinstance(extractor, MaskCLIPExtractor):
                    img_tensor = extractor.preprocess(image_path, resolution=feature_resolution)
                    img_batch = img_tensor.unsqueeze(0)
                    feat = extractor(img_batch).squeeze(0)  # (D, H_p, W_p)
                    target_h, target_w = img_tensor.shape[-2], img_tensor.shape[-1]
                    feat = F.interpolate(
                        feat.unsqueeze(0),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)  # (D, H, W)

                elif isinstance(extractor, DINOFeatureExtractor):
                    img_batch, target_h, target_w = extractor.preprocess(image_path)
                    tokens = extractor(img_batch)
                    feat = extractor.reshape(tokens, target_h, target_w)  # (D, H_p, W_p)
                    feat = F.interpolate(
                        feat.unsqueeze(0),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)  # (D, H, W)

                else:
                    raise ValueError(f"Unsupported extractor type: {type(extractor)}")

                # Upsample feature map to match the depth/image grid resolution
                feat_upsampled = F.interpolate(
                    feat.unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # (D, H, W)

                # Sample features at valid pixel locations
                feat_spatial = feat_upsampled.permute(1, 2, 0).cpu().numpy()  # (H, W, D)
                valid_features = feat_spatial[mask]  # (N_valid, D)
                all_raw_features.append(valid_features)

                # Clear GPU memory
                del feat, feat_upsampled, feat_spatial
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"  View {view_idx}: {len(valid_points):,} valid points")

        # Concatenate all views
        pointcloud = {
            "points": np.concatenate(all_points, axis=0),
            "colors": np.concatenate(all_colors, axis=0),
            "confidences": np.concatenate(all_confidences, axis=0),
            "view_indices": np.concatenate(all_view_indices, axis=0),
        }

        # Fit PCA and project features if extracted
        if extract_features and all_raw_features:
            # Concatenate and ensure float32 for memory efficiency
            all_feat_np = np.concatenate(all_raw_features, axis=0).astype(np.float32)  # (N_total, D)
            del all_raw_features  # Free memory immediately
            pytorch_gc()

            actual_components = min(n_components, all_feat_np.shape[0], all_feat_np.shape[1])

            n_points = all_feat_np.shape[0]
            feat_dim = all_feat_np.shape[1]
            print(f"Fitting IncrementalPCA: {feat_dim}-d → {actual_components}-d on {n_points:,} points...")
            print(f"  Memory usage: ~{(n_points * feat_dim * 4 / 1e9):.2f} GB for features")

            # Use smaller batch size to avoid memory spikes
            batch_size = min(5000, max(500, n_points // 20))  # More conservative batch size
            pca = IncrementalPCA(n_components=actual_components, batch_size=batch_size)

            # Fit in batches with progress bar
            n_batches = (n_points + batch_size - 1) // batch_size
            with tqdm(total=n_points, desc="  PCA fitting", unit="pts", unit_scale=True) as pbar:
                for i in range(0, n_points, batch_size):
                    batch = all_feat_np[i:i + batch_size]
                    pca.partial_fit(batch)
                    pbar.update(len(batch))
                    del batch  # Free batch memory

            pytorch_gc()  # Clean up after fitting

            # Transform in batches (memory-efficient) with progress bar
            projected_batches = []
            with tqdm(total=n_points, desc="  PCA transform", unit="pts", unit_scale=True) as pbar:
                for i in range(0, n_points, batch_size):
                    batch = all_feat_np[i:i + batch_size]
                    projected_batches.append(pca.transform(batch))
                    pbar.update(len(batch))
                    del batch  # Free batch memory

            del all_feat_np  # Free original features array
            projected = np.concatenate(projected_batches, axis=0)
            del projected_batches  # Free intermediate list
            pytorch_gc()  # Final cleanup

            # L2-normalise so dot-product == cosine similarity at query time
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            projected = projected / norms

            self._pca = pca

            pointcloud["features"] = projected.astype(np.float32)
            pointcloud["feature_extractor"] = extractor_name
            pointcloud["feature_dim"] = actual_components

            explained = pca.explained_variance_ratio_.sum() * 100
            print(f"✓ IncrementalPCA complete: {explained:.1f}% variance retained in {actual_components} components")

        print(f"✓ Pointcloud created")
        print(f"  • Total points: {len(pointcloud['points']):,}")
        print(f"  • Points per view (avg): {len(pointcloud['points']) / S:,.0f}")
        if confidence_threshold is not None:
            print(f"  • Confidence threshold: {confidence_threshold}")
        if subsample_factor > 1:
            print(f"  • Subsample factor: {subsample_factor}")
        if extract_features:
            print(f"  • Feature dim: {pointcloud['feature_dim']} (extractor: {extractor_name})")

        return pointcloud

    def query_pointcloud(
        self,
        pointcloud: Dict[str, np.ndarray],
        positive: List[str],
        negative: Optional[List[str]] = None,
        softmax_temp: float = 0.05,
        method: str = "standard",
    ) -> np.ndarray:
        """Query a CLIP-featured pointcloud with text prompts.

        Only works when the pointcloud was created with a CLIP-based feature extractor
        (e.g. "maskclip"). Text embeddings are projected through the same PCA that was
        fitted during pointcloud creation, ensuring alignment with per-point features.

        Args:
            pointcloud: Pointcloud dict returned by feedforward_to_pointcloud() with features.
                        Must have keys: "features", "feature_extractor", "feature_dim".
            positive: List of positive text queries (e.g. ["chair", "furniture"]).
            negative: List of negative text queries (e.g. ["background", "floor"]).
                      Defaults to ["object"] if None.
            softmax_temp: Temperature for softmax. Lower values make the distribution peakier.
            method: Similarity aggregation method. "standard" sums positive probabilities after
                    softmax over all queries. "pairwise" computes pairwise softmax between
                    averaged positive and each negative, then takes the minimum.

        Returns:
            (N,) float32 array of similarity scores in [0, 1] for each point.

        Raises:
            ValueError: If pointcloud lacks features, or if the extractor is not CLIP-based.
            RuntimeError: If PCA was not fitted during feature extraction.

        Example:
            >>> pcd = rec.feedforward_to_pointcloud(feature_extractor="maskclip", n_components=64)
            >>> scores = rec.query_pointcloud(pcd, positive=["chair"], negative=["floor"])
            >>> chair_points = pcd["points"][scores > 0.7]
        """
        # Validate pointcloud structure
        if "features" not in pointcloud:
            raise ValueError(
                "Pointcloud does not contain features. "
                "Call feedforward_to_pointcloud() with a feature_extractor argument."
            )
        if "feature_extractor" not in pointcloud:
            raise ValueError("Pointcloud missing 'feature_extractor' metadata key.")

        # Validate extractor is CLIP-based
        extractor_name = pointcloud["feature_extractor"]
        if not isinstance(self._feature_extractor, MaskCLIPExtractor):
            raise ValueError(
                f"query_pointcloud requires a CLIP-based extractor (e.g. 'maskclip'). "
                f"Got: {extractor_name} ({type(self._feature_extractor).__name__})"
            )

        # Validate PCA exists
        if self._pca is None:
            raise RuntimeError(
                "PCA not found. Ensure feedforward_to_pointcloud() with features was called "
                "before querying."
            )

        # Default negative queries
        if negative is None:
            negative = ["object"]

        # Encode text with CLIP
        all_queries = positive + negative
        text_embed_raw = self._feature_extractor.encode_text(all_queries)  # (num_queries, D_clip)
        text_embed_raw_np = text_embed_raw.cpu().numpy()

        # Project text embeddings through the same PCA used for point features
        text_embed_pca = self._pca.transform(text_embed_raw_np)  # (num_queries, n_components)

        # L2-normalise so dot product == cosine similarity
        text_norms = np.linalg.norm(text_embed_pca, axis=1, keepdims=True)
        text_norms = np.where(text_norms == 0, 1.0, text_norms)
        text_embed_pca = text_embed_pca / text_norms

        # Get normalised point features (already normalised during feedforward_to_pointcloud)
        point_features = pointcloud["features"]  # (N, n_components), already L2-normalised

        # Compute raw similarities: (N, num_queries)
        raw_similarities = point_features @ text_embed_pca.T  # cosine similarity

        # Apply method-specific aggregation
        num_positive = len(positive)

        if method == "standard":
            # Softmax over all queries, sum positive probabilities
            probs = np.exp(raw_similarities / softmax_temp)
            probs = probs / probs.sum(axis=1, keepdims=True)  # (N, num_queries)
            similarity = probs[:, :num_positive].sum(axis=1)  # (N,)

        elif method == "pairwise":
            # Pairwise softmax: average positive vs each negative
            pos_similarities = raw_similarities[:, :num_positive]  # (N, num_positive)
            neg_similarities = raw_similarities[:, num_positive:]  # (N, num_negative)

            # Average positive similarities
            avg_pos_similarity = pos_similarities.mean(axis=1, keepdims=True)  # (N, 1)

            # Broadcast to match negative shape
            broadcasted_pos = np.tile(avg_pos_similarity, (1, neg_similarities.shape[1]))  # (N, num_negative)

            # Stack pairs: positive vs each negative
            paired_similarities = np.stack([broadcasted_pos, neg_similarities], axis=-1)  # (N, num_negative, 2)

            # Pairwise softmax
            probs = np.exp(paired_similarities / softmax_temp)
            probs = probs / probs.sum(axis=-1, keepdims=True)  # (N, num_negative, 2)

            # Extract positive probabilities and take minimum across pairs
            pos_pair_probs = probs[:, :, 0]  # (N, num_negative)
            similarity = pos_pair_probs.min(axis=1)  # (N,)

            # Handle NaN values
            similarity = np.nan_to_num(similarity, nan=0.0)

        else:
            raise ValueError(
                f"Unknown method: {method}. Choose 'standard' or 'pairwise'."
            )

        return similarity.astype(np.float32)

    def save_pointcloud(
        self,
        output_path: Union[str, Path],
        pointcloud: Optional[Dict[str, np.ndarray]] = None,
        confidence_threshold: Optional[float] = None,
        subsample_factor: int = 1,
    ) -> None:
        """Save pointcloud to PLY file.

        Args:
            output_path: Path to save the pointcloud (.ply extension)
            pointcloud: Optional pre-computed pointcloud. If None, will compute it.
            confidence_threshold: Optional confidence threshold (only used if pointcloud is None)
            subsample_factor: Subsample factor (only used if pointcloud is None)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate pointcloud if not provided
        if pointcloud is None:
            pointcloud = self.feedforward_to_pointcloud(
                confidence_threshold=confidence_threshold,
                subsample_factor=subsample_factor
            )

        points = pointcloud["points"]
        colors = pointcloud["colors"]

        # Ensure colors are in 0-255 range and uint8
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

        # Write PLY file
        print(f"Saving pointcloud to {output_path}")

        with open(output_path, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Data
            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {r} {g} {b}\n")

        print(f"✓ Pointcloud saved: {len(points):,} points")

    ### TO-DO: Implement pointcloud refinement
    # 1. Backproject points to world based on depth ✓
    # 2. Filter points based on confidence ✓
    # 3. Bundle adjustment?
    # MIGHT WANT TO CONSIDER USING COLMAP HERE