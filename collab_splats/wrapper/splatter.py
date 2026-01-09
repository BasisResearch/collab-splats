import os
import glob
import json
import pickle
import subprocess
from pathlib import Path
from typing import Optional, TypedDict, Set, Dict, Any, Union, List
import cv2
import torch
import numpy as np
import pyvista as pv
import open3d as o3d

DEFAULT_TIMEOUT = 3600


class SplatterConfig(TypedDict):
    """Configuration for the Splatter class.

    Required Keys:
        file_path: Path to the input file for processing (e.g. video, images, etc.)
        dtype: Specifies if the data is 2D or 3D
        method: Processing method to use (different methods for 2D and 3D)

    Optional Keys:
        output_path: Path for output data
            - preproc: preprocessed images
            - model_ckpts: derived model images
            - If output_path is not specified, will default to the grandparent directory of the input file
        overwrite: If True, will rerun preprocessing even if transforms.json exists
    """

    file_path: Union[str, Path]
    method: str
    output_path: Optional[Union[str, Path]]
    frame_proportion: Optional[float]
    min_frames: Optional[int]
    websocket_port: Optional[int]


class ValidationError(Exception):
    """Raised when environment configuration is invalid."""

    pass


class Splatter:
    # Valid processing methods for each dtype

    SPLATTING_METHODS: Set[str] = {
        "splatfacto",
        "feature-splatting",
        "rade-gs",
        "rade-features",
    }

    def __init__(self, config: SplatterConfig):
        """Initialize the splatter with configuration.

        Args:
            config: Configuration dictionary specifying environment parameters
        """
        # Validate config before initialization
        validated_config = self.validate_config(config)
        self.config: Dict[str, Any] = dict(validated_config)

        # Optional pipeline configs (set by from_config_file)
        self._preprocess_config: Optional[Dict[str, Any]] = None
        self._training_config: Optional[Dict[str, Any]] = None
        self._meshing_config: Optional[Dict[str, Any]] = None

    @classmethod
    def validate_config(cls, config: SplatterConfig) -> SplatterConfig:
        """Validate the splatter configuration.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValidationError: If the configuration is invalid
        """

        ############################################
        ######### Check fields of config ###########
        ############################################

        required_fields = {"file_path", "method"}
        missing_fields = required_fields - set(config.keys())

        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")

        # Validate method based on dtype
        valid_methods = cls.SPLATTING_METHODS
        if config["method"] not in valid_methods:
            raise ValidationError(
                f"Invalid method '{config['method']}'. "
                f"Valid methods are: {sorted(valid_methods)}"
            )

        ############################################
        ############ Set up file paths #############
        ############################################

        # Set the file path -> turn to a Path object for easy structuring
        file_path = Path(config["file_path"])

        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")

        # If so, set the file path
        config["file_path"] = file_path

        # If we don't specify an output path, default to the grandparent directory of the input file
        if config.get("output_path") is None:
            default_output_path = os.path.join(
                file_path.parent.parent, "environment", file_path.stem
            )
            config.setdefault("output_path", Path(default_output_path))

        if config.get("min_frames") is None:
            config.setdefault(
                "min_frames", 300
            )  # Default number of video frames to use for COLMAP

        return config

    @classmethod
    def available_methods(cls) -> None:
        """Print the available methods.

        Args:
        """
        print("Available methods:")
        print("  ", sorted(cls.SPLATTING_METHODS))

    @classmethod
    def from_config_file(
        cls,
        dataset: str,
        config_dir: Union[str, Path],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "Splatter":
        """
        Create Splatter instance from YAML configuration.

        Args:
            dataset: Dataset config name (from datasets/ subdirectory)
            config_dir: Directory containing config files (base.yaml and datasets/)
            overrides: Optional runtime overrides

        Returns:
            Configured Splatter instance with pipeline configs attached

        Example:
            >>> splatter = Splatter.from_config_file(
            ...     dataset='ants_001',
            ...     config_dir='docs/splats/configs'
            ... )
            >>> splatter.run_pipeline(overwrite=True)
        """
        from collab_splats.wrapper.config import ConfigLoader

        loader = ConfigLoader(config_dir)
        config = loader.load(dataset=dataset, overrides=overrides)

        # Store full config for later use
        full_config = config.copy()

        # Extract SplatterConfig fields
        splatter_fields: Dict[str, Any] = {
            "file_path": config["file_path"],
            "method": config["method"],
        }
        # Add optional fields if present
        if "frame_proportion" in config:
            splatter_fields["frame_proportion"] = config["frame_proportion"]
        if "min_frames" in config:
            splatter_fields["min_frames"] = config["min_frames"]
        if "output_path" in config:
            splatter_fields["output_path"] = config["output_path"]

        splatter_config: SplatterConfig = splatter_fields  # type: ignore
        instance = cls(splatter_config)

        # Attach configs for pipeline methods
        instance._preprocess_config = full_config.get("preprocess", {})
        instance._training_config = full_config.get("training", {})
        instance._meshing_config = full_config.get("meshing", {})

        return instance

    def run_pipeline(self, overwrite: bool = False) -> None:
        """
        Run complete pipeline using stored configurations.

        This method runs preprocessing, training, and meshing using
        configurations loaded via from_config_file().

        Args:
            overwrite: Whether to overwrite existing outputs

        Raises:
            ValueError: If pipeline configs not found (must use from_config_file)
        """
        if self._preprocess_config is None:
            raise ValueError(
                "Pipeline configs not found. Use Splatter.from_config_file() "
                "to load configurations before calling run_pipeline()"
            )

        print(f"\n{'=' * 80}")
        print(f"Running {self.config['method']} pipeline")
        print(f"File: {Path(self.config['file_path']).name}")
        print(f"{'=' * 80}\n")

        # Step 1: Preprocessing
        print("[1/3] Preprocessing...")
        self.preprocess(overwrite=overwrite)

        # Step 2: Training --> uses _training_config by default
        print("\n[2/3] Training...")
        self.extract_features(overwrite=overwrite)

        # Step 3: Meshing --> uses _meshing_config by default
        print("\n[3/3] Meshing...")
        self.mesh(overwrite=overwrite)

        print(f"\n{'=' * 80}")
        print("Pipeline complete!")
        print(f"{'=' * 80}\n")

    def preprocess(
        self,
        overwrite: bool = False,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Preprocess the data in the splatter.

        This function handles any necessary data preprocessing steps based on the
        configured method.

        Args:
            overwrite: If True, rerun preprocessing even if transforms.json exists
            kwargs: Additional arguments to pass to ns-process-data
        """
        file_path = self.config["file_path"]
        output_path = self.config["output_path"]

        assert self._preprocess_config is not None, "Preprocess config not found"

        preprocess_config = (self._preprocess_config or {}).copy()
        
        if kwargs is not None:
            preprocess_config.update(kwargs)

        # Determine input type based on file extension
        ext = file_path.suffix.lower()
        if ext in [".mp4", ".mov", ".avi"]:
            input_type = "video"
        elif ext in [".jpg", ".jpeg", ".png"]:
            if "360" in str(file_path):
                input_type = (
                    "images --camera-type equirectangular --images-per-equirect 14"
                )
            else:
                input_type = "images"
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Set the output path to same directory as input fil
        preproc_data_path = output_path / "preproc"
        transforms_path = preproc_data_path / "transforms.json"

        # If the transforms exists and we don't want to overwrite
        # Return and store the processed_data_path
        if transforms_path.exists() and not overwrite:
            print(f"transforms.json already exists at {transforms_path}")
            print("To rerun preprocessing, set overwrite=True")
            self.config["preproc_data_path"] = preproc_data_path
            return

        if self.config.get("frame_proportion") is not None:
            video_capture = cv2.VideoCapture(file_path.as_posix())
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            n_samples = int(n_frames * self.config["frame_proportion"])

            # If we have less than the minimum number of frames, as many as possible
            n_samples = n_frames if n_samples < self.config["min_frames"] else n_samples

            print("Number of frames to sample: ", n_samples)

            # Create the command
            num_frames_target = f"--num-frames-target {n_samples}"
        else:
            num_frames_target = ""

        # TLB --> we should bump up number of frames to max
        cmd = (
            f"ns-process-data "
            f"{input_type} "
            f"--data {file_path.as_posix()} "
            f"--output-dir {preproc_data_path.as_posix()} "
            f"{num_frames_target} "
        )

        # Use preprocess config for ns-process-data
        kwargs_cmds = " ".join([f"--{k} {v}" for k, v in preprocess_config.items()])
        cmd += kwargs_cmds

        subprocess.run(cmd, shell=True)

        # Store the preprocessed data path in the config
        self.config["preproc_data_path"] = preproc_data_path

    def extract_features(
        self, overwrite: bool = False, kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Extract features from the preprocessed data.

        Feature extraction is performed according to the configured dtype and method.
        """
        method = self.config["method"]

        training_config = (self._training_config or {}).copy()
        
        if kwargs is not None:
            training_config.update(kwargs)

        # Check that preprocessing was completed before extracting features
        if self.config["preproc_data_path"] is None:
            raise ValueError("preprocess_data() must be run before extracting features")

        # Set the model path (where it outputs results of model training)
        model_path = self.config["output_path"] / method
        model_exists = any(model_path.glob("**/*.ckpt"))

        if model_exists and not overwrite:
            print(f"Output already exists for {method}")
            print("To rerun feature extraction, set overwrite=True")
            self.config["model_path"] = model_path
            return

        cmd = (
            f"ns-train "
            f"{method} "
            f"--data {self.config['preproc_data_path'].as_posix()} "
            f"--output-dir {self.config['output_path'].as_posix()} "
            f"--experiment-name '' "  # This keeps our file structure as environment/BASE_NAME/method/
            "--viewer.quit-on-train-completion True "  # This quits the function once training is complete
        )

        # Use training config for ns-train
        kwargs_cmds = " ".join([f"--{k} {v}" for k, v in training_config.items()])
        cmd += kwargs_cmds

        self.config["model_path"] = model_path
        subprocess.run(cmd, shell=True)

    def viewer(self, use_latest_run: bool = False) -> None:
        """Display or visualize the splatter data.

        This function handles feature extraction from the preprocessed data.
        The specific feature extraction pipeline depends on config['dtype']:
            - 2D: Image-based feature extraction
            - 3D: Volume-based feature extraction

        Args:
            use_latest_run: If True, automatically select the most recent run without prompting
        """

        self._select_run(use_latest_run=use_latest_run)

        cmd = f"ns-viewer --load-config {self.config['model_config_path']} "

        if self.config.get("websocket_port") is not None:
            cmd += f"--websocket-port {self.config['websocket_port']} "

        subprocess.run(cmd, shell=True, timeout=DEFAULT_TIMEOUT)

    def _select_run(self, use_latest_run: bool = False) -> None:
        """Select a run from the available runs.

        Args:
            use_latest_run: If True, automatically select the most recent run without prompting
        """
        # Find all runs with config.yml files
        output_dir = Path(str(self.config["output_path"]), self.config["method"])

        # Grab all directories with a config.yml file --> convert to paths
        run_dirs_glob = glob.glob(os.path.join(output_dir, "**/config.yml"))
        run_dirs: List[Path] = [Path(run_dir).parent for run_dir in run_dirs_glob]

        if not run_dirs:
            raise ValueError(f"No runs with config.yml found in {output_dir}")

        # Sort runs by directory name (which contains timestamp)
        sorted_runs = sorted(run_dirs)

        # Print available runs
        print("\nAvailable runs:")
        for i, run in enumerate(sorted_runs):
            print(f"[{i}] {run.name}")

        if len(sorted_runs) == 1 or use_latest_run:
            # Automatically select the most recent run
            selected_run = sorted_runs[-1]
            if use_latest_run:
                print(f"\nUsing latest run: {selected_run.name}")
        else:
            # Prompt user to select a run
            while True:
                try:
                    selection = input(
                        "\nSelect run number (or press Enter for most recent): "
                    ).strip()
                    if selection == "":
                        selected_run = sorted_runs[-1]
                        break
                    idx = int(selection)
                    if 0 <= idx < len(sorted_runs):
                        selected_run = sorted_runs[idx]
                        break
                    print(f"Please enter a number between 0 and {len(sorted_runs) - 1}")
                except ValueError:
                    print("Please enter a valid number")

        self.config["model_path"] = selected_run.as_posix()
        self.config["model_config_path"] = (selected_run / "config.yml").as_posix()

    def load_model(
        self,
        config_path: Optional[Union[str, Path]] = None,
        test_mode: str = "inference",
        use_latest_run: bool = False,
    ):
        """
        Load a trained nerfstudio model.

        Args:
            config_path: Path to config.yml. If None, uses model_config_path from config or prompts selection
            test_mode: Evaluation mode - "test", "val", or "inference" (default)
            use_latest_run: If True, automatically select the most recent run without prompting

        Returns:
            Tuple of (config, pipeline, model)

        Example:
            >>> splatter = Splatter(config)
            >>> config, pipeline, model = splatter.load_model("outputs/scene/rade-gs/config.yml")
            >>> outputs = model.get_outputs(camera)
        """
        from collab_splats.utils import load_checkpoint

        # Determine config path
        if config_path is None:
            if not self.config.get("model_config_path"):
                self._select_run(use_latest_run=use_latest_run)
            config_path = self.config["model_config_path"]

        print(f"Loading model from {config_path}")

        # Load using utility function
        config, pipeline, checkpoint_path, step = load_checkpoint(
            config_path, test_mode=test_mode
        )

        # Store in instance
        self.model = pipeline.model
        self.pipeline = pipeline
        self.model_config = config
        self.checkpoint_path = checkpoint_path
        self.training_step = step

        print(f"✓ Model loaded: {type(self.model).__name__} (step {step})")

        return config, pipeline, self.model

    def mesh(
        self,
        kwargs: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """Generate a mesh from the splatter data.

        This function handles mesh generation from the preprocessed data.
        If a pipeline is already loaded (via load_model), it will be passed
        to the mesher to avoid duplicate loading and reduce memory usage.
        """
        mesher_config = (self._meshing_config or {}).copy()
        use_latest_run = mesher_config.pop("use_latest_run", False)
        mesher_type = mesher_config.pop("mesher_type", "Open3DTSDFFusion")

        if kwargs is not None:
            # Extract use_latest_run from kwargs before updating mesher_config
            use_latest_run = kwargs.pop("use_latest_run", use_latest_run)
            mesher_config.update(kwargs)

        self._select_run(use_latest_run=use_latest_run)

        # Save mesh under the selected run directory
        mesh_dir = Path(self.config["model_path"]) / "mesh"

        # Create the mesh
        if not mesh_dir.exists() or overwrite:
            from collab_splats.utils import mesh

            print(f"Initializing mesher {mesher_type}")

            
            # If pipeline is already loaded, pass it to avoid duplicate loading
            # This significantly reduces memory usage
            if hasattr(self, "pipeline") and self.pipeline is not None:
                print("Using pre-loaded pipeline (memory-efficient mode)")
                mesher = getattr(mesh, mesher_type)(output_dir=mesh_dir,
                    pipeline=self.pipeline,
                    **mesher_config,
                )
            else:
                print("Loading pipeline from config")
                mesher = getattr(mesh, mesher_type)(
                    load_config=Path(self.config["model_config_path"]),
                    output_dir=mesh_dir,
                    **mesher_config,
                )

            self.config["mesh_info"] = mesher.main()
        else:
            # HARDCODING FOR NOW TLB FIX
            self.config["mesh_info"] = {
                "mesh": mesh_dir / "mesh.ply",
                "features": mesh_dir / "mesh_features.pt",
            }

    def query_mesh(
        self,
        positive_queries: List[str] = [""],
        negative_queries: List[str] = ["object"],
        method: str = "pairwise",
        output_fn: Optional[str] = None,
    ) -> None:
        """Query the mesh for features."""

        if getattr(self, "model", None) is None:
            self.load_model()

        mesh_info = self.config.get("mesh_info")
        if mesh_info is None:
            raise ValueError("Mesh information not found. Please run mesh() first.")
        elif mesh_info.get("features") is None:
            raise ValueError(
                "Features not found. Please run mesh() with features_name specified."
            )

        features = torch.load(self.config["mesh_info"]["features"])

        decoded_features = self.model.decoder.per_gaussian_forward(
            features.to(self.model.device).to(torch.float32)
        )

        similarity_map = (
            self.model.similarity_fx(
                features=decoded_features[self.model.main_features_name]
                .unsqueeze(0)
                .permute(2, 1, 0),
                positive=positive_queries,
                negative=negative_queries,
                method=method,
            )
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )

        del features

        if output_fn is not None:
            output_dir = self.config["mesh_info"]["mesh"].parent
            output_path = output_dir / output_fn

            # # Map to open3d format
            # if similarity_map.ndim == 1:
            #     similarity_cast = similarity_map[:, np.newaxis]

            # Normalize and pad to RGB
            similarity_colors = np.zeros((len(similarity_map), 3))
            similarity_cast = similarity_map.astype(np.float64)
            if np.max(similarity_cast) > 0:
                similarity_cast /= np.max(similarity_cast)

            # Map it to colors
            similarity_colors[:, : similarity_map.shape[1]] = similarity_cast

            # Load the mesh and add the similarity map as a vertex color
            mesh = o3d.io.read_triangle_mesh(self.config["mesh_info"]["mesh"])
            mesh.vertex_colors = o3d.utility.Vector3dVector(similarity_colors)
            o3d.io.write_triangle_mesh(output_path, mesh)

        return similarity_colors

    def plot_mesh(
        self, attribute: Optional[Union[str, np.ndarray]] = None, rgb: bool = True
    ) -> None:
        """Plot the mesh."""
        mesh_info = self.config.get("mesh_info")
        if mesh_info is None:
            raise ValueError("Mesh information not found. Please run mesh() first.")
        elif mesh_info.get("mesh") is None:
            raise ValueError("Mesh not found. Please run mesh() first.")

        mesh_path = self.config["mesh_info"]["mesh"]
        mesh = pv.read(mesh_path)

        # Print basic information about the mesh
        print(f"Number of points: {mesh.n_points}")
        print(f"Number of cells: {mesh.n_cells}")
        print(f"Bounds: {mesh.bounds}")
        # Create a plotter and add the mesh
        mesh.plot(scalars=attribute, rgb=rgb)

    #########################################################
    ############ Load mesh transforms / cameras #############
    #########################################################

    def load_mesh_transform(self):
        mesh_dir = self.config["mesh_info"]["mesh"].parent
        mesh_transform_fn = mesh_dir / "transforms.pkl"
        with open(mesh_transform_fn, "rb") as f:
            mesh_transform = pickle.load(f)
        return mesh_transform

    def load_aligned_cameras(self, align_mesh: bool = False):
        """
        Load the colmap cameras and align them to the splat or mesh (if specified).
        """

        # Get the preproc and model directories
        preproc_dir = Path(self.config["preproc_data_path"])
        model_dir = Path(self.config["model_path"])

        # Load the transforms
        transforms_json = preproc_dir / "transforms.json"
        nerfstudio_transforms_path = (
            model_dir / "dataparser_transforms.json"
        )  # This is the nerfstudio transform

        # Camera transforms (poses)
        with open(transforms_json, "r") as f:
            transforms = json.load(f)

        # Nerfstudio transforms aligned to camera
        with open(nerfstudio_transforms_path, "r") as f:
            nerfstudio_transforms = json.load(f)

        # Compose into 4x4 matrix
        transform = np.stack(nerfstudio_transforms["transform"])

        # Add the translation to the transform
        transform = np.concatenate(
            [transform, np.array([0, 0, 0, 1])[np.newaxis]], axis=0
        )

        # Apply to cameras
        camera_poses = np.stack(
            [f["transform_matrix"] for f in transforms["frames"]]
        )  # Load from your camera pose files or nerfstudio transforms
        camera_poses[..., :3, 3] *= nerfstudio_transforms["scale"]

        if align_mesh:
            mesh_transform = self.load_mesh_transform()
            camera_poses = mesh_transform["mesh_transform"] @ camera_poses

        return camera_poses
