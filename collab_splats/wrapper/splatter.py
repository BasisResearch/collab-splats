import os
import glob
import json
import pickle
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Literal, TypedDict, Set, Dict, Any, Union, List
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pyvista as pv
import open3d as o3d
from collab_splats.semantics.features import BaseFeatureExtractor
from collab_splats.geometry.transforms import extrinsics_to_homogeneous
from collab_splats.preproc.qa import load_video_quality
from collab_splats.preproc.sampling import sample_optical_flow
from collab_splats.utils.torch_utils import get_device
from nerfstudio.utils.eval_utils import eval_setup

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
    pointcloud_method: Optional[str]  # "sfm" | "feedforward"; default "sfm"
    frame_selection: Optional[Literal["fps", "optical_flow"]]  # default "fps"


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
        warnings.warn(
            "Splatter is deprecated and will be removed in a future release. "
            "Use collab_splats.wrapper.Reconstructor instead with pointcloud.method='nerfstudio'.",
            DeprecationWarning,
            stacklevel=2,
        )
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
                f"Invalid method '{config['method']}'. " f"Valid methods are: {sorted(valid_methods)}"
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
            default_output_path = os.path.join(file_path.parent.parent, "environment", file_path.stem)
            config.setdefault("output_path", Path(default_output_path))

        if config.get("min_frames") is None:
            config.setdefault("min_frames", 300)  # Default number of video frames to use for COLMAP

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
        if "pointcloud_method" in config:
            splatter_fields["pointcloud_method"] = config["pointcloud_method"]

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
        self.preprocess(kwargs=self._preprocess_config, overwrite=overwrite)

        # Step 2: Training
        print("\n[2/3] Training...")
        self.extract_features(kwargs=self._training_config, overwrite=overwrite)

        # Step 3: Meshing
        print("\n[3/3] Meshing...")
        mesher_config = (self._meshing_config or {}).copy()
        mesher_type = mesher_config.pop("mesher_type", "Open3DTSDFFusion")
        self.mesh(mesher_type=mesher_type, mesher_kwargs=mesher_config, overwrite=overwrite)

        print(f"\n{'=' * 80}")
        print("Pipeline complete!")
        print(f"{'=' * 80}\n")

    def preprocess(self, overwrite: bool = False, kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Preprocess the data in the splatter.

        This function handles any necessary data preprocessing steps based on the
        configured method.
        """
        file_path = self.config["file_path"]
        output_path = self.config["output_path"]

        # Determine input type based on file extension
        ext = file_path.suffix.lower()
        if ext in [".mp4", ".mov", ".avi"]:
            input_type = "video"
        elif ext in [".jpg", ".jpeg", ".png"]:
            if "360" in str(file_path):
                input_type = "images --camera-type equirectangular --images-per-equirect 14"
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

            # Create the command
            num_frames_target = f"--num-frames-target {n_samples}"
        else:
            num_frames_target = ""

        # If optical_flow frame selection is requested, pre-extract frames and redirect
        # ns-process-data to images mode using the sampled frame directory.
        if input_type == "video" and self.config.get("frame_selection") == "optical_flow":
            tmp_dir = Path(self.config["output_path"]) / "tmp_frames"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # Determine target frame count (mirrors frame_proportion logic above)
            video_capture = cv2.VideoCapture(file_path.as_posix())
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_capture.release()
            if self.config.get("frame_proportion") is not None:
                n_samples = int(n_frames * self.config["frame_proportion"])
                n_samples = n_frames if n_samples < self.config["min_frames"] else n_samples
            else:
                n_samples = n_frames

            # Measure once into the scene's tmp dir, then select from the report
            report = load_video_quality(file_path, tmp_dir / "video_quality_report.json")

            sampled_frames, _ = sample_optical_flow(file_path.as_posix(), max_frames=min(n_samples, 200), report=report)

            for i, frame in enumerate(sampled_frames):
                cv2.imwrite(str(tmp_dir / f"{i:05d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Switch to images mode pointing at the pre-extracted frame directory
            input_type = "images"
            data_source = tmp_dir.as_posix()
            num_frames_target = ""  # already sampled exactly what we need
        else:
            data_source = file_path.as_posix()

        # TLB --> we should bump up number of frames to max
        cmd = (
            f"ns-process-data "
            f"{input_type} "
            f"--data {data_source} "
            f"--output-dir {preproc_data_path.as_posix()} "
            f"{num_frames_target} "
        )

        if kwargs is not None:
            kwargs_cmds = " ".join([f"--{k} {v}" for k, v in kwargs.items()])
            cmd += kwargs_cmds

        subprocess.run(cmd, shell=True)

        # Store the preprocessed data path in the config
        self.config["preproc_data_path"] = preproc_data_path

    def extract_features(self, overwrite: bool = False, kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Extract features from the preprocessed data.

        Feature extraction is performed according to the configured dtype and method.
        """
        method = self.config["method"]

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

        if kwargs is not None:
            kwargs_cmds = " ".join([f"--{k} {v}" for k, v in kwargs.items()])
            cmd += kwargs_cmds

        self.config["model_path"] = model_path
        subprocess.run(cmd, shell=True)

    def viewer(self) -> None:
        """Display or visualize the splatter data.

        This function handles feature extraction from the preprocessed data.
        The specific feature extraction pipeline depends on config['dtype']:
            - 2D: Image-based feature extraction
            - 3D: Volume-based feature extraction
        """

        self._select_run()

        cmd = f"ns-viewer --load-config {self.config['model_config_path']} "

        if self.config.get("websocket_port") is not None:
            cmd += f"--websocket-port {self.config['websocket_port']} "

        subprocess.run(cmd, shell=True, timeout=DEFAULT_TIMEOUT)

    def _select_run(self) -> None:
        """Select a run from the available runs."""
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

        if len(sorted_runs) == 1:
            selected_run = sorted_runs[0]
        else:
            # Prompt user to select a run
            while True:
                try:
                    selection = input("\nSelect run number (or press Enter for most recent): ").strip()
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
                except EOFError:
                    selected_run = sorted_runs[-1]
                    print(f"Non-interactive mode — selecting most recent: {selected_run.name}")
                    break

        self.config["model_path"] = selected_run.as_posix()
        self.config["model_config_path"] = (selected_run / "config.yml").as_posix()

    def mesh(
        self,
        mesher_type: str = "Open3DTSDFFusion",
        mesher_kwargs: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """Generate a mesh from the splatter data.

        This function handles mesh generation from the preprocessed data.
        """
        self._select_run()

        mesh_dir = self.config["output_path"] / self.config["method"] / "mesh"

        # Create the mesh
        if not mesh_dir.exists() or overwrite:
            from collab_splats.mesh import get_mesh_creator
            from collab_splats.nerfstudio.utils.mesh_adapter import extract_mesh_inputs

            print(f"Initializing mesher {mesher_type}")

            mesher_kwargs = mesher_kwargs or {}
            depth_name = mesher_kwargs.pop("depth_name", "median_depth")
            features_name = mesher_kwargs.pop("features_name", None)
            mesher_kwargs.pop("normals_name", None)
            mesher_kwargs.pop("align_floor", None)
            depths, rgbs, c2w, intrinsics = extract_mesh_inputs(
                load_config=Path(self.config["model_config_path"]),
                depth_name=depth_name,
            )
            creator = get_mesh_creator(mesher_type, output_dir=mesh_dir, **mesher_kwargs)
            result = creator.create(depths, rgbs, c2w, intrinsics)
            self.config["mesh_info"] = {"mesh": result.mesh_path}
            if features_name is not None:
                self._extract_mesh_features(features_name=features_name)
            self._export_gaussian_splats(mesh_dir, overwrite=overwrite)
            splats_path = mesh_dir / "splats.ply"
            if splats_path.exists():
                self.config["mesh_info"]["splats"] = splats_path
        else:
            # mesh.ply is the only name any mesher in this repo writes; clean_repair rewrites
            # that same file in place rather than producing a mesh_clean.ply (mesh/tsdf.py).
            mesh_path = mesh_dir / "mesh.ply"
            if not mesh_path.exists():
                raise FileNotFoundError(
                    f"No mesh found in {mesh_dir}. Re-run mesh() with overwrite=True."
                )
            self.config["mesh_info"] = {"mesh": mesh_path}
            features_path = mesh_dir / "mesh_features.pt"
            if features_path.exists():
                self.config["mesh_info"]["features"] = features_path
            splats_path = mesh_dir / "splats.ply"
            if splats_path.exists():
                self.config["mesh_info"]["splats"] = splats_path
            # Bootstrap decoder for datasets created before mesh_decoder.pt was introduced
            decoder_path = mesh_dir / "mesh_decoder.pt"
            if self.config["mesh_info"].get("features") and not decoder_path.exists():
                self._save_mesh_decoder()

    def _save_mesh_decoder(self) -> None:
        """Save decoder weights to mesh_decoder.pt (one-time bootstrap for the fast-path).

        Reads weights directly from the training checkpoint — no eval_setup, no dataset load.
        """
        decoder_path = self.config["mesh_info"]["mesh"].parent / "mesh_decoder.pt"
        if decoder_path.exists():
            return
        if not self.config.get("model_config_path"):
            self._select_run()
        config_path = Path(self.config["model_config_path"])
        ckpt_dir = config_path.parent / "nerfstudio_models"
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        prefix = "_model.decoder."
        decoder_state = {
            k[len(prefix):]: v
            for k, v in ckpt["pipeline"].items()
            if k.startswith(prefix)
        }
        torch.save(decoder_state, decoder_path)
        print(f"Saved mesh decoder → {decoder_path}")

    def _extract_mesh_features(self, features_name: str = "distill_features") -> None:
        """Map per-Gaussian latent features to mesh vertices and save as mesh_features.pt."""
        from collab_splats.mesh.utils import features2vertex

        if getattr(self, "model", None) is None:
            _, pipeline, _, _ = eval_setup(Path(self.config["model_config_path"]))
            self.model = pipeline.model

        mesh_path = self.config["mesh_info"]["mesh"]
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        vertices = np.asarray(mesh.vertices).astype(np.float32)

        with torch.no_grad():
            gauss_pos = self.model.means.detach().cpu().numpy()
            gauss_feat = self.model.gauss_params[features_name].detach().cpu().numpy()

        vertex_features = features2vertex(vertices, gauss_pos, gauss_feat)

        features_path = mesh_path.parent / "mesh_features.pt"
        torch.save(torch.from_numpy(vertex_features).float(), features_path)
        torch.save(self.model.decoder.state_dict(), features_path.parent / "mesh_decoder.pt")
        self.config["mesh_info"]["features"] = features_path
        print(f"Saved mesh features → {features_path}")

    def _export_gaussian_splats(self, mesh_dir: Path, overwrite: bool = False) -> Optional[Path]:
        """Export the trained Gaussian splat cloud to splats.ply via ns-export.

        Skipped (returns existing path) if splats.ply already exists and overwrite is False.
        """
        out = mesh_dir / "splats.ply"
        if out.exists() and not overwrite:
            return out
        if not self.config.get("model_config_path"):
            self._select_run()
        try:
            subprocess.run(
                [
                    "ns-export", "gaussian-splat",
                    "--load-config", str(self.config["model_config_path"]),
                    "--output-dir", str(mesh_dir),
                    "--output-filename", "splats.ply",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"[warning] ns-export gaussian-splat failed ({e}); splats.ply not written")
            return None
        return out

    def query_mesh(
        self,
        positive_queries: Optional[List[str]] = None,
        negative_queries: Optional[List[str]] = None,
        output_fn: Optional[str] = None,
        temperature: float = 0.05,
    ) -> np.ndarray:
        """Query the mesh for features.

        Returns:
            similarity_colors: (N_vertices, 3) float64 array. R channel holds the
            normalised similarity score [0, 1]; G and B are zero unless multiple
            positive queries are provided.
        """

        if positive_queries is None:
            positive_queries = [""]
        if negative_queries is None:
            negative_queries = ["object"]

        if getattr(self, "model", None) is None:
            if not self.config.get("model_config_path"):
                self._select_run()
            mesh_dir = self.config["mesh_info"]["mesh"].parent
            decoder_path = mesh_dir / "mesh_decoder.pt"
            if decoder_path.exists():
                from collab_splats.nerfstudio.models.rade_features import (
                    TwoLayerMLP, _QUERYABLE_FEATURE_TYPES, _TEXT_ENCODER_NAME,
                )
                state = torch.load(decoder_path, map_location="cpu", weights_only=True)
                input_dim  = state["hidden_conv.weight"].shape[1]
                hidden_dim = state["hidden_conv.weight"].shape[0]
                feat_dims  = {
                    k.split(".")[1]: (v.shape[0], 1, 1)
                    for k, v in state.items()
                    if k.startswith("feature_branch_dict.") and k.endswith(".weight")
                }
                device = get_device()
                decoder = TwoLayerMLP(input_dim, hidden_dim, feat_dims)
                decoder.load_state_dict(state)
                decoder = decoder.to(device)
                feature_type = next(k for k in feat_dims if k in _QUERYABLE_FEATURE_TYPES)
                encoder_name = _TEXT_ENCODER_NAME.get(feature_type, feature_type)
                text_encoder = BaseFeatureExtractor.get(encoder_name)(device=device)
                self.model = SimpleNamespace(
                    decoder=decoder,
                    similarity_fx=text_encoder.score_queries,
                    main_features_name=feature_type,
                    device=device,
                )
                self.model._text_encoder = text_encoder
            else:
                print(f"Loading model from {self.config['model_config_path']}")
                _, pipeline, _, _ = eval_setup(Path(self.config["model_config_path"]))
                self.model = pipeline.model

        mesh_info = self.config.get("mesh_info")
        if mesh_info is None:
            raise ValueError("Mesh information not found. Please run mesh() first.")
        elif mesh_info.get("features") is None:
            raise ValueError("Features not found. Please run mesh() with features_name specified.")

        features = torch.load(self.config["mesh_info"]["features"])

        decoded_features = self.model.decoder.per_gaussian_forward(features.to(self.model.device).to(torch.float32))

        # Decoder regresses unit-norm targets but does not enforce them — observed norms span
        # [0.85, 110] in trained talk2dino runs, which makes the einsum a scaled dot product
        # rather than cosine similarity. Renormalize to recover cosine.
        feats = F.normalize(decoded_features[self.model.main_features_name], dim=-1)
        similarity_map = (
            self.model.similarity_fx(
                features=feats.unsqueeze(0).permute(2, 1, 0),
                positive=positive_queries,
                negative=negative_queries,
                temperature=temperature,
            )
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )

        del features

        # Normalise and pack into (N, 3) — R = score, G = B = 0
        similarity_colors = np.zeros((len(similarity_map), 3))
        similarity_cast = similarity_map.astype(np.float64)
        if similarity_cast.ndim == 1:
            similarity_cast = similarity_cast[:, np.newaxis]
        sim_max = float(np.max(similarity_cast))
        if sim_max != 0:
            similarity_cast /= sim_max
        similarity_cast = np.clip(similarity_cast, 0.0, 1.0)
        similarity_colors[:, : similarity_cast.shape[1]] = similarity_cast

        if output_fn is not None:
            output_dir = self.config["mesh_info"]["mesh"].parent
            output_path = output_dir / output_fn
            mesh = o3d.io.read_triangle_mesh(self.config["mesh_info"]["mesh"])
            mesh.vertex_colors = o3d.utility.Vector3dVector(similarity_colors)
            o3d.io.write_triangle_mesh(output_path, mesh)

        return similarity_colors

    def plot_mesh(self, attribute: Optional[Union[str, np.ndarray]] = None, rgb: bool = True) -> None:
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
        if not mesh_transform_fn.exists():
            print(f"No transforms.pkl in {mesh_dir} — mesh is in nerfstudio world frame, using identity.")
            return {"mesh_transform": np.eye(4, dtype=np.float32)}
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
        nerfstudio_transforms_path = model_dir / "dataparser_transforms.json"  # This is the nerfstudio transform

        # Camera transforms (poses)
        with open(transforms_json, "r") as f:
            transforms = json.load(f)

        # Nerfstudio transforms aligned to camera
        with open(nerfstudio_transforms_path, "r") as f:
            nerfstudio_transforms = json.load(f)

        # Compose into 4x4 matrix
        transform = np.stack(nerfstudio_transforms["transform"])

        # Add the translation to the transform
        transform = extrinsics_to_homogeneous(transform)

        # Apply to cameras
        camera_poses = np.stack(
            [f["transform_matrix"] for f in transforms["frames"]]
        )  # Load from your camera pose files or nerfstudio transforms
        camera_poses[..., :3, 3] *= nerfstudio_transforms["scale"]

        if align_mesh:
            mesh_transform = self.load_mesh_transform()
            camera_poses = mesh_transform["mesh_transform"] @ camera_poses

        return camera_poses
