import os   
import glob
import subprocess
from pathlib import Path
from typing import Optional, TypedDict, Set, Dict, Any, Union, List
import cv2
import torch
import numpy as np

from nerfstudio.utils.eval_utils import eval_setup
from ns_extension.utils.mesh import Open3DTSDFFusion
from ns_extension.utils.plotting import load_and_plot_ply

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
        'rade-gs',
        'rade-features'
    }

    def __init__(self, config: SplatterConfig):
        """Initialize the splatter with configuration.
        
        Args:
            config: Configuration dictionary specifying environment parameters
        """
        # Validate config before initialization
        validated_config = self.validate_config(config)
        self.config: Dict[str, Any] = dict(validated_config)

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
        config['file_path'] = file_path
        
        # If we don't specify an output path, default to the grandparent directory of the input file
        if config.get('output_path') is None:
            default_output_path = os.path.join(file_path.parent.parent, 'environment', file_path.stem)
            config.setdefault('output_path', Path(default_output_path)) # type: ignore

        if config.get('min_frames') is None:
            config.setdefault('min_frames', 300) # Default number of video frames to use for COLMAP
        
        return config
    
    @classmethod
    def available_methods(cls) -> None:
        """Print the available methods.
        
        Args:
        """
        print("Available methods:")
        print("  ", sorted(cls.SPLATTING_METHODS))

    def preprocess(self) -> None:
        """Preprocess the data in the splatter.
        
        This function handles any necessary data preprocessing steps based on the
        configured method.
        """
        file_path = self.config['file_path']
        output_path = self.config['output_path']
        
        # Determine input type based on file extension
        ext = file_path.suffix.lower()
        if ext in ['.mp4', '.mov', '.avi']:
            input_type = 'video'
        elif ext in ['.jpg', '.jpeg', '.png']:
            if '360' in str(file_path):
                input_type = 'images --camera-type equirectangular --images-per-equirect 14'
            else:
                input_type = 'images'
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Set the output path to same directory as input fil    
        preproc_data_path = output_path / 'preproc'
        transforms_path = preproc_data_path / "transforms.json"

        # If the transforms exists and we don't want to overwrite
        # Return and store the processed_data_path
        if transforms_path.exists() and not self.config.get('overwrite', False):
            print(f"transforms.json already exists at {transforms_path}")
            print("To rerun preprocessing, set overwrite=True")
            self.config['preproc_data_path'] = preproc_data_path
            return

        if self.config.get('frame_proportion') is not None:
            video_capture = cv2.VideoCapture(file_path.as_posix())
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            n_samples = int(n_frames*self.config['frame_proportion'])

            # If we have less than the minimum number of frames, as many as possible
            n_samples = n_frames if n_samples < self.config['min_frames'] else n_samples

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
            f"{num_frames_target}"
        )

        subprocess.run(cmd, shell=True)
        
        # Store the preprocessed data path in the config
        self.config['preproc_data_path'] = preproc_data_path

    def extract_features(self, overwrite: bool = False, kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Extract features from the preprocessed data.
        
        Feature extraction is performed according to the configured dtype and method.
        """
        method = self.config["method"]

        # Check that preprocessing was completed before extracting features
        if self.config['preproc_data_path'] is None:
            raise ValueError("preprocess_data() must be run before extracting features")

        # Set the model path (where it outputs results of model training)
        model_path = self.config['output_path'] / method
        model_exists = any(model_path.glob("**/*.ckpt"))

        if model_exists and not overwrite:
            print(f"Output already exists for {method}")
            print("To rerun feature extraction, set overwrite=True")
            self.config['model_path'] = model_path
            return

        cmd = (
            f"ns-train "
            f"{method} "
            f"--data {self.config['preproc_data_path'].as_posix()} "
            f"--output-dir {self.config['output_path'].as_posix()} "
            f"--experiment-name '' " # This keeps our file structure as environment/BASE_NAME/method/
            "--viewer.quit-on-train-completion True " # This quits the function once training is complete
        )

        if kwargs is not None:
            kwargs_cmds = ' '.join([f"--{k} {v}" for k, v in kwargs.items()])
            cmd += kwargs_cmds

        self.config['model_path'] = model_path
        subprocess.run(cmd, shell=True)
    
    def viewer(self) -> None:
        """Display or visualize the splatter data.
        
        This function handles feature extraction from the preprocessed data.
        The specific feature extraction pipeline depends on config['dtype']:
            - 2D: Image-based feature extraction
            - 3D: Volume-based feature extraction
        """

        self._select_run()

        cmd = (
            f"ns-viewer "
            f"--load-config {self.config['model_config_path'] } "
        )
        
        subprocess.run(cmd, shell=True, timeout=DEFAULT_TIMEOUT)

    def _select_run(self) -> None:
        """Select a run from the available runs."""
        # Find all runs with config.yml files
        output_dir = Path(str(self.config['output_path']), self.config['method'])

        # Grab all directories with a config.yml file --> convert to paths
        run_dirs = glob.glob(os.path.join(output_dir, "**/config.yml"))
        run_dirs = [Path(run_dir).parent for run_dir in run_dirs]
        
        if not run_dirs:
            raise ValueError(f"No runs with config.yml found in {output_dir}")
            
        # Sort runs by directory name (which contains timestamp)
        sorted_runs = sorted(run_dirs)
        
        # Print available runs
        print("\nAvailable runs:")
        for i, run in enumerate(sorted_runs):
            print(f"[{i}] {run.name}")
            
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
                print(f"Please enter a number between 0 and {len(sorted_runs)-1}")
            except ValueError:
                print("Please enter a valid number")

        self.config['model_path'] = selected_run.as_posix()
        self.config['model_config_path'] = (selected_run / "config.yml").as_posix()

    def mesh(
        self, 
        depth_name: str = "depth",
        normals_name: str = "normals",
        features_name: Optional[str] = "distill_features", 
        sdf_trunc: Optional[float] = 0.03,
        depth_trunc: Optional[float] = 3.0, 
        overwrite: bool = False,
    ) -> None:
        """Generate a mesh from the splatter data.
        
        This function handles mesh generation from the preprocessed data.
        """
        self._select_run()

        mesh_dir = self.config['output_path'] / self.config['method'] / "mesh" 

        # Create the mesh
        if not mesh_dir.exists() or overwrite:

            # Initialize the mesher
            mesher = Open3DTSDFFusion(
                load_config=Path(self.config['model_config_path']),
                depth_name=depth_name,
                normals_name=normals_name,
                features_name=features_name,
                depth_trunc=depth_trunc,
                sdf_trunc=sdf_trunc,
                output_dir=mesh_dir
            )

            self.config['mesh_info'] = mesher.main()
        else:
            # HARDCODING FOR NOW TLB FIX
            self.config['mesh_info'] = {
                'mesh': mesh_dir / "Open3dTSDFfusion_mesh.ply",
                'features': mesh_dir / "mesh_features.pt"
            }

    def query_mesh(
        self, 
        positive_queries: List[str] = [""], 
        negative_queries: List[str] = ["object"], 
        method: str = "pairwise"
    ) -> None:
        """Query the mesh for features."""

        if not self.config.get('model_config_path'):
            self._select_run()
        elif getattr(self, 'model', None) is None:
            print(f"Loading model from {self.config['model_config_path']}")
            _, pipeline, _,  _ = eval_setup(Path(self.config['model_config_path']))
            self.model = pipeline.model

        if self.config.get('mesh_info') is None:
            raise ValueError("Mesh information not found. Please run mesh() first.")
        elif self.config.get('mesh_info').get('features') is None:
            raise ValueError("Features not found. Please run mesh() with features_name specified.")

        features = torch.load(self.config['mesh_info']['features'])

        decoded_features = self.model.decoder.per_gaussian_forward(
            features.to(self.model.device).to(torch.float32)
        )

        similarity_map = self.model.similarity_fx(
            features=decoded_features[self.model.main_features_name].unsqueeze(0).permute(2, 1, 0), 
            positive=positive_queries, 
            negative=negative_queries,
            method=method
        ).squeeze(-1).detach().cpu().numpy()

        del features

        return similarity_map


    def plot_mesh(self, attribute: Optional[np.ndarray] = None) -> None:
        """Plot the mesh."""
        if self.config.get('mesh_info') is None:
            raise ValueError("Mesh information not found. Please run mesh() first.")
        elif self.config.get('mesh_info').get('mesh') is None:
            raise ValueError("Mesh not found. Please run mesh() first.")

        mesh_path = self.config['mesh_info']['mesh']
        load_and_plot_ply(mesh_path, attribute)