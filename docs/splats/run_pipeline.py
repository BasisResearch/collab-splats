from pathlib import Path
from collab_splats.wrapper import Splatter, SplatterConfig

test_configs = {
    # 'birds_001': {
    #     'file_path': '/workspace/fieldwork-data/birds/2024-05-23/SplatsSD/GH010070.MP4',
    #     'frame_proportion': 0.125,
    # },

    # 'birds_002': {
    #     'file_path': '/workspace/fieldwork-data/birds/2024-05-18/SplatsSD/C0065.MP4',
    #     'frame_proportion': 0.25,
    # },
    # 'birds_003': {
    #     'file_path': '/workspace/fieldwork-data/birds/2024-05-19/SplatsSD/C0067.MP4',
    #     'frame_proportion': 0.25,
    # },
    # 'birds_004': {
    #     'file_path': '/workspace/fieldwork-data/birds/2023-11-05/SplatsSD/PXL_20231105_154956078.mp4',
    #     'frame_proportion': 0.25,
    # },
    # 'birds_005': {
    #     'file_path': '/workspace/fieldwork-data/birds/2024-06-01/SplatsSD/GH010164.MP4',
    #     'frame_proportion': 0.1,
    # },
    # 'birds_006': {
    #     'file_path': '/workspace/fieldwork-data/birds/2024-05-27/SplatsSD/GH010097.MP4',
    #     'frame_proportion': 0.14,
    # },
    # 'birds_007': {
    #     'file_path': '/workspace/fieldwork-data/birds/2024-05-27/SplatsSD/GH010105.MP4',
    #     'frame_proportion': 0.25,
    # },
    # 'birds_008': {
    #     'file_path': '/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4',
    #     'frame_proportion': 0.25,
    # },
    'rats_001': {
        'file_path': '/workspace/fieldwork-data/rats/2024-07-11/SplatsSD/C0119.MP4',
        'frame_proportion': 0.25,
    },
}

METHODS = ['rade-features'] #'rade-gs'] #'feature-splatting',

if __name__ == "__main__":
    

    for species, config in test_configs.items():
        for method in METHODS:
            print (f"Running {method} for {species} video {Path(config['file_path']).name}")
            
            config['method'] = method

            # Create the splatter
            config = SplatterConfig(**config)
            splatter = Splatter(config)

            # Create the colmap
            preproc_kwargs = {
                "sfm_tool": "hloc",
                # "refine_pixsfm": "", # This is a flag currently so doesn't need to be given "True" afai can tell
            }

            splatter.preprocess(kwargs=preproc_kwargs) #, overwrite=True)

            # Train the splatting model -- can pass additional arguments to ns-train
            feature_kwargs = {
                "pipeline.model.output-depth-during-training": True,
                "pipeline.model.rasterize-mode": "antialiased",
                "pipeline.model.use_scale_regularization": True,
            }

            splatter.extract_features(kwargs=feature_kwargs) #, overwrite=True)

            # Mesh the splatting model
            mesher_kwargs = {
                'depth_name': "median_depth",
                'depth_trunc': 1.0, # Should be between 1.0 and 3.0
                'voxel_size': 0.005, 
                'normals_name': "normals",
                'features_name': "distill_features", 
                'sdf_trunc': 0.03,
                'k': 20,
                'clean_repair': True,
                'align_floor': True,
            }

            splatter.mesh(
                mesher_type="Open3DTSDFFusion",
                mesher_kwargs=mesher_kwargs,
                overwrite=True
            )