import sys
from collab_splats.wrapper import Splatter, SplatterConfig

test_configs = {
    # 'rats': {
    #     'file_path': '/workspace/fieldwork-data/rats/2024-07-11/SplatsSD/C0119.MP4',
    #     'frame_proportion': 0.25,
    # },
    'birds_001': {
        'file_path': '/workspace/fieldwork-data/birds/2024-05-18/SplatsSD/C0065.MP4',
        'frame_proportion': 0.25,
    },
    'birds_002': {
        'file_path': '/workspace/fieldwork-data/birds/2024-05-19/SplatsSD/C0067.MP4',
        'frame_proportion': 0.25,
    }
}

METHODS = ['rade-features'] #'rade-gs'] #'feature-splatting',

if __name__ == "__main__":

    for species, config in test_configs.items():
        for method in METHODS:
            print (f"Running {method} for {species}")
            
            config['method'] = method

            # Create the splatter
            config = SplatterConfig(**config)
            splatter = Splatter(config)

            # Create the colmap
            preproc_kwargs = {
                "sfm_tool": "hloc",
                "refine_pixsfm": True,
            }

            splatter.preprocess(kwargs=preproc_kwargs)

            # Train the splatting model -- can pass additional arguments to ns-train
            feature_kwargs = {
                "pipeline.model.output-depth-during-training": True,
                "pipeline.model.rasterize-mode": "antialiased",
                "pipeline.model.use_scale_regularization": True,
            }

            splatter.extract_features(overwrite=True, kwargs=feature_kwargs)

        # sys.exit()