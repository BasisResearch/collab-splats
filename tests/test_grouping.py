import os, sys
from pathlib import Path

from collab_splats.utils.grouping import GroupingClassifier, GroupingParams

# Path to the config for a trained model
load_config = '/workspace/fieldwork-data/rats/2024-07-11/environment/C0119/rade-features/2025-07-25_074037/config.yml'
load_config = Path(load_config)

grouping_params = GroupingParams(segmentation_backend='mobilesamv2', segmentation_strategy='object', front_percentage=0.2, iou_threshold=0.1, num_patches=32)
grouping_classifier = GroupingClassifier(load_config=load_config, params=grouping_params)