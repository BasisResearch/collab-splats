"""
VGGT utilities - DEPRECATED

DEPRECATION NOTICE:
    VGGT is now integrated directly into nerfstudio (nerfstudio/process_data/vggt_utils.py).
    This module is kept only for backward compatibility and will be removed in a future version.

    New code should use:
    - splatter.preprocess(sfm_tool='vggt') instead of splatter.preprocess_vggt()
    - Or call ns-process-data directly with --sfm-tool vggt

    The nerfstudio integration provides:
    - Better performance
    - Direct integration with ns-process-data
    - Maintained alongside other SfM tools (COLMAP, hloc)
    - Automatic transforms.json generation

For reference, see:
    - nerfstudio/process_data/vggt_utils.py (new implementation)
    - https://github.com/facebookresearch/vggt (VGGT repository)
"""

import warnings

warnings.warn(
    "collab_splats.utils.vggt_utils is deprecated. "
    "VGGT is now integrated into nerfstudio. "
    "Use splatter.preprocess(sfm_tool='vggt') instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Legacy imports are kept for backward compatibility, but all functions
# now redirect to nerfstudio's implementation or raise deprecation errors.

def __getattr__(name):
    """Intercept any function calls to show deprecation message."""
    raise ImportError(
        f"The function '{name}' has been moved to nerfstudio. "
        f"Please use splatter.preprocess(sfm_tool='vggt') instead of "
        f"direct vggt_utils calls. See nerfstudio/process_data/vggt_utils.py "
        f"for the new implementation."
    )
