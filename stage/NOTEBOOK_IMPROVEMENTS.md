# Optical Flow Frame Selection Demo - Improvements Summary

## Issues Fixed

### 1. **Metrics Access Bug**
- **Problem**: Code tried to access `metrics['disparities']` which doesn't exist
- **Root Cause**: The `process_video()` method returns metrics with structure:
  ```python
  metrics = {
      'scores': [...],
      'components': [{'disparity': x, 'motion': y, 'coverage': z, ...}, ...],
      'total_frames': n,
      'selected_frames': m,
      ...
  }
  ```
- **Solution**: Added helper function `extract_metrics()` that properly extracts arrays from `metrics['components']`

### 2. **Component Key Names**
- **Problem**: Code used `components.get('motion_score')` and `components.get('coverage_score')`
- **Correct Keys**: `'motion'` and `'coverage'` (not `'motion_score'` or `'coverage_score'`)
- **Fixed**: Updated all component access to use correct keys

### 3. **Section Numbering**
- **Problem**: Sections jumped from 3 → 6 → 6.5 → 7, etc.
- **Fixed**: Renumbered all sections sequentially from 1-10

### 4. **Code Complexity**
- **Problem**: Added ~400 lines of overly complex diagnostic code
- **Solution**: Removed 5 cells with redundant/complex visualizations
- **Result**: Reduced from 27 to 23 cells (15% reduction)

## Final Notebook Structure

**Total: 23 cells** (11 code, 12 markdown)

### Sections:
1. **Understanding the Algorithm** - Explains VGGT-SLAM approach and our enhancements
2. **Basic Usage Example** - Initialize selector with default parameters
3. **Process a Video File** - Load configuration and process video
4. **Visualize Selection Metrics** - Built-in summary visualization
5. **Detailed Disparity Analysis** - 4-panel custom analysis (NEW/FIXED)
   - Helper function to extract metrics properly
   - Disparity timeline, distribution, motion vs coverage, scores
6. **Visualize Optical Flow** - Flow vectors at different video sections
7. **Compare Different Parameter Settings** - Test 4 configurations on sample frames
8. **Visualize Selected Frames** - Show grid of selected keyframes
9. **Recommendations for Real Usage** - Configuration examples for different scenarios
10. **Export Selected Frames** - Helper functions to save indices and create scripts

## Code Quality Improvements

### Modularity
✅ Helper function `extract_metrics()` for clean metric extraction
✅ Local functions in cells (e.g., `load_frame()`, `test_parameter_configuration()`)
✅ Clear separation of concerns

### Simplicity
✅ Reduced section 5 from ~200 lines to ~60 lines
✅ Removed redundant visualizations
✅ Focused on most useful diagnostics

### Correctness
✅ All metrics access uses correct structure
✅ All component keys use correct names
✅ No undefined variables
✅ Logical cell ordering (helper before usage)

## Final Bug Fixes (Session 2)

### Critical Bug: Incorrect Data Extraction
**Problem**: Helper function tried to extract disparity from `metrics['components']`, but:
- First frame component only has 3 keys: `{'motion', 'coverage', 'combined'}`
- Subsequent frames have all 6 keys but we should use `metrics['stats']` for raw data

**Root Cause**:
- Line 387-392 in `preproc_utils.py` shows first frame returns limited component dict
- Disparity values are stored in `metrics['stats']['disparities']` (raw pixel values)
- Components contain *normalized scores*, not raw measurements

**Solution**:
- Changed `extract_metrics()` to use `metrics['stats']` for disparity/rotation/histogram data
- This gives us the actual pixel displacements, not normalized scores
- Added proper documentation explaining the data structure

### Added: Frame Selection Explanation
**Enhancement**: Expanded Section 1 with detailed explanation of HOW frames are selected:
1. Feature Detection (Shi-Tomasi corners)
2. Optical Flow Tracking (Lucas-Kanade)
3. Motion Analysis (translation + rotation)
4. Coverage Analysis (histogram similarity)
5. Scoring and Selection (weighted combination)
6. Keyframe Update (reference frame management)

Also added explanation of why this matters for 3D reconstruction.

### Added: Video Rotation Correction
**Enhancement**: Added automatic detection and correction of video rotation:
- Helper function `get_video_rotation()` reads rotation metadata from video using ffprobe
- Helper function `rotate_frame()` applies the rotation to frames before display
- Applied to all visualization cells (optical flow, selected frames)
- Fixes issue where drone footage recorded in portrait is displayed rotated

### Code Refactoring: Created optical_flow.py Module
**Refactoring**: Consolidated all code into a new `optical_flow.py` module:
- Moved `OpticalFlowFrameSelector` class from `preproc_utils.py`
- Added all visualization functions:
  - `visualize_optical_flow` - Draw flow vectors on frames
  - `visualize_flow_between_frames` - Flow between two frames
  - `create_selection_summary_plot` - 4-panel diagnostic summary
  - `create_configuration_comparison_plot` - Compare multiple configurations (NEW)
- Added utility functions:
  - `get_video_rotation`, `rotate_frame` - Video rotation handling
  - `load_frame` - Frame loading helper
  - `extract_metrics` - Metrics extraction for analysis
  - `test_parameter_configuration` - Test selector configurations (NEW)
- Added export utilities (`save_selected_indices`, `create_copy_script`)
- Updated notebook to import from `optical_flow` module instead of inline definitions
- Removed ALL inline function definitions from notebook cells
- **Result**: Ultra-clean notebook (22 cells), fully reusable module, ~150 lines of code moved to module

## Verification Results

✅ Helper function extracts data from correct source (`metrics['stats']`)
✅ Disparity visualization now works correctly
✅ Proper handling of array length differences (disparities start from frame 1)
✅ Sequential section numbering (1-10)
✅ All cells properly structured
✅ Comprehensive algorithm explanation added
✅ No syntax errors or undefined variables

## Key Takeaways

1. **Always check API return structures** - Don't assume dictionary keys exist
2. **Use helper functions** - Makes code modular and reusable
3. **Simplify visualizations** - Focus on most informative plots
4. **Verify cell dependencies** - Ensure variables are defined before use
5. **Keep notebooks clean** - Remove redundant or overly complex cells

## Usage

The notebook now works correctly end-to-end:
1. Run all cells sequentially
2. Section 5 properly extracts and visualizes disparity data
3. All visualizations work without errors
4. Clean, professional output for analysis
