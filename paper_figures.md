# Figures from the paper

## Figure 3 

### Panel A = rasterized gaussian splatting viewer
 * Run in `examples/derive_splats.ipynb`
 * Created via `examples/run_pipeline.py`  — default arguments of RaDe-Features model along with the following:
    - hloc (for preprocessing)
    - Anti-aliasing
    - Scale regularization
 * Mesh created also via run_pipeline w/ default arguments

### Panels B/C/D
  Parts created by `examples/create_mesh.ipynb` notebook
  * Feeder query
    - Positive = "feeder"
    - Negative = "ground", "leaves", "rocks"
  * Tree query
    - Positive = "tree"
    - Negative = "ground", "leaves"
  * Semantic clustering
    - Similarity threshold = 0.95
    - Spatial radius = 0.01