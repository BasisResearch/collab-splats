# Paper

***I. Aitsahalia et al., "Inferring cognitive strategies from groups of animals in natural environments," presented at the NeurIPS Workshop on Data on the Brain \& Mind Findings, 2025.***

## Figures from the paper

### Figure 3

#### Panel A = rasterized gaussian splatting viewer

 * Run in [derive_splats.ipynb](splats/derive_splats.ipynb)
 * Created via [run_pipeline.py](splats/run_pipeline.py)  — default arguments of RaDe-Features model along with the following:
    - hloc (for preprocessing)
    - Anti-aliasing
    - Scale regularization
 * Mesh created also via run_pipeline w/ default arguments

#### Panels B/C/D

  Parts created by [create_mesh.ipynb](splats/create_mesh.ipynb) notebook
  * Feeder query
    - Positive = "feeder"
    - Negative = "ground", "leaves", "rocks"
  * Tree query
    - Positive = "tree"
    - Negative = "ground", "leaves"
  * Semantic clustering
    - Similarity threshold = 0.95
    - Spatial radius = 0.01