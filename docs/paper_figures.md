# Paper

***I. Aitsahalia et al.,  ["Inferring cognitive strategies from groups of animals in natural environments"](https://openreview.net/pdf?id=mUPCoCIxLK), presented at the NeurIPS Workshop on Data on the Brain \& Mind Findings, 2025.***

## Figures from the paper

The figures came from the pre-refactor code; links point at that snapshot.

### Figure 3

#### Panel A = rasterized gaussian splatting viewer

 * Run in [derive_splats.ipynb](https://github.com/BasisResearch/collab-splats/blob/966ed460e825ecce783637ddfd90813f724770ae/docs/splats/derive_splats.ipynb)
 * Created via [run_pipeline.py](https://github.com/BasisResearch/collab-splats/blob/966ed460e825ecce783637ddfd90813f724770ae/docs/splats/run_pipeline.py)  — default arguments of RaDe-Features model along with the following:
    - hloc (for preprocessing)
    - Anti-aliasing
    - Scale regularization
 * Mesh created also via run_pipeline w/ default arguments

#### Panels B/C/D

  Parts created by [create_mesh.ipynb](https://github.com/BasisResearch/collab-splats/blob/966ed460e825ecce783637ddfd90813f724770ae/docs/splats/create_mesh.ipynb) notebook
  * Feeder query
    - Positive = "feeder"
    - Negative = "ground", "leaves", "rocks"
  * Tree query
    - Positive = "tree"
    - Negative = "ground", "leaves"
  * Semantic clustering
    - Similarity threshold = 0.95
    - Spatial radius = 0.01
