# Topological graph embedding

Topological graph embedding represents a noisy point cloud with a small,
smooth network of one-dimensional routes. The learned network can contain
endpoints, branches, junctions, open paths, and loops. Each observation is
assigned to a route, given a position along that route, and described by the
residual that remains after projection.

The main idea is useful when the data have a **graph-shaped latent structure**:
cell lineages, trajectories, branching processes, road-like geometry, or
scientific measurements concentrated around curves and networks. It is not
intended to replace a general-purpose nonlinear embedding. Instead, it makes
the one-dimensional structure explicit and keeps the off-route variation
available for analysis.

## The underlying idea

Suppose observations $x_i \\in \\mathbb{R}^d$ are sampled around an unknown
graph of smooth curves. We learn a route network

$$
\Gamma = \{\gamma_r : [0,1] \rightarrow \mathbb{R}^d\}_{r=1}^R.
$$

Each observation is projected to its closest route coordinate:

$$
(r_i, u_i) = \arg\min_{r,u} \|x_i - \gamma_r(u)\|_2.
$$

The resulting representation separates three kinds of information:

- `route_id` identifies the branch or loop;
- `position` gives longitudinal position along that route;
- `residual` and `residual_norm` describe displacement away from the route.

The public result also contains `projected`, the point on the fitted route,
and `tangent`, the local route direction. In high-dimensional data, the graph
is fitted in the original feature space; PCA, classical MDS, or UMAP is used only to display
that fitted graph.

This distinction matters. A two-dimensional display can make a complicated
high-dimensional cloud look compact or tangled, but it does not change the
route assignments or residuals learned in the original metric.

## How fitting works

The estimator uses a coarse-to-fine pipeline:

1. **Prepare the metric.** Features are optionally standardized. Constant
   features receive unit scale so degenerate inputs remain finite.
2. **Compress the observations.** A deterministic, blocked k-means procedure
   produces at most `n_centroids` landmarks. The landmarks are used for graph
   construction instead of building a dense graph over every observation.
3. **Estimate topology.** Persistent homology in dimension 1 estimates how
   many meaningful cycles are present. Ripser is used when available, with a
   NumPy Vietoris–Rips fallback.
4. **Build a sparse graph.** A minimum spanning tree provides connectivity.
   Local edges from a symmetrized landmark k-nearest-neighbor graph are then
   considered as cycle-closing candidates. Candidate edges are selected using
   the contrast between their graph path length and direct geometric length.
5. **Simplify the graph.** Degree-2 landmarks are collapsed into route chains;
   short terminal branches may be pruned and nearby junctions merged.
6. **Fit route geometry.** Open and closed chains are represented by dense
   sampled curves. SciPy smoothing splines are preferred, with a deterministic
   NumPy fallback for unsupported or numerically difficult cases.
7. **Project observations.** Each observation is assigned to its closest
   sampled route. Deterministic normal frames provide stable off-route
   coordinates, including when only a subset is transformed.

The topology is deliberately sparse and approximate. Diagnostics on the
fitted estimator make that approximation visible rather than hiding it.

## Installation

The core package requires NumPy and SciPy. For development and the notebooks:

```bash
python -m pip install -e ".[dev,notebooks]"
```

The optional notebook dependencies include scikit-learn, UMAP, Matplotlib,
Plotly, ipywidgets, and Ripser. The package can still use its NumPy topology
and curve fallbacks when optional backends are unavailable.

## Quick start

```python
import numpy as np

from topological_graph_embedding import SplineGraphEmbedding
from topological_graph_embedding.datasets import generate_synthetic_datasets

datasets = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)
X = datasets["loop-branch"]

model = SplineGraphEmbedding(
    n_centroids=32,
    max_cycles=5,
    topology_neighbors=6,
    random_state=0,
)
result = model.fit_transform(X)

print("routes:", len(model.routes_))
print("cycles:", model.realized_cycle_count_)
print("median residual:", float(np.median(result.residual_norm)))

# Stable coordinates in the local hyperplane normal to each route.
normal = model.normal_coordinates(result)

# Direct feature-space visualization for a two-dimensional dataset.
axis = model.plot_network(X, show_projections=True)
```

The repository also contains a command-line demo:

```bash
python run_demo.py --output-dir outputs
```

It generates synthetic line, branch, loop, and figure-eight examples and
writes figures plus a CSV summary to `outputs/`.

## Interpreting `EmbeddingResult`

`fit_transform` returns a frozen `EmbeddingResult` with six fields:

| Field | Meaning |
| --- | --- |
| `route_id` | Integer identity of the assigned route |
| `position` | Longitudinal coordinate in approximately `[0, 1]` |
| `projected` | Nearest point on the fitted route in original coordinates |
| `residual` | `X - projected` in original coordinates |
| `residual_norm` | Euclidean magnitude of the residual |
| `tangent` | Local route tangent in standardized fitting coordinates |

The result is attribute-based; it intentionally does not implement dictionary
indexing or legacy field aliases.

For a route with tangent $v$, the normal coordinates form an orthonormal
basis of the $(d-1)$-dimensional space perpendicular to $v$:

$$
e_i \approx \sum_{k=1}^{d-1} z_{ik} n_k(u_i).
$$

Use them when the residual direction matters, for example when modeling
branch-specific variation or building downstream features:

```python
normal = model.normal_coordinates(result)
print(normal.shape)  # (n_samples, n_features - 1)
```

## Parameters that control the learned graph

The most important estimator parameters are:

| Parameter | Role |
| --- | --- |
| `n_centroids` | Resolution of the landmark graph; larger values preserve more detail but cost more |
| `max_cycles` | Maximum number of cycles allowed in the fitted graph |
| `persistence_threshold` | Threshold for treating an H1 bar as meaningful; `None` selects a local-scale default |
| `topology_neighbors` | Number of local kNN neighbors considered for cycle candidates |
| `spline_smoothing` | Smoothing strength for route curves |
| `merge_junction_distance` | Distance used to merge nearby graph junctions; `None` selects an automatic value |
| `prune_short_branches` | Whether very short terminal branches are removed |
| `standardize` | Whether fitting distances are computed after featurewise standardization |
| `persistence_max_points` | Cap used by the persistence calculation for large point clouds |
| `random_state` | Reproducibility for landmark initialization and capped persistence sampling |

There are two different notions of scale in the project:

- estimator parameters control the graph learned from the data;
- visualization parameters control how that graph is drawn.

For example, `MetroLayout(residual_width=0.02)` changes the apparent thickness
of the observation dispersion around a metro route. It does not alter route
assignments or projection residuals.

## Visualization

The visualization package provides three complementary views:

- a direct feature-space or reduced-space view of observations and fitted
  routes;
- a graph-coordinate view showing `route_id` against `position`;
- a metro-style view showing route connectivity, stations, and residual
  dispersion around each route.

For high-dimensional data, the display reducer is separate from the fitted
graph:

```python
from topological_graph_embedding.visualization.reduction import fit_reducer

reducer = fit_reducer(X, method="umap", n_neighbors=15)
```

UMAP's `n_neighbors` changes the visual layout's local/global emphasis only.
It does not refit the topological graph. PCA and classical MDS are available
as deterministic display alternatives. The high-dimensional notebook exposes the UMAP
neighbor count and metro dispersion width through interactive sliders.

The shared four-panel renderer is available through
`topological_graph_embedding.visualization.plot_embedding_row`. For direct
control of the metro layout:

```python
from topological_graph_embedding.visualization import MetroLayout

layout = MetroLayout(model, residual_width=0.04).fit(result)
```

The Plotly view in `visualization.interactive` places the metro map in the XY
plane and uses local residual principal components for lateral and vertical
coordinates.

## Scikit-learn integration

The optional adapters expose route information as ordinary numerical features:

```python
from sklearn.ensemble import RandomForestClassifier

from topological_graph_embedding.sklearn import (
    SplineEmbeddingClassifier,
    SplineEmbeddingTransformer,
)

transformer = SplineEmbeddingTransformer(n_centroids=32, random_state=0)
features = transformer.fit_transform(X)
result = transformer.transform_result(X)

classifier = SplineEmbeddingClassifier(
    estimator=RandomForestClassifier(n_estimators=200, random_state=0),
    n_centroids=32,
    random_state=0,
)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

The transformer emits route indicators, longitudinal position, residual norm,
and scaled residual components. The classifier uses route/position features
and deterministic normal coordinates before delegating to the cloned
downstream estimator.

## Diagnostics and fallbacks

Useful fitted attributes include:

- `persistent_cycle_count_`: significant H1 bars;
- `requested_cycle_count_`: target after applying `max_cycles`;
- `realized_cycle_count_`: cycle rank of the fitted landmark graph;
- `topology_shortfall_`: requested cycles that local candidates could not realize;
- `persistence_backend_`: `ripser`, `numpy`, or `numpy-after-ripser-error`;
- `route_backends_`: backend used for each route;
- `route_chains_`: graph chains that became routes;
- `junctions_` and `endpoints_`: landmark nodes classified by degree.

Fallbacks are recorded in these attributes so a result remains inspectable.
Unexpected backend failures are warned about; expected optional-dependency
fallbacks remain quiet.

## Limitations

This is a one-dimensional graph-skeleton model. It does not currently model
branch-specific density, uncertainty, a full continuous projection
optimization, or a higher-dimensional manifold with a thick intrinsic cross
section. The sparse topology is an approximation and can under-realize a
persistence target; inspect `topology_shortfall_` when cycles matter.

Metro coordinates are a display transform, not distances in the original
feature space. UMAP, PCA, and classical MDS plots are also display transforms and should not be
used as substitutes for the original-space `projected`, `residual`, or
`position` values.

## Repository guide

The main implementation is organized as follows:

```text
topological_graph_embedding/
├── embedding.py                 # public estimator and projection API
├── results.py                   # immutable EmbeddingResult
├── _topology.py                 # landmarks, graph construction, persistence
├── _curves.py                   # route fitting and projection
├── _frames.py                   # deterministic normal frames
├── sklearn.py                   # optional sklearn adapters
├── datasets.py                  # synthetic point clouds
└── visualization/
    ├── metro.py                 # schematic metro layout
    ├── network.py               # feature-space network rendering
    ├── plots.py                 # static plotting and evaluation
    ├── interactive.py           # Plotly 3D view
    ├── reduction.py             # PCA/MDS/UMAP display reducers
    └── workflows/               # reusable notebook workflows
```

The notebooks in `notebooks/` are thin wrappers around the installed package.
The longer algorithm description is in
[`docs/whitepaper.md`](docs/whitepaper.md), and the reusable notebook
workflow notes are in [`notebooks/README.md`](notebooks/README.md).

Run the tests after installing the development extras:

```bash
python -m pytest
```
