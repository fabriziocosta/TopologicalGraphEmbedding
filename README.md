# SkeletalEmbedding

SkeletalEmbedding extracts a compact skeleton from a noisy point cloud as a
small, smooth network of backbone splines, optional coverage ribs, and local
residual subspaces. The learned network can contain
endpoints, branches, junctions, open paths, and loops. Each observation is
assigned to a route, given a position along that route, and described by the
residual that remains after projection.

The backbone is useful when the data have a **graph-shaped latent structure**;
coverage refinement extends it to surfaces and other higher-dimensional
manifolds:
cell lineages, trajectories, branching processes, road-like geometry, or
scientific measurements concentrated around curves and networks. It is not
intended to replace a general-purpose nonlinear embedding. Instead, it makes
the one-dimensional structure explicit and keeps the off-route variation
available for analysis.

## The underlying idea

Suppose observations $x_i \in \mathbb{R}^d$ are sampled around an unknown
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
- `residual` and `residual_norm` describe displacement away from the route;
- optional residual-PCA coordinates capture leading smooth transverse
  directions; and
- `unexplained_residual` contains the remaining reconstruction error.

The public result also contains `projected`, the point on the fitted route,
and `tangent`, the local route direction. When `max_residual_dim > 0`, the
learned representation is

$$
x \approx \gamma_r(u) + U_r(u)z + \epsilon.
$$

Residual-PCA fields are learned in standardized fitting coordinates, so
`residual_coordinates` are in those coordinates. `projected`, `reconstructed`,
`residual`, and `unexplained_residual` remain in the original input units. In
high-dimensional data, the graph
is fitted in the original feature space; PCA, classical MDS, or UMAP is used only to display
that fitted graph.

This distinction matters. A two-dimensional display can make a complicated
high-dimensional cloud look compact or tangled, but it does not change the
route assignments or residuals learned in the original metric.

## How fitting works

The estimator uses a selectable coarse-to-fine pipeline. The default
`initialization="skeletal"` path treats the dense weighted
observation kNN graph is treated as a routing substrate and the backbone is
selected from explicit topology and connectivity constraints:

Most repository notebook workflows explicitly select
`initialization="skeletal"` so their figures and summaries use the
topology-aware initializer. The synthetic binary-tree example is the
exception: it uses the cycle-free coarsening path so noise between nearby
branches is not promoted to spurious loops. Select
`initialization="legacy_coarsen"` explicitly when comparing with the legacy
initializer. The synthetic notebook runs the six easy
examples with `persistence_max_points=60` and puts the polygon/hypercube
examples in a separate cell using cap `300` and normalized H1 threshold `4.0`.
Electrical resistance/current terms remain opt-in; mutual-neighbour routing
and Euclidean-MST augmentation are enabled by default.

1. **Prepare the metric.** Features are optionally standardized. Constant
   features receive unit scale so degenerate inputs remain finite.
2. **Build the routing substrate.** Topological mode constructs a weighted,
   symmetric observation kNN graph with Euclidean lengths and affinity-based
   conductances. With `mutual_knn=True`, only reciprocal neighbor pairs are
   retained. With `add_mst=True`, the exact Euclidean MST is added to the
   selected kNN edges. Disconnected kNN components receive a bridge only for
   electrical computations; route selection keeps their backbones separate.
   Coarsening mode instead continues with the existing centroid graph.
3. **Estimate topology and local geometry.** Persistent H1 estimates the
   cycle rank. Multiscale annulus components identify junction and endpoint
   regions, while local PCA estimates ordinary tangents and one outgoing
   direction per junction arm.
4. **Select the backbone.** Candidate landmark routes are scored using length,
   tangent consistency, density, and optional effective-resistance/current
   support. A small MIP selector enforces connectivity, endpoint/junction
   degrees, and the requested persistent cycle rank, with a deterministic
   fallback when the solver is disabled or infeasible.
5. **Simplify the graph.** Maximal degree-2 paths are collapsed into route
   support paths. The existing coarsening mode may still prune short terminal
   branches and merge nearby graph junctions.
6. **Fit route geometry.** Open and closed chains are represented by dense
   sampled curves. SciPy smoothing splines are preferred, with a deterministic
   NumPy fallback for unsupported or numerically difficult cases.
7. **Fit residual subspaces and ribs.** Tangent-orthogonal local PCA fields
   provide transverse coordinates. When post-PCA reconstruction error exceeds
   the requested coverage tolerance, stable high-error regions seed candidate
   transverse or parallel ribs. Candidates are penalized for complexity and
   selected iteratively.
8. **Project observations.** Each observation is assigned to its closest
   sampled route. Deterministic normal frames provide stable off-route
   coordinates, including when only a subset is transformed.

The topology is deliberately sparse and approximate. Diagnostics on the
fitted estimator make that approximation visible rather than hiding it.

For a detected $d$-dimensional hypercube, `realized_cycle_count_` is the
independent cycle rank, not the number of geometric faces. In particular, a
3D cube has 8 degree-3 junctions, 12 edges, 5 independent cycles, and 6 square
faces. The latter is reported separately as `face_cycle_count_`, with the
detected dimension in `hypercube_dimension_`.

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

from skeletalembedding import SkeletalEmbedding
from skeletalembedding.datasets import generate_synthetic_datasets

datasets = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)
X = datasets["loop-branch"]

model = SkeletalEmbedding(
    n_centroids=32,
    initialization="skeletal",
    max_cycles=5,
    topology_neighbors=6,
    random_state=0,
)
result = model.fit_transform(X)

print("routes:", len(model.routes_))
print("cycles:", model.realized_cycle_count_)
print("median residual:", float(np.median(result.residual_norm)))

# Optional smooth transverse coordinates and reconstruction error.
model = SkeletalEmbedding(max_residual_dim=2, random_state=0).fit(X)
result = model.transform(X)
print(result.residual_coordinates.shape)

# Optional sparse wire-frame refinement for higher-dimensional structure.
model = SkeletalEmbedding(
    max_residual_dim=1,
    coverage_refinement=True,
    coverage_error_tolerance=0.10,
    random_state=0,
).fit(X)
print("ribs:", len(model.rib_paths_))

# Stable coordinates in the local hyperplane normal to each route.
normal = model.normal_coordinates(result)

# Direct feature-space visualization for a two-dimensional dataset.
axis = model.plot_network(X, show_projections=True)
```

The repository also contains a command-line demo:

```bash
python run_demo.py --output-dir outputs
```

It generates synthetic line, star, binary-tree, loop, figure-eight, and
polygon/radial-circle examples and writes figures plus a CSV summary to
`outputs/`.

## Interpreting `EmbeddingResult`

`fit_transform` returns a frozen `EmbeddingResult`:

| Field | Meaning |
| --- | --- |
| `route_id` | Integer identity of the assigned route |
| `position` | Longitudinal coordinate in approximately `[0, 1]` |
| `projected` | Nearest point on the fitted route in original coordinates |
| `residual` | `X - projected` in original coordinates |
| `residual_norm` | Euclidean magnitude of the residual |
| `tangent` | Local route tangent in standardized fitting coordinates |
| `residual_coordinates` | Learned local PCA coordinates in standardized fitting coordinates; empty when disabled |
| `reconstructed` | `projected + U @ residual_coordinates`, in original feature units |
| `unexplained_residual` | `X - reconstructed`, in original feature units |
| `unexplained_residual_norm` | Euclidean magnitude of `unexplained_residual` |

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

Set `max_residual_dim` to a positive value to fit a fixed-dimensional,
Gaussian-weighted PCA field along every route. Neighboring subspaces can be
regularized with `residual_subspace_smoothness`; `residual_pca_bandwidth`
controls the Gaussian neighborhood in normalized route position. With the
default `max_residual_dim=0`, reconstruction equals `projected` and the
unexplained residual equals the compatibility `residual`.

## Parameters that control the learned graph

The most important estimator parameters are:

| Parameter | Role |
| --- | --- |
| `n_centroids` | Resolution of the landmark graph; larger values preserve more detail but cost more |
| `n_backbone_nodes` | Exact target for the final skeletal backbone node count; topology-preserving when set |
| `backbone_node_spacing` | Optional maximum fitted-space edge length; subdivides the final backbone automatically |
| `backbone_node_policy` | `topology_preserving` protects junctions, endpoints, and cycle rank; `allow_topology_relaxation` permits cycle loss when contracting |
| `max_cycles` | Maximum number of cycles allowed in the fitted graph |
| `persistence_threshold` | H1 significance threshold; topological mode interprets it in normalized nearest-neighbour units |
| `topology_neighbors` | Number of local kNN neighbors considered for cycle candidates |
| `mutual_knn` | Retain an observation edge only when both endpoints select each other; default `True` |
| `add_mst` | Add the exact Euclidean minimum spanning tree to the routing graph; default `True` |
| `max_residual_dim` | Number of learned transverse residual-PCA coordinates; default `0` |
| `residual_pca_bandwidth` | Gaussian bandwidth in normalized route position |
| `residual_subspace_smoothness` | Non-negative neighboring-subspace smoothing strength |
| `initialization` | `legacy_coarsen` for the legacy initializer or `skeletal` for topology-aware routing |
| `junction_scales` / `junction_inner_fraction` | Multiscale annulus settings for local branch detection |
| `junction_confidence` | Minimum stable branch-count confidence |
| `use_local_pca` / `local_pca_neighbors` | Enable local tangent and branch-direction estimation |
| `max_branch_angle_degrees` | Maximum allowed departure angle at a detected junction |
| `use_effective_resistance` / `use_electrical_flow` / `use_kron_reduction` | Opt-in electrical connectivity diagnostics and routing support |
| `routing_*_weight` | Relative length, tangent, density, resistance, and current costs |
| `use_tangent_boundary_conditions` | Add PCA-aligned virtual spline control points at open route boundaries |
| `spline_smoothing` | Smoothing strength for route curves |
| `spline_control_mode` | `support` preserves dense support geometry; `backbone` anchors splines to simplified backbone vertices |
| `merge_junction_distance` | Distance used to merge nearby graph junctions; `None` selects an automatic value |
| `prune_short_branches` | Whether very short terminal branches are removed |
| `standardize` | Whether fitting distances are computed after featurewise standardization |
| `persistence_max_points` | Cap used by the persistence calculation for large point clouds |
| `random_state` | Reproducibility for landmark initialization and capped persistence sampling |
| `use_mip` | Use the small SciPy mixed-integer backbone selector; default `True` |
| `coverage_refinement` | Add coverage ribs after residual-PCA fitting; default `False` |
| `coverage_error_tolerance` / `coverage_quantile` | Post-PCA coverage stopping criterion |
| `coverage_max_iterations` / `coverage_max_ribs` | Limits on adaptive refinement |
| `coverage_selection` | `greedy` or `mip` rib selection |
| `rib_candidate_type` | `transverse`, `parallel`, or `both` candidate generation |
| `stability_selection` / `stability_runs` | Optional matched subsampling-based structural support |

There are two different notions of scale in the project:

- estimator parameters control the graph learned from the data;
- visualization parameters control how that graph is drawn.

For example, `MetroLayout(residual_width=0.02)` changes the apparent thickness
of the observation dispersion around a schematic route. It does not alter
route assignments or projection residuals.

## Visualization

The visualization package provides four complementary views:

- a direct feature-space or reduced-space view of observations and fitted
  routes;
- a graph-coordinate view showing `route_id` against `position`;
- a schematic route view showing route connectivity, stations, and residual
  dispersion around each route; and
- an ambient 3D PCA view showing the extracted skeleton as thick bones. Each
  bone is built from sampled one-standard-deviation ellipses in the tangent
  space orthogonal to its spline.

For high-dimensional data, the display reducer is separate from the fitted
graph:

```python
from skeletalembedding.visualization.reduction import fit_reducer

reducer = fit_reducer(X, method="umap", n_neighbors=15)
```

UMAP's `n_neighbors` changes the visual layout's local/global emphasis only.
It does not refit the topological graph. PCA and classical MDS are available
as deterministic display alternatives. The high-dimensional notebook exposes the UMAP
neighbor count, metro dispersion width, and (for topology-aware initialization)
the final backbone node count through interactive sliders. A value of zero
leaves the backbone node count automatic.

The shared four-panel renderer is available through
`skeletalembedding.visualization.plot_embedding_row`. For direct
control of the metro layout:

```python
from skeletalembedding.visualization import MetroLayout

layout = MetroLayout(model, residual_width=0.04).fit(result)
```

The Plotly view in `visualization.interactive` projects the observations and
fitted splines into the first three PCA components of the input. It samples
each spline at normalized positions in `[0, 1]`; at each position it estimates
the local residual covariance in the hyperplane orthogonal to the spline
tangent, draws its 1σ ellipse, and projects that ellipse into the same PCA
space. The result is a data skeleton with thick bones rather than a flattened
metro map. For example:

```python
from skeletalembedding.visualization import plot_spline_3d

figure = plot_spline_3d(
    model,
    result,
    n_spline_samples=24,
    ellipse_bandwidth=0.08,
)
figure.show()
```

`n_spline_samples` controls how many normalized positions receive a cross
section. `ellipse_bandwidth` controls the local neighborhood in normalized
route position, while `ellipse_scale=1.0` is the one-standard-deviation
default.

## Scikit-learn integration

The optional adapters expose route information as ordinary numerical features:

```python
from sklearn.ensemble import RandomForestClassifier

from skeletalembedding.sklearn import (
    SkeletalEmbeddingClassifier,
    SkeletalEmbeddingTransformer,
)

transformer = SkeletalEmbeddingTransformer(n_centroids=32, random_state=0)
features = transformer.fit_transform(X)
result = transformer.transform_result(X)

classifier = SkeletalEmbeddingClassifier(
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
- `cycle_count_`, `persistence_diagram_`, and `normalized_persistence_diagram_`;
- `junctions_` and `endpoints_`: topological regions in topological mode;
- `branch_counts_`, `branch_confidence_`, `local_tangents_`, and
  `junction_branch_directions_`;
- `face_cycle_count_` and `hypercube_dimension_` for verified hypercube-like
  clouds; `junction_degree_shortfall_` and `endpoint_degree_violations_` for
  constraints that could not be realized;
- `effective_resistance_`, `edge_leverage_`, `electrical_traffic_`,
  `backbone_graph_`, and `backbone_paths_` in topological mode;
- `routing_components_`, `component_cycle_counts_`, and `candidate_paths_`;
  these make disconnected-component handling and constrained route candidates
  inspectable.

Fallbacks are recorded in these attributes so a result remains inspectable.
Unexpected backend failures are warned about; expected optional-dependency
fallbacks remain quiet.

## Limitations

SkeletalEmbedding models a sparse one-dimensional backbone and can add
coverage ribs for higher-dimensional manifolds. Residual PCA fields remain
local manifold coordinates; the final rib-filled graph is a geometric
wire-frame approximation, not a claim of topological identity. It does not
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
skeletalembedding/
├── embedding.py                 # public estimator and projection API
├── results.py                   # immutable EmbeddingResult
├── _topology.py                 # graph construction, persistence, local topology
├── _local_geometry.py           # local PCA tangents and branch directions
├── _electrical.py               # resistance, flow, and Kron diagnostics
├── _curves.py                   # route fitting and projection
├── _frames.py                   # deterministic normal frames
├── sklearn.py                   # optional sklearn adapters
├── datasets.py                  # synthetic point clouds
└── visualization/
    ├── metro.py                 # schematic metro layout
    ├── network.py               # feature-space network rendering
    ├── plots.py                 # static plotting and evaluation
    ├── interactive.py           # Plotly PCA skeleton and thick-bone view
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
