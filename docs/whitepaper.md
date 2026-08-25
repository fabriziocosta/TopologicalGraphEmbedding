# Sparse topological spline graph embeddings

## Abstract

Topological graph embedding represents a noisy point cloud with a compact
network of smooth routes. The network is intended to preserve both intrinsic
position and branching or looping structure: an observation is assigned to a
route, projected to a longitudinal position on that route, and accompanied by
its off-route residual. The implementation is designed for two-dimensional
visual analysis and for high-dimensional scientific data, where the route
network is fitted in the original feature space and visualized separately.

The method combines landmark compression, persistent H1 diagnostics, a sparse
symmetrized k-nearest-neighbor topology, spline route fitting, deterministic
normal frames, and a metro-style visualization. Numerical fallbacks are
explicitly reported so that a result remains inspectable when optional
backends are unavailable.

## 1. Problem formulation

Let

$$
X = \{x_1,\ldots,x_n\}, \qquad x_i \in \mathbb{R}^d
$$

be observations sampled around an unknown one-dimensional structure. The
structure may contain endpoints, junctions, open routes, and closed loops. We
seek a small route network

$$
\Gamma = \{\gamma_r : [0,1] \rightarrow \mathbb{R}^d\}_{r=1}^R
$$

such that every observation has a nearest route coordinate

$$
(r_i, u_i) = \arg\min_{r,u}\|x_i - \gamma_r(u)\|_2.
$$

The public result for an observation contains:

$$
(r_i, u_i, \hat{x}_i, e_i, \|e_i\|_2, v_i),
$$

where `route_id` is the route index, `position` is the longitudinal
coordinate, `projected` is \(\hat{x}_i\), `residual` is
\(e_i=x_i-\hat{x}_i\), `residual_norm` is its magnitude, and `tangent` is the
local route tangent in fitting coordinates.

## 2. Fitting pipeline

### 2.1 Validation and metric preparation

The estimator accepts finite, non-empty two-dimensional numeric arrays. Empty
inputs, zero-feature inputs, non-finite values, feature-count mismatches, and
invalid parameter values fail with explicit errors.

When `standardize=True`, the point cloud is centered and scaled featurewise
for distance, persistence, landmark, and spline calculations. Constant
features receive unit scale, preserving finite coordinates for degenerate
clouds. The original coordinates and the fitted affine transform are retained
for public projections and plotting.

The fitted local scale is the median non-zero nearest-neighbor distance. A
fully duplicated cloud has no non-zero neighbor distance, so the implementation
uses a small finite floor instead of producing `NaN`.

### 2.2 Initialization strategies

`SplineGraphEmbedding` preserves the original `coarsen` initializer by
default. It compresses observations with deterministic k-means, builds an
MST, adds local cycle-closing edges, and then extracts spline chains.

The optional `topological` initializer separates topology, connectivity, and
geometry. It builds a weighted symmetric kNN graph over standardized
observations. Edge lengths provide geometric cost; affinity weights provide
conductances for electrical diagnostics. The graph is a dense routing
substrate and is not itself returned as the manifold backbone. If it has
disconnected natural components, a bridge is retained for electrical
diagnostics only; candidate routes remain within their original component.

All repository notebook workflows explicitly select this initializer. The
estimator’s Python API retains `backbone_initialization="coarsen"` as its
compatibility default; callers can select `"topological"` explicitly.

Persistent H1 is computed on the existing capped persistence backend. Raw bars
remain available in `persistence_diagram_`; `normalized_persistence_diagram_`
expresses birth and death values in median-nearest-neighbour units. The
significant normalized bars determine the requested cycle rank.

Multiscale annuli around k-means prototype candidates provide local branch
counts. Stable counts of one, two, or at least three identify endpoints,
regular points, and junctions. Nearby candidates are clustered into one
`JunctionRegion` or `EndpointRegion`, each retaining a center, confidence, and
source members. Coarse landmarks are used only to stabilize singular-region
candidates; final routes are selected through the dense graph.

### 2.3 Landmark compression and local geometry

The point cloud is compressed to at most `n_centroids` landmarks by a
deterministic NumPy k-means implementation with k-means++ initialization. Its
distance calculations are blocked: the implementation forms point-anchor
Gram blocks rather than a dense `(n, k, d)` tensor. This keeps memory usage
linear in the number of observations and features for each block.

The landmark graph is the computational representation of the topology. Each
landmark retains its standardized coordinate, and graph edges store Euclidean
lengths in the fitting metric.

### 2.4 Sparse topology and electrical connectivity

In topological mode, local PCA supplies unoriented tangent fields at ordinary
vertices and independently estimated outward directions for each junction
arm. Candidate path departures are rejected when their oriented angle exceeds
`max_branch_angle_degrees`; ordinary edges use the sign-invariant tangent
consistency term (1-|u_i^T u_j|).

When enabled, the conductance Laplacian supplies effective resistance,
leverage (w_eR_e), aggregate source-target current, and an optional Kron
reduction on retained landmarks. These quantities are connectivity evidence,
not topology; their routing weights default to zero.

The selector first builds a low-cost connected landmark structure, completes
missing junction arms, removes redundant cycles when necessary, and adds
cycle-closing candidates until the requested rank is reached. Maximal degree-2
paths are retained as support points but collapsed into `backbone_graph_` edges
before spline fitting.

In coarsen mode, the initial graph remains a minimum spanning tree. The MST is
used only to guarantee connectivity; it is not a global cycle generator.
Additional candidate edges come from the symmetrized landmark kNN graph,
controlled by `topology_neighbors` and defaulting to six.

For each candidate edge, the estimator rejects microscopic chords and edges
that do not close a sufficiently long local path. Among eligible local edges,
the route with the strongest path-to-chord contrast is selected until the
requested H1 target is reached or candidates are exhausted. This restricts
topological shortcuts to local landmark neighborhoods and avoids the dense
all-pairs shortcut heuristic used by the prototype.

Persistent homology supplies the target. Ripser is used when available;
otherwise a NumPy Vietoris–Rips H1 fallback is used. Backend selection and
unexpected Ripser failures are recorded; unexpected failures are warned about.

After fitting, topology diagnostics are available as:

| Attribute | Meaning |
| --- | --- |
| `persistent_cycle_count_` | Significant H1 bars under the persistence threshold |
| `requested_cycle_count_` | Target after applying `max_cycles` |
| `realized_cycle_count_` | Cycle rank of the fitted landmark graph |
| `topology_shortfall_` | Requested cycles that local candidates could not realize |
| `persistence_backend_` | `ripser`, `numpy`, or `numpy-after-ripser-error` |
| `topology_candidate_edges_` | Symmetrized local kNN candidates |
| `cycle_count_` | Significant normalized H1 cycle count |
| `junction_regions_` / `endpoint_regions_` | Clustered local-topology regions |
| `backbone_graph_` / `backbone_paths_` | Selected abstract graph and point-level route supports |
| `effective_resistance_` / `edge_leverage_` | Optional edge electrical diagnostics |
| `electrical_traffic_` | Optional normalized aggregate current support |
| `routing_components_` / `component_cycle_counts_` | Natural routing components and their persistent cycle counts |
| `candidate_paths_` | Constrained dense-substrate routes considered by the selector |

Linear structure detection produces an ordered path graph for noisy lines. For
other geometries, short terminal branches can be pruned and nearby junctions
merged using validated geometric thresholds. The resulting route chains are
stored in `route_chains_`, while `junctions_` and `endpoints_` identify graph
landmarks by degree.

## 3. Route geometry and projection

Each route chain is represented by a dense sampled curve. SciPy smoothing
splines are preferred for open and closed chains. If SciPy fitting is
unavailable or fails numerically, a deterministic NumPy Catmull–Rom/polyline
fallback is used. Every route records its backend in `route_backends_`.

Projection is batched per route. Each batch is compared with the sampled
route using squared distances, and the closest route wins. The implementation
does not return an invalid route identifier: if no route can be selected, it
raises an explicit diagnostic containing the invalid observation count.

Projection is expressed publicly in the original feature coordinates. The
tangent is retained in standardized fitting coordinates because that is the
metric used to define the local normal hyperplane.

## 4. Deterministic normal coordinates

For a route tangent \(v(u)\), the normal coordinate system is an orthonormal
basis of the \((d-1)\)-dimensional complement of \(v(u)\). A naïve basis built
from the query batch can change when a subset is transformed. The estimator
therefore learns a frame grid for every fitted route:

1. initialize the first frame by deterministic Gram–Schmidt against the
   coordinate axes;
2. transport the frame along a fixed route parameter grid using projection and
   QR re-orthogonalization;
3. interpolate the stored grid for each query tangent;
4. re-orthogonalize against the query tangent for numerical stability.

Closed routes use a periodic parameter grid and explicitly reuse the first
frame at the seam. Consequently, transforming the full dataset and then a
subset produces identical normal coordinates for the shared observations in
2D, 3D, and higher dimensions.

The estimator exposes these values through:

```python
normal = model.normal_coordinates(result)
```

The returned array has shape `(n_samples, n_features - 1)`. In one dimension,
the normal coordinate array has zero columns.

## 5. Visualization architecture

Visualization is part of the main package rather than a notebook-local copy.
The package layout is:

```text
topological_graph_embedding/
├── embedding.py                 # public estimator
├── results.py                   # typed immutable result
├── _topology.py                 # landmarks, graph, persistence
├── _curves.py                   # route fitting and projection
├── _frames.py                   # deterministic normal frames
├── sklearn.py                   # optional sklearn adapters
├── datasets.py                  # synthetic point clouds
└── visualization/
    ├── metro.py                 # MetroLayout schematic routing
    ├── network.py               # fitted-network Matplotlib rendering
    ├── plots.py                 # static plots and route evaluation
    ├── interactive.py           # Plotly 3D rendering
    ├── reduction.py             # PCA/MDS/UMAP display reducers
    └── paul.py                  # optional Paul et al. display helpers
```

`MetroLayout` converts graph connectivity, route arc length, junction discs,
and endpoint directions into a readable schematic. Observations are placed at
their route positions and offset laterally using residual magnitude and local
residual PCA directions. `plot_network` provides a direct feature-space view;
`plots.py` provides the shared four-panel renderer used by the notebooks; and
`interactive.py` adds a Plotly route/residual-plane view.

Visualization functions consume `EmbeddingResult` attributes directly. This
keeps route identity, position, projection, and residual semantics consistent
across static, schematic, and interactive views.

## 6. Estimator interfaces

The core estimator is:

```python
from topological_graph_embedding import SplineGraphEmbedding

model = SplineGraphEmbedding(
    n_centroids=32,
    backbone_initialization="topological",
    max_cycles=5,
    topology_neighbors=6,
    random_state=0,
)
result = model.fit_transform(X)
normal = model.normal_coordinates(result)
```

`EmbeddingResult` is a frozen dataclass with six fields:
`route_id`, `position`, `projected`, `residual`, `residual_norm`, and
`tangent`. It is attribute-based and intentionally does not implement mapping
or legacy field aliases.

The optional sklearn adapters are:

```python
from topological_graph_embedding.sklearn import (
    SplineEmbeddingClassifier,
    SplineEmbeddingTransformer,
)

transformer = SplineEmbeddingTransformer()
features = transformer.fit_transform(X)
result = transformer.transform_result(X)

classifier = SplineEmbeddingClassifier(estimator=downstream_estimator)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

The transformer emits numeric route indicators, position, residual norm, and
scaled residual components. The classifier delegates to a cloned downstream
estimator using route/position features plus deterministic normal coordinates.

## 7. Computational considerations

Let `m` be the number of landmarks, `d` the feature dimension, and `b` a
distance block size. Landmark k-means uses blocked distance products of size
`b × m`; topology calculations operate on the much smaller landmark graph.
Route projection is batched over observations and route samples rather than
materializing a global observation-route-feature tensor. The persistence
fallback is intended for moderate point clouds and may be capped with
`persistence_max_points`.

The sparse topology is an approximation. Its diagnostics make approximation
visible: a nonzero `topology_shortfall_` means the persistence target was not
realized by the available local candidates. A backend string beginning with
`numpy` means the approximate persistence implementation was used.

## 8. Reproducibility and limitations

Randomized stages use `random_state`, including landmark initialization and
subsampling for capped persistence. Numerical outputs can still depend on the
installed SciPy/Ripser versions when optional backends are available; the
chosen backend is recorded on the fitted estimator.

The route network is a one-dimensional approximation. It does not model
branch-specific density, uncertainty, or a full continuous projection
optimization. Visualization is also a display transform: metro coordinates
are not intended to preserve distances in the original feature space.

The test suite covers synthetic line, branch, circle, figure-eight, and loop
topologies; projection validity; degenerate and one-dimensional data;
batch-independent normal frames; backend reporting; sklearn compatibility;
visualization; package installation; and the absence of removed legacy names.
