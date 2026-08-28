# Implementation details

This document describes the current implementation of
`SplineGraphEmbedding`. The [whitepaper](whitepaper.md) gives the conceptual
description; this document records the data flow, algorithms, public fields,
fallbacks, and implementation-specific trade-offs.

## 1. Objects and coordinate conventions

The implementation uses four related graph objects:

- The observation graph is a weighted, symmetrized k-nearest-neighbor graph
  over the input observations.
- The landmark graph is built after compressing observations into at most
  `n_centroids` landmarks. It is the main graph used for route selection.
- The backbone graph is the selected abstract graph after topology and
  connectivity constraints have been applied.
- A route chain is a maximal degree-2 path in the backbone graph. Each chain
  is fitted as one open or closed curve and becomes a public route.

These objects should not be treated as interchangeable. The observation and
landmark graphs provide candidate structure; the backbone graph is the
selected structure; and fitted routes provide the geometry used for projection.

Let $x_i$ denote an observation in the original feature coordinates. With
standardization enabled, the fitting coordinates are
$z_i=(x_i-\mu)/s$, applied featurewise, where constant features receive unit
scale. Without standardization, $\mu=0$ and $s=1$. Routes and tangents are
fitted in $z$-coordinates, but public projections and residuals are returned
in $x$-coordinates.

For an observation, `EmbeddingResult` contains:

| Field | Meaning |
| --- | --- |
| `route_id` | Integer index of the selected fitted route |
| `position` | Normalized route parameter in $[0,1]$ |
| `projected` | Projected point in the original feature coordinates |
| `residual` | `X - projected` in the original feature coordinates |
| `residual_norm` | Euclidean norm of `residual` |
| `tangent` | Local unit tangent in fitting coordinates |
| `residual_coordinates` | Learned local PCA coordinates in fitting units; shape `(n_samples, residual_dim_)` |
| `reconstructed` | Route projection plus learned residual-PCA reconstruction, in original units |
| `unexplained_residual` | Original observation minus `reconstructed` |
| `unexplained_residual_norm` | Euclidean norm of `unexplained_residual` |

The position parameter follows the cumulative length of the fitted support
points, but it is normalized to $[0,1]$ and should not be interpreted as a
physical distance.

## 2. Input validation and metric preparation

The estimator accepts finite, non-empty, two-dimensional numeric arrays. It
rejects empty inputs, zero-feature inputs, non-finite values, feature-count
mismatches, and invalid parameter values before fitting begins.

When `standardize=True`, each feature is centered and scaled before distance,
persistence, landmark, graph, and spline calculations. A constant feature is
assigned unit scale. This keeps degenerate inputs finite while preserving the
constant coordinate in the original-space projection.

The local scale is the median non-zero nearest-neighbor distance. If every
observation is duplicated and no such distance exists, the implementation uses
a small finite floor instead of producing `NaN` values.

## 3. Landmark compression and initialization

### 3.1 Coarsening mode

`backbone_initialization="coarsen"` is the compatibility default. The
implementation compresses observations with deterministic NumPy k-means using
k-means++ initialization, constructs a landmark minimum spanning tree, adds
local cycle-closing candidates, and extracts route chains.

The MST guarantees a connected initial structure. It is not treated as a
global cycle generator. Additional candidate edges come from the
symmetrized landmark k-nearest-neighbor graph, controlled by
`topology_neighbors`.

### 3.2 Topological mode

`backbone_initialization="topological"` builds a weighted, symmetric kNN graph
over the fitting coordinates. Edge lengths supply geometric costs. Affinity
weights supply conductances for optional electrical diagnostics. With
`mutual_knn=True`, only reciprocal neighbor pairs are retained. The kNN graph
is a routing substrate, not the final manifold backbone. With `add_mst=True`,
the exact Euclidean minimum spanning tree is added to the selected kNN edges
before natural components are recorded. The MST pass uses linear memory but
quadratic time in the number of observations, so it is an opt-in setting.

If the natural kNN graph is disconnected, a bridge may be added for electrical
calculations. Route selection still keeps the original components separate.
Candidate routes therefore cannot cross a bridge introduced only for
diagnostics.

Landmark compression is used to keep topology selection tractable. Distance
calculations are blocked: point-anchor Gram blocks are formed instead of a
dense `(n, k, d)` tensor. A block of `b` observations and `m` landmarks uses
memory proportional to the block rather than to the full observation set.

## 4. Local geometry and topology selection

Persistent homology provides a cycle target. Ripser is used when available;
otherwise the implementation uses a capped NumPy Vietoris–Rips H1 fallback.
Raw and normalized persistence diagrams are retained on the fitted estimator.
The normalized diagram expresses birth and death values in median-neighbor
units. Significant bars provide a requested cycle count, while the selected
landmark graph has its own realized cycle rank. These values may differ.

Multiscale annuli around prototype candidates estimate local branch counts.
Stable counts distinguish endpoint, regular, and junction candidates. Nearby
candidates are clustered into `JunctionRegion` and `EndpointRegion` records
with a center, confidence, and source members.

In topological mode, local PCA supplies unoriented tangent fields at ordinary
vertices and outward directions for individual junction arms. Candidate path
departures are rejected when their oriented angle exceeds
`max_branch_angle_degrees`. For ordinary edges, tangent consistency is
sign-invariant and uses

$1-|u_i^T u_j|$.

When enabled, the conductance Laplacian supplies effective resistance, edge
leverage, aggregate source-target current, and an optional Kron reduction.
These quantities are connectivity evidence. They do not define topology, and
their routing weights default to zero.

The selector then:

1. builds a low-cost connected landmark structure;
2. completes missing junction arms;
3. removes redundant cycles when necessary;
4. adds eligible cycle-closing candidates until the requested rank is reached
   or candidates are exhausted; and
5. collapses maximal degree-2 paths into `backbone_graph_` edges before spline
   fitting.

Candidate edges are rejected when they form microscopic chords or fail to close
a sufficiently long local path. The remaining candidates are ranked by their
path-to-chord contrast. This limits shortcuts to local landmark neighborhoods
instead of considering all landmark pairs.

Hypercube-like clouds receive one additional structural diagnostic. For a
3D cube, `face_cycle_count_` is 6 (the square faces), while
`realized_cycle_count_` remains 5 (the independent cycle rank
`E - V + 1`); `hypercube_dimension_` records the detected dimension. The
specialized candidate routes preserve all eight degree-3 corners when the
observed arms are supported. Degree and endpoint constraint failures remain
visible through `junction_degree_shortfall_` and
`endpoint_degree_violations_`.

The main topology diagnostics are:

| Attribute | Meaning |
| --- | --- |
| `persistent_cycle_count_` | Significant H1 bars before `max_cycles` is applied |
| `requested_cycle_count_` | Cycle target after applying `max_cycles` |
| `realized_cycle_count_` | Cycle rank of the selected landmark graph |
| `topology_shortfall_` | Requested cycles not realized by available candidates |
| `persistence_backend_` | `ripser`, `numpy`, or `numpy-after-ripser-error` |
| `topology_candidate_edges_` | Symmetrized local kNN candidates |
| `cycle_count_` | Significant normalized H1 cycle count |
| `face_cycle_count_` / `hypercube_dimension_` | Geometric face count and dimension for verified hypercube-like clouds |
| `junction_regions_` / `endpoint_regions_` | Clustered local-geometry regions |
| `backbone_graph_` / `backbone_paths_` | Selected graph and route supports |
| `effective_resistance_` / `edge_leverage_` | Optional electrical diagnostics |
| `electrical_traffic_` | Optional normalized aggregate current support |
| `routing_components_` / `component_cycle_counts_` | Natural routing components and their cycle counts |
| `candidate_paths_` | Constrained routes considered by the selector |

## 5. Route fitting and projection

Each route chain is represented by a dense sampled curve. SciPy smoothing
splines are preferred for open and closed chains. If SciPy fitting is
unavailable or fails numerically, the implementation uses a deterministic
NumPy Catmull–Rom or polyline fallback. The selected backend for each route is
stored in `route_backends_`. `spline_control_mode="support"` fits the dense
route support path, preserving the historical behavior. The optional
`"backbone"` mode keeps that ordered support path for geometry but snaps its
nearest points to the simplified landmark-backbone vertices and gives those
anchors high fitting weight. This lets smoothing remove observation-level
zigzags without allowing the spline to drift away from the selected backbone.

Projection is performed in batches. For each route, observations are compared
with the route's sampled line segments using squared distances. The closest
route and segment determine `route_id`, `position`, and the fitting-space
projection. That projection is then mapped back to the original feature
coordinates.

If no route can be selected for an observation, `transform` raises an explicit
diagnostic containing the invalid observation count. It does not return a
sentinel route identifier.

## 6. Residual PCA fields

Residual PCA is optional and is disabled by the compatibility default
`max_residual_dim=0`. After route projection, training observations are
assigned to their fitted route and normalized position. Their centerline
residuals are expressed in standardized fitting coordinates and projected into
the deterministic route-normal frame. At each route-grid position, a
Gaussian-weighted second-moment matrix is eigendecomposed and its leading
`min(max_residual_dim, n_features - 1)` directions are stored as an ambient
orthonormal basis in `residual_bases_`; the corresponding eigenvalues are in
`residual_eigenvalues_` and the normalized grids are in
`residual_parameter_grids_`.

The basis is orthogonal to the spline tangent and is re-orthonormalized after
interpolation. Neighboring PCA frames are aligned by a Procrustes rotation.
When `residual_subspace_smoothness > 0`, five fixed passes average neighboring
projectors with weight `lambda / (1 + lambda)`, including cyclic neighbors for
closed routes. Transformation uses only these fitted grids, so full-dataset
and subset transforms are deterministic. The learned coordinates are in
standardized fitting units; reconstructed and unexplained residual arrays are
mapped back to original feature units.

The decomposition is

$$
x = \gamma_r(u) + U_r(u)z + \epsilon,
$$

where the compatibility fields `projected` and `residual` continue to refer
to the centerline projection and centerline residual. With zero learned
dimensions, `reconstructed == projected` and `unexplained_residual == residual`.

## 7. Deterministic normal frames

For a route tangent $v(u)$, normal coordinates use an orthonormal basis of the
$(d-1)$-dimensional complement of $v(u)$. Building a basis independently from
each query batch would make the result dependent on which observations were
transformed together. The estimator instead stores a frame grid for every
fitted route:

1. initialize the first frame by deterministic Gram–Schmidt against the
   coordinate axes;
2. transport the frame along a fixed route-parameter grid using projection and
   QR re-orthogonalization;
3. interpolate the stored grid for each query tangent; and
4. re-orthogonalize against the query tangent for numerical stability.

Closed routes use a periodic parameter grid and reuse the first frame at the
seam. As a result, transforming the full dataset and then a subset gives the
same normal coordinates for shared observations, up to floating-point
roundoff.

The public method is:

```python
normal = model.normal_coordinates(result)
```

The returned array has shape `(n_samples, n_features - 1)`. In one dimension,
it has zero columns.

## 8. Visualization

Visualization consumes `EmbeddingResult` directly. This keeps route identity,
position, projection, and residual semantics consistent across views.

`plot_network` shows the fitted network in feature space. Static plotting
workflows provide shared multi-panel views, while `interactive.py` provides a
Plotly PCA skeleton view with thick-bone cross-sections. `MetroLayout` converts graph connectivity,
route arc length, junctions, and endpoint directions into a schematic layout.
Observations are placed at their route positions and offset laterally using
residual magnitude and local residual directions.

The Plotly skeleton view fits PCA to the observations represented by the
result, projects each fitted spline into the first three components, and
samples normalized route positions. At every sample it estimates residual
covariance in the feature-space hyperplane orthogonal to the spline tangent,
renders the one-standard-deviation ellipse, and projects that ellipse into
the ambient PCA coordinates. `n_spline_samples` controls the number of
cross-sections and `ellipse_bandwidth` controls their local neighborhood.

PCA, classical MDS, and UMAP are display reducers. They are not used to fit the
route network and do not change route assignments or residuals.

## 9. Public interfaces

The core estimator can be used as follows:

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

`EmbeddingResult` is a frozen dataclass. In addition to the six compatibility
fields, it provides `residual_coordinates`, `reconstructed`,
`unexplained_residual`, and `unexplained_residual_norm`. Legacy six-field
construction is backfilled with empty coordinates, `projected`, `residual`,
and its norm. It is attribute-based and does not implement mapping behavior or
legacy field aliases.

The optional scikit-learn adapters are:

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

The transformer emits route indicators, position, residual norm, and either
the legacy scaled residual components or learned `residual_pca_*` components
when enabled. The classifier delegates to a cloned downstream estimator using
route/position features and either learned residual-PCA coordinates or the
legacy deterministic normal coordinates.

## 10. Computational considerations

Let `m` be the number of landmarks, `d` the feature dimension, and `b` the
distance block size. Landmark k-means forms blocked distance products of size
`b × m`. Topology calculations operate on the smaller landmark graph. Route
projection is batched over observations and route samples rather than
materializing a global observation-route-feature tensor.

The persistence fallback is intended for moderate point clouds and can be
capped with `persistence_max_points`. A nonzero `topology_shortfall_` means
that the requested persistence-derived cycle target was not realized by the
available local candidates. A backend name beginning with `numpy` indicates
that the approximate persistence implementation was used.

The synthetic notebook runs simple examples with `persistence_max_points=60`
and isolates its detailed polygon/hypercube cell at cap `300` with normalized
threshold `4.0`. This is a workflow setting, not an estimator-default change.
When Ripser is unavailable, the pure-NumPy fallback internally caps its sample
at 120 because its full Rips 2-skeleton is cubic. Effective resistance,
electrical flow, and their routing weights remain opt-in.

## 11. Reproducibility and limitations

Randomized stages use `random_state`, including landmark initialization and
subsampling for capped persistence. Numerical outputs can still depend on the
installed SciPy and Ripser versions when optional backends are available. The
chosen backend is recorded on the fitted estimator.

The route network is a one-dimensional approximation. It does not model
branch-specific density or uncertainty, and projection is not a full
continuous optimization. Metro coordinates are for display and are not
intended to preserve distances in the original feature space.
