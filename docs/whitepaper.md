# Sparse spline graph embeddings for noisy point clouds

## Abstract

This work describes a spline-based representation for point clouds whose
latent structure is approximately one-dimensional and may contain branches or
cycles. The method constructs a sparse neighborhood graph, estimates local
geometric and topological structure, selects a graph backbone, and fits smooth
routes to its paths. Each observation is then assigned to an approximate
nearest route coordinate and represented by its route identity, longitudinal
parameter, projection, local tangent, and off-route residual.

Fitting is performed in the original feature space, after an optional affine
standardization. Dimensionality-reduction methods and the metro-style layout
are used only for visualization. The implementation exposes diagnostics for
the distinction between the persistence-derived cycle target and the cycle
rank realized by the selected graph, and records numerical fallbacks when
optional backends are unavailable. Backend details and the complete software
interface are documented separately in [Implementation details](implementation.md).

## 1. Problem formulation

Let

$$
X = \{x_1,\ldots,x_n\}, \qquad x_i\in\mathbb{R}^d,
$$

be a finite set of observations. We assume that the observations are sampled
near an unknown one-dimensional structure that may contain endpoints,
junctions, open paths, and closed loops. The objective is to construct a
compact route network that provides both intrinsic coordinates and an
off-network error for each observation.

Let $T$ denote the optional featurewise affine standardization used during
fitting, and let $z_i=T(x_i)$. When standardization is disabled, $T$ is the
identity. The fitted route network is

$$\widetilde{\Gamma}=\{\widetilde{\gamma}_r:[0,1]\rightarrow\mathbb{R}^d\}_{r=1}^{R}.$$

where the curves are expressed in fitting coordinates. For each observation,
the implementation approximates

$(r_i,u_i)\approx\arg\min_{r,u}\|z_i-widetilde{\gamma}_r(u)\|_2$.

The minimization is evaluated on a dense piecewise-linear representation of
each route rather than by continuous optimization. Consequently, the route
assignment and projection are approximate. The parameter $u_i\in[0,1]$ is a
normalized route parameter; it represents progress along a route and is not a
physical distance.

## 2. Method

The method separates graph selection from route geometry. A sparse graph is
used to identify connectivity, branches, and cycles. Smooth curves are then
fitted to the selected graph paths. This separation allows the topology and
the geometry to be inspected independently.

### 2.1 Metric preparation

The estimator accepts finite, non-empty, two-dimensional numeric arrays. It
rejects invalid shapes, non-finite values, incompatible feature counts, and
invalid parameter values.

With standardization enabled, each feature is centered and scaled before
distance, persistence, graph, and spline calculations. Constant features
receive unit scale, which preserves finite coordinates for degenerate inputs.
The inverse affine transform is retained so that public projections and
residuals can be reported in the original feature units. A local scale is
estimated from non-zero nearest-neighbor distances; a small finite floor is
used when all observations are duplicated.

### 2.2 Neighborhood graph and landmarks

The primary graph substrate is a symmetrized $k$-nearest-neighbor graph. Each
observation is connected to nearby observations, and an edge is retained when
the neighborhood relation is present in either direction. Edge lengths provide
geometric costs. Affinity weights can additionally be interpreted as
conductances for connectivity diagnostics.

For larger datasets, observations are compressed into at most `n_centroids`
landmarks using deterministic k-means with k-means++ initialization. The
landmark graph is used for most topology-selection operations. Distance
calculations are blocked so that the implementation does not materialize a
full observation-by-landmark-by-feature tensor.

Two initialization schemes are available. The `coarsen` scheme constructs a
landmark minimum spanning tree and then considers local cycle-closing edges.
The `topological` scheme uses the neighborhood graph together with local
geometry and topology diagnostics when selecting the backbone. `coarsen`
remains the compatibility default in the Python API.

### 2.3 Local geometry and topology

Local principal component analysis (PCA) estimates unoriented tangent fields
at ordinary graph vertices. At candidate junctions, separate outward
directions are estimated for the individual arms. Candidate paths are rejected
when their direction is inconsistent with these local estimates. Ordinary-edge
tangent consistency is sign-invariant and uses $1-|u_i^T u_j|$.

Branch structure is estimated from neighborhoods examined at several distance
scales. Connected components in thin annular regions provide evidence for
endpoints, regular points, and junctions. Nearby candidates are clustered into
regions to reduce sensitivity to sampling noise.

Loop structure is estimated with persistent homology. Persistent homology
examines connectivity over a range of distance scales; its first homology
group, $H_1$, records one-dimensional holes. Long-lived $H_1$ features are
used as evidence for cycles. The resulting persistence estimate supplies a
requested cycle count, but it does not guarantee that the selected graph will
realize that count.

Optional electrical diagnostics use a conductance Laplacian to compute
effective resistance, edge leverage, and aggregate current support. These
quantities provide additional connectivity evidence; they do not define the
topology, and their routing weights default to zero.

### 2.4 Backbone selection

The selector constructs a low-cost connected landmark structure, completes
missing junction arms, and considers local cycle-closing candidates. Candidate
paths combine geometric length with tangent consistency, density, and optional
connectivity terms. Locality constraints reject microscopic chords and
shortcuts that do not close a sufficiently long path.

The selected graph is simplified before route fitting. Maximal paths whose
internal vertices have degree two are collapsed into backbone edges while
retaining their support points. The resulting `backbone_graph_` records the
abstract connectivity, and `backbone_paths_` records the point-level supports
used to fit routes.

The implementation records both the persistence-derived target and the cycle
rank of the selected graph. Their difference is exposed as a topology
shortfall, rather than being hidden as a successful reconstruction.

### 2.5 Route fitting and projection

Each backbone path is fitted as an open or closed route. Smoothing splines are
preferred when the numerical backend supports them. A deterministic NumPy
Catmull–Rom or polyline representation is used when spline fitting is
unavailable or fails numerically.

The fitted routes are sampled densely. For each observation, squared distances
to the sampled segments of every route are evaluated in batches. The closest
valid route determines the route identifier, normalized position, and fitting-
space projection. The projection is then mapped back to the original feature
coordinates. If no valid route can be selected, the transformation raises an
explicit error rather than returning a sentinel route identifier.

## 3. Public embedding representation

For an observation, the public result corresponds to

$$
(r_i,u_i,\hat{x}_i,e_i,\|e_i\|_2,v_i),
$$

with the following field mapping:

| Field | Mathematical meaning |
| --- | --- |
| `route_id` | Route index $r_i$ |
| `position` | Normalized route parameter $u_i$ |
| `projected` | Original-space projection $\hat{x}_i$ |
| `residual` | $e_i=x_i-\hat{x}_i$ |
| `residual_norm` | Euclidean norm $\|e_i\|_2$ |
| `tangent` | Local unit tangent $v_i$ in fitting coordinates |

The coordinate distinction is intentional. The projection and residual are
reported in the units of the input data, whereas the tangent is retained in
fitting coordinates because it defines the local normal geometry used by the
normal-coordinate transform.

## 4. Deterministic normal coordinates

At a route point, the tangent defines the along-route direction. Its
orthogonal complement has dimension $d-1$ and provides local normal
coordinates for the residual. The estimator constructs a deterministic
orthonormal frame along each route rather than building a new frame for every
query batch.

The frame is initialized against the coordinate axes, transported along a
fixed route-parameter grid, and re-orthogonalized for numerical stability.
Closed routes use a periodic parameter grid and reuse the first frame at the
seam. This construction makes normal coordinates invariant to whether the
full dataset or a subset is transformed, up to floating-point roundoff.

The resulting array has shape `(n_samples, n_features - 1)`. In one feature
dimension, it has zero columns.

## 5. Visualization

Visualization is downstream of fitting. PCA, MDS, and Uniform Manifold
Approximation and Projection (UMAP) may be used to display high-dimensional
observations and the fitted routes, but they do not change route assignments,
projections, or residuals.

The metro-style layout converts graph connectivity, route arc length, junctions,
and endpoint directions into a schematic drawing. Observations are placed at
their route positions and offset using residual magnitude and local residual
directions. Metro coordinates are intended to improve readability and are not
a metric embedding of the original data.

## 6. Computational considerations

Landmark compression reduces the graph on which topology selection operates.
Blocked distance calculations avoid materializing a global
observation-landmark-feature tensor, and route projection is batched over
observations and route samples. The persistence fallback is intended for
moderate point clouds and can be capped with `persistence_max_points`.

The synthetic notebook keeps its seven easy examples at cap `60` and isolates
the higher-detail polygon/hypercube demonstrations in a separate cell at cap
`300` with normalized H1 threshold `4.0`; electrical resistance/current
support remains disabled by default. The NumPy persistence fallback uses a
smaller internal cap because its full Rips 2-skeleton is cubic.

The method has several approximation sources: landmark compression, sparse
neighborhood construction, sampled route projection, and the persistence
fallback. The topology shortfall diagnostic identifies one important failure
mode: the persistence-derived cycle target was not realized by the available
local candidates.

## 7. Reproducibility and limitations

Randomized stages accept `random_state`, including landmark initialization and
subsampling for capped persistence. Numerical outputs may depend on installed
SciPy or Ripser versions when optional backends are available; the selected
backend is recorded by the fitted estimator.

The representation is one-dimensional and does not model branch-specific
density or uncertainty. Projection is not a full continuous optimization, and
the inferred topology is an approximation controlled by graph construction,
local geometric thresholds, persistence thresholds, and cycle limits. These
parameters should therefore be reported with empirical results.

The complete API, diagnostic attributes, backend behavior, and usage examples
are maintained in [Implementation details](implementation.md).
