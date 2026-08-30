# Skeletal embeddings for noisy point clouds

## Abstract

This work describes a spline-based representation for point clouds whose
latent structure may contain graph-like branches, cycles, or higher-dimensional
manifold regions. The method constructs a sparse neighborhood graph, estimates
stable topology and local geometry, selects a topology-driven backbone, and
fits smooth splines. Smooth tangent-orthogonal residual-PCA fields add local
manifold coordinates. When those coordinates do not meet a coverage target,
adaptive stable ribs form a sparse geometric wire frame.

Fitting is performed in the original feature space, after an optional affine
standardization. Dimensionality-reduction methods and the schematic layout
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

$$
\tilde{\Gamma}=\{\tilde{\gamma}_r:[0,1]\rightarrow\mathbb{R}^d\}_{r=1}^{R}.
$$

where the curves are expressed in fitting coordinates. For each observation,
the implementation approximates

$(r_i,u_i)\approx\arg\min_{r,u}\|z_i-\tilde{\gamma}_r(u)\|_2$.

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

The topology-aware scheme uses the neighborhood graph together with local
geometry and topology diagnostics when selecting the backbone. A small MIP
solves the connectivity, degree, and cycle constraints, with deterministic
fallback when the solver cannot produce a feasible solution.

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

For a verified hypercube-like sample, geometric faces are reported separately
from independent graph cycles. A three-dimensional cube therefore has eight
degree-three junctions and six square faces, but its graph has cycle rank five
because $E-V+1=12-8+1$. The estimator exposes the face count as
`face_cycle_count_` and keeps `realized_cycle_count_` as the graph-theoretic
rank.

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
| `residual_coordinates` | Learned transverse coordinates $z_i$ in fitting coordinates |
| `reconstructed` | $\hat{x}_i + U_i z_i$ in original feature units |
| `unexplained_residual` | $\epsilon_i=x_i-\mathrm{reconstructed}_i$ in original feature units |
| `unexplained_residual_norm` | Euclidean norm of $\epsilon_i$ |

The coordinate distinction is intentional. The projection and residual are
reported in the units of the input data, whereas the tangent is retained in
fitting coordinates because it defines the local normal geometry used by the
normal-coordinate transform.

When `max_residual_dim > 0`, the residual is further decomposed as

$$
x_i \approx \hat{x}_i + U_{r_i}(u_i)z_i + \epsilon_i.
$$

The basis $U_r(u)$ is learned from Gaussian-weighted second moments of the
standardized centerline residuals along each route. It is constrained to the
normal hyperplane of the spline tangent and interpolated from a fixed route
grid at transformation time. Neighboring projectors are optionally averaged
for five passes with weight $\lambda/(1+\lambda)$; closed routes use cyclic
neighbors and share their basis at the seam. The effective dimension is
$\min(\texttt{max\_residual\_dim},d-1)$. With the default zero dimension, the
new fields are backfilled so the decomposition reduces exactly to the legacy
centerline projection and residual.

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

The schematic layout converts graph connectivity, route arc length, junctions,
and endpoint directions into a readable drawing. The interactive 3D skeleton
view instead projects the fitted splines into the first three PCA components.
At sampled normalized positions it estimates residual covariance in the tangent
space orthogonal to each spline, draws its one-standard-deviation ellipse, and
projects the ellipse into PCA space. These cross-sections form thick bones
around the extracted data skeleton. Neither display is a metric embedding of
the original data or a change to the fitted route assignments.

## 6. Computational considerations

Landmark compression reduces the graph on which topology selection operates.
Blocked distance calculations avoid materializing a global
observation-landmark-feature tensor, and route projection is batched over
observations and route samples. The persistence fallback is intended for
moderate point clouds and can be capped with `persistence_max_points`.

The synthetic notebook keeps its six easy examples at cap `60` and isolates
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

The backbone targets persistent coarse topology; the rib-filled skeleton is a
geometric coverage approximation and need not have the same cycle rank as the
underlying manifold. Projection is not a full continuous optimization, and the
inferred structure is controlled by graph construction, local geometric
thresholds, persistence, coverage penalties, and stability settings. These
parameters should therefore be reported with empirical results.

The complete API, diagnostic attributes, backend behavior, and usage examples
are maintained in [Implementation details](implementation.md).
