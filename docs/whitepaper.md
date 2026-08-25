# Sparse spline graph embeddings for noisy point clouds

## Abstract

Many datasets are stored as point clouds: each observation is a vector of
numbers, but the observations may be arranged around a simpler shape. That
shape might be a line, a loop, or a branching structure such as a Y-shaped
trajectory.

This project represents such data with a small graph whose edges are smooth
curves. It first builds a graph from nearby observations, estimates which
branches and loops are supported by the data, and selects a sparse graph
backbone. It then fits a curve to each backbone route. Every observation is
assigned to its nearest route and receives a route identifier, a position, a
projected point, a local tangent, and a residual showing how far it lies from
the route.

The routes are fitted in the data's feature space. Principal component
analysis (PCA), multidimensional scaling (MDS), Uniform Manifold Approximation
and Projection (UMAP), and the metro-style layout are used only to display the
result. They do not determine the fitted routes.

The detailed implementation reference, including backends, diagnostics, and
all public attributes, is available in [Implementation details](implementation.md).

## 1. The problem

Suppose the input is a collection of $n$ observations, each with $d$ features:

$$
X = \{x_1,\ldots,x_n\}, \qquad x_i\in\mathbb{R}^d.
$$

The data may be high-dimensional, but we assume that its important structure
is approximately one-dimensional. In other words, nearby observations should
mostly lie along a small number of connected curves. Those curves can meet at
junctions and can form loops.

The goal is not simply to create a two-dimensional picture. The goal is to
recover useful coordinates for the original observations:

- which branch or route contains an observation;
- where the observation lies along that route; and
- how far the observation is from the route.

### A simple example

Imagine observations sampled around a noisy Y-shaped trajectory. A standard
dimensionality-reduction plot may show the Y shape, but it does not necessarily
say which branch each observation belongs to or where it lies along that
branch. A spline graph embedding makes those quantities explicit.

## 2. Representation and returned values

The method represents the hidden structure as a collection of routes. Route
$r$ is a curve $\gamma_r(u)$, where the parameter $u$ ranges from $0$ to $1$.
For an observation $x_i$, the algorithm approximately solves

$(r_i,u_i)\approx\arg\min_{r,u}\|z_i-\gamma_r(u)\|_2$,

where $z_i$ is the coordinate used during fitting. The implementation evaluates
this search on densely sampled route segments, so it is an approximation to a
continuous projection.

The position $u_i$ is normalized progress along a route. It is not a physical
distance and should not be compared across routes with different lengths
without additional scaling.

### Original and fitting coordinates

When standardization is enabled, each feature is centered and scaled before
distances and routes are computed. This prevents a feature with large numeric
units from dominating the geometry. Constant features receive unit scale so
that duplicated or degenerate data remain finite.

The algorithm fits routes in these fitting coordinates, but converts projected
points and residuals back to the original feature units. Tangents remain in
fitting coordinates because they define the local geometry used for normal
coordinates.

### Public result

For each observation, `EmbeddingResult` provides:

| Field | Meaning |
| --- | --- |
| `route_id` | Integer identifying the selected route |
| `position` | Normalized position on that route, in $[0,1]$ |
| `projected` | Nearest point on the fitted route, in original feature units |
| `residual` | Difference between the observation and its projection |
| `residual_norm` | Euclidean length of the residual |
| `tangent` | Local unit tangent in fitting coordinates |

If $\hat{x}_i$ is the projected point, the residual is
$e_i=x_i-\hat{x}_i$ and `residual_norm` is $\|e_i\|_2$.

## 3. End-to-end algorithm

The complete process can be summarized as follows:

```text
Input: point cloud X
1. Validate X and optionally standardize its features.
2. Build a graph connecting nearby observations.
3. Compress the data into landmarks when needed.
4. Estimate endpoints, junctions, and loops.
5. Select a sparse graph backbone.
6. Fit a smooth route to each backbone path.
7. Project every observation onto its nearest route.
8. Return route coordinates, projections, tangents, and residuals.
```

### 3.1 Prepare the data

The estimator first checks that the input is a finite, non-empty, two-
dimensional numeric array. It also checks feature counts and parameter values.

If standardization is requested, the data is centered and scaled feature by
feature. The implementation also computes a local scale from non-zero nearest-
neighbor distances. A small finite fallback is used when all observations are
duplicates.

### 3.2 Build a neighborhood graph

The method connects each observation to nearby observations. This is called a
*k-nearest-neighbor graph* or kNN graph: each observation chooses its $k$
closest neighbors, and the resulting edges are symmetrized so that an edge is
available in either direction.

The graph is a source of candidate structure, not necessarily the final
answer. For larger datasets, observations are first compressed into
*landmarks*. Landmarks are representative points found with deterministic
k-means. Working with landmarks makes later graph operations less expensive.

The implementation supports two initialization modes:

- `coarsen` builds a landmark minimum spanning tree (MST) and then considers
  local edges that can add loops. An MST connects the landmarks with low total
  edge length but contains no cycles.
- `topological` uses the neighborhood graph together with local geometric and
  topology signals to select the backbone. It is more explicit about branches
  and loops.

The Python API keeps `coarsen` as its compatibility default. Applications can
select `topological` when topology-aware initialization is desired.

### 3.3 Detect branches and loops

The method uses two kinds of evidence.

First, local neighborhoods are examined at several distance scales. Around a
candidate point, the algorithm counts how many connected pieces appear in thin
rings. Stable patterns help distinguish an endpoint, an ordinary point, and a
junction. Nearby candidates are grouped into one region so that a noisy
junction is not treated as many separate junctions.

Second, the method uses *persistent homology* to estimate loops. Persistent
homology repeats a connectivity analysis at many distance scales. Its H1
component tracks one-dimensional holes, which correspond to loops in the
data. A feature that persists across many scales is stronger evidence of a
real loop than a feature that appears at only one scale.

The persistence estimate supplies a target cycle count. The selected graph may
realize fewer cycles if the available local candidate edges cannot support all
of them. The implementation records both the requested and realized counts.

### 3.4 Select the backbone

The selector combines edge length, local tangent agreement, density, and
optional connectivity evidence. A path that makes a sharp, unsupported turn
is less attractive than one that follows the local geometry.

Some optional calculations treat graph edges like electrical resistors. The
resulting effective resistance and current measures describe how strongly an
edge contributes to connectivity. They are supporting evidence only; they do
not define the topology by themselves.

The selector starts with a connected structure, adds missing junction arms,
and considers local cycle-closing edges. Long chains of ordinary degree-two
vertices are compressed into single backbone edges before curve fitting.

### 3.5 Fit smooth routes

Each backbone path becomes one open or closed route. The preferred
implementation uses smoothing splines. If SciPy is unavailable or numerical
fitting fails, a deterministic NumPy Catmull–Rom or polyline fallback is used.

The route is sampled densely. This sampled representation is used both for
projection and for computing the local tangent, so the geometry used for
assignment stays consistent with the geometry used for visualization.

### 3.6 Project observations

For every observation, the implementation tests its distance to the sampled
segments of every route. The closest valid route wins. The projected point is
returned in the original feature coordinates, while the route tangent is
returned in fitting coordinates.

If no route can be selected, the transform operation raises an error with the
number of invalid observations instead of returning a misleading route ID.

## 4. Normal coordinates

The residual tells us how far an observation is from a route, but in
high-dimensional data it can also be useful to describe the direction of that
residual. At a point on a curve, the tangent gives the along-route direction.
The remaining $d-1$ directions form the local normal space.

The implementation stores a deterministic orthonormal basis for this normal
space along every route. It constructs the first basis from the coordinate
axes, transports it along the route, and repeatedly re-orthogonalizes it for
numerical stability. Reusing the stored frame means that transforming a
dataset and transforming a subset produce the same normal coordinates for
shared observations, up to floating-point roundoff.

The public method is:

```python
normal = model.normal_coordinates(result)
```

The returned array has shape `(n_samples, n_features - 1)`. In one feature
dimension, it has zero columns because there is no normal direction.

## 5. Visualization

The fitted routes can be displayed in several ways.

- A feature-space plot shows the data and the fitted network directly when
  the data has two or three dimensions.
- PCA, MDS, or UMAP can reduce high-dimensional data for display. These are
  visualization steps; they do not refit the route network.
- The metro-style layout draws a readable schematic of branches, junctions,
  and loops. Its coordinates are designed for legibility, not for preserving
  distances in the original data.

All visualizations consume the same `EmbeddingResult` fields. This keeps route
identity, position, projection, and residual meaning consistent across views.

## 6. Minimal Python example

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

The result can be inspected directly:

```python
print(result.route_id)
print(result.position)
print(result.residual_norm)
```

The optional scikit-learn adapters and their feature conventions are described
in [Implementation details](implementation.md).

## 7. Cost, reproducibility, and limitations

Landmark compression limits the size of the graph used for topology selection.
Distance calculations are processed in blocks rather than building one large
observation-by-landmark-by-feature tensor. Route projection is also batched.
The persistence fallback is intended for moderate point clouds and can be
capped with `persistence_max_points`.

Randomized stages accept `random_state`. Results can still vary slightly with
the installed SciPy or Ripser versions when optional backends are available;
the selected backend is recorded by the estimator.

The method is an approximate one-dimensional representation. It does not
model branch-specific density or uncertainty, and projection is not a full
continuous optimization. A topology target may not be realized when local
candidate edges are insufficient. Finally, a metro layout is a schematic
display and should not be interpreted as a metric embedding.
