# Topology-aware spline skeletons for noisy point clouds

## Abstract

Many point clouds are not well described by a single line, plane, or generic
low-dimensional coordinate system. Their latent structure may contain
branches, junctions, loops, or several disconnected routes. This paper
describes SkeletalEmbedding, a representation for such data based on a sparse
graph of smooth spline routes.

The method separates two decisions that are often conflated. First, it
infers a coarse graph topology from a weighted neighborhood graph, persistent
one-dimensional topology, and local geometric evidence. Second, it fits smooth
curves to the selected graph paths. A mixed-integer program can enforce
connectivity, endpoint and junction degrees, and a requested cycle rank while
choosing among candidate paths. Each observation is then projected to its
nearest sampled route and represented by a route identity, a normalized
longitudinal position, and an off-route residual. Optional tangent-orthogonal
residual-PCA fields and adaptive ribs describe transverse variation when a
one-dimensional centerline is not enough.

The result is a topology-aware coordinate system for graph-shaped data. It is
intended to make route structure explicit and to preserve off-route variation
for downstream analysis. The primary object remains a one-dimensional graph
skeleton; surface-like coverage and higher-dimensional structure are optional
refinements rather than the default interpretation.

## 1. Motivation and scope

Dimensionality reduction is often introduced as a way to replace a cloud of
points in a high-dimensional space with a small number of coordinates. That
description is useful, but it can hide an important distinction between data
sets. Some clouds are organized around a single smooth direction. Others are
organized around a graph: observations travel along several paths, meet at
junctions, separate again, or form closed loops.

A two-dimensional display can show these structures, but a display alone does
not define them. A generic projection may overlap unrelated branches, turn a
loop into an open arc, or make an ordinary bend look like a junction. It also
does not say which route an observation belongs to or how far the observation
lies from the inferred structure.

SkeletalEmbedding addresses this narrower problem by constructing an
explicit geometric skeleton. The core object is a collection of smooth
curves connected according to a sparse graph. The representation has three
parts:

- a discrete route identity for branch or loop membership;
- a scalar position describing progress along that route; and
- a residual describing the variation not explained by the route.

This makes the method useful for data with trajectories, lineages,
branching processes, road-like geometry, or other graph-shaped latent
structure. The same representation can be used in two or more ambient
dimensions because the graph is fitted in the input feature space after an
optional featurewise standardization.

The method is deliberately not presented as a replacement for general-purpose
nonlinear embedding. It is not a probabilistic generative model, and its
default output is not a learned surface or volume. It is a topology-aware
one-dimensional representation with optional transverse structure. PCA, MDS,
and UMAP may be used to display the fitted object, but they are not used to
fit the backbone and do not redefine the learned coordinates.

## 2. The representation

Let the observations be $x_1,\ldots,x_n$ in $d$-dimensional feature space.
The fitted backbone consists of routes

$$
\Gamma = \{\gamma_r:[0,1]\rightarrow\mathbb{R}^d\}_{r=1}^{R}.
$$

For each observation, the estimator finds the closest route and the closest
point along that route. In implementation this search is performed on a
dense piecewise-linear sampling of the fitted curves. The resulting primary
representation is

```text
(route_id, position, projected, residual, residual_norm)
```

where `route_id` identifies the selected route, `position` is normalized
progress from the beginning to the end of an open route or around a closed
route, `projected` is the nearest point on the fitted route, and
`residual = x - projected` is the unexplained displacement from the
centerline. `residual_norm` is its Euclidean magnitude.

The position is a route parameter, not a physical distance. Two routes can
therefore have the same position value without having the same length, and a
position value should not be compared across routes as if all routes had the
same metric scale. If physical progress is needed, the route arc length can
be recovered from the sampled curve.

The default representation is centerline-only. When a positive residual
dimension is requested, the residual is decomposed into smooth local
transverse coordinates and a remaining error:

$$
x_i \approx \gamma_{r_i}(u_i) + U_{r_i}(u_i)z_i + \epsilon_i.
$$

Here $U_r(u)$ is an orthonormal basis in the hyperplane normal to the route
tangent, $z_i$ contains the learned residual-PCA coordinates, and
$\epsilon_i$ is the post-PCA residual. This decomposition is optional. The
compatibility fields `projected` and `residual` always refer to the
centerline projection and centerline residual, even when the additional
transverse reconstruction is enabled.

The coordinate systems are kept explicit. Distances and spline calculations
use standardized fitting coordinates when standardization is enabled. Public
`projected`, `residual`, `reconstructed`, and `unexplained_residual` arrays
are returned in the original input units. Tangents and residual-PCA
coordinates are retained in fitting coordinates because they define local
directions and transverse geometry.

## 3. Method overview

The estimator follows a topology-aware coarse-to-fine pipeline. Before routing,
it constructs a recursive hierarchy $X_0 \to X_1 \to \cdots \to X_L$ of
observed medoid representatives with complete original-row ancestry. Grouping
uses local distance normalization and data-derived quantile thresholds,
independently of topology and rib decisions. This geometric compression is
[MILK-inspired](https://github.com/yachielab/milk/blob/main/README.md); it imports
recursive grouping, medoids, and percentile-derived threshold ideas rather
than claiming to reproduce MILK itself.

Coarse adjacent levels supply consensus backbone evidence. Three quantities
remain distinct: filtration persistence, random-subsample support, and
representative-resolution support. Their combination affects structural
selection; none substitutes for the others. An untested quantity is unavailable,
not evidence of perfect reproducibility.

The subsequent stages are:

1. **Prepare the metric.** Validate the point cloud and optionally center and
   scale each feature. Constant features receive unit scale so degenerate
   inputs remain finite.
2. **Build the routing substrate.** Construct a weighted symmetric
   representative kNN graph at eligible hierarchy levels. Reciprocal-neighbor filtering and Euclidean-MST
   augmentation are enabled by default. The graph is used for local routing,
   not returned as the final skeleton.
3. **Estimate topology and local geometry.** Use persistent H1 features as
   evidence for cycles. Use multiscale annulus components to identify
   endpoint and junction regions. Use local PCA to estimate ordinary tangents
   and separate outgoing directions at junctions.
4. **Generate candidate paths.** Compress observations into landmarks for
   tractable selection, then generate ordered paths through the dense routing
   graph between logical landmarks. Candidate paths are scored using length,
   tangent consistency, density, and optional electrical or stability terms.
5. **Select the backbone.** A mixed-integer program can choose candidate paths
   subject to connectivity, degree, and cycle constraints. A deterministic
   topology-aware selector is used when the MIP backend is unavailable or
   cannot find a feasible solution.
6. **Refine and fit route geometry.** Expand selected coarse paths through
   descendant corridors while preserving their ordered anchors and logical
   topology. Collapse maximal degree-two graph paths into fine/full-data route
   supports and fit smooth open or closed curves. Spline fitting is separate
   from discrete graph selection.
7. **Model transverse variation.** Optionally fit local tangent-orthogonal
   residual-PCA fields. If the remaining reconstruction error is high,
   residual-driven and unresolved fine-hierarchy paths seed coverage ribs.
   Reconstruction gain, sampling support, resolution support, and complexity
   determine selection. Fine structure does not automatically become a rib.
8. **Project observations.** Assign every observation to its closest sampled
   route, returning route identity, normalized position, projection, tangent,
   and residual fields.

This ordering is the main design choice. Topology is selected before smooth
geometry is fitted, so the spline stage is not asked to discover branches or
loops by itself. Conversely, the selected graph remains inspectable before it
is converted into curves.

## 4. Topology-aware backbone selection

### 4.1 Metric, neighborhoods, and landmarks

After optional standardization, the estimator builds a weighted observation
kNN graph. Euclidean edge lengths provide geometric costs. Gaussian affinity
weights are retained as conductances for optional connectivity diagnostics.
With `mutual_knn=True`, an observation edge is retained only when both
endpoints select each other. With `add_mst=True`, the exact Euclidean minimum
spanning tree is added to preserve a connected routing substrate when local
reciprocal neighborhoods are too sparse.

The MST is a routing safeguard, not an instruction that the final skeleton
must be a tree. In particular, when persistent cycles are present, augmented
MST edges are penalized during cycle-aware routing so that a natural local
cycle is preferred when one is available. Disconnected natural components
remain separate for route selection. A bridge may be introduced for optional
electrical calculations, but that diagnostic bridge is not allowed to create a
route between otherwise separate backbones.

For larger clouds, deterministic k-means compresses the observations into at
most `n_centroids` landmarks. The dense observation graph continues to supply
the ordered support paths, while the landmark representation keeps topology
selection small enough to inspect and optimize. This distinction prevents
landmark compression from replacing the observed geometry with a collection
of straight centroid-to-centroid chords.

### 4.2 Cycles and local branch structure

Persistent homology estimates whether the cloud contains robust one-
dimensional holes. Significant H1 bars provide a persistence-derived cycle
count. The count is normalized by a local nearest-neighbor scale so that the
threshold has a comparable interpretation across differently scaled data.

Persistent homology supplies evidence, not a complete graph reconstruction.
The estimator therefore records three separate quantities:

- `persistent_cycle_count_`: significant H1 bars before the user limit;
- `requested_cycle_count_`: that count after applying `max_cycles`; and
- `realized_cycle_count_`: the independent cycle rank of the selected graph.

Their difference is exposed as `topology_shortfall_`. A shortfall means that
the available candidate paths could not realize the requested persistent
topology; it is not silently reported as a successful fit.

Branch structure is estimated independently of the cycle count. Neighborhoods
are examined at several scales, and connected components in thin annuli around
candidate centers provide evidence for endpoints, regular regions, and
junctions. Nearby detections are clustered into endpoint and junction
regions. At ordinary vertices, local PCA estimates an unoriented tangent. At
junctions, local angular sectors estimate one outgoing direction per arm.
Candidate paths that depart from a junction in an incompatible direction are
rejected or penalized. Along ordinary edges, tangent inconsistency is
sign-invariant, so reversing a route does not change its score.

### 4.3 Candidate paths and MIP selection

The selector does not give the mixed-integer program every possible pair of
observations. It first constructs a finite candidate set from shortest paths
through the dense routing substrate, junction-arm completion, cycle-closing
opportunities, and special structural handling for supported hypercube-like
data. Each candidate is an ordered path between two logical landmarks and
retains its point-level support.

For candidate path $c$, the routing cost combines normalized path length,
tangent inconsistency, and inverse local density. Effective-resistance
support, aggregate electrical-flow support, and stability support can adjust
the ranking. Electrical terms are opt-in and have zero weight by default.

The MIP uses a binary variable `y_c` for each candidate path. A value of one
means that the whole path is selected. It does not mean that an individual
raw kNN edge, spline control point, or observation is selected. The main hard
constraints are:

1. Endpoint landmarks must have degree one.
2. Junction landmarks must have their detected branch count.
3. Cycle-anchor landmarks must have degree two.
4. The selected path count must be compatible with the requested cycle rank.
5. At most one alternative path may be selected for the same logical endpoint
   pair.
6. Tagged persistent cycle classes must receive supporting selected paths when
   the candidate set contains those tags.

Connectivity is enforced with continuous directed-flow variables. A root
landmark sends one unit of flow to every other logical landmark, and the flow
capacity on a candidate is tied to its binary selection variable. A collection
of locally valid but disconnected paths therefore cannot satisfy the model.

The objective is a weighted route-selection objective rather than a spline
fitting objective:

```text
minimize  sum over candidates c of
          y_c * (cost_c - 0.05 * electrical_c
                       - 0.05 * current_c
                       - 0.05 * stability_c)
```

When SciPy's MIP solver returns an optimal solution, that solution is optimal
only over the generated candidate set. It does not discover a route that was
never proposed by the dense-graph routing stage. If the solver is unavailable
or infeasible, the deterministic selector preserves the same topology-aware
intent and records the outcome in `mip_status_`.

The selected graph is simplified after selection. Maximal paths whose
internal vertices have degree two are collapsed into backbone edges while
their ordered support points are retained. This produces an abstract
`backbone_graph_` for topology and `backbone_paths_` for geometry. The
backbone cycle rank is measured before optional coverage ribs are added.

## 5. Smooth route geometry and projection

Each selected support path becomes an open or closed route. For an open path,
the first and last support points are preserved as endpoint anchors. For a
closed path, the route is treated periodically so that the seam does not
introduce a false endpoint. SciPy smoothing splines are preferred when the
backend supports the route dimension and numerical fit. Shape-preserving
periodic interpolation is used for closed routes when it avoids overshoot,
and deterministic Catmull–Rom or polyline representations provide fallbacks
for unsupported or difficult fits.

The spline stage has a different responsibility from the topology stage. Its
smoothing parameter controls how closely the curve follows the selected
support path. It does not add a branch because the data cloud appears wide,
and it does not decide whether a loop should exist. A `backbone` control mode
can give the selected landmark vertices stronger fitting weight when the
route should remain close to the discrete backbone.

Routes are sampled densely after fitting. For every observation, squared
distances to the sampled line segments of every route are evaluated in
batches. The closest valid route and segment determine `route_id`,
`position`, and the fitting-space projection. The projection is mapped back
to original feature units before it is returned. This makes the public
residual an interpretable displacement in the units of the supplied data.

Projection is intentionally approximate. It uses sampled segments rather
than continuous optimization over every spline parameter. The sampling
resolution is therefore part of the numerical approximation, although the
route identity and position are made deterministic for a fixed fitted model.
If no valid route can be selected, transformation raises an explicit error
rather than returning a sentinel route identifier.

## 6. Optional transverse structure and coverage

### 6.1 Residual-PCA fields

The centerline residual contains more information than a scalar distance. Its
direction can indicate branch-specific variation, a local sheet around a
trajectory, or measurement noise concentrated in particular directions.

When `max_residual_dim` is positive, residuals are expressed in deterministic
normal frames along each route. At each route-grid position, a Gaussian-
weighted covariance of the local residual coordinates is eigendecomposed.
The leading directions form a local orthonormal residual basis. Neighboring
subspaces can be smoothed by averaging their projectors and re-orthogonalizing
against the route tangent.

This gives a fixed-dimensional transverse coordinate system along a graph
route. With zero residual dimensions, reconstruction equals the centerline
projection and the unexplained residual equals the original centerline
residual. The option therefore extends the representation without changing
the meaning of its primary fields.

### 6.2 Coverage ribs

Residual-PCA explains the most stable local transverse directions, but a
single centerline plus a low-rank local basis may still leave coherent regions
of high reconstruction error. Optional coverage refinement uses those regions
to propose local ribs. A rib follows observed residual support through the
routing graph and is fitted as an additional spline. Candidates are scored by
reconstruction gain, support, length, and complexity penalties.

Ribs are geometric coverage elements. They are not fed back into persistent
topology inference, and they do not provide evidence that the underlying
manifold has additional cycles. The default selection policy is greedy; an
optional MIP can select a cardinality-limited set of ribs. This is deliberately
different from backbone selection: the backbone encodes persistent coarse
topology, whereas ribs reduce local reconstruction error.

The distinction matters for surface-like data. A torus, for example, is not
faithfully represented by a single one-dimensional centerline. The backbone
can still provide a useful route-oriented summary, while residual fields and
ribs expose the limits of that summary and add sparse coverage where it is
supported by observations. A true jointly learned two-dimensional manifold is
outside the default model.

## 7. Eight synthetic illustrations

The repository includes eight controlled point-cloud examples designed to
show the range of structures addressed by the representation:

- a line;
- a four-arm star;
- a circle;
- a figure-eight with two cycles sharing a junction;
- a torus surface;
- a branching binary tree;
- a loop with an attached branch; and
- a polygon with radial rays and attached circles.

![Eight synthetic point clouds with fitted spline skeletons](figures/synthetic_eight_datasets.png)

**Figure 1.** Illustrative behavior of the topology-aware spline pipeline on
eight noisy synthetic data sets. Each panel shows the observations and the
fitted backbone in a two-dimensional display. The line, star, and
loop-with-branch make open routes and junction markers easy to inspect. The
circle and figure-eight illustrate closed routes and cycle rank, while the
binary tree shows that a branching-shaped cloud can remain difficult for a
sparse one-dimensional fit under the standard settings. The polygon/rays/
circles example illustrates mixed graph structure. The torus and other
surface-like examples mark the boundary of a one-dimensional backbone and
motivate optional residual-PCA and coverage-rib refinement. The figure is a
conceptual illustration, not a quantitative benchmark.

The figure is generated with 500 observations per data set, noise scale
0.045, a fixed random seed, 32 landmarks for ordinary examples, and 64
landmarks for the binary tree. These settings are inherited from the
repository demo so that the illustration is tied to a reproducible workflow.
The panels are displays of fits made in the ambient fitting space; the
display itself does not replace the learned route coordinates.

## 8. Relation to principal graphs and manifolds

SkeletalEmbedding belongs to the broader family of principal-object methods,
which approximate a data set by an object of lower dimension or complexity
and project observations onto that object. It is especially close in spirit
to principal curves, elastic maps, and elastic principal graphs.

The conceptual difference is where topology enters the procedure. In
elastic-principal-graph approaches, a graph is embedded by minimizing a data
approximation term together with stretching and bending energies. Graph
grammars can grow, split, prune, or otherwise modify the structure under
explicit complexity limits. Junctions are regularized toward ideal or
pluriharmonic configurations, and EM-style projection and re-estimation
updates the embedded graph.

The present method uses a topology-first decomposition. Persistent H1
provides cycle evidence, local annulus analysis provides branch evidence, and
the routing substrate supplies candidate paths. A MIP or deterministic
selector then chooses a graph satisfying discrete connectivity, degree, and
cycle constraints. Smooth splines are fitted after that choice. There is no
single elastic energy jointly optimized over all candidate topologies and
spline coordinates.

This difference affects the interpretation of the output. The method here is
designed around explicit route identity, normalized longitudinal position,
and residual displacement. Elastic maps and related principal objects provide
a broader framework for controlled-complexity approximation, including
regular-grid manifolds and factorized higher-dimensional complexes. Those
objects are useful points of comparison, but the current implementation keeps
its primary claim narrower: it extracts and parameterizes a sparse graph
skeleton, with optional transverse coverage.

## 9. Limitations and reproducibility

The fitted structure is an approximation whose behavior depends on several
scales and thresholds. The main approximation sources are feature
standardization, sparse neighborhood construction, landmark compression,
candidate-path generation, persistent-homology estimation, sampled spline
projection, and optional backend fallbacks. The persistence fallback is
intended for moderate point clouds and may subsample or cap the data.

Persistent topology is evidence rather than certainty. Short noisy H1 bars
may be rejected, while a genuine cycle may be absent from the available
candidate routes. The estimator reports this distinction through persistence,
requested, realized, and shortfall diagnostics. Likewise, local annulus votes
and tangent estimates can be ambiguous near crossings, sparse endpoints, or
strongly anisotropic noise.

The backbone and ribs also have different meanings. The backbone is the
topology-constrained representation of the dominant graph-like structure.
Ribs are optional geometric refinements chosen to improve coverage. Adding
ribs should not be interpreted as discovering new topological cycles.

Randomized stages accept `random_state`, including landmark initialization and
capped persistence subsampling. The fitted estimator records persistence and
spline backends, solver status, cycle diagnostics, route metadata, and
coverage decisions. The complete parameter-level behavior and diagnostic
attributes are documented in [Implementation details](implementation.md).

## 10. References

1. Gorban, A. N., and Zinovyev, A. Y. *Principal Graphs and Manifolds*.
   Supplied chapter on principal objects, elastic maps, elastic principal
   graphs, graph grammars, and principal trees.
2. Pearson, K. “On Lines and Planes of Closest Fit to Systems of Points in
   Space.” *Philosophical Magazine*, 1901.
3. Edelsbrunner, H., and Harer, J. *Computational Topology: An Introduction*.
   American Mathematical Society, 2010.
4. Dierckx, P. *Curve and Surface Fitting with Splines*. Oxford University
   Press, 1993.
