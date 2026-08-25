"""Public spline graph embedding estimator."""

from __future__ import annotations

import heapq
import warnings
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from ._curves import _fit_curve
from ._electrical import _effective_resistance, _electrical_flow, _kron_reduction
from ._frames import (
    _fit_normal_frame_grid,
    _normal_coordinates,
)
from ._local_geometry import (
    _attach_junction_directions,
    _departure_angle,
    _estimate_local_tangents,
    _tangent_inconsistency,
)
from ._topology import (
    CandidatePath,
    EndpointRegion,
    JunctionRegion,
    _as_point_cloud,
    _cycle_anchor_vertices,
    _estimate_local_topology,
    _estimate_persistence,
    _extract_chains,
    _is_nearly_linear,
    _kmeans,
    _LandmarkGraph,
    _local_scale,
    _merge_nearby_junctions,
    _minimum_spanning_tree,
    _normalize_persistence_diagram,
    _ordered_path_graph,
    _prune_short_terminal_branches,
    _standardize,
    _symmetric_knn_edges,
    _weighted_symmetric_knn_graph,
)
from .results import EmbeddingResult

Array = np.ndarray


def _geodesic_diameter_endpoints(
    graph: Any,
    component: Sequence[int],
) -> tuple[int, int]:
    """Return the two ends of an open graph component.

    Euclidean extremes are unreliable for curved strands: an interior bend can
    be farther from one end than the actual second endpoint.  Two Dijkstra
    sweeps recover the diameter endpoints while keeping the search inside the
    natural kNN component.
    """
    allowed = {int(vertex) for vertex in component}
    if len(allowed) < 2:
        vertex = next(iter(allowed))
        return vertex, vertex

    def farthest(source: int) -> tuple[int, dict[int, float]]:
        distances = {source: 0.0}
        queue: list[tuple[float, int]] = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances.get(node, np.inf) + 1e-12:
                continue
            for neighbour in graph.adjacency[node]:
                if neighbour not in allowed:
                    continue
                edge = graph.key(node, neighbour)
                candidate = distance + float(graph.edges[edge])
                if candidate < distances.get(neighbour, np.inf):
                    distances[neighbour] = candidate
                    heapq.heappush(queue, (candidate, int(neighbour)))
        endpoint = max(distances, key=lambda node: (distances[node], -node))
        return int(endpoint), distances

    first, _ = farthest(min(allowed))
    second, distances = farthest(first)
    if second not in distances:
        second = min(allowed - {first})
    return first, second

class SplineGraphEmbedding:
    """Fit a smooth graph of spline routes to a point cloud."""

    def __init__(
        self,
        n_centroids: int = 32,
        persistence_threshold: float | None = None,
        spline_smoothing: float = 0.02,
        max_cycles: int = 5,
        random_state: int = 0,
        standardize: bool = True,
        merge_junction_distance: float | None = None,
        prune_short_branches: bool = True,
        prune_branch_factor: float = 0.5,
        persistence_max_points: int = 60,
        spline_samples_per_node: int = 12,
        linear_structure_tolerance: float = 0.12,
        topology_neighbors: int = 6,
        backbone_initialization: str = "coarsen",
        detect_cycles: bool = True,
        detect_junctions: bool = True,
        junction_scales: int | Sequence[float] = 6,
        junction_inner_fraction: float = 0.25,
        junction_confidence: float = 0.7,
        use_local_pca: bool = True,
        local_pca_neighbors: int = 20,
        max_branch_angle_degrees: float = 45.0,
        use_effective_resistance: bool = False,
        use_electrical_flow: bool = False,
        use_kron_reduction: bool = False,
        routing_length_weight: float = 1.0,
        routing_tangent_weight: float = 1.0,
        routing_density_weight: float = 0.5,
        routing_resistance_weight: float = 0.0,
        routing_current_weight: float = 0.0,
        use_tangent_boundary_conditions: bool = True,
    ) -> None:
        if n_centroids < 3:
            raise ValueError("n_centroids must be at least 3")
        if max_cycles < 0:
            raise ValueError("max_cycles must be non-negative")
        if persistence_max_points < 1:
            raise ValueError("persistence_max_points must be at least 1")
        if spline_samples_per_node < 1:
            raise ValueError("spline_samples_per_node must be at least 1")
        if topology_neighbors < 2:
            raise ValueError("topology_neighbors must be at least 2")
        if backbone_initialization not in {"coarsen", "topological"}:
            raise ValueError("backbone_initialization must be 'coarsen' or 'topological'")
        if local_pca_neighbors < 2:
            raise ValueError("local_pca_neighbors must be at least 2")
        if not 0.0 < junction_inner_fraction < 1.0:
            raise ValueError("junction_inner_fraction must be in (0, 1)")
        if not 0.0 <= junction_confidence <= 1.0:
            raise ValueError("junction_confidence must be in [0, 1]")
        if max_branch_angle_degrees <= 0.0 or max_branch_angle_degrees > 180.0:
            raise ValueError("max_branch_angle_degrees must be in (0, 180]")
        routing_weights = (
            routing_length_weight,
            routing_tangent_weight,
            routing_density_weight,
            routing_resistance_weight,
            routing_current_weight,
        )
        if any(value < 0.0 or not np.isfinite(value) for value in routing_weights):
            raise ValueError("routing weights must be finite and non-negative")
        if spline_smoothing < 0 or not np.isfinite(spline_smoothing):
            raise ValueError("spline_smoothing must be finite and non-negative")
        if prune_branch_factor < 0 or not np.isfinite(prune_branch_factor):
            raise ValueError("prune_branch_factor must be finite and non-negative")
        if linear_structure_tolerance < 0 or not np.isfinite(linear_structure_tolerance):
            raise ValueError("linear_structure_tolerance must be finite and non-negative")
        if persistence_threshold is not None and (
            persistence_threshold < 0 or not np.isfinite(persistence_threshold)
        ):
            raise ValueError("persistence_threshold must be finite and non-negative")
        if merge_junction_distance is not None and (
            merge_junction_distance < 0 or not np.isfinite(merge_junction_distance)
        ):
            raise ValueError("merge_junction_distance must be finite and non-negative")
        self.n_centroids = int(n_centroids)
        self.persistence_threshold = persistence_threshold
        self.spline_smoothing = float(spline_smoothing)
        self.max_cycles = int(max_cycles)
        self.random_state = int(random_state)
        self.standardize = bool(standardize)
        self.merge_junction_distance = merge_junction_distance
        self.prune_short_branches = prune_short_branches
        self.prune_branch_factor = float(prune_branch_factor)
        self.persistence_max_points = int(persistence_max_points)
        self.spline_samples_per_node = int(spline_samples_per_node)
        self.linear_structure_tolerance = float(linear_structure_tolerance)
        self.topology_neighbors = int(topology_neighbors)
        self.backbone_initialization = str(backbone_initialization)
        self.detect_cycles = bool(detect_cycles)
        self.detect_junctions = bool(detect_junctions)
        self.junction_scales = junction_scales
        self.junction_inner_fraction = float(junction_inner_fraction)
        self.junction_confidence = float(junction_confidence)
        self.use_local_pca = bool(use_local_pca)
        self.local_pca_neighbors = int(local_pca_neighbors)
        self.max_branch_angle_degrees = float(max_branch_angle_degrees)
        self.use_effective_resistance = bool(use_effective_resistance)
        self.use_electrical_flow = bool(use_electrical_flow)
        self.use_kron_reduction = bool(use_kron_reduction)
        self.routing_length_weight = float(routing_length_weight)
        self.routing_tangent_weight = float(routing_tangent_weight)
        self.routing_density_weight = float(routing_density_weight)
        self.routing_resistance_weight = float(routing_resistance_weight)
        self.routing_current_weight = float(routing_current_weight)
        self.use_tangent_boundary_conditions = bool(use_tangent_boundary_conditions)
        self._fitted = False

    def fit(self, X: Array | Sequence[Sequence[float]]) -> SplineGraphEmbedding:
        original = _as_point_cloud(X)
        if original.shape[0] < 3:
            raise ValueError(
                f"n_samples = {original.shape[0]}; X must contain at least three "
                "observations for fitting"
            )
        if self.standardize:
            points, mean, scale = _standardize(original)
        else:
            points = original.copy()
            mean = np.zeros(original.shape[1])
            scale = np.ones(original.shape[1])
        self.n_features_in_ = points.shape[1]
        self.mean_ = mean
        self.scale_ = scale
        self._original_X_ = original

        self.persistence_diagram_, self.persistence_backend_ = _estimate_persistence(
            points, max_points=self.persistence_max_points, random_state=self.random_state
        )
        self.local_scale_ = _local_scale(points)
        self.normalized_persistence_diagram_ = _normalize_persistence_diagram(
            self.persistence_diagram_, self.local_scale_
        )
        if self.backbone_initialization == "topological":
            # Topological mode uses scale-free persistence.  The legacy mode
            # retains its historical distance-unit threshold semantics.
            threshold = 3.0 if self.persistence_threshold is None else float(self.persistence_threshold)
            self.persistence_threshold_ = float(threshold)
            diagram = self.normalized_persistence_diagram_
        elif self.persistence_threshold is None:
            threshold = 4.0 * self.local_scale_
            self.persistence_threshold_ = float(threshold)
            diagram = self.persistence_diagram_
        else:
            self.persistence_threshold_ = float(self.persistence_threshold)
            diagram = self.persistence_diagram_
        if len(diagram):
            lifetimes = diagram[:, 1] - diagram[:, 0]
            significant = np.isfinite(lifetimes) & (lifetimes >= self.persistence_threshold_)
            self.persistent_cycle_count_ = int(np.sum(significant))
        else:
            self.persistent_cycle_count_ = 0
        self.cycle_count_ = self.persistent_cycle_count_
        self.requested_cycle_count_ = min(self.max_cycles, self.persistent_cycle_count_)
        if self.backbone_initialization == "topological" and not self.detect_cycles:
            self.requested_cycle_count_ = 0

        self.centroids_ = _kmeans(points, self.n_centroids, self.random_state)
        # Structure detection is evaluated in the original metric.  This is
        # important for noisy one-dimensional clouds: feature standardization
        # can otherwise turn small orthogonal noise into an artificial branch.
        centroids_original = self.centroids_ * scale + mean
        self.linear_structure_ = _is_nearly_linear(
            centroids_original, self.linear_structure_tolerance
        )
        if self.linear_structure_:
            # Estimate the axis before feature standardization.  Standardizing
            # each coordinate makes a thin noisy line look artificially
            # isotropic and can rotate the PCA axis into the noise band.
            original_center = np.mean(original, axis=0)
            centered = original - original_center
            _, _, components = np.linalg.svd(centered, full_matrices=False)
            self.linear_center_ = (original_center - mean) / scale
            direction = components[0] / scale
            self.linear_direction_ = direction / max(float(np.linalg.norm(direction)), 1e-12)
        else:
            self.linear_center_ = None
            self.linear_direction_ = None
        if self.backbone_initialization == "topological" and self.linear_structure_:
            # A noisy line can carry a small numerical H1 bar in a subsample;
            # its one-dimensional geometry is a stronger constraint than that
            # artifact and must not create a cycle anchor.
            self.requested_cycle_count_ = 0
        self.backbone_paths_ = None
        self.central_junction_locked_ = False
        self.central_junction_center_ = None
        if self.backbone_initialization == "topological":
            graph, backbone_paths = self._topological_backbone(points, self.centroids_)
            self.backbone_paths_ = backbone_paths
            self.merge_junction_distance_ = 0.0
        elif self.linear_structure_:
            # A noisy sample of a line can produce a branched MST and a
            # borderline H1 bar.  A PCA-ordered chain is the appropriate
            # low-complexity skeleton for this geometry.
            self.requested_cycle_count_ = 0
            self.merge_junction_distance_ = 0.0
            graph = _ordered_path_graph(self.centroids_, ordering_points=centroids_original)
        else:
            graph = _minimum_spanning_tree(self.centroids_)
            if self.prune_short_branches:
                _prune_short_terminal_branches(graph, self.prune_branch_factor)
            if self.merge_junction_distance is None:
                merge_distance = 13.0 * self.local_scale_
            else:
                merge_distance = self.merge_junction_distance
            self.merge_junction_distance_ = float(merge_distance)
            graph = _merge_nearby_junctions(graph, merge_distance)
            self.backbone_paths_ = None
        self.landmark_graph_ = graph
        self.backbone_graph_ = graph
        if self.backbone_paths_ is None:
            self.topology_candidate_edges_ = _symmetric_knn_edges(
                self.landmark_graph_, self.topology_neighbors,
            )
        else:
            self.topology_candidate_edges_ = set()

        target_cycles = self.requested_cycle_count_
        if self.backbone_paths_ is None:
            while self.landmark_graph_.cycle_rank() < target_cycles:
                candidate = self._best_cycle_edge()
                if candidate is None:
                    break
                left, right, weight = candidate
                self.landmark_graph_.add_edge(left, right, weight)

        self.realized_cycle_count_ = self.landmark_graph_.cycle_rank()
        self.topology_shortfall_ = max(0, target_cycles - self.realized_cycle_count_)
        if self.topology_shortfall_:
            warnings.warn(
                "The sparse landmark graph could not realize all requested cycles "
                f"({self.topology_shortfall_} shortfall).",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.backbone_paths_ is None:
            self.junctions_ = [node for node in self.landmark_graph_.nodes if self.landmark_graph_.degree(node) >= 3]
            self.endpoints_ = [node for node in self.landmark_graph_.nodes if self.landmark_graph_.degree(node) == 1]
            self.junction_regions_ = []
            self.endpoint_regions_ = []
        self.junction_node_ids_ = [
            region.node_id if isinstance(region, JunctionRegion) else int(region)
            for region in self.junctions_
        ]
        self.endpoint_node_ids_ = [
            region.node_id if isinstance(region, EndpointRegion) else int(region)
            for region in self.endpoints_
        ]
        self.route_chains_ = _extract_chains(self.landmark_graph_)
        if not self.route_chains_:
            # Pruning can collapse a very small or fully duplicated graph to a
            # single geometric landmark.  Keep the transform contract valid by
            # retaining one degenerate route instead of returning -1 IDs.
            nodes = sorted(self.landmark_graph_.nodes)
            if nodes:
                self.route_chains_ = [{"nodes": [nodes[0]], "closed": False}]
        for chain in self.route_chains_:
            if self.backbone_paths_ is None:
                chain["points"] = np.asarray([self.landmark_graph_.nodes[node] for node in chain["nodes"]])
            else:
                chain["points"] = self._chain_support_points(chain)
            if self.central_junction_locked_ and chain.get("closed"):
                chain["points"] = self._figure_eight_support_points(points, chain)
            elif (
                chain.get("closed")
                and self.backbone_paths_ is not None
                and all(
                    self.landmark_graph_.degree(node) == 2
                    for node in set(chain["nodes"])
                )
            ):
                chain["points"] = self._simple_cycle_support_points(points, chain)
            if (
                self.linear_structure_
                and self.linear_center_ is not None
                and self.linear_direction_ is not None
                and len(chain["points"]) >= 2
            ):
                # The dense routing path intentionally keeps every support
                # observation, but a noisy one-dimensional cloud must not
                # turn measurement noise into a zig-zag spline.  Project the
                # support points onto the fitted PCA axis before the spline
                # stage; this leaves projection and residuals meaningful
                # while making the linear backbone actually linear.
                support = np.asarray(chain["points"], dtype=float)
                coordinates = (support - self.linear_center_) @ self.linear_direction_
                chain["points"] = (
                    self.linear_center_
                    + coordinates[:, None] * self.linear_direction_
                )
        self.routes_ = []
        for chain in self.route_chains_:
            start_tangent, end_tangent = self._chain_boundary_tangents(chain)
            self.routes_.append(
                _fit_curve(
                    chain["points"],
                    closed=chain["closed"],
                    smoothing=self.spline_smoothing,
                    sample_count=max(64, len(chain["nodes"]) * self.spline_samples_per_node),
                    start_tangent=start_tangent,
                    end_tangent=end_tangent,
                )
            )
        self._anchor_closed_junctions()
        self.route_backends_ = [spline.backend for spline in self.routes_]
        self.normal_frame_grids_ = [
            _fit_normal_frame_grid(spline) for spline in self.routes_
        ]
        self._fitted = True
        return self

    def _topological_backbone(
        self,
        points: Array,
        centroids: Array,
    ) -> tuple[_LandmarkGraph, dict[tuple[int, int], CandidatePath]]:
        """Infer a constrained landmark graph from the dense routing substrate."""
        routing_graph, local_scales = _weighted_symmetric_knn_graph(
            points, self.topology_neighbors
        )
        self.routing_graph_ = routing_graph
        self.local_graph_scales_ = local_scales
        routing_components = [
            np.asarray(component, dtype=int)
            for component in getattr(routing_graph, "original_components", [])
        ]
        self.routing_components_ = routing_components
        component_by_vertex = {
            int(vertex): component_id
            for component_id, component in enumerate(routing_components)
            for vertex in component
        }
        self.routing_component_by_vertex_ = component_by_vertex.copy()
        component_cycle_counts: list[int] = []
        for component_id, component in enumerate(routing_components):
            if len(routing_components) == 1 or len(component) < 8:
                component_cycle_counts.append(0)
                continue
            component_diagram, _ = _estimate_persistence(
                points[component],
                max_points=self.persistence_max_points,
                random_state=self.random_state + component_id + 1,
            )
            component_normalized = _normalize_persistence_diagram(
                component_diagram,
                _local_scale(points[component]),
            )
            lifetimes = (
                component_normalized[:, 1] - component_normalized[:, 0]
                if len(component_normalized) else np.empty(0)
            )
            component_cycle_counts.append(int(np.sum(
                np.isfinite(lifetimes) & (lifetimes >= self.persistence_threshold_)
            )))
        self.component_cycle_counts_ = component_cycle_counts
        if len(routing_components) > 1:
            component_cycle_total = min(
                self.max_cycles,
                int(sum(component_cycle_counts)),
            )
            if component_cycle_total > self.requested_cycle_count_:
                self.requested_cycle_count_ = component_cycle_total
                self.persistent_cycle_count_ = component_cycle_total
                self.cycle_count_ = component_cycle_total
        if self.use_local_pca:
            self.local_tangents_ = _estimate_local_tangents(
                points, routing_graph, self.local_pca_neighbors
            )
        else:
            self.local_tangents_ = np.zeros_like(points)

        if self.linear_structure_:
            # A one-dimensional cloud is a special case of the local detector:
            # small annulus gaps caused by noise should not become junctions.
            junctions = []
            direction = self.linear_direction_
            coordinates = (points - self.linear_center_) @ direction
            endpoint_vertices = [int(np.argmin(coordinates)), int(np.argmax(coordinates))]
            endpoints = [
                EndpointRegion(
                    (self.linear_center_ + coordinates[vertex] * direction).copy(),
                    1.0,
                    [vertex],
                )
                for vertex in endpoint_vertices
            ]
            branch_counts = np.full(len(centroids), 2, dtype=int)
            branch_confidence = np.ones(len(centroids), dtype=float)
        elif self.detect_junctions:
            junctions, endpoints, branch_counts, branch_confidence = _estimate_local_topology(
                points,
                routing_graph,
                centroids,
                local_scales=local_scales,
                inner_radius_fraction=self.junction_inner_fraction,
                min_junction_confidence=self.junction_confidence,
                scales=self.junction_scales,
                merge_distance=self.merge_junction_distance,
            )
        else:
            junctions, endpoints = [], []
            branch_counts = np.empty(0, dtype=int)
            branch_confidence = np.empty(0, dtype=float)
        if (
            not junctions
            and not self.linear_structure_
            and len(routing_components) > 1
            and self.requested_cycle_count_ == 0
        ):
            # A global centroid MST connects disconnected clouds and can
            # manufacture junctions on curved components.  Local annulus
            # votes are also ambiguous on a bent open strand: an interior bend
            # can look like an endpoint at one scale.  Use the weighted graph
            # diameter of each natural component instead, which returns the
            # actual two terminals of every open arc.
            consolidated: list[EndpointRegion] = []
            for component in routing_components:
                left, right = _geodesic_diameter_endpoints(routing_graph, component)
                consolidated.extend([
                    EndpointRegion(points[left].copy(), 1.0, [left]),
                    EndpointRegion(points[right].copy(), 1.0, [right]),
                ])
            endpoints = consolidated
        central_junction_locked = False
        if (
            self.detect_junctions
            and not self.linear_structure_
            and self.requested_cycle_count_ == 2
            and points.shape[1] == 2
            and len(routing_components) == 1
            and len(points)
        ):
            # Multi-cycle figure-eight-like samples often produce an unstable
            # coarse-tree junction on one lobe.  The shared singularity is the
            # dense central region, so use angular sectors there as the local
            # four-arm constraint.
            center = np.mean(points, axis=0)
            center_index = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
            centered = points - center
            if points.shape[1] >= 2:
                _, _, components = np.linalg.svd(centered, full_matrices=False)
                projection = centered @ components[:2].T
                angles = np.arctan2(projection[:, 1], projection[:, 0])
            else:
                angles = centered[:, 0]
            radii = np.linalg.norm(centered, axis=1)
            annulus = np.flatnonzero(
                (radii >= 0.5 * self.local_scale_)
                & (radii <= 8.0 * self.local_scale_)
            )
            if len(annulus) >= 4:
                order = annulus[np.argsort(angles[annulus])]
                arms = [array.astype(int) for array in np.array_split(order, 4) if len(array)]
                if len(arms) == 4:
                    junctions = [JunctionRegion(center.copy(), 4, 0.9, [center_index], arms)]
                    central_junction_locked = True
        self.central_junction_locked_ = central_junction_locked
        self.central_junction_center_ = (
            junctions[0].center.copy() if central_junction_locked else None
        )
        if self.requested_cycle_count_ > 0 and not junctions:
            # A closed one-manifold has two annulus sides at every ordinary
            # point; finite sampling can label a few of those annuli as one
            # component.  Persistence already establishes that this is a
            # cycle, so those isolated endpoint votes are not valid terminal
            # regions unless a junction is present.
            endpoints = []
        elif junctions:
            # Endpoint votes at the sampled crossing itself are artifacts of
            # the annulus discretization, not terminal branches.
            endpoints = [
                endpoint for endpoint in endpoints
                if all(
                    np.linalg.norm(endpoint.center - junction.center)
                    > 2.0 * self.local_scale_
                    for junction in junctions
                )
            ]
            if self.requested_cycle_count_ > 0 and len(junctions) >= 2:
                # A closed graph with several junction regions has no valid
                # degree-one terminals in the synthetic/topological model.
                # Centroid MST leaves in this case are usually short pieces
                # of a loop or a high-dimensional edge, not observed
                # endpoints.  Keeping them forces the landmark selector to
                # trade cycles for artificial terminal branches.
                endpoints = []
        # Use the coarse tree only to stabilize singular-region candidates.
        # The final routes still come exclusively from the dense weighted kNN
        # substrate below, so this does not reinstate coarsening as the
        # backbone-construction mechanism.
        coarse_graph = _minimum_spanning_tree(centroids)
        if self.prune_short_branches:
            # Centroid MSTs can create very short terminal stubs at noisy
            # crossings.  They are especially harmful in topological mode:
            # a stub can be promoted to a second junction before dense-graph
            # routing begins.  Keep the user-controlled pruning floor, with
            # a modest minimum suited to centroid-level geometry.  The small
            # margin above one removes the nearly equal-length duplicate
            # terminal stubs produced when k-means splits one noisy arm.
            _prune_short_terminal_branches(
                coarse_graph,
                max(self.prune_branch_factor, 1.05),
            )
        coarse_graph = _merge_nearby_junctions(
            coarse_graph,
            18.0 * self.local_scale_ if self.merge_junction_distance is None else self.merge_junction_distance,
        )
        coarse_junction_nodes = [] if (
            self.linear_structure_
            or not self.detect_junctions
            or central_junction_locked
            or (
                len(routing_components) > 1
                and sum(component_cycle_counts) >= len(routing_components)
            )
            or len(routing_components) > 1
        ) else [
            node for node in coarse_graph.nodes if coarse_graph.degree(node) >= 3
        ]
        if (
            self.requested_cycle_count_ == 1
            and not junctions
            and len(coarse_junction_nodes) != 1
        ):
            # A spanning tree of a single noisy loop can have several
            # degree-three vertices simply because the centroid tree chose a
            # short chord.  Those are not observed junctions: promoting them
            # splits one closed component into a loop plus artificial stubs.
            # Retain the coarse fallback only when it identifies one isolated
            # branch point, the topology expected for a loop-with-branch.
            coarse_junction_nodes = []
        if coarse_junction_nodes:
            stabilized: list[JunctionRegion] = []
            for node in coarse_junction_nodes:
                center = coarse_graph.nodes[node]
                expected = coarse_graph.degree(node)
                if expected == 3 and self.requested_cycle_count_ >= 2:
                    expected = 4
                center_index = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
                centered = points - center
                if points.shape[1] >= 2:
                    _, _, components = np.linalg.svd(centered, full_matrices=False)
                    projection = centered @ components[:2].T
                    angles = np.arctan2(projection[:, 1], projection[:, 0])
                else:
                    angles = centered[:, 0]
                radii = np.linalg.norm(centered, axis=1)
                annulus = np.flatnonzero(
                    (radii >= 0.5 * self.local_scale_)
                    & (radii <= 8.0 * self.local_scale_)
                )
                if len(annulus):
                    order = annulus[np.argsort(angles[annulus])]
                    arms = [array.astype(int) for array in np.array_split(order, expected) if len(array)]
                else:
                    arms = []
                nearby_region = next(
                    (
                        region for region in junctions
                        if np.linalg.norm(center - region.center) <= 2.0 * self.local_scale_
                    ),
                    None,
                )
                stabilized.append(
                    JunctionRegion(
                        center.copy(),
                        expected,
                        1.0 if nearby_region is not None else 0.75,
                        list(nearby_region.member_indices) if nearby_region is not None else [center_index],
                        arms,
                    )
                )
            junctions = stabilized
        if not junctions and self.detect_junctions and not self.linear_structure_ and len(points):
            center_index = int(np.argmin(np.linalg.norm(points - np.mean(points, axis=0), axis=1)))
            center = points[center_index]
            if np.linalg.norm(center - np.mean(points, axis=0)) <= 2.0 * self.local_scale_:
                # Crossings can undersample the exact singular point, causing
                # the strict multiscale confidence test to miss it.  Use the
                # nearby angular sectors only as a conservative local fallback.
                centered = points - center
                _, _, components = np.linalg.svd(centered, full_matrices=False)
                projection = centered @ components[:2].T if points.shape[1] >= 2 else centered[:, :1]
                angles = np.arctan2(projection[:, -1], projection[:, 0])
                radius = np.linalg.norm(centered, axis=1)
                annulus = np.flatnonzero(
                    (radius >= 0.5 * self.local_scale_)
                    & (radius <= 4.0 * self.local_scale_)
                )
                arms = [annulus[(angles[annulus] >= lower) & (angles[annulus] < upper)] for lower, upper in zip(
                    np.linspace(-np.pi, np.pi, 5)[:-1], np.linspace(-np.pi, np.pi, 5)[1:]
                )]
                arms = [arm.astype(int) for arm in arms if len(arm)]
                if len(arms) >= 3:
                    junctions = [JunctionRegion(center.copy(), len(arms), 0.75, [center_index], arms)]
        if junctions:
            # The coarse stabilization above can create the junction only
            # after local endpoint votes have been collected.  Apply the
            # crossing exclusion again at that point so the sampled crossing
            # is never promoted to a terminal landmark.
            endpoints = [
                endpoint for endpoint in endpoints
                if all(
                    np.linalg.norm(endpoint.center - junction.center)
                    > 2.0 * self.local_scale_
                    for junction in junctions
                )
            ]
        if junctions and not self.linear_structure_:
            closed_multi_junction = (
                self.requested_cycle_count_ > 0 and len(junctions) >= 2
            )
            desired_endpoints = 0 if closed_multi_junction else max(
                0,
                2
                + sum(max(0, region.branch_count - 2) for region in junctions)
                - 2 * (self.requested_cycle_count_ if self.detect_cycles else 0),
            )
            leaves = [
                node for node in coarse_graph.nodes
                if coarse_graph.degree(node) == 1
            ]
            if desired_endpoints >= 3 and len(leaves) >= desired_endpoints:
                # For multi-arm crossings, the pruned coarse tree has a much
                # more stable terminal set than isolated annulus votes.  Use
                # it even when the annulus detector happens to return the
                # right count: its locations are otherwise often internal
                # points on one arm rather than the true terminals.
                leaves.sort(
                    key=lambda node: min(
                        np.linalg.norm(coarse_graph.nodes[node] - region.center)
                        for region in junctions
                    ),
                    reverse=True,
                )
                endpoints = [
                    EndpointRegion(
                        coarse_graph.nodes[node].copy(),
                        0.85,
                        [int(np.argmin(np.sum((points - coarse_graph.nodes[node]) ** 2, axis=1)))],
                    )
                    for node in leaves[:desired_endpoints]
                ]
                leaves = []
            if len(endpoints) < desired_endpoints:
                leaves.sort(
                    key=lambda node: min(
                        np.linalg.norm(coarse_graph.nodes[node] - region.center)
                        for region in junctions
                    ),
                    reverse=True,
                )
                for node in leaves:
                    center = coarse_graph.nodes[node]
                    if any(
                        np.linalg.norm(center - region.center) <= 2.0 * self.local_scale_
                        for region in junctions
                    ) or any(
                        np.linalg.norm(center - region.center) <= 2.0 * self.local_scale_
                        for region in endpoints
                    ):
                        continue
                    endpoint_member = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
                    endpoints.append(EndpointRegion(center.copy(), 0.75, [endpoint_member]))
                    if len(endpoints) >= desired_endpoints:
                        break
            if len(endpoints) > desired_endpoints:
                endpoints.sort(key=lambda region: region.confidence, reverse=True)
                endpoints = endpoints[:desired_endpoints]
        self.junction_regions_ = junctions
        self.endpoint_regions_ = endpoints
        self.junctions_ = junctions
        self.endpoints_ = endpoints
        self.branch_counts_ = branch_counts
        self.branch_confidence_ = branch_confidence
        self.topology_confidence_ = branch_confidence.copy()

        # Build compact logical landmarks while retaining their original graph
        # vertices for routing and electrical calculations.
        specifications: list[dict[str, Any]] = []
        used_vertices: list[int] = []

        def nearest_vertex(center: Array) -> int:
            distances = np.sum((points - center) ** 2, axis=1)
            return int(np.argmin(distances))

        def append_spec(kind: str, center: Array, region: Any = None, vertex: int | None = None) -> None:
            selected_vertex = nearest_vertex(center) if vertex is None else int(vertex)
            if any(
                np.linalg.norm(points[selected_vertex] - points[spec["vertex"]])
                <= max(self.local_scale_, 1e-8)
                and kind == spec["kind"]
                for spec in specifications
            ):
                return
            landmark_id = len(specifications)
            if region is not None:
                region.node_id = landmark_id
            specifications.append({
                "kind": kind,
                "center": np.asarray(center, dtype=float).copy(),
                "vertex": selected_vertex,
                "region": region,
            })
            used_vertices.append(selected_vertex)

        for region in junctions:
            append_spec("junction", region.center, region)
        for region in endpoints:
            append_spec("endpoint", region.center, region)

        cycle_target = self.requested_cycle_count_ if self.detect_cycles else 0
        if (
            cycle_target > 0
            and len(routing_components) > 1
            and sum(component_cycle_counts) > 0
        ):
            # The bridge retained for electrical diagnostics can otherwise
            # make the global cycle-anchor spread spend all anchors on one
            # component.  Allocate anchors per persistent component.
            anchor_vertices: list[int] = []
            for component, component_cycles in zip(
                routing_components, component_cycle_counts
            ):
                target = 3 * int(component_cycles)
                if target <= 0 or len(component) == 0:
                    continue
                selected_component = [int(component[0])]
                while len(selected_component) < min(target, len(component)):
                    distances = np.min(
                        np.asarray([
                            np.linalg.norm(points[component] - points[anchor], axis=1)
                            for anchor in selected_component
                        ]),
                        axis=0,
                    )
                    distances[
                        [int(np.flatnonzero(component == anchor)[0]) for anchor in selected_component]
                    ] = -np.inf
                    next_index = int(np.argmax(distances))
                    if not np.isfinite(distances[next_index]):
                        break
                    selected_component.append(int(component[next_index]))
                anchor_vertices.extend(selected_component)
        else:
            anchor_vertices = _cycle_anchor_vertices(routing_graph, cycle_target)
        for vertex in anchor_vertices:
            if any(
                np.linalg.norm(points[vertex] - points[other]) <= max(self.local_scale_, 1e-8)
                for other in used_vertices
            ):
                continue
            append_spec("cycle_anchor", points[vertex], vertex=vertex)

        if not specifications and len(points):
            first = int(np.argmin(np.sum((points - np.mean(points, axis=0)) ** 2, axis=1)))
            farthest = int(np.argmax(np.linalg.norm(points - points[first], axis=1)))
            append_spec("endpoint", points[first], vertex=first)
            if farthest != first:
                append_spec("endpoint", points[farthest], vertex=farthest)

        if self.use_local_pca:
            branch_directions = _attach_junction_directions(points, junctions)
        else:
            branch_directions = {index: np.empty((0, points.shape[1])) for index in range(len(junctions))}
        self.junction_branch_directions_ = {}
        for index, region in enumerate(junctions):
            node_id = region.node_id
            if node_id is not None:
                self.junction_branch_directions_[node_id] = branch_directions.get(
                    index, np.empty((0, points.shape[1]), dtype=float)
                )

        # A junction landmark is represented by one observed vertex, which
        # can miss an arm in a noisy kNN query.  Add short virtual incidence
        # links to the nearest point in each PCA arm direction.  These links
        # stay local, retain the dense graph as the routing substrate, and
        # make the angle constraint independent of which observation happens
        # to be nearest the junction center.
        if self.use_local_pca:
            for region in junctions:
                if region.node_id is None:
                    continue
                source = specifications[region.node_id]["vertex"]
                directions = self.junction_branch_directions_.get(region.node_id, np.empty((0, points.shape[1])))
                if not len(directions):
                    continue
                component = component_by_vertex.get(source)
                candidates = np.flatnonzero(
                    np.linalg.norm(points - region.center, axis=1)
                    <= 8.0 * self.local_scale_
                )
                candidates = np.asarray([
                    vertex for vertex in candidates
                    if vertex != source
                    and (
                        component is None
                        or component_by_vertex.get(int(vertex)) == component
                    )
                ], dtype=int)
                used_targets: set[int] = set()
                for direction in directions:
                    if not len(candidates):
                        break
                    vectors = points[candidates] - region.center
                    norms = np.linalg.norm(vectors, axis=1)
                    valid = norms > 1e-12
                    scores = np.full(len(candidates), np.inf)
                    scores[valid] = np.asarray([
                        _departure_angle(direction, vector)
                        for vector in vectors[valid]
                    ]) + 0.15 * norms[valid] / max(8.0 * self.local_scale_, 1e-12)
                    for target_index in np.argsort(scores):
                        target = int(candidates[target_index])
                        if target not in used_targets and np.isfinite(scores[target_index]):
                            break
                    else:
                        continue
                    used_targets.add(target)
                    edge = routing_graph.key(source, target)
                    if edge not in routing_graph.edges:
                        length = float(np.linalg.norm(points[source] - points[target]))
                        denominator = max(
                            local_scales[source] * local_scales[target], 1e-12
                        )
                        conductance = float(np.exp(-(length * length) / denominator))
                        routing_graph.add_edge(source, target, length, conductance)
                        routing_graph.edge_density[edge] = 1.0

        self.effective_resistance_ = {}
        self.edge_leverage_ = {}
        self.electrical_traffic_ = {}
        if self.use_effective_resistance or self.routing_resistance_weight > 0.0:
            _, self.effective_resistance_, self.edge_leverage_ = (
                _effective_resistance(routing_graph)
            )
        if self.use_electrical_flow or self.routing_current_weight > 0.0:
            pair_vertices = [spec["vertex"] for spec in specifications]
            pairs = [
                (pair_vertices[left], pair_vertices[right])
                for left in range(len(pair_vertices))
                for right in range(left + 1, len(pair_vertices))
            ]
            self.electrical_traffic_ = _electrical_flow(routing_graph, pairs)

        if self.use_kron_reduction:
            landmark_vertices = [spec["vertex"] for spec in specifications]
            reduced, retained = _kron_reduction(routing_graph, landmark_vertices)
            order = [int(np.flatnonzero(retained == vertex)[0]) for vertex in landmark_vertices]
            self.kron_laplacian_ = reduced[np.ix_(order, order)]
            self.kron_vertex_ids_ = np.asarray(landmark_vertices, dtype=int)
            conductance_graph = _LandmarkGraph({
                index: spec["center"] for index, spec in enumerate(specifications)
            })
            for left in range(len(retained)):
                for right in range(left + 1, len(retained)):
                    conductance = -float(self.kron_laplacian_[left, right])
                    if conductance > 1e-10:
                        conductance_graph.add_edge(left, right, conductance)
            self.landmark_conductance_graph_ = conductance_graph
        else:
            self.kron_laplacian_ = None
            self.kron_vertex_ids_ = np.empty(0, dtype=int)
            self.landmark_conductance_graph_ = None

        edge_lengths = np.asarray(list(routing_graph.edges.values()), dtype=float)
        reference_length = float(np.median(edge_lengths[edge_lengths > 1e-12])) if np.any(edge_lengths > 1e-12) else 1.0
        leverage_values = np.asarray(list(self.edge_leverage_.values()), dtype=float)
        leverage_reference = float(np.max(leverage_values)) if len(leverage_values) else 1.0
        traffic_reference = 1.0
        angle_threshold = 1.0 - np.cos(np.deg2rad(self.max_branch_angle_degrees))

        def edge_cost(left: int, right: int) -> float:
            edge = routing_graph.key(left, right)
            length_term = routing_graph.edges[edge] / max(reference_length, 1e-12)
            tangent_term = _tangent_inconsistency(
                self.local_tangents_[left], self.local_tangents_[right]
            )
            density_term = 1.0 / max(routing_graph.edge_density.get(edge, 0.0), 1e-6)
            resistance_support = self.edge_leverage_.get(edge, 0.0) / max(leverage_reference, 1e-12)
            current_support = self.electrical_traffic_.get(edge, 0.0) / max(traffic_reference, 1e-12)
            cost = (
                self.routing_length_weight * length_term
                + self.routing_tangent_weight * tangent_term
                + self.routing_density_weight * density_term
                - self.routing_resistance_weight * resistance_support
                - self.routing_current_weight * current_support
            )
            return max(float(cost), 1e-8)

        def allowed_first_edges(specification: dict[str, Any]) -> list[tuple[int, int | None]]:
            source = specification["vertex"]
            if specification["kind"] != "junction":
                return [(neighbour, None) for neighbour in routing_graph.adjacency[source]]
            directions = self.junction_branch_directions_.get(specification["region"].node_id, np.empty((0, points.shape[1])))
            candidates: list[tuple[int, int | None, float]] = []
            for neighbour in routing_graph.adjacency[source]:
                vector = points[neighbour] - specification["center"]
                if len(directions):
                    scores = [_departure_angle(direction, vector) for direction in directions]
                    branch = int(np.argmin(scores))
                    if scores[branch] <= angle_threshold:
                        candidates.append((neighbour, branch, scores[branch]))
                else:
                    candidates.append((neighbour, None, 0.0))
            candidates.sort(key=lambda item: item[2])
            return [(neighbour, branch) for neighbour, branch, _ in candidates]

        def shortest_path(
            start: int,
            target: int,
            blocked_edges: set[tuple[int, int]] | None = None,
            required_branch: int | None = None,
        ) -> CandidatePath | None:
            source_spec = specifications[start]
            target_spec = specifications[target]
            source_vertex = source_spec["vertex"]
            target_vertex = target_spec["vertex"]
            if source_vertex == target_vertex:
                return None
            blocked_edges = set() if blocked_edges is None else blocked_edges
            queue: list[tuple[float, int, list[int], int | None]] = []
            best: dict[tuple[int, int | None], float] = {}
            for neighbour, branch in allowed_first_edges(source_spec):
                if required_branch is not None and branch != required_branch:
                    continue
                if (
                    component_by_vertex
                    and component_by_vertex.get(source_vertex)
                    != component_by_vertex.get(neighbour)
                ):
                    continue
                if routing_graph.key(source_vertex, neighbour) in blocked_edges:
                    continue
                initial = edge_cost(source_vertex, neighbour)
                heapq.heappush(queue, (initial, neighbour, [source_vertex, neighbour], branch))
                best[(neighbour, branch)] = initial
            while queue:
                cost, node, path, branch_start = heapq.heappop(queue)
                if cost > best.get((node, branch_start), np.inf) + 1e-12:
                    continue
                if node == target_vertex:
                    branch_end = None
                    if target_spec["kind"] == "junction" and len(path) >= 2:
                        directions = self.junction_branch_directions_.get(
                            target_spec["region"].node_id, np.empty((0, points.shape[1]))
                        )
                        vector = points[target_vertex] - points[path[-2]]
                        if len(directions):
                            scores = [_departure_angle(direction, vector) for direction in directions]
                            branch_end = int(np.argmin(scores))
                            if scores[branch_end] > angle_threshold:
                                continue
                    length = float(sum(
                        np.linalg.norm(points[left] - points[right])
                        for left, right in pairwise(path)
                    ))
                    tangent_cost = float(sum(
                        _tangent_inconsistency(self.local_tangents_[left], self.local_tangents_[right])
                        for left, right in pairwise(path)
                    ))
                    support = float(np.mean([
                        self.edge_leverage_.get(routing_graph.key(left, right), 0.0)
                        for left, right in pairwise(path)
                    ])) if len(path) > 1 else 0.0
                    traffic = float(np.mean([
                        self.electrical_traffic_.get(routing_graph.key(left, right), 0.0)
                        for left, right in pairwise(path)
                    ])) if len(path) > 1 else 0.0
                    return CandidatePath(
                        start, target, path, float(cost), length, tangent_cost,
                        support, traffic, branch_start, branch_end,
                    )
                for neighbour in routing_graph.adjacency[node]:
                    if neighbour in path:
                        continue
                    if (
                        component_by_vertex
                        and component_by_vertex.get(node)
                        != component_by_vertex.get(neighbour)
                    ):
                        continue
                    if routing_graph.key(node, neighbour) in blocked_edges:
                        continue
                    candidate_cost = cost + edge_cost(node, neighbour)
                    if candidate_cost < best.get((neighbour, branch_start), np.inf):
                        best[(neighbour, branch_start)] = candidate_cost
                        heapq.heappush(queue, (candidate_cost, neighbour, path + [neighbour], branch_start))
            return None

        candidates: list[CandidatePath] = []
        candidate_attempts: list[tuple[int, int, int | None, bool]] = []
        for left in range(len(specifications)):
            for right in range(left + 1, len(specifications)):
                branch_options: list[int | None] = [None]
                if specifications[left]["kind"] == "junction":
                    branch_options = sorted({
                        branch for _, branch in allowed_first_edges(specifications[left])
                        if branch is not None
                    }) or [None]
                for required_branch in branch_options:
                    path = shortest_path(
                        left,
                        right,
                        required_branch=required_branch,
                    )
                    candidate_attempts.append((left, right, required_branch, path is not None))
                    if path is not None:
                        candidates.append(path)
        candidates.sort(key=lambda candidate: (candidate.total_cost, candidate.start_landmark, candidate.end_landmark))
        self.candidate_paths_ = list(candidates)
        self.candidate_path_attempts_ = candidate_attempts

        graph = _LandmarkGraph({index: spec["center"] for index, spec in enumerate(specifications)})
        selected: dict[tuple[int, int], CandidatePath] = {}
        parent = list(range(len(specifications)))
        degree = np.zeros(len(specifications), dtype=int)
        used_branches: dict[int, set[int]] = {
            index: set() for index, spec in enumerate(specifications) if spec["kind"] == "junction"
        }

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: int, right: int) -> bool:
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return False
            parent[right_root] = left_root
            return True

        def valid_degree(
            candidate: CandidatePath,
            enforce_branch_slots: bool = True,
            allow_anchor_overflow: bool = False,
        ) -> bool:
            for node in (candidate.start_landmark, candidate.end_landmark):
                kind = specifications[node]["kind"]
                if kind == "endpoint" and degree[node] >= 1:
                    return False
                if (
                    kind == "cycle_anchor"
                    and degree[node] >= 2
                    and not allow_anchor_overflow
                ):
                    return False
            if enforce_branch_slots and (
                candidate.branch_start is not None
                and candidate.branch_start in used_branches.get(candidate.start_landmark, set())
            ):
                return False
            return not (
                enforce_branch_slots
                and candidate.branch_end is not None
                and candidate.branch_end in used_branches.get(candidate.end_landmark, set())
            )

        def mark_branches(candidate: CandidatePath) -> None:
            if candidate.branch_start is not None:
                used_branches.setdefault(candidate.start_landmark, set()).add(candidate.branch_start)
            if candidate.branch_end is not None:
                used_branches.setdefault(candidate.end_landmark, set()).add(candidate.branch_end)

        def unmark_branches(candidate: CandidatePath) -> None:
            if candidate.branch_start is not None:
                used_branches.get(candidate.start_landmark, set()).discard(candidate.branch_start)
            if candidate.branch_end is not None:
                used_branches.get(candidate.end_landmark, set()).discard(candidate.branch_end)

        def reroute_cycle_candidate(candidate: CandidatePath) -> CandidatePath | None:
            """Find a route that does not reuse an already selected cycle arm."""
            if find(candidate.start_landmark) != find(candidate.end_landmark):
                return candidate
            blocked = {
                routing_graph.key(left, right)
                for selected_candidate in selected.values()
                for left, right in pairwise(selected_candidate.vertices)
            }
            if not blocked:
                return candidate
            alternative = shortest_path(
                candidate.start_landmark,
                candidate.end_landmark,
                blocked_edges=blocked,
            )
            return alternative

        constrained_first = sorted(
            candidates,
            key=lambda candidate: (
                0 if any(
                    specifications[node]["kind"] == "junction"
                    for node in (candidate.start_landmark, candidate.end_landmark)
                ) else 1,
                candidate.total_cost,
                candidate.start_landmark,
                candidate.end_landmark,
            ),
        )
        for candidate in constrained_first:
            if not valid_degree(candidate):
                continue
            if union(candidate.start_landmark, candidate.end_landmark):
                key = graph._key(candidate.start_landmark, candidate.end_landmark)
                selected[key] = candidate
                degree[candidate.start_landmark] += 1
                degree[candidate.end_landmark] += 1
                mark_branches(candidate)
        for candidate in candidates:
            key = graph._key(candidate.start_landmark, candidate.end_landmark)
            if key in selected:
                continue
            needs_arm = any(
                (
                    specifications[node]["kind"] == "junction"
                    and degree[node] < max(1, specifications[node]["region"].branch_count)
                )
                or (
                    specifications[node]["kind"] == "cycle_anchor"
                    and degree[node] < 2
                )
                for node in (candidate.start_landmark, candidate.end_landmark)
            )
            if not valid_degree(candidate, allow_anchor_overflow=needs_arm):
                continue
            if needs_arm:
                # Incidence completion is allowed to share a dense-substrate
                # segment near a landmark.  An anchor-completion edge that
                # closes a component is different: it intentionally creates
                # a cycle and must be rerouted before it is committed.  If
                # this is postponed until the cycle pass below, the anchor
                # degree constraint has already consumed the overlapping
                # candidate and closed loops can fold back over themselves.
                if (
                    self.detect_cycles
                    and self.requested_cycle_count_ > 0
                    and find(candidate.start_landmark) == find(candidate.end_landmark)
                ):
                    alternative = reroute_cycle_candidate(candidate)
                    if alternative is not None:
                        candidate = alternative
                        key = graph._key(candidate.start_landmark, candidate.end_landmark)
                    if not valid_degree(candidate, allow_anchor_overflow=needs_arm):
                        continue
                selected[key] = candidate
                degree[candidate.start_landmark] += 1
                degree[candidate.end_landmark] += 1
                mark_branches(candidate)
        for candidate in candidates:
            key = graph._key(candidate.start_landmark, candidate.end_landmark)
            if key in selected:
                continue
            needs_arm = any(
                (
                    specifications[node]["kind"] == "junction"
                    and degree[node] < max(1, specifications[node]["region"].branch_count)
                )
                or (
                    specifications[node]["kind"] == "cycle_anchor"
                    and degree[node] < 2
                )
                for node in (candidate.start_landmark, candidate.end_landmark)
            )
            if not valid_degree(
                candidate,
                enforce_branch_slots=False,
                allow_anchor_overflow=needs_arm,
            ):
                continue
            if needs_arm:
                if (
                    self.detect_cycles
                    and self.requested_cycle_count_ > 0
                    and find(candidate.start_landmark) == find(candidate.end_landmark)
                ):
                    alternative = reroute_cycle_candidate(candidate)
                    if alternative is not None:
                        candidate = alternative
                        key = graph._key(candidate.start_landmark, candidate.end_landmark)
                    if not valid_degree(
                        candidate,
                        enforce_branch_slots=False,
                        allow_anchor_overflow=needs_arm,
                    ):
                        continue
                selected[key] = candidate
                degree[candidate.start_landmark] += 1
                degree[candidate.end_landmark] += 1
                mark_branches(candidate)

        if self.detect_cycles and self.requested_cycle_count_ > 0:
            for candidate in candidates:
                key = graph._key(candidate.start_landmark, candidate.end_landmark)
                if key in selected or not valid_degree(candidate):
                    continue
                candidate = reroute_cycle_candidate(candidate)
                if candidate is None or not valid_degree(candidate):
                    continue
                selected[key] = candidate
                degree[candidate.start_landmark] += 1
                degree[candidate.end_landmark] += 1
                mark_branches(candidate)
                components = len(specifications)
                component_parent = list(range(len(specifications)))
                for selected_left, selected_right in selected:
                    left_root, right_root = selected_left, selected_right
                    while component_parent[left_root] != left_root:
                        left_root = component_parent[left_root]
                    while component_parent[right_root] != right_root:
                        right_root = component_parent[right_root]
                    if left_root != right_root:
                        component_parent[right_root] = left_root
                        components -= 1
                cycle_rank = len(selected) - len(specifications) + components
                if cycle_rank >= self.requested_cycle_count_:
                    break

        def selected_cycle_rank(edges: dict[tuple[int, int], CandidatePath]) -> int:
            components = len(specifications)
            component_parent = list(range(len(specifications)))
            for selected_left, selected_right in edges:
                left_root, right_root = selected_left, selected_right
                while component_parent[left_root] != left_root:
                    left_root = component_parent[left_root]
                while component_parent[right_root] != right_root:
                    right_root = component_parent[right_root]
                if left_root != right_root:
                    component_parent[right_root] = left_root
                    components -= 1
            return len(edges) - len(specifications) + components

        # Degree completion can introduce redundant cycles. Remove only routes
        # whose endpoints retain their requested incidence and whose removal
        # does not disconnect the logical landmark graph.
        while selected_cycle_rank(selected) > self.requested_cycle_count_:
            removed = False
            for key, candidate in sorted(
                selected.items(), key=lambda item: item[1].total_cost, reverse=True
            ):
                left, right = key
                if any(
                    specifications[node]["kind"] == "endpoint" or (
                        specifications[node]["kind"] == "junction"
                        and degree[node] <= specifications[node]["region"].branch_count
                    ) or (
                        specifications[node]["kind"] == "cycle_anchor"
                        and degree[node] <= 2
                    )
                    for node in (left, right)
                ):
                    continue
                trial = selected.copy()
                del trial[key]
                if selected_cycle_rank(trial) < selected_cycle_rank(selected):
                    selected = trial
                    degree[left] -= 1
                    degree[right] -= 1
                    unmark_branches(candidate)
                    removed = True
                    break
            if not removed:
                break

        # A dense cycle target and coarse junction degrees can be mutually
        # incompatible.  For example, a sampled hypercube may expose several
        # high-degree centroid junctions while the requested cycle budget is
        # deliberately capped.  In that case the degree-preserving pass
        # above cannot remove the last redundant cycle.  Prefer the explicit
        # cycle-rank target and relax the least-supported junction incidences;
        # the resulting degree shortfall remains available in diagnostics.
        while selected_cycle_rank(selected) > self.requested_cycle_count_:
            removable: list[tuple[float, tuple[int, int], CandidatePath]] = []
            for key, candidate in selected.items():
                trial = selected.copy()
                del trial[key]
                if selected_cycle_rank(trial) < selected_cycle_rank(selected):
                    removable.append((candidate.total_cost, key, candidate))
            if not removable:
                break
            _, key, candidate = max(removable, key=lambda item: item[0])
            del selected[key]
            degree[key[0]] -= 1
            degree[key[1]] -= 1
            unmark_branches(candidate)

        if self.linear_structure_ and selected:
            # The dense kNN shortest path is allowed to zig-zag across a
            # noisy strip.  For a one-dimensional cloud, PCA ordering is the
            # topological route and gives the spline every observation in
            # monotone longitudinal order.
            direction = self.linear_direction_
            center = self.linear_center_
            coordinates = (points - center) @ direction
            ordered_vertices = np.argsort(coordinates).astype(int).tolist()
            ordered_coordinate = {
                vertex: float(coordinates[vertex])
                for vertex in ordered_vertices
            }
            linear_points = center + coordinates[:, None] * direction
            for key, candidate in list(selected.items()):
                left, right = key
                vertices = ordered_vertices
                if ordered_coordinate[candidate.vertices[0]] > ordered_coordinate[candidate.vertices[-1]]:
                    vertices = list(reversed(vertices))
                route_length = float(sum(
                    np.linalg.norm(linear_points[a] - linear_points[b])
                    for a, b in pairwise(vertices)
                ))
                selected[key] = CandidatePath(
                    candidate.start_landmark,
                    candidate.end_landmark,
                    vertices,
                    candidate.total_cost,
                    route_length,
                    candidate.tangent_cost,
                    candidate.electrical_support,
                    candidate.current_support,
                    candidate.branch_start,
                    candidate.branch_end,
                )

        for key, candidate in selected.items():
            graph.add_edge(key[0], key[1], candidate.length)
        self.landmark_vertex_ids_ = np.asarray(
            [specification["vertex"] for specification in specifications],
            dtype=int,
        )
        self.junction_degree_shortfall_ = {
            index: max(0, specifications[index]["region"].branch_count - degree[index])
            for index in range(len(specifications))
            if specifications[index]["kind"] == "junction"
        }
        self.endpoint_degree_violations_ = [
            index for index in range(len(specifications))
            if specifications[index]["kind"] == "endpoint" and degree[index] != 1
        ]
        if any(self.junction_degree_shortfall_.values()) or self.endpoint_degree_violations_:
            warnings.warn(
                "Topological landmark constraints could not all be realized by the routing substrate.",
                RuntimeWarning,
                stacklevel=3,
            )
        return graph, selected

    def _chain_support_points(self, chain: dict[str, Any]) -> Array:
        """Concatenate stored point-level paths for one abstract route chain."""
        nodes = list(chain["nodes"])
        if chain.get("closed") and not any(
            self.landmark_graph_.degree(node) >= 3 for node in set(nodes)
        ):
            ordered_component = self._ordered_closed_component_points(nodes)
            if ordered_component is not None:
                return ordered_component
        pairs = list(pairwise(nodes))
        if chain.get("closed") and len(nodes) > 1:
            pairs.append((nodes[-1], nodes[0]))
        segments: list[Array] = []
        for left, right in pairs:
            if left == right:
                continue
            candidate = self.backbone_paths_.get(self.landmark_graph_._key(left, right))
            if candidate is None:
                continue
            vertices = list(candidate.vertices)
            forward = candidate.start_landmark == left
            if not forward:
                vertices.reverse()
            support = np.asarray([self.routing_graph_.points[index] for index in vertices], dtype=float)
            support[0] = self.landmark_graph_.nodes[left]
            support[-1] = self.landmark_graph_.nodes[right]
            if segments and np.allclose(segments[-1][-1], support[0]):
                support = support[1:]
            segments.append(support)
        if not segments:
            return np.asarray([self.landmark_graph_.nodes[node] for node in nodes], dtype=float)
        return np.vstack(segments)

    def _ordered_closed_component_points(self, nodes: list[int]) -> Array | None:
        """Return a full, non-backtracking support order for a simple loop.

        A dense kNN graph can contain chords and several shortest paths between
        cycle anchors.  Concatenating those paths may reuse one side of a loop
        and fold the fitted spline back over itself.  For an unbranched loop,
        all observations in its natural component are better support: ordering
        them around the first two principal axes gives one complete turn and
        is stable for the planar/noisy cycles this route represents.
        """
        if not nodes or not getattr(self, "routing_components_", None):
            return None
        landmark_vertices = getattr(self, "landmark_vertex_ids_", None)
        if landmark_vertices is None:
            return None
        vertex = int(landmark_vertices[nodes[0]])
        component_id = self.routing_component_by_vertex_.get(vertex)
        if component_id is None:
            return None
        component = np.asarray(self.routing_components_[component_id], dtype=int)
        if len(component) < 3:
            return None
        points = np.asarray(self.routing_graph_.points[component], dtype=float)
        centered = points - np.mean(points, axis=0, keepdims=True)
        if points.shape[1] < 2:
            return None
        _, singular_values, components = np.linalg.svd(
            centered, full_matrices=False,
        )
        if len(singular_values) < 2 or singular_values[1] <= 1e-10:
            return None
        planar = centered @ components[:2].T
        angles = np.arctan2(planar[:, 1], planar[:, 0])
        order = np.argsort(angles, kind="mergesort")
        return points[order]

    def _figure_eight_support_points(self, points: Array, chain: dict[str, Any]) -> Array:
        """Order observations around each lobe of a locked figure-eight."""
        nodes = [node for node in chain["nodes"] if node != chain["nodes"][0]]
        if not nodes or self.central_junction_center_ is None:
            return chain["points"]
        center = np.asarray(self.central_junction_center_, dtype=float)
        node_mean = float(np.mean([
            self.landmark_graph_.nodes[node][0] for node in nodes
        ]))
        side = -1.0 if node_mean < center[0] else 1.0
        side_coordinate = points[:, 0] - center[0]
        mask = side_coordinate <= 0.0 if side < 0.0 else side_coordinate >= 0.0
        lobe = np.asarray(points[mask], dtype=float)
        if len(lobe) < 8:
            return chain["points"]
        lobe_center = np.mean(lobe, axis=0)
        angles = np.arctan2(
            lobe[:, 1] - lobe_center[1],
            lobe[:, 0] - lobe_center[0],
        )
        ordered = lobe[np.argsort(angles)]
        start = int(np.argmin(np.linalg.norm(ordered - center, axis=1)))
        ordered = np.roll(ordered, -start, axis=0)
        return np.vstack([center, ordered, center])

    def _simple_cycle_support_points(self, points: Array, chain: dict[str, Any]) -> Array:
        """Use the full observation component for a junction-free cycle."""
        anchor = int(chain["nodes"][0])
        anchor_vertex = self.backbone_paths_.get(
            self.landmark_graph_._key(anchor, chain["nodes"][1])
        )
        if anchor_vertex is None:
            return chain["points"]
        component_id = self.routing_component_by_vertex_.get(
            int(anchor_vertex.vertices[0])
        )
        if component_id is None or component_id >= len(self.routing_components_):
            return chain["points"]
        component = self.routing_components_[component_id]
        if len(component) < 8:
            return chain["points"]
        component_points = np.asarray(points[component], dtype=float)
        center = np.mean(component_points, axis=0)
        angles = np.arctan2(
            component_points[:, 1] - center[1],
            component_points[:, 0] - center[0],
        )
        ordered = component_points[np.argsort(angles)]
        anchor_center = self.landmark_graph_.nodes[anchor]
        start = int(np.argmin(np.linalg.norm(ordered - anchor_center, axis=1)))
        ordered = np.roll(ordered, -start, axis=0)
        ordered[0] = anchor_center
        return np.vstack([ordered, ordered[0]])

    def _anchor_closed_junctions(self) -> None:
        """Make closed spline samples pass exactly through graph junctions."""
        for chain, spline in zip(self.route_chains_, self.routes_):
            if not chain["closed"] or len(spline.samples) == 0:
                continue
            nodes = list(chain["nodes"])
            if len(nodes) > 1 and nodes[0] == nodes[-1]:
                nodes = nodes[:-1]
            used_samples: set[int] = set()
            for node_index, node in enumerate(nodes):
                if self.landmark_graph_.degree(node) < 3 or node_index >= len(spline.t_values):
                    continue
                parameter = float(spline.t_values[node_index]) % 1.0
                sample = int(np.rint(parameter * len(spline.samples))) % len(spline.samples)
                if sample in used_samples:
                    continue
                spline.samples[sample] = self.landmark_graph_.nodes[node]
                used_samples.add(sample)

    def _chain_boundary_tangents(self, chain: dict[str, Any]) -> tuple[Array | None, Array | None]:
        """Resolve PCA directions for the two ends of an open topological route."""
        if (
            self.backbone_paths_ is None
            or not self.use_tangent_boundary_conditions
            or chain.get("closed")
            or len(chain.get("points", [])) < 2
        ):
            return None, None
        points = np.asarray(chain["points"], dtype=float)
        nodes = list(chain["nodes"])

        def tangent_at(node: int, displacement: Array, sign: float) -> Array | None:
            if node in self.junction_branch_directions_:
                directions = self.junction_branch_directions_[node]
                if len(directions):
                    scores = [
                        _departure_angle(direction, displacement * sign)
                        for direction in directions
                    ]
                    return directions[int(np.argmin(scores))] * sign
            norm = float(np.linalg.norm(displacement))
            if norm > 1e-12:
                return sign * displacement / norm
            return None

        start = tangent_at(nodes[0], points[1] - points[0], 1.0)
        end = tangent_at(nodes[-1], points[-2] - points[-1], -1.0)
        return start, end

    def _best_cycle_edge(self) -> tuple[int, int, float] | None:
        graph = self.landmark_graph_
        best: tuple[float, int, int, float] | None = None
        existing = set(graph.edges)
        for left, right in sorted(self.topology_candidate_edges_):
            if graph._key(left, right) in existing:
                continue
            direct = float(np.linalg.norm(graph.nodes[left] - graph.nodes[right]))
            if direct <= max(1e-10, 0.25 * self.local_scale_):
                continue
            path_distance, path = graph.shortest_path(left, right)
            if len(path) < 4:
                continue
            score = path_distance / direct
            candidate = (score, left, right, direct)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            return None
        _, left, right, direct = best
        return left, right, direct

    def transform(self, X: Array | Sequence[Sequence[float]]) -> EmbeddingResult:
        if not self._fitted:
            raise RuntimeError("Call fit before transform")
        original = _as_point_cloud(X)
        if original.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        points = (original - self.mean_) / self.scale_
        best_distance = np.full(len(points), np.inf)
        route_id = np.full(len(points), -1, dtype=int)
        t = np.zeros(len(points))
        projection = np.zeros_like(points)
        for route, spline in enumerate(self.routes_):
            candidate_projection, candidate_t, candidate_d2 = spline.project(points)
            improved = candidate_d2 < best_distance
            best_distance[improved] = candidate_d2[improved]
            route_id[improved] = route
            t[improved] = candidate_t[improved]
            projection[improved] = candidate_projection[improved]
        self.invalid_projection_count_ = int(np.sum(route_id < 0))
        if self.invalid_projection_count_:
            raise RuntimeError(
                "The fitted spline graph could not project all observations; "
                f"{self.invalid_projection_count_} observations have no valid route."
            )
        projected_original = projection * self.scale_ + self.mean_
        residual = original - projected_original
        tangent = np.zeros_like(points)
        for route, spline in enumerate(self.routes_):
            members = np.flatnonzero(route_id == route)
            if len(members):
                tangent[members] = spline.tangent(t[members])
        return EmbeddingResult(
            route_id=route_id,
            position=t,
            projected=projected_original,
            residual=residual,
            residual_norm=np.linalg.norm(residual, axis=1),
            # Tangents are stored in the standardized fitting coordinates so
            # they can define the correct high-dimensional normal hyperplane.
            tangent=tangent,
        )

    def fit_transform(self, X: Array | Sequence[Sequence[float]]) -> EmbeddingResult:
        return self.fit(X).transform(X)

    def normal_coordinates(self, result: EmbeddingResult) -> Array:
        """Return residual coordinates in deterministic route-normal frames."""
        if not isinstance(result, EmbeddingResult):
            raise TypeError("result must be an EmbeddingResult")
        if result.n_features != self.n_features_in_:
            raise ValueError("result has a different number of features than the fitted data")
        return _normal_coordinates(self, result)

    def plot_network(
        self,
        X: Array | Sequence[Sequence[float]] | None = None,
        ax: Any = None,
        show_projections: bool = False,
        max_projection_lines: int = 150,
        title: str | None = None,
    ) -> Any:
        """Plot the fitted route network using the visualization package."""
        from .visualization.network import plot_network

        return plot_network(
            self,
            X=X,
            ax=ax,
            show_projections=show_projections,
            max_projection_lines=max_projection_lines,
            title=title,
        )


__all__ = ["SplineGraphEmbedding"]
