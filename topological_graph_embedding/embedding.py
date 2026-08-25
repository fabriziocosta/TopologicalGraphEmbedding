"""Public spline graph embedding estimator."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
import heapq
from typing import Any

import numpy as np

from ._curves import _fit_curve
from ._frames import (
    _fit_normal_frame_grid,
    _normal_coordinates,
)
from ._topology import (
    CandidatePath,
    EndpointRegion,
    JunctionRegion,
    _LandmarkGraph,
    _as_point_cloud,
    _cycle_anchor_vertices,
    _estimate_persistence,
    _estimate_local_topology,
    _extract_chains,
    _is_nearly_linear,
    _kmeans,
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
from ._electrical import _electrical_flow, _effective_resistance, _kron_reduction
from ._local_geometry import (
    _attach_junction_directions,
    _departure_angle,
    _estimate_local_tangents,
    _tangent_inconsistency,
)
from .results import EmbeddingResult

Array = np.ndarray

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
            threshold = 4.0 if self.persistence_threshold is None else float(self.persistence_threshold)
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

        self.centroids_ = _kmeans(points, self.n_centroids, self.random_state)
        # Structure detection is evaluated in the original metric.  This is
        # important for noisy one-dimensional clouds: feature standardization
        # can otherwise turn small orthogonal noise into an artificial branch.
        centroids_original = self.centroids_ * scale + mean
        self.linear_structure_ = _is_nearly_linear(
            centroids_original, self.linear_structure_tolerance
        )
        self.backbone_paths_ = None
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

        if self.backbone_paths_ is None:
            target_cycles = self.requested_cycle_count_
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
        self.routes_ = [
            _fit_curve(
                chain["points"],
                closed=chain["closed"],
                smoothing=self.spline_smoothing,
                sample_count=max(64, len(chain["nodes"]) * self.spline_samples_per_node),
            )
            for chain in self.route_chains_
        ]
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
        if self.use_local_pca:
            self.local_tangents_ = _estimate_local_tangents(
                points, routing_graph, self.local_pca_neighbors
            )
        else:
            self.local_tangents_ = np.zeros_like(points)

        if self.detect_junctions:
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

        self.effective_resistance_ = {}
        self.edge_leverage_ = {}
        self.electrical_traffic_ = {}
        electrical_pseudoinverse = None
        if self.use_effective_resistance or self.routing_resistance_weight > 0.0:
            electrical_pseudoinverse, self.effective_resistance_, self.edge_leverage_ = (
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
            self.kron_laplacian_, retained = _kron_reduction(routing_graph, landmark_vertices)
            self.kron_vertex_ids_ = retained
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

        def shortest_path(start: int, target: int) -> CandidatePath | None:
            source_spec = specifications[start]
            target_spec = specifications[target]
            source_vertex = source_spec["vertex"]
            target_vertex = target_spec["vertex"]
            if source_vertex == target_vertex:
                return None
            queue: list[tuple[float, int, list[int], int | None]] = []
            best: dict[int, float] = {}
            for neighbour, branch in allowed_first_edges(source_spec):
                initial = edge_cost(source_vertex, neighbour)
                heapq.heappush(queue, (initial, neighbour, [source_vertex, neighbour], branch))
                best[neighbour] = initial
            while queue:
                cost, node, path, branch_start = heapq.heappop(queue)
                if cost > best.get(node, np.inf) + 1e-12:
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
                        for left, right in zip(path[:-1], path[1:])
                    ))
                    tangent_cost = float(sum(
                        _tangent_inconsistency(self.local_tangents_[left], self.local_tangents_[right])
                        for left, right in zip(path[:-1], path[1:])
                    ))
                    support = float(np.mean([
                        self.edge_leverage_.get(routing_graph.key(left, right), 0.0)
                        for left, right in zip(path[:-1], path[1:])
                    ])) if len(path) > 1 else 0.0
                    traffic = float(np.mean([
                        self.electrical_traffic_.get(routing_graph.key(left, right), 0.0)
                        for left, right in zip(path[:-1], path[1:])
                    ])) if len(path) > 1 else 0.0
                    return CandidatePath(
                        start, target, path, float(cost), length, tangent_cost,
                        support, traffic, branch_start, branch_end,
                    )
                for neighbour in routing_graph.adjacency[node]:
                    if neighbour in path:
                        continue
                    candidate_cost = cost + edge_cost(node, neighbour)
                    if candidate_cost < best.get(neighbour, np.inf):
                        best[neighbour] = candidate_cost
                        heapq.heappush(queue, (candidate_cost, neighbour, path + [neighbour], branch_start))
            return None

        candidates: list[CandidatePath] = []
        for left in range(len(specifications)):
            for right in range(left + 1, len(specifications)):
                path = shortest_path(left, right)
                if path is not None:
                    candidates.append(path)
        candidates.sort(key=lambda candidate: (candidate.total_cost, candidate.start_landmark, candidate.end_landmark))

        graph = _LandmarkGraph({index: spec["center"] for index, spec in enumerate(specifications)})
        selected: dict[tuple[int, int], CandidatePath] = {}
        parent = list(range(len(specifications)))
        degree = np.zeros(len(specifications), dtype=int)

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

        def valid_degree(candidate: CandidatePath) -> bool:
            for node in (candidate.start_landmark, candidate.end_landmark):
                kind = specifications[node]["kind"]
                if kind == "endpoint" and degree[node] >= 1:
                    return False
            return True

        for candidate in candidates:
            if not valid_degree(candidate):
                continue
            if union(candidate.start_landmark, candidate.end_landmark):
                key = graph._key(candidate.start_landmark, candidate.end_landmark)
                selected[key] = candidate
                degree[candidate.start_landmark] += 1
                degree[candidate.end_landmark] += 1
        for candidate in candidates:
            key = graph._key(candidate.start_landmark, candidate.end_landmark)
            if key in selected or not valid_degree(candidate):
                continue
            needs_arm = any(
                specifications[node]["kind"] == "junction"
                and degree[node] < max(1, specifications[node]["region"].branch_count)
                for node in (candidate.start_landmark, candidate.end_landmark)
            )
            if needs_arm:
                selected[key] = candidate
                degree[candidate.start_landmark] += 1
                degree[candidate.end_landmark] += 1

        if self.detect_cycles and self.requested_cycle_count_ > 0:
            for candidate in candidates:
                key = graph._key(candidate.start_landmark, candidate.end_landmark)
                if key in selected or not valid_degree(candidate):
                    continue
                selected[key] = candidate
                degree[candidate.start_landmark] += 1
                degree[candidate.end_landmark] += 1
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

        for key, candidate in selected.items():
            graph.add_edge(key[0], key[1], candidate.length)
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
        pairs = list(zip(nodes[:-1], nodes[1:]))
        if chain.get("closed") and len(nodes) > 1:
            pairs.append((nodes[-1], nodes[0]))
        segments: list[Array] = []
        for left, right in pairs:
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
