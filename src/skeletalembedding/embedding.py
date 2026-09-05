"""Public spline graph embedding estimator."""

from __future__ import annotations

import heapq
import inspect
import warnings
from collections.abc import Sequence
from itertools import combinations, pairwise
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
from ._multiresolution import (
    finish_hierarchy_diagnostics,
    infer_hierarchy_topology,
    initialize_hierarchy,
    refine_backbone,
    validate_hierarchy_parameters,
)
from ._optimization import select_backbone_mip
from ._residual_pca import attach_residual_pca, fit_residual_pca
from ._ribs import prepare_rib_candidates, propose_ribs, select_ribs
from ._stability import (
    jitter_points,
    match_cycles,
    match_regions,
    match_routes,
    subsample_indices,
    subspace_principal_angle,
)
from ._topology import (
    CandidatePath,
    EndpointRegion,
    JunctionRegion,
    PersistentCycle,
    _approximate_cycle_representatives,
    _as_point_cloud,
    _cycle_anchor_vertices,
    _estimate_local_topology,
    _estimate_persistence,
    _extract_chains,
    _hypercube_junction_regions,
    _is_nearly_linear,
    _kmeans,
    _LandmarkGraph,
    _local_scale,
    _merge_nearby_junctions,
    _minimum_spanning_tree,
    _normalize_persistence_diagram,
    _prune_short_terminal_branches,
    _standardize,
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

class SkeletalEmbedding:
    """Fit a stable nonlinear backbone, residual fields, and coverage ribs."""

    def __init__(
        self,
        n_centroids: int = 32,
        n_backbone_nodes: int | None = None,
        backbone_node_spacing: float | None = None,
        backbone_node_policy: str = "topology_preserving",
        n_neighbors: int = 6,
        persistence_threshold: float | None = None,
        spline_smoothing: float = 0.02,
        spline_control_mode: str = "support",
        max_cycles: int = 5,
        random_state: int = 0,
        standardize: bool = True,
        merge_junction_distance: float | None = None,
        prune_short_branches: bool = True,
        prune_branch_factor: float = 0.5,
        persistence_max_points: int = 60,
        spline_samples_per_node: int = 12,
        linear_structure_tolerance: float = 0.12,
        topology_neighbors: int | None = None,
        mutual_knn: bool = True,
        add_mst: bool = True,
        max_residual_dim: int = 0,
        residual_pca_bandwidth: float = 0.1,
        residual_subspace_smoothness: float = 0.0,
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
        coverage_refinement: bool = False,
        coverage_error_tolerance: float | None = None,
        coverage_relative_tolerance: float | None = None,
        coverage_quantile: float = 0.95,
        coverage_max_iterations: int = 10,
        coverage_max_ribs: int | None = None,
        coverage_max_candidates_per_iteration: int = 20,
        coverage_candidate_spacing: float | None = None,
        coverage_min_error: float | None = None,
        coverage_min_gain: float = 0.0,
        coverage_length_penalty: float = 0.0,
        coverage_rib_penalty: float = 0.0,
        coverage_junction_penalty: float = 0.0,
        coverage_selection: str = "greedy",
        rib_candidate_type: str = "transverse",
        stability_selection: bool = False,
        stability_runs: int = 30,
        stability_fraction: float = 0.7,
        stability_min_support: float = 0.75,
        stability_jitter: float = 0.0,
        rib_stability_runs: int | None = None,
        rib_min_support: float = 0.6,
        stability_residual_subspaces: bool = False,
        use_multiresolution: bool = True,
        hierarchy_max_levels: int = 8,
        hierarchy_target_size: int = 1000,
        hierarchy_min_reduction: float = 0.15,
        representative_method: str = "medoid",
        hierarchy_distance_quantile: float = 0.1,
        hierarchy_local_neighbors: int = 10,
        backbone_level: int | str = "auto",
        backbone_max_representatives: int = 2000,
        backbone_consensus_levels: int = 3,
        route_resolution_weight: float = 0.1,
        rib_resolution_weight: float = 0.1,
        rib_seed_source: str = "both",
        n_jobs: int | None = None,
    ) -> None:
        if n_centroids < 3:
            raise ValueError("n_centroids must be at least 3")
        if n_backbone_nodes is not None and (
            isinstance(n_backbone_nodes, bool)
            or not isinstance(n_backbone_nodes, (int, np.integer))
            or n_backbone_nodes < 1
        ):
            raise ValueError("n_backbone_nodes must be a positive integer or None")
        if backbone_node_spacing is not None and (
            backbone_node_spacing <= 0.0 or not np.isfinite(backbone_node_spacing)
        ):
            raise ValueError("backbone_node_spacing must be positive and finite or None")
        if n_backbone_nodes is not None and backbone_node_spacing is not None:
            raise ValueError(
                "n_backbone_nodes and backbone_node_spacing are mutually exclusive"
            )
        if backbone_node_policy not in {
            "topology_preserving",
            "allow_topology_relaxation",
        }:
            raise ValueError(
                "backbone_node_policy must be 'topology_preserving' or "
                "'allow_topology_relaxation'"
            )
        if max_cycles < 0:
            raise ValueError("max_cycles must be non-negative")
        if persistence_max_points < 1:
            raise ValueError("persistence_max_points must be at least 1")
        if spline_samples_per_node < 1:
            raise ValueError("spline_samples_per_node must be at least 1")
        if topology_neighbors is not None and topology_neighbors < 2:
            raise ValueError("topology_neighbors must be at least 2")
        if n_neighbors < 2:
            raise ValueError("n_neighbors must be at least 2")
        if isinstance(max_residual_dim, bool) or not isinstance(max_residual_dim, (int, np.integer)):
            raise ValueError(  # noqa: TRY004
                "max_residual_dim must be a non-negative integer"
            )
        if max_residual_dim < 0:
            raise ValueError("max_residual_dim must be non-negative")
        if residual_pca_bandwidth <= 0.0 or not np.isfinite(residual_pca_bandwidth):
            raise ValueError("residual_pca_bandwidth must be positive and finite")
        if residual_subspace_smoothness < 0.0 or not np.isfinite(residual_subspace_smoothness):
            raise ValueError("residual_subspace_smoothness must be finite and non-negative")
        if spline_control_mode not in {"support", "backbone"}:
            raise ValueError("spline_control_mode must be 'support' or 'backbone'")
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
        if coverage_error_tolerance is not None and (
            coverage_error_tolerance < 0 or not np.isfinite(coverage_error_tolerance)
        ):
            raise ValueError("coverage_error_tolerance must be finite and non-negative")
        if coverage_relative_tolerance is not None and (
            coverage_relative_tolerance < 0 or not np.isfinite(coverage_relative_tolerance)
        ):
            raise ValueError("coverage_relative_tolerance must be finite and non-negative")
        if not 0.0 < coverage_quantile <= 1.0:
            raise ValueError("coverage_quantile must be in (0, 1]")
        if coverage_max_iterations < 1:
            raise ValueError("coverage_max_iterations must be at least 1")
        if coverage_max_ribs is not None and coverage_max_ribs < 0:
            raise ValueError("coverage_max_ribs must be non-negative")
        if coverage_max_candidates_per_iteration < 1:
            raise ValueError("coverage_max_candidates_per_iteration must be at least 1")
        coverage_values = (
            coverage_min_gain, coverage_length_penalty, coverage_rib_penalty,
            coverage_junction_penalty, rib_min_support,
        )
        if any(value < 0 or not np.isfinite(value) for value in coverage_values):
            raise ValueError("coverage penalties and supports must be finite and non-negative")
        if coverage_selection not in {"greedy", "mip"}:
            raise ValueError("coverage_selection must be 'greedy' or 'mip'")
        if rib_candidate_type not in {"transverse", "parallel", "both"}:
            raise ValueError("rib_candidate_type must be 'transverse', 'parallel', or 'both'")
        if stability_runs < 1 or not 0.0 < stability_fraction <= 1.0:
            raise ValueError("stability_runs must be positive and stability_fraction must be in (0, 1]")
        if not 0.0 <= stability_min_support <= 1.0:
            raise ValueError("stability_min_support must be in [0, 1]")
        if stability_jitter < 0 or not np.isfinite(stability_jitter):
            raise ValueError("stability_jitter must be finite and non-negative")
        if rib_stability_runs is not None and rib_stability_runs < 1:
            raise ValueError("rib_stability_runs must be positive when provided")
        self.n_centroids = int(n_centroids)
        self.n_backbone_nodes = (
            None if n_backbone_nodes is None else int(n_backbone_nodes)
        )
        self.backbone_node_spacing = (
            None if backbone_node_spacing is None else float(backbone_node_spacing)
        )
        self.backbone_node_policy = str(backbone_node_policy)
        self.n_neighbors = int(n_neighbors)
        self.persistence_threshold = persistence_threshold
        self.spline_smoothing = float(spline_smoothing)
        self.spline_control_mode = str(spline_control_mode)
        self.max_cycles = int(max_cycles)
        self.random_state = int(random_state)
        self.standardize = bool(standardize)
        self.merge_junction_distance = merge_junction_distance
        self.prune_short_branches = prune_short_branches
        self.prune_branch_factor = float(prune_branch_factor)
        self.persistence_max_points = int(persistence_max_points)
        self.spline_samples_per_node = int(spline_samples_per_node)
        self.linear_structure_tolerance = float(linear_structure_tolerance)
        self.topology_neighbors = topology_neighbors
        self.topology_neighbors_ = int(n_neighbors if topology_neighbors is None else topology_neighbors)
        self.mutual_knn = bool(mutual_knn)
        self.add_mst = bool(add_mst)
        self.max_residual_dim = int(max_residual_dim)
        self.residual_pca_bandwidth = float(residual_pca_bandwidth)
        self.residual_subspace_smoothness = float(residual_subspace_smoothness)
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
        self.coverage_refinement = bool(coverage_refinement)
        self.coverage_error_tolerance = coverage_error_tolerance
        self.coverage_relative_tolerance = coverage_relative_tolerance
        self.coverage_quantile = float(coverage_quantile)
        self.coverage_max_iterations = int(coverage_max_iterations)
        self.coverage_max_ribs = None if coverage_max_ribs is None else int(coverage_max_ribs)
        self.coverage_max_candidates_per_iteration = int(coverage_max_candidates_per_iteration)
        self.coverage_candidate_spacing = coverage_candidate_spacing
        self.coverage_min_error = coverage_min_error
        self.coverage_min_gain = float(coverage_min_gain)
        self.coverage_length_penalty = float(coverage_length_penalty)
        self.coverage_rib_penalty = float(coverage_rib_penalty)
        self.coverage_junction_penalty = float(coverage_junction_penalty)
        self.coverage_selection = str(coverage_selection)
        self.rib_candidate_type = str(rib_candidate_type)
        self.stability_selection = bool(stability_selection)
        self.stability_runs = int(stability_runs)
        self.stability_fraction = float(stability_fraction)
        self.stability_min_support = float(stability_min_support)
        self.stability_jitter = float(stability_jitter)
        self.rib_stability_runs = rib_stability_runs
        self.rib_min_support = float(rib_min_support)
        self.stability_residual_subspaces = bool(stability_residual_subspaces)
        self.use_multiresolution = use_multiresolution
        self.hierarchy_max_levels = hierarchy_max_levels
        self.hierarchy_target_size = hierarchy_target_size
        self.hierarchy_min_reduction = hierarchy_min_reduction
        self.representative_method = representative_method
        self.hierarchy_distance_quantile = hierarchy_distance_quantile
        self.hierarchy_local_neighbors = hierarchy_local_neighbors
        self.backbone_level = backbone_level
        self.backbone_max_representatives = backbone_max_representatives
        self.backbone_consensus_levels = backbone_consensus_levels
        self.route_resolution_weight = route_resolution_weight
        self.rib_resolution_weight = rib_resolution_weight
        self.rib_seed_source = rib_seed_source
        self.n_jobs = n_jobs
        validate_hierarchy_parameters(self)
        self._fitted = False

    def fit(self, X: Array | Sequence[Sequence[float]]) -> SkeletalEmbedding:
        # A repeated fit must never reuse fitted graph or rib state.
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        self.topology_neighbors_ = self.topology_neighbors or self.n_neighbors
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
        self.residual_dim_ = min(self.max_residual_dim, max(0, self.n_features_in_ - 1))
        self.mean_ = mean
        self.scale_ = scale
        self._original_X_ = original

        initialize_hierarchy(self, points)
        infer_hierarchy_topology(self)
        structure_points = self.levels_[self.selected_backbone_level_].points
        self.backbone_input_size_ = len(structure_points)
        self.persistence_diagram_, self.persistence_backend_ = _estimate_persistence(
            structure_points, max_points=self.persistence_max_points, random_state=self.random_state
        )
        self.local_scale_ = _local_scale(structure_points)
        self.normalized_persistence_diagram_ = _normalize_persistence_diagram(
            self.persistence_diagram_, self.local_scale_
        )
        # Topology fitting uses scale-free persistence in normalized
        # nearest-neighbour units.
        threshold = 3.0 if self.persistence_threshold is None else float(self.persistence_threshold)
        self.persistence_threshold_ = float(threshold)
        diagram = self.normalized_persistence_diagram_
        if len(diagram):
            lifetimes = diagram[:, 1] - diagram[:, 0]
            significant = np.isfinite(lifetimes) & (lifetimes >= self.persistence_threshold_)
            self.persistent_cycle_count_ = int(np.sum(significant))
        else:
            self.persistent_cycle_count_ = 0
        self.persistent_cycles_ = [
            PersistentCycle(
                float(birth),
                float(death),
                float(death - birth),
                representative=None,
            )
            for birth, death in np.asarray(diagram, dtype=float)[
                np.isfinite(np.asarray(diagram, dtype=float)).all(axis=1)
                & ((np.asarray(diagram, dtype=float)[:, 1] - np.asarray(diagram, dtype=float)[:, 0]) >= self.persistence_threshold_)
            ]
        ]
        if len(self.levels_) > 1:
            self.persistent_cycles_ = sorted(self.persistent_cycles_, key=lambda cycle: -cycle.persistence)[:self.max_cycles]
        self.cycle_support_ = np.ones(len(self.persistent_cycles_), dtype=float)
        self.cycle_count_ = self.persistent_cycle_count_
        self.requested_cycle_count_ = min(self.max_cycles, self.persistent_cycle_count_)
        if not self.detect_cycles:
            self.requested_cycle_count_ = 0

        self.centroids_ = _kmeans(structure_points, self.n_centroids, self.random_state)
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
        if self.linear_structure_:
            # A noisy line can carry a small numerical H1 bar in a subsample;
            # its one-dimensional geometry is a stronger constraint than that
            # artifact and must not create a cycle anchor.
            self.requested_cycle_count_ = 0
        self.backbone_paths_ = None
        self.central_junction_locked_ = False
        self.central_junction_center_ = None
        self.face_cycle_count_ = 0
        self.hypercube_dimension_ = None
        from ._stability import prepare_structural_subsamples
        prepare_structural_subsamples(self, points)
        if len(self.levels_) > 1:
            from ._stability import structural_feature_evidence
            evidence = self.topology_by_level_.get(self.selected_backbone_level_, {})
            for cycle, detected in zip(self.persistent_cycles_, evidence.get("persistent_cycles", [])):
                cycle.representative = detected.representative
            stable_cycles = structural_feature_evidence(self, cycles=self.persistent_cycles_)
            self.requested_cycle_count_ = min(self.requested_cycle_count_, len(stable_cycles))
        graph, backbone_paths = self._topological_backbone(structure_points, self.centroids_)
        refine_backbone(self, graph, backbone_paths)
        graph, backbone_paths = self._resize_skeletal_backbone(
            graph, backbone_paths, structure_points
        )
        self.backbone_paths_ = backbone_paths
        self.merge_junction_distance_ = 0.0
        representatives = _approximate_cycle_representatives(
            self.routing_graph_, self.persistent_cycle_count_
        )
        for cycle, representative in zip(self.persistent_cycles_, representatives):
            cycle.representative = representative
        if not hasattr(self, "mip_status_"):
            self.mip_status_ = "not_applicable"
        self.landmark_graph_ = graph
        self.backbone_graph_ = graph
        self.topology_candidate_edges_ = set()

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
            if self.selected_backbone_level_ > 0:
                chain["preserve_support_order"] = True
            if self.backbone_paths_ is None:
                chain["points"] = np.asarray([self.landmark_graph_.nodes[node] for node in chain["nodes"]])
            else:
                chain["points"] = self._chain_support_points(chain, points)
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
            if self.spline_control_mode == "backbone":
                spline_points = np.asarray(chain["points"], dtype=float).copy()
                spline_weights = np.ones(len(spline_points), dtype=float)
                for node in chain["nodes"]:
                    landmark = np.asarray(self.landmark_graph_.nodes[node], dtype=float)
                    anchor = int(np.argmin(np.sum((spline_points - landmark) ** 2, axis=1)))
                    spline_points[anchor] = landmark
                    spline_weights[anchor] = 100.0
                chain["spline_points"] = spline_points
                chain["spline_weights"] = spline_weights
            else:
                chain["spline_points"] = np.asarray(chain["points"], dtype=float)
        self.routes_ = []
        for chain in self.route_chains_:
            start_tangent, end_tangent = self._chain_boundary_tangents(chain)
            self.routes_.append(
                _fit_curve(
                    chain.get("spline_points", chain["points"]),
                    closed=chain["closed"],
                    smoothing=self.spline_smoothing,
                    sample_count=max(64, len(chain["nodes"]) * self.spline_samples_per_node),
                    start_tangent=start_tangent,
                    end_tangent=end_tangent,
                    weights=chain.get("spline_weights"),
                    preserve_support_order=bool(chain.get("preserve_support_order", False)),
                )
            )
        self._anchor_closed_junctions()
        self.route_backends_ = [spline.backend for spline in self.routes_]
        self.normal_frame_grids_ = [
            _fit_normal_frame_grid(spline) for spline in self.routes_
        ]
        self.splines_ = self.routes_
        centerline_result = self._project_centerline(original)
        fit_residual_pca(self, points, centerline_result)
        self._initialize_skeleton_metadata()
        self._refine_coverage(original, points)
        self._initialize_skeleton_metadata()
        self._run_stability_selection(original)
        self._fit_final_diagnostics(original, points)
        finish_hierarchy_diagnostics(self)
        self._fitted = True
        return self

    def _topological_backbone(
        self,
        points: Array,
        centroids: Array,
        *,
        topology_only: bool = False,
    ) -> tuple[_LandmarkGraph, dict[tuple[int, int], CandidatePath]]:
        """Infer a constrained landmark graph from the dense routing substrate."""
        routing_graph, local_scales = _weighted_symmetric_knn_graph(
            points,
            self.topology_neighbors_,
            mutual_knn=self.mutual_knn,
            add_mst=self.add_mst,
        )
        self.routing_graph_ = routing_graph
        self.local_graph_scales_ = local_scales
        routing_components = [
            np.asarray(component, dtype=int)
            for component in getattr(
                routing_graph,
                "topology_components",
                getattr(routing_graph, "original_components", []),
            )
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
            if component_cycle_total != self.requested_cycle_count_:
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
        hypercube_junctions_detected = False
        if self.detect_junctions and not self.linear_structure_:
            hypercube_junctions, face_count, hypercube_dimension = (
                _hypercube_junction_regions(points, self.local_scale_)
            )
            if hypercube_junctions:
                junctions = hypercube_junctions
                endpoints = []
                hypercube_junctions_detected = True
                self.face_cycle_count_ = face_count
                self.hypercube_dimension_ = hypercube_dimension
        if (
            not self.linear_structure_
            and len(routing_components) > 1
            and self.requested_cycle_count_ == 0
        ):
            # A global centroid MST connects disconnected clouds and can
            # manufacture junctions on curved components.  Local annulus
            # votes are also ambiguous on a bent open strand: an interior bend
            # can look like an endpoint at one scale.  Use the weighted graph
            # diameter of each natural component instead, which returns the
            # actual two terminals of every open arc.
            # A disconnected open component cannot contain a junction that
            # connects the components.  Annulus overlaps between nearby
            # strands are therefore treated as false local branch votes.
            junctions = []
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
            and points.shape[1] >= 2
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
            # A planar synthetic graph may have independent noise dimensions.
            # Use the two dominant PCA coordinates for the annulus as well, so
            # that orthogonal measurement noise cannot split the crossing into
            # several false junction regions.
            radii = np.linalg.norm(projection, axis=1)
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
            or hypercube_junctions_detected
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
        loop_branch_endpoint_center: Array | None = None
        if junctions and not self.linear_structure_:
            closed_multi_junction = (
                self.requested_cycle_count_ > 0 and len(junctions) >= 2
            )
            closed_high_order_junction = (
                self.requested_cycle_count_ > 0
                and len(junctions) == 1
                and junctions[0].branch_count >= 4
                and self.persistent_cycle_count_ >= 1
            )
            if closed_high_order_junction:
                # A high-order junction on a cyclic graph is a crossing or
                # cycle attachment, not evidence of terminal branches.  In
                # this case the annulus endpoint votes are artificial leaves
                # created by the sparse routing approximation.
                endpoints = []
            desired_endpoints = 0 if (
                closed_multi_junction or closed_high_order_junction
            ) else max(
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
            if (
                self.requested_cycle_count_ == 1
                and desired_endpoints == 1
                and len(junctions) == 1
                and junctions[0].branch_count == 3
                and leaves
            ):
                # A centroid MST breaks a loop at two arbitrary leaves.  The
                # remaining leaf is the actual terminal arm of a
                # loop-with-branch, but local annulus votes can prefer one of
                # the artificial loop breaks.  Select the coarse leaf nearest
                # the junction and keep its region out of cycle anchoring.
                junction_center = junctions[0].center
                branch_leaf = min(
                    leaves,
                    key=lambda node: np.linalg.norm(
                        coarse_graph.nodes[node] - junction_center
                    ),
                )
                loop_branch_endpoint_center = coarse_graph.nodes[branch_leaf].copy()
                endpoint_member = int(np.argmin(
                    np.sum((points - loop_branch_endpoint_center) ** 2, axis=1)
                ))
                endpoints = [EndpointRegion(
                    loop_branch_endpoint_center.copy(),
                    0.9,
                    [endpoint_member],
                )]
        if (
            not junctions
            and not self.linear_structure_
            and self.requested_cycle_count_ == 0
            and len(routing_components) == 1
            and len(routing_components[0]) >= 2
        ):
            # A curved open manifold has no reliable local endpoint vote at
            # every scale: bends and the noisy inner end of a spiral can look
            # like short terminal regions.  The weighted graph diameter gives
            # the two terminals of the whole connected component and lets
            # the dense routing substrate carry the spline across the full
            # curve rather than fitting a short landmark-to-landmark chord.
            left, right = _geodesic_diameter_endpoints(
                routing_graph,
                routing_components[0],
            )
            endpoints = [
                EndpointRegion(points[left].copy(), 1.0, [left]),
                EndpointRegion(points[right].copy(), 1.0, [right]),
            ]
        if (
            len(routing_components) > 1
            and self.requested_cycle_count_ >= len(routing_components)
            and sum(component_cycle_counts) >= len(routing_components)
        ):
            # Independent persistent loops have no intrinsic junction. A
            # coarse MST can create a spurious branch vote where its bridge
            # touches one loop; keep that bridge in the substrate, but do not
            # promote it to skeletal topology.
            junctions = []
            endpoints = []
        if self.use_multiresolution and (len(self.levels_) > 1 or getattr(self, "_evaluating_hierarchy_", False)):
            # Compression must not turn an empty region into an intrinsic
            # junction merely because nearest-neighbor spacing increased.
            if central_junction_locked:
                junctions = [region for region in junctions
                             if np.min(np.linalg.norm(points - region.center, axis=1)) <= self.local_scale_]
            if not junctions:
                central_junction_locked = False
                self.central_junction_locked_ = False
                self.central_junction_center_ = None
                loop_branch_endpoint_center = None
                if self.requested_cycle_count_ > 0:
                    endpoints = []
            if hypercube_junctions_detected and len(junctions) != 2 ** int(self.hypercube_dimension_ or 0):
                hypercube_junctions_detected = False
        self.loop_branch_endpoint_center_ = loop_branch_endpoint_center
        if len(self.levels_) > 1:
            from ._stability import structural_feature_evidence
            junctions = structural_feature_evidence(self, junctions=junctions)
        self.junction_regions_ = junctions
        self.endpoint_regions_ = endpoints
        self.junctions_ = junctions
        self.endpoints_ = endpoints
        self.branch_counts_ = branch_counts
        self.branch_confidence_ = branch_confidence
        self.topology_confidence_ = branch_confidence.copy()

        if topology_only:
            return None, None

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
            excluded_vertices: set[int] = set()
            if loop_branch_endpoint_center is not None:
                branch_vector = (
                    loop_branch_endpoint_center - junctions[0].center
                )
                branch_length = float(np.linalg.norm(branch_vector))
                branch_direction = branch_vector / max(branch_length, 1e-12)
                offsets = points - junctions[0].center
                longitudinal = offsets @ branch_direction
                orthogonal = np.linalg.norm(
                    offsets - longitudinal[:, None] * branch_direction[None, :],
                    axis=1,
                )
                corridor_width = max(4.0 * self.local_scale_, 0.25 * branch_length)
                in_branch_corridor = (
                    (longitudinal >= 0.05 * branch_length)
                    & (longitudinal <= 1.15 * branch_length)
                    & (orthogonal <= corridor_width)
                )
                excluded_vertices = set(np.flatnonzero(in_branch_corridor).astype(int))
            anchor_vertices = _cycle_anchor_vertices(
                routing_graph,
                cycle_target,
                excluded_vertices=excluded_vertices,
            )
        if hypercube_junctions_detected:
            # The eight cube corners already provide the complete logical
            # vertex set.  Adding persistence anchors between them would
            # obscure the face/edge structure with support landmarks.
            anchor_vertices = []
        for vertex in anchor_vertices:
            if any(
                np.linalg.norm(points[vertex] - points[other]) <= max(self.local_scale_, 1e-8)
                for other in used_vertices
            ):
                continue
            append_spec("cycle_anchor", points[vertex], vertex=vertex)

        self.cycle_connector_node_ids_ = []
        if (len(self.levels_) > 1 and cycle_target > 1 and not junctions and not endpoints
                and len(routing_components) == 1 and len(specifications) >= 2 * cycle_target + 1):
            # Multiple surface cycles need a shared skeletal connector even
            # when the sampled manifold has no intrinsic branch junction.
            # Leave its degree to the existing connectivity / cycle-rank MIP
            # constraints; do not mislabel it as a detected junction.
            specifications[0]["kind"] = "cycle_connector"
            self.cycle_connector_node_ids_ = [0]

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
        path_graph = getattr(routing_graph, "topology_graph", routing_graph)
        leverage_values = np.asarray(list(self.edge_leverage_.values()), dtype=float)
        leverage_reference = float(np.max(leverage_values)) if len(leverage_values) else 1.0
        traffic_reference = 1.0
        angle_threshold = 1.0 - np.cos(np.deg2rad(self.max_branch_angle_degrees))

        def edge_cost(left: int, right: int) -> float:
            edge = routing_graph.key(left, right)
            length = (
                routing_graph.edges[edge]
                if edge in routing_graph.edges
                else path_graph.edges[edge]
            )
            length_term = length / max(reference_length, 1e-12)
            tangent_term = _tangent_inconsistency(
                self.local_tangents_[left], self.local_tangents_[right]
            )
            density_term = 1.0 / max(routing_graph.edge_density.get(edge, 1.0), 1e-6)
            resistance_support = self.edge_leverage_.get(edge, 0.0) / max(leverage_reference, 1e-12)
            current_support = self.electrical_traffic_.get(edge, 0.0) / max(traffic_reference, 1e-12)
            cost = (
                self.routing_length_weight * length_term
                + self.routing_tangent_weight * tangent_term
                + self.routing_density_weight * density_term
                - self.routing_resistance_weight * resistance_support
                - self.routing_current_weight * current_support
            )
            if self.requested_cycle_count_ > 0 and edge in getattr(routing_graph, "mst_edges", set()):
                # MST augmentation keeps the substrate connected but should
                # not shortcut a persistent loop when a natural route exists.
                cost += 10.0
            return max(float(cost), 1e-8)

        def allowed_first_edges(
            specification: dict[str, Any],
            search_graph: Any = path_graph,
        ) -> list[tuple[int, int | None]]:
            source = specification["vertex"]
            if specification["kind"] != "junction":
                return [(neighbour, None) for neighbour in search_graph.adjacency[source]]
            directions = self.junction_branch_directions_.get(specification["region"].node_id, np.empty((0, points.shape[1])))
            candidates: list[tuple[int, int | None, float]] = []
            for neighbour in search_graph.adjacency[source]:
                vector = points[neighbour] - specification["center"]
                if len(directions):
                    scores = [_departure_angle(direction, vector) for direction in directions]
                    branch = int(np.argmin(scores))
                    if scores[branch] <= angle_threshold:
                        candidates.append((neighbour, branch, scores[branch]))
                else:
                    candidates.append((neighbour, None, 0.0))
            if not candidates and len(directions):
                # In high-dimensional clouds, local PCA directions can be
                # noisy enough that the configured angular gate rejects every
                # edge at a detected junction. Preserve connectivity by using
                # the best available arm assignment rather than allowing the
                # later selector to fall back to disconnected endpoint pairs.
                candidates = [
                    (
                        neighbour,
                        int(np.argmin([
                            _departure_angle(direction, points[neighbour] - specification["center"])
                            for direction in directions
                        ])),
                        0.0,
                    )
                    for neighbour in search_graph.adjacency[source]
                ]
            candidates.sort(key=lambda item: item[2])
            return [(neighbour, branch) for neighbour, branch, _ in candidates]

        def shortest_path(
            start: int,
            target: int,
            blocked_edges: set[tuple[int, int]] | None = None,
            required_branch: int | None = None,
            search_graph: Any = path_graph,
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
            for neighbour, branch in allowed_first_edges(source_spec, search_graph):
                if required_branch is not None and branch != required_branch:
                    continue
                if (
                    component_by_vertex
                    and component_by_vertex.get(source_vertex)
                    != component_by_vertex.get(neighbour)
                ):
                    continue
                if search_graph.key(source_vertex, neighbour) in blocked_edges:
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
                for neighbour in search_graph.adjacency[node]:
                    if neighbour in path:
                        continue
                    if (
                        component_by_vertex
                        and component_by_vertex.get(node)
                        != component_by_vertex.get(neighbour)
                    ):
                        continue
                    if search_graph.key(node, neighbour) in blocked_edges:
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
                        blocked_edges=(
                            getattr(routing_graph, "mst_edges", set())
                            if self.requested_cycle_count_ > 0 else None
                        ),
                        required_branch=required_branch,
                        search_graph=path_graph,
                    )
                    if path is None and self.requested_cycle_count_ > 0:
                        # The MST is a connectivity safeguard.  If the
                        # natural reciprocal substrate cannot realize a
                        # landmark route, allow the safeguard as a
                        # deterministic fallback.
                        path = shortest_path(
                            left,
                            right,
                            required_branch=required_branch,
                            search_graph=routing_graph,
                        )
                    candidate_attempts.append((left, right, required_branch, path is not None))
                    if path is not None:
                        candidates.append(path)
        if hypercube_junctions_detected:
            # The dense kNN graph can jump from one cube edge to another near
            # a noisy corner.  For a verified hypercube, order observations on
            # each Hamming-one edge directly.  This retains the observed edge
            # geometry while keeping the eight corners at degree three.
            cube_candidates: list[CandidatePath] = []
            fixed_tolerance = max(0.35, 12.0 * self.local_scale_)
            for left, right in combinations(
                range(len(specifications)), 2,
            ):
                left_sign = np.sign(specifications[left]["center"])
                right_sign = np.sign(specifications[right]["center"])
                changed = np.flatnonzero(left_sign != right_sign)
                if len(changed) != 1:
                    continue
                axis = int(changed[0])
                fixed = np.ones(points.shape[1], dtype=bool)
                fixed[axis] = False
                displacement = np.abs(
                    points[:, fixed] - specifications[left]["center"][fixed]
                )
                valid = np.all(displacement <= fixed_tolerance, axis=1)
                vertices = np.flatnonzero(valid).astype(int)
                if len(vertices) < 4:
                    continue
                coordinate = points[vertices, axis]
                if specifications[left]["center"][axis] > specifications[right]["center"][axis]:
                    order = np.argsort(-coordinate, kind="mergesort")
                else:
                    order = np.argsort(coordinate, kind="mergesort")
                vertices = vertices[order].tolist()
                source = int(specifications[left]["vertex"])
                target = int(specifications[right]["vertex"])
                vertices = [source] + [vertex for vertex in vertices if vertex not in {source, target}] + [target]
                length = float(sum(
                    np.linalg.norm(points[a] - points[b])
                    for a, b in pairwise(vertices)
                ))
                tangent_cost = float(sum(
                    _tangent_inconsistency(self.local_tangents_[a], self.local_tangents_[b])
                    for a, b in pairwise(vertices)
                ))
                cube_candidates.append(
                    CandidatePath(
                        left,
                        right,
                        vertices,
                        length / max(reference_length, 1e-12) + tangent_cost,
                        length,
                        tangent_cost,
                        branch_start=axis,
                        branch_end=axis,
                    )
                )
            # A d-dimensional hypercube has d * 2**(d - 1) edges.  The
            # previous 3/2 shortcut only works for a three-dimensional cube
            # and silently discarded the specialized candidates for a
            # configured four-dimensional hypercube.
            cube_dimension = int(hypercube_dimension or 0)
            expected_cube_edges = len(specifications) * cube_dimension // 2
            if cube_dimension >= 3 and len(cube_candidates) == expected_cube_edges:
                candidates = cube_candidates

        if loop_branch_endpoint_center is not None:
            # In a loop-with-branch, the terminal is a real branch endpoint,
            # not another point on the cycle.  Allowing the optimizer to
            # connect that endpoint through a cycle anchor can satisfy all
            # degree and cycle-count constraints while assigning one arc of
            # the loop to the open stem.  Keep the endpoint's logical route
            # attached directly to the junction; cycle anchors must then be
            # consumed by the closed route.
            junction_ids = {
                index for index, specification in enumerate(specifications)
                if specification["kind"] == "junction"
            }
            endpoint_ids = {
                index for index, specification in enumerate(specifications)
                if specification["kind"] == "endpoint"
            }
            candidates = [
                candidate for candidate in candidates
                if not (
                    (
                        candidate.start_landmark in endpoint_ids
                        or candidate.end_landmark in endpoint_ids
                    )
                    and not (
                        candidate.start_landmark in junction_ids
                        or candidate.end_landmark in junction_ids
                    )
                )
            ]
        candidates.sort(key=lambda candidate: (candidate.total_cost, candidate.start_landmark, candidate.end_landmark))
        self._tag_persistent_cycle_candidates(candidates)
        from ._stability import score_multiresolution_candidates
        score_multiresolution_candidates(self, candidates, specifications)
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

        cycle_class_count = (
            len(self.persistent_cycles_)
            if self.requested_cycle_count_ > 0
            else 0
        )
        self.mip_candidate_count_ = len(candidates)
        mip_selected, mip_status = select_backbone_mip(
            candidates,
            specifications,
            self.requested_cycle_count_,
            cycle_class_count=cycle_class_count,
        )
        self.mip_status_ = mip_status
        if mip_status == "optimal":
            selected = mip_selected
            degree = np.zeros(len(specifications), dtype=int)
            for candidate in selected.values():
                degree[candidate.start_landmark] += 1
                degree[candidate.end_landmark] += 1

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

    @staticmethod
    def _polyline_length(points: Array) -> float:
        """Return the length of an ordered support polyline."""
        points = np.asarray(points, dtype=float)
        if len(points) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    def _candidate_support_points(
        self,
        graph: _LandmarkGraph,
        candidate: CandidatePath,
        left: int,
        right: int,
        points: Array,
    ) -> Array:
        """Return a candidate path's support in ``left`` to ``right`` order."""
        if candidate.support_points is not None:
            support = np.asarray(candidate.support_points, dtype=float).copy()
            if candidate.start_landmark != left:
                support = support[::-1].copy()
        else:
            vertices = np.asarray(candidate.vertices, dtype=int)
            support = np.asarray(points[vertices], dtype=float).copy()
            if candidate.start_landmark != left:
                support = support[::-1].copy()
        if len(support) == 0:
            support = np.asarray([graph.nodes[left], graph.nodes[right]], dtype=float)
        elif len(support) == 1:
            support = np.vstack((graph.nodes[left], graph.nodes[right]))
        else:
            support[0] = graph.nodes[left]
            support[-1] = graph.nodes[right]
        return support

    @staticmethod
    def _interpolate_polyline(points: Array, distance: float) -> Array:
        """Interpolate one point at arclength ``distance`` on a polyline."""
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            return np.empty(0, dtype=float)
        if len(points) == 1:
            return points[0].copy()
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        distance = float(np.clip(distance, 0.0, cumulative[-1]))
        segment = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment = min(max(segment, 0), len(points) - 2)
        span = cumulative[segment + 1] - cumulative[segment]
        if span <= 1e-12:
            return points[segment].copy()
        fraction = (distance - cumulative[segment]) / span
        return points[segment] + fraction * (points[segment + 1] - points[segment])

    @classmethod
    def _split_polyline(cls, points: Array, parts: int) -> list[Array]:
        """Split a support polyline into equal-arclength pieces."""
        points = np.asarray(points, dtype=float)
        parts = max(1, int(parts))
        if parts == 1:
            return [points.copy()]
        if len(points) < 2:
            points = np.vstack((points, points)) if len(points) else np.zeros((2, 1))
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        total = float(cumulative[-1])
        boundaries = [
            cls._interpolate_polyline(points, total * index / parts)
            for index in range(parts + 1)
        ]
        result: list[Array] = []
        for index in range(parts):
            start_distance = total * index / parts
            end_distance = total * (index + 1) / parts
            interior = points[
                (cumulative > start_distance + 1e-12)
                & (cumulative < end_distance - 1e-12)
            ]
            piece = np.vstack((boundaries[index], interior, boundaries[index + 1]))
            result.append(piece)
        return result

    @staticmethod
    def _support_candidate(
        left: int,
        right: int,
        support: Array,
        template: CandidatePath,
        total_cost: float | None = None,
    ) -> CandidatePath:
        """Create a resized candidate while retaining selection diagnostics."""
        length = SkeletalEmbedding._polyline_length(support)
        return CandidatePath(
            start_landmark=int(left),
            end_landmark=int(right),
            vertices=list(template.vertices),
            total_cost=float(length if total_cost is None else total_cost),
            length=length,
            tangent_cost=float(template.tangent_cost),
            electrical_support=float(template.electrical_support),
            current_support=float(template.current_support),
            persistent_cycle_classes=tuple(template.persistent_cycle_classes),
            stability_support=float(template.stability_support),
            support_points=np.asarray(support, dtype=float).copy(),
            subsample_support=template.subsample_support,
            resolution_support=template.resolution_support,
            geometry_cost=template.geometry_cost,
            descendant_original_indices=template.descendant_original_indices,
        )

    def _contract_backbone_node(
        self,
        graph: _LandmarkGraph,
        paths: dict[tuple[int, int], CandidatePath],
        node: int,
        points: Array,
        preserve_topology: bool,
    ) -> bool:
        """Contract one degree-two node and merge its support paths."""
        neighbours = graph.neighbors(node)
        if len(neighbours) != 2 or neighbours[0] == neighbours[1]:
            return False
        left, right = neighbours
        merged_edge = graph._key(left, right)
        if preserve_topology and merged_edge in graph.edges:
            # Removing this node would collapse two parallel cycle sides into
            # one simple-graph edge and therefore lower the cycle rank.
            return False
        first_edge = graph._key(left, node)
        second_edge = graph._key(node, right)
        first = paths.get(first_edge)
        second = paths.get(second_edge)
        if first is None or second is None:
            return False
        first_support = self._candidate_support_points(
            graph, first, left, node, points
        )
        second_support = self._candidate_support_points(
            graph, second, node, right, points
        )
        merged_support = np.vstack((first_support, second_support[1:]))
        graph.remove_node(node)
        paths.pop(first_edge, None)
        paths.pop(second_edge, None)
        if merged_edge not in graph.edges:
            graph.add_edge(left, right, self._polyline_length(merged_support))
            paths[merged_edge] = self._support_candidate(
                left, right, merged_support, first,
                total_cost=first.total_cost + second.total_cost,
            )
        return True

    def _resize_skeletal_backbone(
        self,
        graph: _LandmarkGraph,
        backbone_paths: dict[tuple[int, int], CandidatePath],
        points: Array,
    ) -> tuple[_LandmarkGraph, dict[tuple[int, int], CandidatePath]]:
        """Resize the selected skeletal graph without rerunning topology selection."""
        target = self.n_backbone_nodes
        spacing = self.backbone_node_spacing
        if target is None and spacing is None:
            self.backbone_node_count_ = len(graph.nodes)
            self.backbone_node_target_ = None
            self.backbone_node_spacing_ = None
            self.backbone_node_minimum_ = len(graph.nodes)
            self.backbone_node_vertex_ids_ = {
                int(node): int(vertex)
                for node, vertex in enumerate(getattr(self, "landmark_vertex_ids_", []))
                if node in graph.nodes
            }
            return graph, backbone_paths

        graph = graph.copy()
        paths = dict(backbone_paths)
        original_vertex_ids = np.asarray(
            getattr(self, "landmark_vertex_ids_", np.full(len(graph.nodes), -1)),
            dtype=int,
        )
        node_vertex_ids = {
            int(node): int(original_vertex_ids[node])
            for node in graph.nodes
            if node < len(original_vertex_ids)
        }

        # First contract degree-two nodes when an exact target is smaller than
        # the selected graph.  Junctions and endpoints are never contracted.
        if target is not None and target < len(graph.nodes):
            preserve_topology = self.backbone_node_policy == "topology_preserving"
            while len(graph.nodes) > target:
                candidates = []
                for node in graph.nodes:
                    if graph.degree(node) != 2:
                        continue
                    neighbours = graph.neighbors(node)
                    if len(neighbours) != 2 or neighbours[0] == neighbours[1]:
                        continue
                    edge = graph._key(neighbours[0], node)
                    other = graph._key(node, neighbours[1])
                    first = paths.get(edge)
                    second = paths.get(other)
                    if first is None or second is None:
                        continue
                    support = self._candidate_support_points(
                        graph, first, neighbours[0], node, points
                    )
                    support = np.vstack((
                        support,
                        self._candidate_support_points(
                            graph, second, node, neighbours[1], points
                        )[1:],
                    ))
                    if preserve_topology and graph._key(*neighbours) in graph.edges:
                        continue
                    candidates.append((self._polyline_length(support), int(node)))
                if not candidates:
                    break
                _, node = min(candidates)
                neighbours = graph.neighbors(node)
                if not self._contract_backbone_node(
                    graph, paths, node, points, preserve_topology
                ):
                    break
                node_vertex_ids.pop(node, None)

            if len(graph.nodes) < target:
                minimum = len(graph.nodes)
                self.backbone_node_minimum_ = minimum
                raise ValueError(
                    f"n_backbone_nodes={target} cannot be reached while preserving "
                    f"the selected backbone; the minimum is {minimum}"
                )

        if target is not None and target > len(graph.nodes):
            if not graph.edges:
                raise ValueError(
                    "n_backbone_nodes cannot exceed the selected backbone node count "
                    "when the backbone has no edges"
                )
            parts = {edge: 1 for edge in graph.edges}
            supports = {
                edge: self._candidate_support_points(
                    graph, paths[edge], edge[0], edge[1], points
                )
                for edge in graph.edges
            }
            lengths = {edge: self._polyline_length(support) for edge, support in supports.items()}
            while len(graph.nodes) + sum(parts.values()) - len(parts) < target:
                edge = max(
                    parts,
                    key=lambda item: (
                        lengths[item] / parts[item],
                        lengths[item],
                        tuple(-value for value in item),
                    ),
                )
                parts[edge] += 1
            target_parts = parts
        elif spacing is not None:
            target_parts = {}
            for edge in graph.edges:
                support = self._candidate_support_points(
                    graph, paths[edge], edge[0], edge[1], points
                )
                target_parts[edge] = max(
                    1, int(np.ceil(self._polyline_length(support) / spacing))
                )
        else:
            target_parts = {edge: 1 for edge in graph.edges}

        if any(parts > 1 for parts in target_parts.values()):
            resized = _LandmarkGraph(graph.nodes)
            next_node = max(graph.nodes, default=-1) + 1
            resized_paths: dict[tuple[int, int], CandidatePath] = {}
            for edge, old_candidate in paths.items():
                if edge not in target_parts:
                    continue
                left, right = edge
                support = self._candidate_support_points(
                    graph, old_candidate, left, right, points
                )
                pieces = self._split_polyline(support, target_parts[edge])
                boundary_nodes = [left]
                for piece in pieces[:-1]:
                    node = next_node
                    next_node += 1
                    boundary_nodes.append(node)
                    resized.nodes[node] = piece[-1].copy()
                    node_vertex_ids[node] = int(
                        np.argmin(np.sum((points - piece[-1]) ** 2, axis=1))
                    )
                boundary_nodes.append(right)
                total_length = max(self._polyline_length(support), 1e-12)
                for index, piece in enumerate(pieces):
                    subedge = resized._key(boundary_nodes[index], boundary_nodes[index + 1])
                    piece_length = self._polyline_length(piece)
                    resized.add_edge(boundary_nodes[index], boundary_nodes[index + 1], piece_length)
                    resized_paths[subedge] = self._support_candidate(
                        boundary_nodes[index],
                        boundary_nodes[index + 1],
                        piece,
                        old_candidate,
                        total_cost=old_candidate.total_cost * piece_length / total_length,
                    )
            graph = resized
            paths = resized_paths

        if target is not None and len(graph.nodes) != target:
            self.backbone_node_minimum_ = len(graph.nodes)
            raise ValueError(
                f"n_backbone_nodes={target} could not be realized; "
                f"the resulting backbone has {len(graph.nodes)} nodes"
            )
        self.backbone_node_count_ = len(graph.nodes)
        self.backbone_node_target_ = target
        self.backbone_node_spacing_ = spacing
        self.backbone_node_minimum_ = len(graph.nodes)
        self.backbone_node_vertex_ids_ = node_vertex_ids
        return graph, paths

    def _tag_persistent_cycle_candidates(self, candidates: list[CandidatePath]) -> None:
        """Associate candidate routes with approximate persistent cycles."""
        cycles = [cycle for cycle in self.persistent_cycles_ if cycle.representative is not None]
        if not cycles or not candidates:
            return
        threshold = 2.5 * max(float(self.local_scale_), 1e-8)
        for cycle_index, cycle in enumerate(cycles):
            representative = np.asarray(cycle.representative, dtype=float)
            scores: list[float] = []
            for candidate in candidates:
                route = self.routing_graph_.points[np.asarray(candidate.vertices, dtype=int)]
                distances = np.min(
                    np.linalg.norm(representative[:, None, :] - route[None, :, :], axis=2),
                    axis=1,
                )
                scores.append(float(np.mean(distances <= threshold)))
            best_score = max(scores)
            if best_score <= 0.0:
                continue
            for candidate, score in zip(candidates, scores):
                if score >= max(0.25, 0.75 * best_score):
                    candidate.persistent_cycle_classes = tuple(sorted({
                        *candidate.persistent_cycle_classes,
                        cycle_index,
                    }))

    def _chain_support_points(self, chain: dict[str, Any], points: Array) -> Array:
        """Concatenate stored point-level paths for one abstract route chain."""
        if chain.get("closed") and getattr(self, "loop_branch_endpoint_center_", None) is not None:
            cycle_support = self._loop_branch_cycle_support_points(points)
            if cycle_support is not None:
                chain["preserve_support_order"] = True
                return cycle_support
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
            if candidate.support_points is not None:
                support = np.asarray(candidate.support_points, dtype=float).copy()
                if candidate.start_landmark != left:
                    support = support[::-1].copy()
            else:
                vertices = list(candidate.vertices)
                forward = candidate.start_landmark == left
                if not forward:
                    vertices.reverse()
                support = np.asarray(
                    [self.routing_graph_.points[index] for index in vertices],
                    dtype=float,
                )
            support[0] = self.landmark_graph_.nodes[left]
            support[-1] = self.landmark_graph_.nodes[right]
            if segments and np.allclose(segments[-1][-1], support[0]):
                support = support[1:]
            segments.append(support)
        if not segments:
            return np.asarray([self.landmark_graph_.nodes[node] for node in nodes], dtype=float)
        return np.vstack(segments)

    def _loop_branch_cycle_support_points(self, points: Array) -> Array | None:
        """Order observations around a loop while excluding its open stem.

        Candidate shortest paths on a noisy kNN graph can share long portions
        of a loop. Concatenating those paths would make the closed support
        sequence visit the same arc twice and forces a spline to bend back on
        itself. The branch corridor is known from the terminal landmark, so
        remove it and order the remaining observations once around their
        principal two-dimensional plane.
        """
        if len(self.junction_regions_) != 1 or len(self.endpoint_regions_) != 1:
            return None
        junction = np.asarray(self.junction_regions_[0].center, dtype=float)
        endpoint = np.asarray(self.loop_branch_endpoint_center_, dtype=float)
        branch = endpoint - junction
        branch_length = float(np.linalg.norm(branch))
        if branch_length <= 1e-12 or len(points) < 8:
            return None
        direction = branch / branch_length
        offsets = np.asarray(points, dtype=float) - junction
        longitudinal = offsets @ direction
        orthogonal = np.linalg.norm(
            offsets - longitudinal[:, None] * direction[None, :], axis=1,
        )
        corridor_width = max(4.0 * self.local_scale_, 0.12 * branch_length)
        stem = (
            (longitudinal >= 0.05 * branch_length)
            & (longitudinal <= 1.15 * branch_length)
            & (orthogonal <= corridor_width)
        )
        cycle_points = np.asarray(points[~stem], dtype=float)
        if len(cycle_points) < 8:
            return None

        centered = cycle_points - np.mean(cycle_points, axis=0, keepdims=True)
        _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
        if len(singular_values) < 2 or singular_values[1] <= 1e-12:
            return None
        plane = centered @ components[:2].T
        angles = np.arctan2(plane[:, 1], plane[:, 0])
        ordered = cycle_points[np.argsort(angles, kind="mergesort")]
        start = int(np.argmin(np.linalg.norm(ordered - junction, axis=1)))
        ordered = np.roll(ordered, -start, axis=0)
        ordered[0] = junction
        return np.vstack([ordered, junction])

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
        node_vertices = getattr(self, "backbone_node_vertex_ids_", None)
        if node_vertices is not None and nodes[0] in node_vertices:
            vertex = int(node_vertices[nodes[0]])
        else:
            landmark_vertices = getattr(self, "landmark_vertex_ids_", None)
            if landmark_vertices is None or nodes[0] >= len(landmark_vertices):
                return None
            vertex = int(landmark_vertices[nodes[0]])
        component_id = self.routing_component_by_vertex_.get(vertex)
        if component_id is None:
            return None
        component = np.asarray(self.routing_components_[component_id], dtype=int)
        if len(component) < 3:
            return None
        if self.selected_backbone_level_ > 0:
            level = self.levels_[self.selected_backbone_level_]
            original_ids = np.unique(np.concatenate([level.descendant_indices[i] for i in component]))
            points = self.levels_[0].points[original_ids]
        else:
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
        if anchor_vertex.vertices:
            anchor_routing_vertex = int(anchor_vertex.vertices[0])
        elif getattr(self, "backbone_node_vertex_ids_", None) is not None:
            anchor_routing_vertex = int(
                self.backbone_node_vertex_ids_.get(anchor, -1)
            )
        else:
            return chain["points"]
        component_id = self.routing_component_by_vertex_.get(anchor_routing_vertex)
        if component_id is None or component_id >= len(self.routing_components_):
            return chain["points"]
        component = self.routing_components_[component_id]
        if len(component) < 8:
            return chain["points"]
        if self.selected_backbone_level_ > 0:
            level = self.levels_[self.selected_backbone_level_]
            component = np.unique(np.concatenate([level.descendant_indices[i] for i in component]))
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

    def _initialize_skeleton_metadata(self) -> None:
        """Expose the typed structural view shared by backbone and ribs."""
        if not hasattr(self, "backbone_element_count_"):
            self.backbone_element_count_ = len(self.routes_)
            self.rib_paths_ = []
            self.rib_support_ = []
            self.rib_stability_ = []
        rib_count = max(0, len(self.routes_) - self.backbone_element_count_)
        self.element_types_ = (
            ["backbone"] * self.backbone_element_count_
            + ["rib"] * rib_count
        )
        self.element_support_ = [1.0] * self.backbone_element_count_ + list(
            getattr(self, "rib_support_", [])
        )
        self.route_support_ = np.asarray(self.element_support_, dtype=float)
        self.element_levels_ = (
            ["backbone"] * self.backbone_element_count_
            + ["major_rib" if support >= 0.75 else "minor_rib"
               for support in getattr(self, "rib_support_", [])]
        )
        for spline, element_type, level in zip(
            self.routes_, self.element_types_, self.element_levels_
        ):
            spline.element_type = element_type
            spline.level = level
        self.backbone_cycle_rank_ = int(self.backbone_graph_.cycle_rank())
        self.rib_graph_ = getattr(self, "rib_graph_", _LandmarkGraph())
        self.skeleton_graph_ = self.backbone_graph_.copy()
        for edge in self.skeleton_graph_.edges:
            self.skeleton_graph_.set_edge_metadata(
                *edge, element_type="backbone", level="backbone", support=1.0
            )
        for node, coordinate in self.rib_graph_.nodes.items():
            if node not in self.skeleton_graph_.nodes:
                self.skeleton_graph_.nodes[node] = coordinate.copy()
        for edge, weight in self.rib_graph_.edges.items():
            self.skeleton_graph_.edges[edge] = weight
            self.skeleton_graph_.set_edge_metadata(
                *edge, element_type="rib", level="rib", support=1.0
            )
        self.skeleton_cycle_rank_ = int(self.skeleton_graph_.cycle_rank())
        self.element_types = list(self.element_types_)
        self.junction_types_ = {
            int(getattr(region, "node_id", index)): "intrinsic"
            for index, region in enumerate(getattr(self, "junctions_", []))
        }
        self.junction_types_.update({
            index: "coverage"
            for index, _ in enumerate(getattr(self, "coverage_intersections_", []), start=len(self.junction_types_))
        })

    def _coverage_tolerance(self, errors: Array) -> float:
        if self.coverage_error_tolerance is not None:
            return float(self.coverage_error_tolerance)
        if self.coverage_relative_tolerance is not None:
            return float(self.coverage_relative_tolerance) * max(self.local_scale_, 1e-12)
        return 3.0 * max(self.local_scale_, 1e-12)

    def _refit_after_ribs(self, original: Array, points: Array) -> None:
        self.normal_frame_grids_ = [
            _fit_normal_frame_grid(spline) for spline in self.routes_
        ]
        self.splines_ = self.routes_
        centerline = self._project_centerline(original)
        fit_residual_pca(self, points, centerline)

    def _coverage_intersections_for(self, candidate: Any, points: Array) -> list[dict[str, Any]]:
        """Validate rib/backbone crossings against the local tangent plane."""
        intersections: list[dict[str, Any]] = []
        backbone_count = int(getattr(self, "backbone_element_count_", len(self.routes_)))
        for element_id, spline in enumerate(self.routes_[:backbone_count]):
            left = np.asarray(spline.samples, dtype=float)
            right = np.asarray(candidate.spline.samples, dtype=float)
            distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
            base_index, rib_index = np.unravel_index(int(np.argmin(distances)), distances.shape)
            distance = float(distances[base_index, rib_index])
            if distance > 3.0 * max(float(self.local_scale_), 1e-8):
                continue
            base_tangent = np.asarray(spline.tangent(spline.t_values[base_index]), dtype=float)
            rib_tangent = np.asarray(candidate.spline.tangent(candidate.spline.t_values[rib_index]), dtype=float)
            if abs(float(base_tangent @ rib_tangent)) > 0.97:
                continue
            center = 0.5 * (left[base_index] + right[rib_index])
            local = points[
                np.linalg.norm(points - center, axis=1)
                <= 4.0 * max(float(self.local_scale_), 1e-8)
            ]
            if len(local) < 5 or points.shape[1] < 2:
                continue
            _, singular_values, components = np.linalg.svd(
                local - np.mean(local, axis=0), full_matrices=False
            )
            if len(singular_values) < 2 or singular_values[1] <= 1e-10:
                continue
            tangent_plane = components[:2].T
            curve_plane = np.column_stack([base_tangent, rib_tangent])
            curve_plane, _ = np.linalg.qr(curve_plane, mode="reduced")
            mismatch = subspace_principal_angle(curve_plane, tangent_plane)
            if mismatch <= np.deg2rad(45.0):
                intersections.append({
                    "element_ids": (element_id, backbone_count + len(self.rib_paths_)),
                    "junction_type": "coverage",
                    "point": center.copy(),
                    "tangent_plane_mismatch": float(mismatch),
                })
        return intersections

    def _refine_coverage(self, original: Array, points: Array) -> None:
        """Iteratively add useful ribs after local residual reconstruction."""
        self.coverage_history_ = []
        self.coverage_iterations_ = 0
        self.rib_candidates_ = []
        self.rib_graph_ = _LandmarkGraph()
        self.coverage_intersections_ = []
        if not self.coverage_refinement:
            return

        for iteration in range(self.coverage_max_iterations):
            centerline = self._project_centerline(original)
            result = attach_residual_pca(self, original, points, centerline)
            errors = np.asarray(result.unexplained_residual_norm, dtype=float)
            tolerance = self._coverage_tolerance(errors)
            quantile_error = float(np.quantile(errors, self.coverage_quantile))
            self.coverage_tolerance_ = tolerance
            self.coverage_history_.append({
                "iteration": iteration,
                "quantile_error": quantile_error,
                "max_error": float(np.max(errors)) if len(errors) else 0.0,
                "tolerance": tolerance,
                "rib_count": len(self.rib_paths_),
            })
            self.coverage_iterations_ = iteration + 1
            if quantile_error <= tolerance:
                break
            if (
                self.coverage_max_ribs is not None
                and len(self.rib_paths_) >= self.coverage_max_ribs
            ):
                break
            candidates = [] if self.rib_seed_source == "hierarchy" else propose_ribs(
                self,
                points,
                result,
                max_candidates=self.coverage_max_candidates_per_iteration,
                candidate_type=self.rib_candidate_type,
            )
            candidates = prepare_rib_candidates(self, points, result, candidates)
            if self.coverage_min_error is not None:
                candidates = [
                    candidate for candidate in candidates
                    if errors[candidate.seed_index] >= self.coverage_min_error
                ]
            self.rib_candidates_.extend(candidates)
            selected = select_ribs(
                candidates,
                max_ribs=(
                    None if self.coverage_max_ribs is None else
                    self.coverage_max_ribs - len(self.rib_paths_)
                ),
                min_gain=self.coverage_min_gain,
                length_penalty=self.coverage_length_penalty,
                rib_penalty=self.coverage_rib_penalty,
                junction_penalty=self.coverage_junction_penalty,
                selection=self.coverage_selection,
                resolution_weight=self.rib_resolution_weight,
                measured_stability_only=len(self.levels_) > 1,
            )
            if not selected:
                break
            for candidate in selected:
                candidate_intersections = self._coverage_intersections_for(candidate, points)
                self.routes_.append(candidate.spline)
                self.route_chains_.append({
                    "nodes": [],
                    "closed": False,
                    "points": candidate.points,
                    "spline_points": candidate.points,
                    "element_type": "rib",
                })
                self.rib_paths_.append(candidate.points.copy())
                self.rib_support_.append(float(candidate.support))
                self.rib_stability_.append(float(candidate.stability))
                self.rib_resolution_support_.append(float(candidate.resolution_support))
                start_id = max(
                    [*self.backbone_graph_.nodes, *self.rib_graph_.nodes, -1]
                ) + 1
                end_id = start_id + 1
                self.rib_graph_.nodes[start_id] = candidate.points[0].copy()
                self.rib_graph_.nodes[end_id] = candidate.points[-1].copy()
                if not np.allclose(candidate.points[0], candidate.points[-1]):
                    self.rib_graph_.add_edge(start_id, end_id, float(
                        np.sum(np.linalg.norm(np.diff(candidate.points, axis=0), axis=1))
                    ))
                    for node, coordinate in self.backbone_graph_.nodes.items():
                        self.rib_graph_.nodes.setdefault(int(node), coordinate.copy())
                    for endpoint_id in (start_id, end_id):
                        nearest = min(
                            self.backbone_graph_.nodes,
                            key=lambda node: np.linalg.norm(
                                self.backbone_graph_.nodes[node]
                                - self.rib_graph_.nodes[endpoint_id]
                            ),
                        )
                        self.rib_graph_.add_edge(
                            endpoint_id,
                            int(nearest),
                            float(np.linalg.norm(
                                self.rib_graph_.nodes[endpoint_id]
                                - self.backbone_graph_.nodes[int(nearest)]
                            )),
                        )
                    self.coverage_intersections_.extend(candidate_intersections)
            self._refit_after_ribs(original, points)
        self._initialize_skeleton_metadata()

    def _fit_final_diagnostics(self, original: Array, points: Array) -> None:
        """Cache the final decomposition in estimator-level fitted fields."""
        self.geometry_fit_n_samples_ = len(original)
        self.geometry_fit_indices_ = np.arange(len(original), dtype=int)
        self.full_data_refit_ = True
        centerline = self._project_centerline(original)
        result = attach_residual_pca(self, original, points, centerline)
        self.centerline_residual_ = np.asarray(result.residual, dtype=float)
        self.centerline_residual_norm_ = np.asarray(result.residual_norm, dtype=float)
        self.residual_coordinates_ = np.asarray(result.residual_coordinates, dtype=float)
        self.reconstruction_ = np.asarray(result.reconstructed, dtype=float)
        self.post_pca_residual_ = np.asarray(result.unexplained_residual, dtype=float)
        self.post_pca_residual_norm_ = np.asarray(result.unexplained_residual_norm, dtype=float)
        self.reconstruction_error_ = float(np.mean(self.post_pca_residual_norm_ ** 2))
        self.stability_summary_ = getattr(self, "stability_summary_", {
            "enabled": False,
            "runs": 0,
            "min_support": self.stability_min_support,
        })
        self.persistent_cycle_count_ = int(getattr(self, "persistent_cycle_count_", 0))
        self.persistent_cycle_count = self.persistent_cycle_count_

    def _run_stability_selection(self, original: Array) -> None:
        """Estimate structural support without averaging fitted geometries.

        Every probe is matched to the full-data model by persistence values,
        landmark corridors, and sampled spline geometry.  This avoids the
        misleading global-count support used by the original prototype.
        """
        junction_count = len(getattr(self, "junctions_", []))
        endpoint_count = len(getattr(self, "endpoints_", []))
        backbone_count = int(getattr(self, "backbone_element_count_", len(self.routes_)))
        self.junction_support_ = np.ones(junction_count, dtype=float)
        self.endpoint_support_ = np.ones(endpoint_count, dtype=float)
        self.route_support_ = np.ones(len(getattr(self, "routes_", [])), dtype=float)
        self.branch_direction_support_ = {}
        self.branch_direction_dispersion_ = {}
        self.junction_consensus_ = []
        self.endpoint_consensus_ = []
        self.route_consensus_ = []
        self.rib_support_ = list(getattr(self, "rib_support_", []))
        self.rib_stability_ = list(getattr(self, "rib_stability_", []))
        self.residual_subspace_stability_ = None
        self.stability_residual_subspaces_ = None
        if not self.stability_selection:
            self.stable_cycle_mask_ = np.ones(len(self.persistent_cycles_), dtype=bool)
            self.stable_junction_mask_ = np.ones(junction_count, dtype=bool)
            self.stable_endpoint_mask_ = np.ones(endpoint_count, dtype=bool)
            self.stable_route_mask_ = np.ones(len(self.routes_), dtype=bool)
            self.stable_rib_mask_ = np.ones(len(self.rib_paths_), dtype=bool)
            self.stability_summary_ = {
                "enabled": False,
                "runs": 0,
                "successful_runs": 0,
                "min_support": self.stability_min_support,
            }
            return

        rng = np.random.default_rng(self.random_state)
        cycle_hits = np.zeros(len(self.persistent_cycles_), dtype=float)
        junction_hits = np.zeros(junction_count, dtype=float)
        endpoint_hits = np.zeros(endpoint_count, dtype=float)
        route_hits = np.zeros(len(self.routes_), dtype=float)
        rib_hits = np.zeros(len(getattr(self, "rib_paths_", [])), dtype=float)
        direction_angles: dict[int, list[float]] = {}
        residual_angles: list[list[float]] = [[] for _ in range(backbone_count)]
        successful_runs = 0
        reference_diagram = np.asarray(
            [[cycle.birth, cycle.death] for cycle in self.persistent_cycles_],
            dtype=float,
        ).reshape(-1, 2)
        reference_junctions = list(getattr(self, "junctions_", []))
        reference_endpoints = list(getattr(self, "endpoints_", []))
        reference_routes = [spline.samples for spline in self.routes_[:backbone_count]]
        reference_ribs = [spline.samples for spline in self.routes_[backbone_count:]]
        run_count = self.stability_runs
        if reference_ribs and self.rib_stability_runs is not None:
            run_count = max(run_count, self.rib_stability_runs)
        run_count = int(run_count)
        tolerance = 8.0 * max(float(self.local_scale_), 1e-8)

        for run in range(run_count):
            indices = subsample_indices(len(original), self.stability_fraction, self.random_state + run + 1)
            sample = jitter_points(
                np.asarray(original[indices], dtype=float),
                jitter=self.stability_jitter,
                local_scale=self.local_scale_,
                rng=rng,
            )
            try:
                probe = SkeletalEmbedding(
                    n_centroids=min(self.n_centroids, len(sample)),
                    n_backbone_nodes=self.n_backbone_nodes,
                    backbone_node_spacing=self.backbone_node_spacing,
                    backbone_node_policy=self.backbone_node_policy,
                    n_neighbors=self.n_neighbors,
                    persistence_threshold=self.persistence_threshold,
                    spline_smoothing=self.spline_smoothing,
                    spline_control_mode=self.spline_control_mode,
                    max_cycles=self.max_cycles,
                    random_state=self.random_state + run + 1,
                    standardize=self.standardize,
                    merge_junction_distance=self.merge_junction_distance,
                    prune_short_branches=self.prune_short_branches,
                    prune_branch_factor=self.prune_branch_factor,
                    persistence_max_points=self.persistence_max_points,
                    spline_samples_per_node=self.spline_samples_per_node,
                    linear_structure_tolerance=self.linear_structure_tolerance,
                    topology_neighbors=self.topology_neighbors,
                    mutual_knn=self.mutual_knn,
                    add_mst=self.add_mst,
                    max_residual_dim=self.max_residual_dim if self.stability_residual_subspaces else 0,
                    residual_pca_bandwidth=self.residual_pca_bandwidth,
                    residual_subspace_smoothness=self.residual_subspace_smoothness,
                    detect_cycles=self.detect_cycles,
                    detect_junctions=self.detect_junctions,
                    junction_scales=self.junction_scales,
                    junction_inner_fraction=self.junction_inner_fraction,
                    junction_confidence=self.junction_confidence,
                    use_local_pca=self.use_local_pca,
                    local_pca_neighbors=self.local_pca_neighbors,
                    max_branch_angle_degrees=self.max_branch_angle_degrees,
                    use_effective_resistance=self.use_effective_resistance,
                    use_electrical_flow=self.use_electrical_flow,
                    use_kron_reduction=self.use_kron_reduction,
                    routing_length_weight=self.routing_length_weight,
                    routing_tangent_weight=self.routing_tangent_weight,
                    routing_density_weight=self.routing_density_weight,
                    routing_resistance_weight=self.routing_resistance_weight,
                    routing_current_weight=self.routing_current_weight,
                    use_tangent_boundary_conditions=self.use_tangent_boundary_conditions,
                    coverage_refinement=bool(self.coverage_refinement and len(reference_ribs)),
                    coverage_error_tolerance=self.coverage_error_tolerance,
                    coverage_relative_tolerance=self.coverage_relative_tolerance,
                    coverage_quantile=self.coverage_quantile,
                    coverage_max_iterations=self.coverage_max_iterations,
                    coverage_max_ribs=self.coverage_max_ribs,
                    coverage_max_candidates_per_iteration=self.coverage_max_candidates_per_iteration,
                    coverage_candidate_spacing=self.coverage_candidate_spacing,
                    coverage_min_error=self.coverage_min_error,
                    coverage_min_gain=self.coverage_min_gain,
                    coverage_length_penalty=self.coverage_length_penalty,
                    coverage_rib_penalty=self.coverage_rib_penalty,
                    coverage_junction_penalty=self.coverage_junction_penalty,
                    coverage_selection=self.coverage_selection,
                    rib_candidate_type=self.rib_candidate_type,
                    use_multiresolution=self.use_multiresolution,
                    hierarchy_max_levels=self.hierarchy_max_levels,
                    hierarchy_target_size=self.hierarchy_target_size,
                    hierarchy_min_reduction=self.hierarchy_min_reduction,
                    representative_method=self.representative_method,
                    hierarchy_distance_quantile=self.hierarchy_distance_quantile,
                    hierarchy_local_neighbors=self.hierarchy_local_neighbors,
                    backbone_max_representatives=self.backbone_max_representatives,
                    backbone_consensus_levels=self.backbone_consensus_levels,
                    route_resolution_weight=self.route_resolution_weight,
                    rib_resolution_weight=self.rib_resolution_weight,
                    rib_seed_source=self.rib_seed_source,
                    backbone_level="auto",
                    stability_selection=False,
                ).fit(sample)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue

            successful_runs += 1
            probe_diagram = np.asarray(
                [[cycle.birth, cycle.death] for cycle in getattr(probe, "persistent_cycles_", [])],
                dtype=float,
            ).reshape(-1, 2)
            cycle_hits += match_cycles(
                reference_diagram,
                probe_diagram,
                tolerance=1.0,
            )
            junction_match = match_regions(
                reference_junctions,
                list(getattr(probe, "junctions_", [])),
                tolerance=tolerance,
                require_branch_count=True,
            )
            endpoint_match = match_regions(
                reference_endpoints,
                list(getattr(probe, "endpoints_", [])),
                tolerance=tolerance,
            )
            junction_hits += junction_match
            endpoint_hits += endpoint_match
            probe_backbone_count = int(getattr(probe, "backbone_element_count_", len(probe.routes_)))
            route_match = match_routes(
                reference_routes,
                [spline.samples for spline in probe.routes_[:probe_backbone_count]],
                tolerance=tolerance,
            )
            route_hits[:len(route_match)] += route_match
            if reference_ribs:
                rib_match = match_routes(
                    reference_ribs,
                    [spline.samples for spline in probe.routes_[probe_backbone_count:]],
                    tolerance=tolerance,
                )
                rib_hits[:len(rib_match)] += rib_match

            for reference_index, matched in enumerate(junction_match):
                if not matched or reference_index >= len(reference_junctions):
                    continue
                reference = reference_junctions[reference_index]
                probe_regions = list(getattr(probe, "junctions_", []))
                if reference_index >= len(probe_regions):
                    continue
                left = np.asarray(self.junction_branch_directions_.get(reference.node_id, []), dtype=float)
                right = np.asarray(probe.junction_branch_directions_.get(probe_regions[reference_index].node_id, []), dtype=float)
                if len(left) and len(right):
                    angles = np.arccos(np.clip(np.abs(left @ right.T), -1.0, 1.0))
                    direction_angles[reference_index] = direction_angles.get(reference_index, []) + [float(np.min(angles, axis=1).mean())]

            if self.stability_residual_subspaces and self.residual_dim_:
                for route, matched in enumerate(route_match):
                    if not matched or route >= len(probe.residual_bases_):
                        continue
                    reference_basis = self.residual_bases_[route]
                    candidate_basis = probe.residual_bases_[route]
                    sample_count = min(len(reference_basis), len(candidate_basis))
                    if sample_count:
                        positions = np.linspace(0, sample_count - 1, min(16, sample_count)).astype(int)
                        residual_angles[route].extend(
                            subspace_principal_angle(reference_basis[index], candidate_basis[index])
                            for index in positions
                        )

        denominator = max(successful_runs, 1)
        self.cycle_support_ = cycle_hits / denominator
        self.junction_support_ = junction_hits / denominator
        self.endpoint_support_ = endpoint_hits / denominator
        self.route_support_ = route_hits / denominator
        self.rib_support_ = (rib_hits / denominator).tolist()
        self.rib_stability_ = list(self.rib_support_)
        self.stable_cycle_mask_ = self.cycle_support_ >= self.stability_min_support
        self.stable_junction_mask_ = self.junction_support_ >= self.stability_min_support
        self.stable_endpoint_mask_ = self.endpoint_support_ >= self.stability_min_support
        self.stable_route_mask_ = self.route_support_ >= self.stability_min_support
        self.stable_rib_mask_ = np.asarray(self.rib_support_) >= self.rib_min_support
        for cycle, support in zip(self.persistent_cycles_, self.cycle_support_):
            cycle.stability_support = float(support)
        self.consensus_cycle_count_ = int(np.sum(self.cycle_support_ >= self.stability_min_support))
        self.consensus_junction_count_ = int(np.sum(self.junction_support_ >= self.stability_min_support))
        self.junction_consensus_ = [
            {"center": region.center.copy(), "branch_count": region.branch_count, "support": float(support)}
            for region, support in zip(reference_junctions, self.junction_support_)
        ]
        self.endpoint_consensus_ = [
            {"center": region.center.copy(), "support": float(support)}
            for region, support in zip(reference_endpoints, self.endpoint_support_)
        ]
        self.route_consensus_ = [
            {"route": index, "support": float(support)}
            for index, support in enumerate(self.route_support_)
        ]
        self.branch_direction_support_ = {
            index: float(np.mean(np.asarray(values) <= np.deg2rad(self.max_branch_angle_degrees)))
            for index, values in direction_angles.items()
        }
        self.branch_direction_dispersion_ = {
            index: float(np.std(values)) for index, values in direction_angles.items()
        }
        if self.stability_residual_subspaces and self.residual_dim_:
            self.residual_subspace_stability_ = {
                route: {
                    "mean_principal_angle": float(np.mean(values)) if values else float("inf"),
                    "dispersion": float(np.std(values)) if values else float("inf"),
                    "support": float(len(values) / max(successful_runs, 1)),
                }
                for route, values in enumerate(residual_angles)
            }
        else:
            self.residual_subspace_stability_ = None
        self.stability_residual_subspaces_ = self.residual_subspace_stability_
        self.stability_summary_ = {
            "enabled": True,
            "runs": run_count,
            "successful_runs": successful_runs,
            "min_support": self.stability_min_support,
            "cycle_support": self.cycle_support_.copy(),
            "junction_support": self.junction_support_.copy(),
            "endpoint_support": self.endpoint_support_.copy(),
            "route_support": self.route_support_.copy(),
            "rib_support": np.asarray(self.rib_support_, dtype=float),
            "consensus_cycle_count": self.consensus_cycle_count_,
            "consensus_junction_count": self.consensus_junction_count_,
        }
        if reference_ribs and self.rib_stability_runs is not None:
            keep = np.asarray(self.rib_support_, dtype=float) >= self.rib_min_support
            if not np.all(keep):
                kept_indices = [index for index, selected in enumerate(keep) if selected]
                self.routes_ = self.routes_[:backbone_count] + [
                    self.routes_[backbone_count + index] for index in kept_indices
                ]
                self.route_chains_ = self.route_chains_[:backbone_count] + [
                    self.route_chains_[backbone_count + index] for index in kept_indices
                ]
                self.rib_paths_ = [self.rib_paths_[index] for index in kept_indices]
                self.rib_support_ = [self.rib_support_[index] for index in kept_indices]
                self.rib_stability_ = [self.rib_stability_[index] for index in kept_indices]
                self.rib_resolution_support_ = [self.rib_resolution_support_[index] for index in kept_indices]
                self.rib_graph_ = _LandmarkGraph()
                self.coverage_intersections_ = []
                standardized = (original - self.mean_) / self.scale_
                self._refit_after_ribs(original, standardized)
                self._initialize_skeleton_metadata()

    def _project_centerline(self, X: Array | Sequence[Sequence[float]]) -> EmbeddingResult:
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

    def transform(self, X: Array | Sequence[Sequence[float]]) -> EmbeddingResult:
        if not self._fitted:
            raise RuntimeError("Call fit before transform")
        original = _as_point_cloud(X)
        if original.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        points = (original - self.mean_) / self.scale_
        centerline_result = self._project_centerline(original)
        return attach_residual_pca(self, original, points, centerline_result)

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

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for scikit-learn compatibility."""
        parameters = inspect.signature(self.__init__).parameters
        return {
            name: getattr(self, name)
            for name in parameters
            if name != "self" and hasattr(self, name)
        }

    def set_params(self, **params: Any) -> SkeletalEmbedding:
        """Set constructor parameters using the scikit-learn convention."""
        valid = self.get_params()
        unknown = sorted(set(params) - set(valid))
        if unknown:
            raise ValueError(f"Invalid parameter(s): {', '.join(unknown)}")
        for name, value in params.items():
            setattr(self, name, value)
        return self


__all__ = ["SkeletalEmbedding"]
