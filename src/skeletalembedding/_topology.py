"""Landmark topology construction and persistent-homology utilities."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise, product
from typing import Any

import numpy as np

Array = np.ndarray


@dataclass
class JunctionRegion:
    """Spatially clustered local-topology evidence for one junction."""

    center: Array
    branch_count: int
    confidence: float
    member_indices: list[int] = field(default_factory=list)
    arm_indices: list[np.ndarray] = field(default_factory=list)
    node_id: int | None = None


@dataclass
class EndpointRegion:
    """Spatially clustered local-topology evidence for one endpoint."""

    center: Array
    confidence: float
    member_indices: list[int] = field(default_factory=list)
    node_id: int | None = None


# Public terminology aliases used by the structural diagnostics API.
Junction = JunctionRegion
Endpoint = EndpointRegion


@dataclass
class TopologyEstimate:
    """Diagnostics produced by the topology initialization stage."""

    cycle_count: int
    persistence_diagram: Array
    normalized_persistence_diagram: Array
    junction_regions: list[JunctionRegion]
    endpoint_regions: list[EndpointRegion]
    branch_counts: Array
    topology_confidence: Array


@dataclass
class PersistentCycle:
    """A persistent H1 feature retained as a backbone constraint."""

    birth: float
    death: float
    persistence: float
    representative: Array | None = None
    stability_support: float = 1.0


@dataclass
class CandidatePath:
    """A route through the dense graph between two logical landmarks."""

    start_landmark: int
    end_landmark: int
    vertices: list[int]
    total_cost: float
    length: float
    tangent_cost: float
    electrical_support: float = 0.0
    current_support: float = 0.0
    branch_start: int | None = None
    branch_end: int | None = None
    persistent_cycle_classes: tuple[int, ...] = ()
    stability_support: float = 1.0


class _WeightedKNNGraph:
    """Weighted symmetrized kNN graph used as a routing substrate."""

    def __init__(self, points: Array) -> None:
        self.points = np.asarray(points, dtype=float)
        self.edges: dict[tuple[int, int], float] = {}
        self.conductances: dict[tuple[int, int], float] = {}
        self.adjacency: dict[int, list[int]] = {index: [] for index in range(len(points))}
        self.edge_density: dict[tuple[int, int], float] = {}

    @staticmethod
    def key(left: int, right: int) -> tuple[int, int]:
        return (left, right) if left < right else (right, left)

    def add_edge(self, left: int, right: int, length: float, conductance: float) -> None:
        if left == right:
            return
        edge = self.key(int(left), int(right))
        if edge in self.edges:
            # Keep the strongest local connection if a symmetrized neighbour
            # query contributes the same edge twice.
            if conductance <= self.conductances[edge]:
                return
            self.adjacency[edge[0]].remove(edge[1])
            self.adjacency[edge[1]].remove(edge[0])
        self.edges[edge] = float(length)
        self.conductances[edge] = max(float(conductance), 1e-12)
        self.adjacency[edge[0]].append(edge[1])
        self.adjacency[edge[1]].append(edge[0])

    def connected_components(self) -> list[list[int]]:
        remaining = set(self.adjacency)
        components: list[list[int]] = []
        while remaining:
            root = min(remaining)
            remaining.remove(root)
            stack = [root]
            component = [root]
            while stack:
                node = stack.pop()
                for neighbour in self.adjacency[node]:
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        stack.append(neighbour)
                        component.append(neighbour)
            components.append(component)
        return components


def _as_point_cloud(X: Array | Sequence[Sequence[float]]) -> Array:
    points = np.asarray(X, dtype=float)
    if points.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_samples, n_features)")
    if points.shape[0] < 1:
        raise ValueError("X must contain at least one observation")
    if points.shape[1] < 1:
        raise ValueError("X must contain at least one feature")
    if not np.all(np.isfinite(points)):
        raise ValueError("X contains NaN or infinite values")
    return points


def _pairwise_distances(X: Array) -> Array:
    differences = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=2))


def _local_scale(X: Array) -> float:
    # Work from Gram blocks instead of materializing an ``n × n × d`` array.
    # The latter is needlessly expensive for single-cell matrices with many
    # genes and can consume several gigabytes before the nearest-neighbor
    # distances are reduced.
    squared_norms = np.sum(X * X, axis=1)
    nearest_squared = np.full(len(X), np.inf)
    block_size = 256
    for start in range(0, len(X), block_size):
        stop = min(start + block_size, len(X))
        distances_squared = (
            squared_norms[start:stop, None]
            + squared_norms[None, :]
            - 2.0 * (X[start:stop] @ X.T)
        )
        distances_squared = np.maximum(distances_squared, 0.0)
        distances_squared[distances_squared <= 1e-24] = np.inf
        row_indices = np.arange(stop - start)
        distances_squared[row_indices, start + row_indices] = np.inf
        nearest_squared[start:stop] = np.min(distances_squared, axis=1)
    nearest = np.sqrt(nearest_squared)
    finite = nearest[np.isfinite(nearest)]
    if not len(finite):
        # A completely duplicated point cloud has no non-zero nearest-neighbor
        # scale.  Keep the standardized metric well-defined without emitting
        # a misleading ``median of empty slice`` warning.
        return 1e-8
    scale = float(np.median(finite))
    return max(scale, 1e-8)


def _standardize(X: Array) -> tuple[Array, Array, Array]:
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale[scale < 1e-12] = 1.0
    return (X - mean) / scale, mean, scale


def _euclidean_mst_edges(points: Array) -> tuple[Array, Array]:
    """Return the exact Euclidean MST using a memory-linear Prim pass."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=float)

    selected = np.zeros(len(points), dtype=bool)
    best_squared = np.full(len(points), np.inf, dtype=float)
    parent = np.full(len(points), -1, dtype=int)
    best_squared[0] = 0.0
    edges: list[tuple[int, int, float]] = []

    for _ in range(len(points)):
        available = np.flatnonzero(~selected)
        current = int(available[np.argmin(best_squared[available])])
        selected[current] = True
        if parent[current] >= 0:
            edges.append((int(parent[current]), current, np.sqrt(best_squared[current])))

        remaining = np.flatnonzero(~selected)
        if not len(remaining):
            continue
        differences = points[remaining] - points[current]
        distances_squared = np.sum(differences * differences, axis=1)
        improved = distances_squared < best_squared[remaining]
        improved_indices = remaining[improved]
        best_squared[improved_indices] = distances_squared[improved]
        parent[improved_indices] = current

    edge_array = np.asarray([(left, right) for left, right, _ in edges], dtype=int)
    lengths = np.asarray([length for _, _, length in edges], dtype=float)
    return edge_array, lengths


def _weighted_symmetric_knn_graph(
    X: Array,
    neighbors: int,
    mutual_knn: bool = False,
    add_mst: bool = False,
) -> tuple[_WeightedKNNGraph, Array]:
    """Build a weighted kNN graph, optionally augmented with its Euclidean MST.

    When ``mutual_knn`` is true, an edge is retained only when both endpoints
    select each other among their local neighbors.  When ``add_mst`` is true,
    the exact Euclidean minimum spanning tree is added before natural
    components are recorded.  Disconnected-component bridges are still added
    after the natural graph is recorded so optional electrical diagnostics
    retain a connected substrate.

    Euclidean lengths are retained for geometry while Gaussian affinities are
    retained as conductances.  A minimum-distance bridge is added between
    disconnected kNN components so shortest paths and electrical quantities
    have a well-defined connected substrate.
    """
    points = np.asarray(X, dtype=float)
    graph = _WeightedKNNGraph(points)
    graph.mst_edges = set()
    if len(points) < 2:
        graph.local_scales = np.ones(len(points), dtype=float)
        return graph, graph.local_scales
    count = min(max(1, int(neighbors)), len(points) - 1)
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(points).query(points, k=count + 1)
        distances = np.asarray(distances, dtype=float)
        indices = np.asarray(indices, dtype=int)
        local_scales = np.maximum(distances[:, -1], 1e-8)
    except (ImportError, TypeError, ValueError):  # pragma: no cover - SciPy is core in normal use.
        distances_all = _pairwise_distances(points)
        order = np.argsort(distances_all, axis=1, kind="mergesort")[:, 1:count + 1]
        distances = np.take_along_axis(distances_all, order, axis=1)
        indices = np.column_stack([np.arange(len(points)), order])
        distances = np.column_stack([np.zeros(len(points)), distances])
        local_scales = np.maximum(distances[:, -1], 1e-8)

    neighbor_sets = None
    if mutual_knn:
        neighbor_sets = [
            {
                int(target)
                for target in row
                if int(target) != source
            }
            for source, row in enumerate(indices)
        ]

    union_graph = _WeightedKNNGraph(points)
    for source in range(len(points)):
        for column in range(1, count + 1):
            target = int(indices[source, column])
            distance = float(distances[source, column])
            denominator = max(local_scales[source] * local_scales[target], 1e-12)
            conductance = float(np.exp(-(distance * distance) / denominator))
            union_graph.add_edge(source, target, distance, conductance)
            if not mutual_knn or source in neighbor_sets[target]:
                graph.add_edge(source, target, distance, conductance)

    natural_graph = _WeightedKNNGraph(points)
    for edge, length in graph.edges.items():
        natural_graph.add_edge(
            edge[0], edge[1], length, graph.conductances[edge],
        )
    graph.natural_graph = natural_graph
    graph.topology_graph = union_graph

    if add_mst:
        mst_edges, mst_lengths = _euclidean_mst_edges(points)
        for (left, right), distance in zip(mst_edges, mst_lengths):
            edge = graph.key(int(left), int(right))
            was_present = edge in graph.edges
            denominator = max(local_scales[left] * local_scales[right], 1e-12)
            conductance = float(np.exp(-(distance * distance) / denominator))
            graph.add_edge(int(left), int(right), float(distance), conductance)
            if not was_present:
                graph.mst_edges.add(edge)

    components = graph.connected_components()
    # ``original_components`` describes the selected graph and is retained for
    # low-level diagnostics.  Topology-aware routing uses the union-neighbour
    # components so reciprocal filtering cannot fragment a genuine branch.
    graph.original_components = [component.copy() for component in components]
    graph.topology_components = [
        component.copy() for component in union_graph.connected_components()
    ]
    while len(components) > 1:
        left_component = components[0]
        best: tuple[float, int, int] | None = None
        for left in left_component:
            for component in components[1:]:
                for right in component:
                    distance = float(np.linalg.norm(points[left] - points[right]))
                    candidate = (distance, left, right)
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            break
        distance, left, right = best
        denominator = max(local_scales[left] * local_scales[right], 1e-12)
        graph.add_edge(left, right, distance, float(np.exp(-(distance * distance) / denominator)))
        components = graph.connected_components()

    positive_scales = local_scales[local_scales > 1e-12]
    reference = float(np.median(positive_scales)) if len(positive_scales) else 1.0
    for edge, conductance in graph.conductances.items():
        graph.edge_density[edge] = float(np.clip(conductance / max(reference, 1e-12), 0.0, 1.0))
    graph.local_scales = local_scales
    return graph, local_scales


def _induced_annulus_components(
    graph: _WeightedKNNGraph,
    center: Array,
    radius: float,
    inner_radius_fraction: float,
    max_edge_length: float | None = None,
) -> list[np.ndarray]:
    """Return connected components in a graph-induced annulus."""
    distances = np.linalg.norm(graph.points - np.asarray(center), axis=1)
    selected = np.flatnonzero(
        (distances <= max(radius, 1e-8))
        & (distances >= inner_radius_fraction * max(radius, 1e-8))
    )
    if len(selected) == 0:
        return []
    selected_set = {int(index) for index in selected}
    remaining = set(selected_set)
    components: list[np.ndarray] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack = [root]
        component = [root]
        while stack:
            node = stack.pop()
            for neighbour in graph.adjacency[node]:
                edge = graph.key(node, neighbour)
                if (
                    neighbour in remaining
                    and neighbour in selected_set
                    and (max_edge_length is None or graph.edges[edge] <= max_edge_length)
                ):
                    remaining.remove(neighbour)
                    stack.append(neighbour)
                    component.append(neighbour)
        components.append(np.asarray(sorted(component), dtype=int))
    return components


def _cluster_region_candidates(
    points: Array,
    candidates: list[tuple[int, int, float, list[np.ndarray]]],
    merge_distance: float,
    junction: bool,
) -> list[JunctionRegion | EndpointRegion]:
    """Cluster nearby stable local-topology candidates into regions."""
    if not candidates:
        return []
    remaining = set(range(len(candidates)))
    regions: list[JunctionRegion | EndpointRegion] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        group = [root]
        changed = True
        while changed:
            changed = False
            for index in list(remaining):
                if any(
                    np.linalg.norm(points[candidates[index][0]] - points[candidates[member][0]])
                    <= merge_distance
                    for member in group
                ):
                    remaining.remove(index)
                    group.append(index)
                    changed = True
        weights = np.asarray([max(candidates[index][2], 1e-8) for index in group])
        centers = np.asarray([points[candidates[index][0]] for index in group])
        center = np.average(centers, axis=0, weights=weights)
        members = [candidates[index][0] for index in group]
        confidence = float(np.average(weights, weights=weights))
        if junction:
            counts = [candidates[index][1] for index in group]
            branch_count = round(float(np.median(counts)))
            arm_indices = max(
                (candidates[index][3] for index in group),
                key=lambda arms: (len(arms), -abs(len(arms) - branch_count)),
            )
            regions.append(JunctionRegion(center, branch_count, confidence, members, arm_indices))
        else:
            regions.append(EndpointRegion(center, confidence, members))
    return regions


def _estimate_local_topology(
    X: Array,
    graph: _WeightedKNNGraph,
    candidate_centers: Array,
    *,
    local_scales: Sequence[float] | None = None,
    inner_radius_fraction: float = 0.25,
    min_junction_confidence: float = 0.7,
    scales: int | Sequence[float] = 6,
    merge_distance: float | None = None,
) -> tuple[list[JunctionRegion], list[EndpointRegion], Array, Array]:
    """Estimate stable branch counts around prototype candidates."""
    if isinstance(scales, int):
        scale_count = max(2, int(scales))
        scale_multipliers = np.linspace(0.85, 2.2, scale_count)
    else:
        scale_multipliers = np.asarray(list(scales), dtype=float)
        scale_multipliers = scale_multipliers[np.isfinite(scale_multipliers) & (scale_multipliers > 0)]
        if len(scale_multipliers) < 2:
            scale_multipliers = np.linspace(0.85, 2.2, 6)
    if local_scales is None:
        local_scales = graph.local_scales
    local_scales = np.asarray(local_scales, dtype=float)
    global_scale = _local_scale(X)
    candidate_indices = np.asarray([
        int(np.argmin(np.sum((X - center) ** 2, axis=1))) for center in candidate_centers
    ])
    merge_distance = float(merge_distance) if merge_distance is not None else 2.5 * global_scale
    junction_candidates: list[tuple[int, int, float, list[np.ndarray]]] = []
    endpoint_candidates: list[tuple[int, int, float, list[np.ndarray]]] = []
    branch_counts = np.zeros(len(candidate_indices), dtype=int)
    confidences = np.zeros(len(candidate_indices), dtype=float)
    for row, point_index in enumerate(candidate_indices):
        base_radius = max(float(local_scales[point_index]), global_scale)
        component_sets = [
            _induced_annulus_components(
                graph,
                X[point_index],
                base_radius * float(multiplier),
                inner_radius_fraction,
                # The annulus is sampled from a noisy cloud.  A short edge
                # cutoff makes ordinary curved strands look disconnected and
                # turns gaps in the annulus into false branches.  Keep the
                # cutoff below the separation of genuinely distinct arms,
                # while allowing neighbouring samples on one strand to join.
                max_edge_length=0.75 * base_radius * float(multiplier),
            )
            for multiplier in scale_multipliers
        ]
        counts = np.asarray([len(components) for components in component_sets], dtype=int)
        counts[counts == 0] = 1
        values, frequencies = np.unique(counts, return_counts=True)
        mode_index = int(np.argmax(frequencies))
        branch_count = int(values[mode_index])
        confidence = float(frequencies[mode_index] / len(counts))
        branch_counts[row] = branch_count
        confidences[row] = confidence
        stable_arms = component_sets[mode_index]
        entry = (int(point_index), branch_count, confidence, stable_arms)
        if confidence < min_junction_confidence:
            continue
        if branch_count >= 3:
            junction_candidates.append(entry)
        elif branch_count == 1:
            endpoint_candidates.append(entry)
    junctions = _cluster_region_candidates(X, junction_candidates, merge_distance, True)
    endpoints = _cluster_region_candidates(X, endpoint_candidates, merge_distance, False)
    return (
        [region for region in junctions if isinstance(region, JunctionRegion)],
        [region for region in endpoints if isinstance(region, EndpointRegion)],
        branch_counts,
        confidences,
    )


def _hypercube_junction_regions(
    X: Array,
    local_scale: float,
) -> tuple[list[JunctionRegion], int, int | None]:
    """Detect cube-like corners from signed coordinate extremes.

    A sampled hypercube is not well served by a centroid MST: removing the
    cycles from the cube leaves an arbitrary tree and therefore hides some of
    the true corner vertices.  In standardized coordinates, every corner is
    characterized by one of the ``2**d`` sign patterns and each incident arm
    changes exactly one coordinate.  Require all sign patterns and all arms to
    have substantial support before using this specialized geometric
    diagnostic; ordinary clouds do not pass that joint test.

    Returns the junction regions, the number of square faces, and the detected
    dimension.  The face count is intentionally separate from H1 cycle rank:
    a 3D cube has six faces but only five independent cycles.
    """
    points = np.asarray(X, dtype=float)
    dimension = points.shape[1]
    if dimension < 3 or dimension > 5 or len(points) < 4 * (2**dimension):
        return [], 0, None
    if not np.all(np.isfinite(points)):
        return [], 0, None

    scale = max(float(local_scale), 1e-8)
    sign_patterns = [np.asarray(signs, dtype=float) for signs in product((-1.0, 1.0), repeat=dimension)]
    corners: list[tuple[np.ndarray, int, np.ndarray]] = []
    corner_scores: list[float] = []
    for signs in sign_patterns:
        scores = np.min(points * signs[None, :], axis=1)
        vertex = int(np.argmax(scores))
        score = float(scores[vertex])
        # Standardized noisy cube corners are normally well beyond 0.8 in
        # every coordinate.  The lower bound leaves room for moderate noise
        # while rejecting sign patterns supported only by an interior cloud.
        if score < 0.65:
            return [], 0, None
        corners.append((points[vertex].copy(), vertex, signs))
        corner_scores.append(score)

    corner_points = np.asarray([corner[0] for corner in corners])
    corner_distances = np.linalg.norm(
        corner_points[:, None, :] - corner_points[None, :, :], axis=2
    )
    corner_distances[np.eye(len(corner_points), dtype=bool)] = np.inf
    if np.min(corner_distances) <= 0.4:
        return [], 0, None
    magnitudes = np.abs(corner_points)
    if np.max(np.std(magnitudes, axis=0) / np.maximum(np.mean(magnitudes, axis=0), 1e-8)) > 0.35:
        return [], 0, None

    regions: list[JunctionRegion] = []
    arm_radius = max(0.28, 12.0 * scale)
    for center, vertex, signs in corners:
        arms: list[np.ndarray] = []
        for axis in range(dimension):
            direction = np.zeros(dimension, dtype=float)
            direction[axis] = -signs[axis]
            displacement = points - center
            projection = displacement @ direction
            orthogonal = np.linalg.norm(
                displacement - projection[:, None] * direction[None, :],
                axis=1,
            )
            valid = np.flatnonzero(
                (projection >= max(2.0 * scale, 0.03))
                & (projection <= 1.9)
                & (orthogonal <= arm_radius)
            )
            if len(valid) < 3:
                return [], 0, None
            order = np.argsort(
                orthogonal[valid] + 0.03 * np.abs(projection[valid] - 0.65),
                kind="mergesort",
            )
            arms.append(valid[order[: min(len(order), 32)].astype(int)])
        regions.append(
            JunctionRegion(
                center=center,
                branch_count=dimension,
                confidence=float(np.clip(np.mean(corner_scores), 0.0, 1.0)),
                member_indices=[vertex],
                arm_indices=arms,
            )
        )
    face_count = (dimension * (dimension - 1) * 2 ** (dimension - 2)) // 2
    return regions, int(face_count), dimension


class _LandmarkGraph:
    """A tiny undirected weighted graph used by the prototype.

    Nodes are integer IDs and store coordinate arrays.  Edges are keyed by
    sorted endpoint tuples and store Euclidean lengths.
    """

    def __init__(self, nodes: dict[int, Array] | None = None) -> None:
        self.nodes: dict[int, Array] = {
            int(key): np.asarray(value, dtype=float).copy()
            for key, value in (nodes or {}).items()
        }
        self.edges: dict[tuple[int, int], float] = {}
        self.edge_metadata: dict[tuple[int, int], dict[str, Any]] = {}

    @staticmethod
    def _key(u: int, v: int) -> tuple[int, int]:
        if u == v:
            raise ValueError("Self-loops are not supported")
        return (u, v) if u < v else (v, u)

    def copy(self) -> _LandmarkGraph:
        result = _LandmarkGraph(self.nodes)
        result.edges = self.edges.copy()
        result.edge_metadata = {
            edge: dict(metadata) for edge, metadata in self.edge_metadata.items()
        }
        return result

    def add_edge(self, u: int, v: int, weight: float | None = None) -> None:
        if u not in self.nodes or v not in self.nodes:
            raise KeyError("Both edge endpoints must already be graph nodes")
        key = self._key(u, v)
        if weight is None:
            weight = float(np.linalg.norm(self.nodes[u] - self.nodes[v]))
        self.edges[key] = float(weight)

    def set_edge_metadata(self, u: int, v: int, **metadata: Any) -> None:
        """Attach structural metadata to an existing edge."""
        self.edge_metadata[self._key(u, v)] = dict(metadata)

    def remove_edge(self, u: int, v: int) -> None:
        self.edges.pop(self._key(u, v), None)

    def remove_node(self, node: int) -> None:
        self.nodes.pop(node)
        for edge in list(self.edges):
            if node in edge:
                del self.edges[edge]

    def neighbors(self, node: int) -> list[int]:
        result = []
        for u, v in self.edges:
            if u == node:
                result.append(v)
            elif v == node:
                result.append(u)
        return result

    def degree(self, node: int) -> int:
        return len(self.neighbors(node))

    def number_of_edges(self) -> int:
        return len(self.edges)

    def cycle_rank(self) -> int:
        if not self.nodes:
            return 0
        components = self._components()
        return self.number_of_edges() - len(self.nodes) + len(components)

    def _components(self) -> list[list[int]]:
        remaining = set(self.nodes)
        components = []
        while remaining:
            root = min(remaining)
            stack = [root]
            remaining.remove(root)
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbor in self.neighbors(node):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
            components.append(component)
        return components

    def shortest_path(self, source: int, target: int) -> tuple[float, list[int]]:
        """Return weighted shortest-path distance and node sequence."""
        if source == target:
            return 0.0, [source]
        distances = {node: np.inf for node in self.nodes}
        previous: dict[int, int] = {}
        unvisited = set(self.nodes)
        distances[source] = 0.0
        while unvisited:
            current = min(unvisited, key=lambda node: distances[node])
            unvisited.remove(current)
            if not np.isfinite(distances[current]):
                break
            if current == target:
                break
            for neighbor in self.neighbors(current):
                if neighbor not in unvisited:
                    continue
                edge = self._key(current, neighbor)
                candidate = distances[current] + self.edges[edge]
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    previous[neighbor] = current
        if not np.isfinite(distances[target]):
            raise ValueError("Graph is disconnected")
        path = [target]
        while path[-1] != source:
            path.append(previous[path[-1]])
        path.reverse()
        return float(distances[target]), path


def _kmeans(X: Array, n_clusters: int, random_state: int, max_iter: int = 80) -> Array:
    """Small deterministic k-means implementation with k-means++ seeds."""
    n_samples = X.shape[0]
    k = min(max(1, int(n_clusters)), n_samples)
    rng = np.random.default_rng(random_state)
    centers = np.empty((k, X.shape[1]), dtype=float)
    first = int(rng.integers(n_samples))
    centers[0] = X[first]
    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for index in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 1e-15:
            centers[index:] = X[rng.choice(n_samples, size=k - index, replace=False)]
            break
        probabilities = closest_sq / total
        chosen = int(rng.choice(n_samples, p=probabilities))
        centers[index] = X[chosen]
        closest_sq = np.minimum(closest_sq, np.sum((X - centers[index]) ** 2, axis=1))

    def squared_distances_to_centers(points: Array, anchors: Array) -> Array:
        point_norms = np.sum(points * points, axis=1)
        anchor_norms = np.sum(anchors * anchors, axis=1)
        distances = np.empty((len(points), len(anchors)), dtype=float)
        block_size = 256
        for start in range(0, len(points), block_size):
            stop = min(start + block_size, len(points))
            block = (
                point_norms[start:stop, None]
                + anchor_norms[None, :]
                - 2.0 * (points[start:stop] @ anchors.T)
            )
            distances[start:stop] = np.maximum(block, 0.0)
        return distances

    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        distances = squared_distances_to_centers(X, centers)
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for cluster in range(k):
            members = X[new_labels == cluster]
            if len(members):
                new_centers[cluster] = np.mean(members, axis=0)
            else:
                new_centers[cluster] = X[int(rng.integers(n_samples))]
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            centers = new_centers
            break
        labels, centers = new_labels, new_centers
    return centers


def _minimum_spanning_tree(centroids: Array) -> _LandmarkGraph:
    graph = _LandmarkGraph({index: point for index, point in enumerate(centroids)})
    distances = _pairwise_distances(centroids)
    selected = np.zeros(len(centroids), dtype=bool)
    best = np.full(len(centroids), np.inf)
    parent = np.full(len(centroids), -1, dtype=int)
    best[0] = 0.0
    for _ in range(len(centroids)):
        available = np.where(~selected)[0]
        current = int(available[np.argmin(best[available])])
        selected[current] = True
        if parent[current] >= 0:
            graph.add_edge(current, int(parent[current]), float(best[current]))
        candidates = ~selected
        improved = candidates & (distances[current] < best)
        best[improved] = distances[current, improved]
        parent[improved] = current
    return graph


def _symmetric_knn_edges(graph: _LandmarkGraph, neighbors: int) -> set[tuple[int, int]]:
    """Return local landmark edges from the symmetrized kNN graph.

    The MST remains responsible for connectivity.  These edges are only
    eligible for adding topology, which prevents the previous all-pairs
    shortcut heuristic from connecting unrelated distant landmarks.
    """
    node_ids = sorted(graph.nodes)
    if len(node_ids) < 2:
        return set()
    coordinates = np.asarray([graph.nodes[node] for node in node_ids])
    distances = _pairwise_distances(coordinates)
    count = min(max(1, int(neighbors)), len(node_ids) - 1)
    edges: set[tuple[int, int]] = set()
    for row, node in enumerate(node_ids):
        order = np.argsort(distances[row], kind="mergesort")
        nearest = [int(column) for column in order if int(column) != row][:count]
        for column in nearest:
            edges.add(graph._key(node, node_ids[int(column)]))
    return edges


def _ordered_path_graph(centroids: Array, ordering_points: Array | None = None) -> _LandmarkGraph:
    """Build a single chain by ordering landmarks along their first PCA axis."""
    reference = centroids if ordering_points is None else ordering_points
    centered_reference = reference - np.mean(reference, axis=0)
    _, _, components = np.linalg.svd(centered_reference, full_matrices=False)
    order = np.argsort(centered_reference @ components[0])
    graph = _LandmarkGraph({index: point for index, point in enumerate(centroids)})
    for left, right in pairwise(order):
        graph.add_edge(int(left), int(right))
    return graph


def _is_nearly_linear(points: Array, tolerance: float) -> bool:
    """Detect a point cloud whose variance is concentrated on one axis."""
    if points.shape[1] == 1:
        return True
    singular_values = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
    if singular_values[0] <= 1e-12:
        return True
    return bool(singular_values[1] / singular_values[0] <= tolerance)


def _project_to_principal_line(points: Array) -> Array:
    """Project points onto their best-fitting affine one-dimensional line."""
    center = np.mean(points, axis=0)
    _, _, components = np.linalg.svd(points - center, full_matrices=False)
    direction = components[0]
    coordinates = (points - center) @ direction
    return center + coordinates[:, None] * direction


def _prune_short_terminal_branches(graph: _LandmarkGraph, factor: float) -> None:
    if factor <= 0 or graph.number_of_edges() == 0:
        return
    median_edge = float(np.median(list(graph.edges.values())))
    threshold = factor * median_edge
    changed = True
    while changed:
        changed = False
        for leaf in list(graph.nodes):
            if leaf not in graph.nodes or graph.degree(leaf) != 1:
                continue
            path = [leaf]
            length = 0.0
            current = leaf
            previous = None
            while graph.degree(current) == 1 or (graph.degree(current) == 2 and previous is not None):
                neighbors = [node for node in graph.neighbors(current) if node != previous]
                if not neighbors:
                    break
                following = neighbors[0]
                length += graph.edges[graph._key(current, following)]
                path.append(following)
                previous, current = current, following
                if graph.degree(current) != 2:
                    break
            if graph.degree(current) >= 3 and length < threshold:
                for node in path[:-1]:
                    if node in graph.nodes:
                        graph.remove_node(node)
                changed = True
                break


def _merge_nearby_junctions(graph: _LandmarkGraph, distance: float | None) -> _LandmarkGraph:
    if distance is None or distance <= 0:
        return graph
    junctions = [node for node in graph.nodes if graph.degree(node) >= 3]
    if len(junctions) < 2:
        return graph

    parent = {node: node for node in junctions}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for index, left in enumerate(junctions):
        for right in junctions[index + 1 :]:
            if np.linalg.norm(graph.nodes[left] - graph.nodes[right]) <= distance:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for node in junctions:
        groups.setdefault(find(node), []).append(node)
    if all(len(group) == 1 for group in groups.values()):
        return graph

    mapping = {node: node for node in graph.nodes}
    for group in groups.values():
        representative = min(group)
        for node in group:
            mapping[node] = representative

    # Contract degree-2 connector chains that became internal when two
    # nearby junctions were merged.  Without this step, two MST junctions
    # sharing a short connector leave that connector as a false terminal:
    # both of its incident edges map to the same representative, so one
    # duplicate edge is discarded and the connector acquires degree one.
    # Only all-degree-2 components are contracted; a component with a side
    # branch may contain a genuine terminal and must remain explicit.
    junction_set = set(junctions)
    eligible = {
        node for node in graph.nodes
        if node not in junction_set and graph.degree(node) == 2
    }
    remaining = set(eligible)
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component = {root}
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbour in graph.neighbors(node):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        boundary_representatives = [
            mapping[neighbour]
            for node in component
            for neighbour in graph.neighbors(node)
            if neighbour not in component and neighbour in junction_set
        ]
        if (
            len(boundary_representatives) >= 2
            and len(set(boundary_representatives)) == 1
        ):
            representative = boundary_representatives[0]
            for node in component:
                mapping[node] = representative

    new_nodes: dict[int, list[Array]] = {}
    for node, point in graph.nodes.items():
        representative = mapping[node]
        new_nodes.setdefault(representative, []).append(point)
    new_graph = _LandmarkGraph({node: np.mean(points, axis=0) for node, points in new_nodes.items()})
    for (left, right), weight in graph.edges.items():
        new_left, new_right = mapping[left], mapping[right]
        if new_left != new_right:
            new_graph.add_edge(new_left, new_right, weight)
    return new_graph


def _extract_chains(graph: _LandmarkGraph) -> list[dict[str, Any]]:
    """Extract maximal edge chains between non-degree-2 nodes."""
    visited: set[tuple[int, int]] = set()
    chains: list[dict[str, Any]] = []

    def mark(left: int, right: int) -> None:
        visited.add(graph._key(left, right))

    def is_visited(left: int, right: int) -> bool:
        return graph._key(left, right) in visited

    boundary = [node for node in graph.nodes if graph.degree(node) != 2]
    for start in boundary:
        for neighbor in graph.neighbors(start):
            if is_visited(start, neighbor):
                continue
            nodes = [start]
            previous, current = start, neighbor
            mark(previous, current)
            nodes.append(current)
            closed = False
            while graph.degree(current) == 2:
                candidates = [node for node in graph.neighbors(current) if node != previous]
                if not candidates:
                    break
                following = candidates[0]
                if is_visited(current, following):
                    break
                mark(current, following)
                previous, current = current, following
                nodes.append(current)
                if current == start:
                    closed = True
                    break
            chains.append({"nodes": nodes, "closed": closed})

    # A connected component made entirely of degree-2 nodes is one closed loop.
    for start in graph.nodes:
        for neighbor in graph.neighbors(start):
            if is_visited(start, neighbor):
                continue
            nodes = [start]
            previous, current = start, neighbor
            mark(previous, current)
            nodes.append(current)
            while current != start:
                candidates = [node for node in graph.neighbors(current) if node != previous]
                if not candidates:
                    break
                following = candidates[0]
                if is_visited(current, following):
                    break
                mark(current, following)
                previous, current = current, following
                if current != start:
                    nodes.append(current)
            chains.append({"nodes": nodes, "closed": True})
    return chains


def _rips_h1_persistence(X: Array, max_points: int, random_state: int) -> Array:
    """Compute H1 Vietoris-Rips bars over Z2 using vertices, edges, triangles.

    This compact fallback is intended for small prototype data sets.  It is
    replaced automatically by Ripser when that package is installed.
    """
    if len(X) > max_points:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=max_points, replace=False)
        X = X[np.sort(indices)]
    distances = _pairwise_distances(X)
    n = len(X)
    edge_data: list[tuple[int, int, float]] = []
    for left in range(n):
        for right in range(left + 1, n):
            edge_data.append((left, right, float(distances[left, right])))
    simplices: list[tuple[float, int, tuple[int, ...]]] = []
    simplices.extend((0.0, 0, (vertex,)) for vertex in range(n))
    simplices.extend((distance, 1, (left, right)) for left, right, distance in edge_data)
    for left in range(n):
        for middle in range(left + 1, n):
            for right in range(middle + 1, n):
                filtration = max(distances[left, middle], distances[left, right], distances[middle, right])
                simplices.append((float(filtration), 2, (left, middle, right)))
    simplices.sort(key=lambda item: (item[0], item[1], item[2]))

    edge_simplex_id: dict[tuple[int, int], int] = {}
    for simplex_id, (_, dimension, vertices) in enumerate(simplices):
        if dimension == 1:
            edge_simplex_id[vertices] = simplex_id

    pivot_columns: dict[int, set[int]] = {}
    h1_births: dict[int, float] = {}
    bars: list[tuple[float, float]] = []
    for simplex_id, (filtration, dimension, vertices) in enumerate(simplices):
        if dimension == 0:
            column: set[int] = set()
        elif dimension == 1:
            left, right = vertices
            column = {left, right}
        else:
            left, middle, right = vertices
            column = {
                edge_simplex_id[(left, middle)],
                edge_simplex_id[(left, right)],
                edge_simplex_id[(middle, right)],
            }
        while column:
            pivot = max(column)
            previous = pivot_columns.get(pivot)
            if previous is None:
                break
            column ^= previous
        if not column:
            if dimension == 1:
                h1_births[simplex_id] = filtration
            continue
        pivot = max(column)
        pivot_columns[pivot] = column
        if dimension == 2:
            birth = h1_births.pop(pivot, None)
            if birth is not None and filtration > birth:
                bars.append((birth, filtration))
    # Any remaining creator is an infinite bar.  Full Vietoris-Rips complexes
    # normally kill these, but retaining them makes the fallback mathematically
    # honest for a deliberately truncated filtration.
    bars.extend((birth, np.inf) for birth in h1_births.values())
    return np.asarray(sorted(bars), dtype=float).reshape((-1, 2))


def _estimate_persistence(
    X: Array,
    max_points: int,
    random_state: int,
) -> tuple[Array, str]:
    try:
        from ripser import ripser  # type: ignore
    except ImportError:
        # Ripser is optional.  The selected backend is exposed on the fitted
        # estimator, so an unavailable optional dependency does not need to
        # interrupt normal fitting with a warning.  The pure-NumPy fallback
        # builds a full Vietoris--Rips 2-skeleton and therefore has cubic
        # memory/time growth.  Bound its internal sample independently of the
        # requested cap so a notebook cannot become unresponsive merely
        # because Ripser is unavailable.
        fallback_max_points = min(int(max_points), 120)
        return (
            _rips_h1_persistence(
                X, max_points=fallback_max_points, random_state=random_state,
            ),
            "numpy",
        )

    try:
        if len(X) > max_points:
            rng = np.random.default_rng(random_state)
            sample_indices = rng.choice(len(X), size=max_points, replace=False)
            sample = X[sample_indices]
        else:
            sample = X
        result = ripser(sample, maxdim=1)
        return np.asarray(result["dgms"][1], dtype=float), "ripser"
    except Exception as exc:  # noqa: BLE001 - persistence backend failure is reported
        warnings.warn(
            "Ripser failed; using the NumPy persistent-homology fallback: "
            f"{exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return (
            _rips_h1_persistence(
                X, max_points=min(int(max_points), 120), random_state=random_state,
            ),
            "numpy-after-ripser-error",
        )


def estimate_topology(
    X: Array,
    *,
    persistence_threshold: float | None = None,
    local_scales: Sequence[float] | None = None,
    inner_radius_fraction: float = 0.25,
    min_junction_confidence: float = 0.7,
    junction_scales: int | Sequence[float] = 6,
    topology_neighbors: int = 6,
    mutual_knn: bool = False,
    add_mst: bool = False,
    persistence_max_points: int = 60,
    random_state: int = 0,
) -> TopologyEstimate:
    """Estimate cycle and local branch constraints for a point cloud.

    This compact internal entry point is useful for diagnostics and unit
    tests.  ``SkeletalEmbedding`` calls the same primitives while retaining
    fitted graph and routing diagnostics on the estimator.
    """
    points = _as_point_cloud(X)
    graph, graph_scales = _weighted_symmetric_knn_graph(
        points,
        topology_neighbors,
        mutual_knn=mutual_knn,
        add_mst=add_mst,
    )
    scale = _local_scale(points)
    diagram, _ = _estimate_persistence(
        points, max_points=persistence_max_points, random_state=random_state
    )
    normalized = _normalize_persistence_diagram(diagram, scale)
    threshold = 3.0 if persistence_threshold is None else float(persistence_threshold)
    lifetimes = normalized[:, 1] - normalized[:, 0] if len(normalized) else np.empty(0)
    cycle_count = int(np.sum(np.isfinite(lifetimes) & (lifetimes >= threshold)))
    prototypes = _kmeans(points, min(32, len(points)), random_state)
    junctions, endpoints, branch_counts, confidences = _estimate_local_topology(
        points,
        graph,
        prototypes,
        local_scales=graph_scales if local_scales is None else local_scales,
        inner_radius_fraction=inner_radius_fraction,
        min_junction_confidence=min_junction_confidence,
        scales=junction_scales,
    )
    return TopologyEstimate(
        cycle_count,
        np.asarray(diagram, dtype=float),
        normalized,
        junctions,
        endpoints,
        branch_counts,
        confidences,
    )


def _normalize_persistence_diagram(diagram: Array, scale: float) -> Array:
    """Express a raw persistence diagram in nearest-neighbour scale units."""
    diagram = np.asarray(diagram, dtype=float)
    if diagram.size == 0:
        return np.empty((0, 2), dtype=float)
    result = diagram.copy()
    result[:, 0] /= max(float(scale), 1e-12)
    finite_deaths = np.isfinite(result[:, 1])
    result[finite_deaths, 1] /= max(float(scale), 1e-12)
    return result


def _cycle_anchor_vertices(
    graph: _WeightedKNNGraph,
    count: int,
    excluded_vertices: set[int] | None = None,
) -> list[int]:
    """Choose graph vertices near the strongest fundamental cycle candidates.

    ``excluded_vertices`` is used when another topological feature has already
    claimed a terminal arm, such as the stem of a loop-with-branch shape.
    Dense kNN graphs contain many short noise cycles, so allowing those points
    to seed the logical cycle can make the stem appear to be part of the loop.
    """
    if count <= 0 or len(graph.points) == 0:
        return []
    excluded = {
        int(vertex) for vertex in (excluded_vertices or set())
        if 0 <= int(vertex) < len(graph.points)
    }
    parent = list(range(len(graph.points)))

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

    tree_adjacency: dict[int, list[int]] = {node: [] for node in range(len(graph.points))}
    non_tree: list[tuple[float, tuple[int, int]]] = []
    for edge, length in sorted(graph.edges.items(), key=lambda item: item[1]):
        left, right = edge
        if union(left, right):
            tree_adjacency[left].append(right)
            tree_adjacency[right].append(left)
        else:
            # A high path-to-chord contrast is a useful cycle-anchor proxy.
            non_tree.append((float(length), edge))

    def tree_path(source: int, target: int) -> list[int]:
        previous: dict[int, int] = {source: -1}
        queue = [source]
        for current in queue:
            if current == target:
                break
            for neighbour in tree_adjacency[current]:
                if neighbour not in previous:
                    previous[neighbour] = current
                    queue.append(neighbour)
        if target not in previous:
            return []
        path = [target]
        while path[-1] != source:
            path.append(previous[path[-1]])
        return path[::-1]

    scored: list[tuple[float, int, int]] = []
    for direct, (left, right) in non_tree:
        path = tree_path(left, right)
        if len(path) < 3:
            continue
        path_length = sum(
            float(graph.edges[graph.key(a, b)]) for a, b in pairwise(path)
        )
        scored.append((path_length / max(direct, 1e-12), left, right))
    scored.sort(reverse=True)
    anchors: list[int] = []
    for _, left, right in scored:
        for vertex in (left, right):
            if vertex in excluded:
                continue
            if vertex not in anchors:
                anchors.append(vertex)
            if len(anchors) >= max(3, 3 * count):
                break
    target = max(3, 3 * count)
    # Use the strongest cycle edge only as a reproducible seed, then spread
    # anchors around the corresponding support rather than retaining several
    # adjacent vertices from the same local kNN chord.
    if not anchors:
        available = [vertex for vertex in range(len(graph.points)) if vertex not in excluded]
        if not available:
            return []
        anchors = [available[0]]
    else:
        anchors = [anchors[0]]
    while len(anchors) < target and len(anchors) < len(graph.points):
        distances = np.min(
            np.asarray([
                np.linalg.norm(graph.points - graph.points[anchor], axis=1)
                for anchor in anchors
            ]),
            axis=0,
        )
        distances[anchors] = -np.inf
        if excluded:
            distances[list(excluded)] = -np.inf
        next_anchor = int(np.argmax(distances))
        if not np.isfinite(distances[next_anchor]):
            break
        anchors.append(next_anchor)
    return anchors[:target]


def _approximate_cycle_representatives(
    graph: _WeightedKNNGraph,
    count: int,
) -> list[Array]:
    """Return deterministic point-level fundamental-cycle representatives.

    Persistent-homology backends used by the lightweight package expose the
    diagram consistently but do not all expose cocycles.  A minimum spanning
    forest plus non-tree chords gives a useful, reproducible geometric
    representative without adding a backend dependency.
    """
    if count <= 0 or not graph.edges:
        return []
    parent = list(range(len(graph.points)))
    tree: dict[int, list[int]] = {node: [] for node in range(len(graph.points))}
    chords: list[tuple[float, tuple[int, int]]] = []

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

    for edge, length in sorted(graph.edges.items(), key=lambda item: (item[1], item[0])):
        left, right = edge
        if union(left, right):
            tree[left].append(right)
            tree[right].append(left)
        else:
            chords.append((float(length), edge))

    def tree_path(source: int, target: int) -> list[int]:
        previous: dict[int, int] = {source: -1}
        queue = [source]
        for node in queue:
            if node == target:
                break
            for neighbour in sorted(tree[node]):
                if neighbour not in previous:
                    previous[neighbour] = node
                    queue.append(neighbour)
        if target not in previous:
            return []
        path = [target]
        while path[-1] != source:
            path.append(previous[path[-1]])
        return path[::-1]

    scored: list[tuple[float, list[int]]] = []
    for chord_length, (left, right) in chords:
        path = tree_path(left, right)
        if len(path) < 3:
            continue
        tree_length = sum(
            float(graph.edges[graph.key(a, b)]) for a, b in pairwise(path)
        )
        scored.append((tree_length / max(chord_length, 1e-12), path + [left]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [graph.points[np.asarray(path, dtype=int)].copy() for _, path in scored[:count]]
