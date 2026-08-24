"""Topological graph skeletonization with smooth spline highways.

The implementation intentionally keeps the graph representation small and
inspectable.  NumPy is required.  SciPy is used when available for smoothing
splines and Ripser is used when available for persistent homology; compact
NumPy fallbacks are provided so the prototype remains runnable in a minimal
environment.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:  # Optional, but recommended.
    from scipy.interpolate import splev, splprep
except ImportError as exc:  # pragma: no cover - exercised without SciPy.
    splev = None
    splprep = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


Array = np.ndarray


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


class SkeletonGraph:
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

    @staticmethod
    def _key(u: int, v: int) -> tuple[int, int]:
        if u == v:
            raise ValueError("Self-loops are not supported")
        return (u, v) if u < v else (v, u)

    def copy(self) -> "SkeletonGraph":
        result = SkeletonGraph(self.nodes)
        result.edges = self.edges.copy()
        return result

    def add_edge(self, u: int, v: int, weight: float | None = None) -> None:
        if u not in self.nodes or v not in self.nodes:
            raise KeyError("Both edge endpoints must already be graph nodes")
        key = self._key(u, v)
        if weight is None:
            weight = float(np.linalg.norm(self.nodes[u] - self.nodes[v]))
        self.edges[key] = float(weight)

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


def _minimum_spanning_tree(centroids: Array) -> SkeletonGraph:
    graph = SkeletonGraph({index: point for index, point in enumerate(centroids)})
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


def _symmetric_knn_edges(graph: SkeletonGraph, neighbors: int) -> set[tuple[int, int]]:
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


def _ordered_path_graph(centroids: Array, ordering_points: Array | None = None) -> SkeletonGraph:
    """Build a single chain by ordering landmarks along their first PCA axis."""
    reference = centroids if ordering_points is None else ordering_points
    centered_reference = reference - np.mean(reference, axis=0)
    _, _, components = np.linalg.svd(centered_reference, full_matrices=False)
    order = np.argsort(centered_reference @ components[0])
    graph = SkeletonGraph({index: point for index, point in enumerate(centroids)})
    for left, right in zip(order[:-1], order[1:]):
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


def _prune_short_terminal_branches(graph: SkeletonGraph, factor: float) -> None:
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


def _merge_nearby_junctions(graph: SkeletonGraph, distance: float | None) -> SkeletonGraph:
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
    new_nodes: dict[int, list[Array]] = {}
    for node, point in graph.nodes.items():
        representative = mapping[node]
        new_nodes.setdefault(representative, []).append(point)
    new_graph = SkeletonGraph({node: np.mean(points, axis=0) for node, points in new_nodes.items()})
    for (left, right), weight in graph.edges.items():
        new_left, new_right = mapping[left], mapping[right]
        if new_left != new_right:
            new_graph.add_edge(new_left, new_right, weight)
    return new_graph


def _extract_chains(graph: SkeletonGraph) -> list[dict[str, Any]]:
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


def _catmull_rom(points: Array, t: Array, closed: bool) -> Array:
    """Fallback cubic curve evaluator used when SciPy is unavailable."""
    n = len(points)
    if n < 3:
        return np.vstack([points[0] + value * (points[-1] - points[0]) for value in t])
    result = []
    if closed:
        for value in t:
            position = (value % 1.0) * n
            index = int(np.floor(position))
            local = position - index
            p0 = points[(index - 1) % n]
            p1 = points[index % n]
            p2 = points[(index + 1) % n]
            p3 = points[(index + 2) % n]
            result.append(
                0.5
                * ((2 * p1) + (-p0 + p2) * local + (2 * p0 - 5 * p1 + 4 * p2 - p3) * local**2
                   + (-p0 + 3 * p1 - 3 * p2 + p3) * local**3)
            )
    else:
        for value in t:
            position = np.clip(value, 0.0, 1.0) * (n - 1)
            index = min(int(np.floor(position)), n - 2)
            local = position - index
            p0 = points[max(index - 1, 0)]
            p1 = points[index]
            p2 = points[index + 1]
            p3 = points[min(index + 2, n - 1)]
            result.append(
                0.5
                * ((2 * p1) + (-p0 + p2) * local + (2 * p0 - 5 * p1 + 4 * p2 - p3) * local**2
                   + (-p0 + 3 * p1 - 3 * p2 + p3) * local**3)
            )
    return np.asarray(result)


@dataclass
class SplineCurve:
    """Dense, parameterized representation of one fitted highway."""

    samples: Array
    t_values: Array
    closed: bool
    tck: Any = None
    backend: str = "numpy"

    def evaluate(self, t: Array | float) -> Array:
        values = np.atleast_1d(np.asarray(t, dtype=float))
        if self.closed:
            values = values % 1.0
        else:
            values = np.clip(values, 0.0, 1.0)
        if self.tck is not None and splev is not None:
            evaluated = np.asarray(splev(values, self.tck)).T
        else:
            evaluated = _catmull_rom(self.samples, values, self.closed)
        if not self.closed:
            evaluated[values <= 0.0] = self.samples[0]
            evaluated[values >= 1.0] = self.samples[-1]
        return evaluated[0] if np.ndim(t) == 0 else evaluated

    def tangent(self, t: Array | float, epsilon: float = 1e-4) -> Array:
        """Return unit tangent vectors in the curve's coordinate system.

        The dense sampled curve is also used as the projection geometry, so a
        centered finite difference gives a consistent tangent for both the
        SciPy and NumPy curve evaluators.  The direction is all that matters
        here; the normal frame is invariant to the tangent's scale.
        """
        values = np.atleast_1d(np.asarray(t, dtype=float))
        if self.closed:
            left = values - epsilon
            right = values + epsilon
        else:
            left = np.clip(values - epsilon, 0.0, 1.0)
            right = np.clip(values + epsilon, 0.0, 1.0)
        derivatives = np.asarray(self.evaluate(right) - self.evaluate(left), dtype=float)
        norms = np.linalg.norm(derivatives, axis=1, keepdims=True)
        invalid = norms[:, 0] < 1e-12
        if np.any(invalid):
            fallback = np.diff(self.samples, axis=0)
            if self.closed:
                fallback = np.vstack([fallback, self.samples[0] - self.samples[-1]])
            fallback_norms = np.linalg.norm(fallback, axis=1)
            valid_fallback = np.flatnonzero(fallback_norms > 1e-12)
            if len(valid_fallback):
                for row in np.flatnonzero(invalid):
                    nearest = int(np.argmin(np.abs(self.t_values[valid_fallback] - values[row])))
                    derivatives[row] = fallback[valid_fallback[nearest]]
                norms = np.linalg.norm(derivatives, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        result = derivatives / norms
        return result[0] if np.ndim(t) == 0 else result

    def project(self, X: Array, batch_size: int = 4096) -> tuple[Array, Array, Array]:
        """Project points onto sampled line segments; return point, t, d2."""
        X = np.asarray(X, dtype=float)
        count = len(self.samples)
        segment_count = count if self.closed else count - 1
        if segment_count <= 0:
            projection = np.repeat(self.samples[:1], len(X), axis=0)
            residual = X - projection
            return projection, np.zeros(len(X)), np.sum(residual * residual, axis=1)
        starts = self.samples[:segment_count]
        ends = self.samples[1:segment_count + 1]
        if self.closed:
            ends = np.vstack([ends, self.samples[0]])
        vectors = ends - starts
        denominators = np.sum(vectors * vectors, axis=1)
        denominators[denominators < 1e-15] = 1.0

        best_d2 = np.full(len(X), np.inf)
        best_projection = np.zeros_like(X)
        best_t = np.zeros(len(X))
        for batch_start in range(0, len(X), max(1, int(batch_size))):
            batch_stop = min(batch_start + max(1, int(batch_size)), len(X))
            batch = X[batch_start:batch_stop]
            batch_best_d2 = np.full(len(batch), np.inf)
            batch_best_projection = np.zeros_like(batch)
            batch_best_t = np.zeros(len(batch))
            for index in range(segment_count):
                offset = batch - starts[index]
                alpha = np.sum(offset * vectors[index], axis=1) / denominators[index]
                alpha = np.clip(alpha, 0.0, 1.0)
                candidate = starts[index] + alpha[:, None] * vectors[index]
                d2 = np.sum((batch - candidate) ** 2, axis=1)
                improved = d2 < batch_best_d2
                batch_best_d2[improved] = d2[improved]
                batch_best_projection[improved] = candidate[improved]
                if self.closed:
                    t = (index + alpha) / segment_count
                else:
                    t = self.t_values[index] + alpha * (
                        self.t_values[index + 1] - self.t_values[index]
                    )
                batch_best_t[improved] = t[improved]
            best_d2[batch_start:batch_stop] = batch_best_d2
            best_projection[batch_start:batch_stop] = batch_best_projection
            best_t[batch_start:batch_stop] = batch_best_t
        return best_projection, best_t, best_d2


def _normal_frame(tangent: Array) -> Array:
    """Return a deterministic orthonormal basis normal to ``tangent``."""
    tangent = np.asarray(tangent, dtype=float).reshape(-1)
    if tangent.size <= 1:
        return np.zeros((tangent.size, 0), dtype=float)
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-12:
        tangent = np.zeros_like(tangent)
        tangent[0] = 1.0
    else:
        tangent = tangent / norm
    columns: list[Array] = []
    for axis in range(tangent.size):
        candidate = np.zeros_like(tangent)
        candidate[axis] = 1.0
        candidate -= tangent * float(tangent @ candidate)
        for previous in columns:
            candidate -= previous * float(previous @ candidate)
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm < 1e-10:
            continue
        candidate /= candidate_norm
        pivot = int(np.argmax(np.abs(candidate)))
        if candidate[pivot] < 0.0:
            candidate *= -1.0
        columns.append(candidate)
        if len(columns) == tangent.size - 1:
            break
    if len(columns) != tangent.size - 1:
        return np.linalg.qr(
            np.eye(tangent.size) - np.outer(tangent, tangent), mode="reduced"
        )[0][:, : tangent.size - 1]
    return np.column_stack(columns)


def _transport_frame(previous: Array, tangent: Array) -> Array:
    """Parallel-transport a normal frame to a new unit tangent."""
    tangent = np.asarray(tangent, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-12:
        tangent = np.zeros_like(tangent)
        tangent[0] = 1.0
    else:
        tangent = tangent / norm
    projected = previous - tangent[:, None] * (tangent @ previous)[None, :]
    frame, _ = np.linalg.qr(projected, mode="reduced")
    if frame.shape[1] != previous.shape[1] or not np.all(np.isfinite(frame)):
        return _normal_frame(tangent)
    for column in range(frame.shape[1]):
        if float(frame[:, column] @ projected[:, column]) < 0.0:
            frame[:, column] *= -1.0
    return frame


def _fit_normal_frame_grid(spline: SplineCurve) -> dict[str, Any]:
    """Fit a deterministic, query-batch-independent frame grid."""
    dimension = spline.samples.shape[1]
    count = max(64, len(spline.samples))
    if spline.closed:
        grid_t = np.linspace(0.0, 1.0, count + 1)
    else:
        grid_t = np.linspace(0.0, 1.0, count)
    if dimension <= 1:
        return {"t": grid_t, "frames": np.zeros((len(grid_t), dimension, 0))}
    tangents = np.asarray(spline.tangent(grid_t), dtype=float)
    frames = np.empty((len(grid_t), dimension, dimension - 1), dtype=float)
    frames[0] = _normal_frame(tangents[0])
    for index in range(1, len(grid_t)):
        frames[index] = _transport_frame(frames[index - 1], tangents[index])
    if spline.closed:
        # The endpoint is the same geometric point as t=0.  Reusing the first
        # frame makes interpolation across the seam periodic and deterministic.
        frames[-1] = frames[0]
    return {"t": grid_t, "frames": frames}


def _frame_from_grid(grid: dict[str, Any], t: float, tangent: Array, closed: bool) -> Array:
    """Interpolate one fixed frame grid and re-orthonormalize it."""
    grid_t = np.asarray(grid["t"], dtype=float)
    grid_frames = np.asarray(grid["frames"], dtype=float)
    if grid_frames.shape[2] == 0:
        return grid_frames[0]
    value = float(t % 1.0) if closed else float(np.clip(t, 0.0, 1.0))
    index = int(np.searchsorted(grid_t, value, side="right") - 1)
    index = min(max(index, 0), len(grid_t) - 2)
    denominator = grid_t[index + 1] - grid_t[index]
    alpha = 0.0 if denominator <= 1e-12 else (value - grid_t[index]) / denominator
    base = (1.0 - alpha) * grid_frames[index] + alpha * grid_frames[index + 1]
    tangent = np.asarray(tangent, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-12:
        tangent = np.zeros_like(tangent)
        tangent[0] = 1.0
    else:
        tangent = tangent / norm
    projected = base - tangent[:, None] * (tangent @ base)[None, :]
    frame, _ = np.linalg.qr(projected, mode="reduced")
    if frame.shape != base.shape or not np.all(np.isfinite(frame)):
        return _normal_frame(tangent)
    for column in range(frame.shape[1]):
        if float(frame[:, column] @ projected[:, column]) < 0.0:
            frame[:, column] *= -1.0
    return frame


def spline_normal_frames(model: Any, result: dict[str, Array]) -> Array:
    """Return one orthonormal normal frame for every projected observation.

    The returned array has shape ``(n_samples, n_features, n_features - 1)``.
    Coordinates in this frame describe only displacement perpendicular to the
    local spline tangent; the longitudinal direction is intentionally omitted.
    """
    highway_ids = np.asarray(result["highway_id"], dtype=int)
    t_values = np.asarray(result["t"], dtype=float)
    n_samples = len(highway_ids)
    n_features = int(np.asarray(result["residual_vector"]).shape[1])
    frames = np.zeros((n_samples, n_features, max(0, n_features - 1)), dtype=float)
    tangent_vectors = result.get("tangent_vector")
    if tangent_vectors is not None:
        tangent_vectors = np.asarray(tangent_vectors, dtype=float)
    frame_grids = getattr(model, "normal_frame_grids_", None)
    for route, spline in enumerate(model.splines_):
        members = np.flatnonzero(highway_ids == route)
        if not len(members):
            continue
        if tangent_vectors is None:
            tangents = np.asarray(spline.tangent(t_values[members]), dtype=float)
        else:
            tangents = tangent_vectors[members]
        if frame_grids is not None and route < len(frame_grids):
            for position, member in enumerate(members):
                tangent = np.asarray(tangents[position], dtype=float)
                if n_features == 2:
                    norm = max(float(np.linalg.norm(tangent)), 1e-12)
                    tangent = tangent / norm
                    frames[member] = np.asarray([[-tangent[1]], [tangent[0]]])
                else:
                    frames[member] = _frame_from_grid(
                        frame_grids[route], t_values[member], tangent, spline.closed,
                    )
            continue

        # Compatibility fallback for lightweight externally-created model
        # objects that predate ``normal_frame_grids_``.
        order = np.argsort(t_values[members])
        previous_frame = None
        for position in order:
            member = members[position]
            tangent = np.asarray(tangents[position], dtype=float)
            norm = float(np.linalg.norm(tangent))
            if norm < 1e-12:
                tangent = np.zeros(n_features, dtype=float)
                tangent[0] = 1.0
            else:
                tangent = tangent / norm
            if n_features == 2:
                frame = np.asarray([[-tangent[1]], [tangent[0]]])
            elif previous_frame is None:
                frame = _normal_frame(tangent)
            else:
                frame = _transport_frame(previous_frame, tangent)
            frames[member] = frame
            previous_frame = frame
    return frames


def spline_normal_coordinates(model: Any, result: dict[str, Array]) -> Array:
    """Project residuals into the local spline-normal hyperplane coordinates."""
    residual = np.asarray(result["residual_vector"], dtype=float)
    scale = np.asarray(getattr(model, "scale_", np.ones(residual.shape[1])), dtype=float)
    residual_scaled = residual / scale
    frames = spline_normal_frames(model, result)
    if frames.shape[2] == 0:
        return np.empty((len(residual), 0), dtype=float)
    return np.einsum("ni,nij->nj", residual_scaled, frames)


def _fit_curve(points: Array, closed: bool, smoothing: float, sample_count: int) -> SplineCurve:
    points = np.asarray(points, dtype=float)
    if closed and len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 2:
        repeated = np.repeat(points, 2, axis=0)
        return SplineCurve(repeated, np.array([0.0, 1.0]), False, backend="degenerate")

    differences = np.diff(points, axis=0)
    if closed:
        differences = np.vstack([differences, points[0] - points[-1]])
    segment_lengths = np.linalg.norm(differences, axis=1)
    if np.sum(segment_lengths) <= 1e-12:
        return SplineCurve(
            np.repeat(points[:1], 2, axis=0),
            np.array([0.0, 1.0]),
            False,
            backend="degenerate",
        )
    if closed:
        t = np.concatenate([[0.0], np.cumsum(segment_lengths[:-1])]) / np.sum(segment_lengths)
    else:
        t = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        t /= t[-1]

    tck = None
    backend = "numpy"
    if splprep is not None and len(points) >= 3:
        degree = min(3, len(points) - 1)
        smoothing_factor = max(0.0, float(smoothing)) * len(points)
        for _ in range(8):
            try:
                # FITPACK can report that a small requested smoothing value is
                # numerically unattainable for a short chain. Increase the
                # value for that chain only; other RuntimeWarnings remain
                # visible and ordinary fitting errors still use the fallback.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'error',
                        message='A theoretically impossible result when finding a smoothing spline.*',
                        category=RuntimeWarning,
                    )
                    tck, _ = splprep(
                        points.T,
                        u=t,
                        s=smoothing_factor,
                        per=closed,
                        k=degree,
                    )
                break
            except RuntimeWarning:
                smoothing_factor = max(1e-6, 2.0 * smoothing_factor)
                tck = None
            except Exception as exc:
                warnings.warn(
                    f"SciPy spline fitting failed; using the NumPy fallback: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                tck = None
                break
        if tck is not None:
            backend = "scipy"
    count = max(32, int(sample_count))
    sample_t = np.linspace(0.0, 1.0, count, endpoint=not closed)
    if tck is not None and splev is not None:
        candidate_samples = np.asarray(splev(sample_t, tck)).T
        control_min = np.min(points, axis=0)
        control_max = np.max(points, axis=0)
        control_span = np.maximum(control_max - control_min, 1e-8)
        margin = np.maximum(0.25 * control_span, 1e-3)
        has_overshoot = np.any(candidate_samples < control_min - margin) or np.any(
            candidate_samples > control_max + margin
        )
        if np.all(np.isfinite(candidate_samples)) and not has_overshoot:
            samples = candidate_samples
        else:
            tck = None
            backend = "numpy-overshoot-fallback"
    if tck is None:
        samples = _catmull_rom(points, sample_t, closed)
    if not closed:
        # Smoothing splines can pull their endpoints away from the graph
        # nodes.  Blend that correction over several samples so the highway
        # reaches the node without creating a sharp one-segment kink.
        if tck is not None:
            window = min(0.12, max(0.06, 8.0 / max(count - 1, 1)))
            start_weight = 1.0 - np.clip(sample_t / window, 0.0, 1.0) ** 2 * (
                3.0 - 2.0 * np.clip(sample_t / window, 0.0, 1.0)
            )
            end_distance = 1.0 - sample_t
            end_weight = 1.0 - np.clip(end_distance / window, 0.0, 1.0) ** 2 * (
                3.0 - 2.0 * np.clip(end_distance / window, 0.0, 1.0)
            )
            samples = samples + start_weight[:, None] * (points[0] - samples[0])
            samples = samples + end_weight[:, None] * (points[-1] - samples[-1])
            # The sampled curve now includes the endpoint constraints.  Use
            # the dense corrected samples for both plotting and evaluation.
            tck = None
        samples[0] = points[0]
        samples[-1] = points[-1]
    return SplineCurve(samples, sample_t, closed, tck, backend=backend)


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
    edge_lookup = {(left, right): index for index, (left, right, _) in enumerate(edge_data)}

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
    except ImportError as exc:
        warnings.warn(
            "Ripser is unavailable; using the NumPy persistent-homology fallback: "
            f"{exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return (
            _rips_h1_persistence(X, max_points=max_points, random_state=random_state),
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
    except Exception as exc:
        warnings.warn(
            "Ripser failed; using the NumPy persistent-homology fallback: "
            f"{exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return (
            _rips_h1_persistence(X, max_points=max_points, random_state=random_state),
            "numpy-after-ripser-error",
        )


class TopologicalSplineGraph:
    """Fit a smooth graph of spline highways to a point cloud."""

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
        self._fitted = False

    def fit(self, X: Array | Sequence[Sequence[float]]) -> "TopologicalSplineGraph":
        original = _as_point_cloud(X)
        if original.shape[0] < 3:
            raise ValueError("X must contain at least three observations for fitting")
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
        if self.persistence_threshold is None:
            threshold = 4.0 * self.local_scale_
            self.persistence_threshold_ = float(threshold)
        else:
            self.persistence_threshold_ = float(self.persistence_threshold)
        if len(self.persistence_diagram_):
            lifetimes = self.persistence_diagram_[:, 1] - self.persistence_diagram_[:, 0]
            significant = np.isfinite(lifetimes) & (lifetimes >= self.persistence_threshold_)
            self.persistent_cycle_count_ = int(np.sum(significant))
        else:
            self.persistent_cycle_count_ = 0
        self.requested_cycle_count_ = min(self.max_cycles, self.persistent_cycle_count_)

        self.centroids_ = _kmeans(points, self.n_centroids, self.random_state)
        self.linear_structure_ = _is_nearly_linear(self.centroids_, self.linear_structure_tolerance)
        if self.linear_structure_:
            # A noisy sample of a line can produce a branched MST and a
            # borderline H1 bar.  A PCA-ordered chain is the appropriate
            # low-complexity skeleton for this geometry.
            self.requested_cycle_count_ = 0
            self.merge_junction_distance_ = 0.0
            graph = _ordered_path_graph(self.centroids_)
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
        self.graph_ = graph
        self.topology_candidate_edges_ = _symmetric_knn_edges(
            self.graph_, self.topology_neighbors,
        )

        target_cycles = self.requested_cycle_count_
        while self.graph_.cycle_rank() < target_cycles:
            candidate = self._best_cycle_edge()
            if candidate is None:
                break
            left, right, weight = candidate
            self.graph_.add_edge(left, right, weight)

        self.cycle_count_ = self.graph_.cycle_rank()
        self.topology_shortfall_ = max(0, target_cycles - self.cycle_count_)
        if self.topology_shortfall_:
            warnings.warn(
                "The sparse landmark graph could not realize all requested cycles "
                f"({self.topology_shortfall_} shortfall).",
                RuntimeWarning,
                stacklevel=2,
            )
        self.junction_nodes_ = [node for node in self.graph_.nodes if self.graph_.degree(node) >= 3]
        self.endpoint_nodes_ = [node for node in self.graph_.nodes if self.graph_.degree(node) == 1]
        self.chains_ = _extract_chains(self.graph_)
        if not self.chains_:
            # Pruning can collapse a very small or fully duplicated graph to a
            # single geometric landmark.  Keep the transform contract valid by
            # retaining one degenerate highway instead of returning -1 IDs.
            nodes = sorted(self.graph_.nodes)
            if nodes:
                self.chains_ = [{"nodes": [nodes[0]], "closed": False}]
        for chain in self.chains_:
            chain["points"] = np.asarray([self.graph_.nodes[node] for node in chain["nodes"]])
            if self.linear_structure_:
                chain["points"] = _project_to_principal_line(chain["points"])
        self.splines_ = [
            _fit_curve(
                chain["points"],
                closed=chain["closed"],
                smoothing=self.spline_smoothing,
                sample_count=max(64, len(chain["nodes"]) * self.spline_samples_per_node),
            )
            for chain in self.chains_
        ]
        self._anchor_closed_junctions()
        if _SCIPY_IMPORT_ERROR is not None:
            warnings.warn(
                "SciPy is unavailable; spline highways use the NumPy fallback: "
                f"{_SCIPY_IMPORT_ERROR}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.spline_backend_ = [spline.backend for spline in self.splines_]
        self.normal_frame_grids_ = [
            _fit_normal_frame_grid(spline) for spline in self.splines_
        ]
        self._fitted = True
        return self

    def _anchor_closed_junctions(self) -> None:
        """Make closed spline samples pass exactly through graph junctions."""
        for chain, spline in zip(self.chains_, self.splines_):
            if not chain["closed"] or len(spline.samples) == 0:
                continue
            nodes = list(chain["nodes"])
            if len(nodes) > 1 and nodes[0] == nodes[-1]:
                nodes = nodes[:-1]
            used_samples: set[int] = set()
            for node_index, node in enumerate(nodes):
                if self.graph_.degree(node) < 3 or node_index >= len(spline.t_values):
                    continue
                parameter = float(spline.t_values[node_index]) % 1.0
                sample = int(np.rint(parameter * len(spline.samples))) % len(spline.samples)
                if sample in used_samples:
                    continue
                spline.samples[sample] = self.graph_.nodes[node]
                used_samples.add(sample)

    def _best_cycle_edge(self) -> tuple[int, int, float] | None:
        graph = self.graph_
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

    def transform(self, X: Array | Sequence[Sequence[float]]) -> dict[str, Array]:
        if not self._fitted:
            raise RuntimeError("Call fit before transform")
        original = _as_point_cloud(X)
        if original.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        points = (original - self.mean_) / self.scale_
        best_distance = np.full(len(points), np.inf)
        highway_id = np.full(len(points), -1, dtype=int)
        t = np.zeros(len(points))
        projection = np.zeros_like(points)
        for highway, spline in enumerate(self.splines_):
            candidate_projection, candidate_t, candidate_d2 = spline.project(points)
            improved = candidate_d2 < best_distance
            best_distance[improved] = candidate_d2[improved]
            highway_id[improved] = highway
            t[improved] = candidate_t[improved]
            projection[improved] = candidate_projection[improved]
        self.invalid_projection_count_ = int(np.sum(highway_id < 0))
        if self.invalid_projection_count_:
            raise RuntimeError(
                "The fitted spline graph could not project all observations; "
                f"{self.invalid_projection_count_} observations have no highway."
            )
        projection_original = projection * self.scale_ + self.mean_
        residual = original - projection_original
        tangent_vector = np.zeros_like(points)
        for highway, spline in enumerate(self.splines_):
            members = np.flatnonzero(highway_id == highway)
            if len(members):
                tangent_vector[members] = spline.tangent(t[members])
        result = {
            "highway_id": highway_id,
            "t": t,
            "projection": projection_original,
            "residual_vector": residual,
            "residual_norm": np.linalg.norm(residual, axis=1),
            # Tangents are stored in the standardized fitting coordinates so
            # they can define the correct high-dimensional normal hyperplane.
            "tangent_vector": tangent_vector,
        }
        return result

    def fit_transform(self, X: Array | Sequence[Sequence[float]]) -> dict[str, Array]:
        return self.fit(X).transform(X)

    def plot(
        self,
        X: Array | Sequence[Sequence[float]] | None = None,
        ax: Any = None,
        show_projections: bool = False,
        max_projection_lines: int = 150,
        title: str | None = None,
    ) -> Any:
        """Plot a fitted 2D graph and optionally return the Matplotlib axes."""
        if not self._fitted:
            raise RuntimeError("Call fit before plot")
        if self.n_features_in_ != 2:
            raise ValueError("plot is only available for two-dimensional data")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - depends on environment.
            raise ImportError("plot requires matplotlib") from exc
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        data = self._original_X_ if X is None else _as_point_cloud(X)
        ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.22, color="tab:blue", label="observations")
        for index, spline in enumerate(self.splines_):
            curve = spline.samples * self.scale_ + self.mean_
            if spline.closed:
                curve = np.vstack([curve, curve[0]])
            ax.plot(curve[:, 0], curve[:, 1], linewidth=2.8, label="spline" if index == 0 else None)
        junctions = np.asarray([self.graph_.nodes[node] for node in self.junction_nodes_])
        endpoints = np.asarray([self.graph_.nodes[node] for node in self.endpoint_nodes_])
        if len(junctions):
            junctions = junctions * self.scale_ + self.mean_
            ax.scatter(junctions[:, 0], junctions[:, 1], s=70, color="tab:red", zorder=5, label="junction")
        if len(endpoints):
            endpoints = endpoints * self.scale_ + self.mean_
            ax.scatter(endpoints[:, 0], endpoints[:, 1], s=65, marker="s", color="tab:orange", zorder=5, label="endpoint")
        if show_projections:
            result = self.transform(data)
            count = min(max_projection_lines, len(data))
            indices = np.linspace(0, len(data) - 1, count, dtype=int)
            for index in indices:
                ax.plot(
                    [data[index, 0], result["projection"][index, 0]],
                    [data[index, 1], result["projection"][index, 1]],
                    color="0.35",
                    linewidth=0.35,
                    alpha=0.35,
                )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")
        if title:
            ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        return ax


__all__ = [
    "SkeletonGraph",
    "SplineCurve",
    "TopologicalSplineGraph",
    "spline_normal_coordinates",
    "spline_normal_frames",
]
