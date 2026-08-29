"""Local PCA geometry for topology-aware graph routing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._topology import JunctionRegion, _WeightedKNNGraph

Array = np.ndarray


def _unit(vector: Array) -> Array:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def _pca_direction(points: Array, center: Array | None = None) -> Array:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.zeros(points.shape[1], dtype=float)
    if center is None:
        center = np.mean(points, axis=0)
    centered = points - center
    if np.allclose(centered, 0.0):
        return np.zeros(points.shape[1], dtype=float)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return _unit(components[0])


def _estimate_local_tangents(
    X: Array,
    graph: _WeightedKNNGraph,
    neighbors: int,
) -> Array:
    """Estimate one unoriented tangent direction per graph vertex."""
    points = np.asarray(X, dtype=float)
    result = np.zeros_like(points)
    count = max(2, int(neighbors))
    for node in range(len(points)):
        distances = [
            (float(np.linalg.norm(points[node] - points[other])), other)
            for other in graph.adjacency[node]
        ]
        distances.sort(key=lambda item: item[0])
        selected = [other for _, other in distances[:count]]
        if len(selected) < 2:
            selected = graph.adjacency[node]
        if selected:
            result[node] = _pca_direction(points[selected], points[node])
    return result


def _junction_branch_directions(
    X: Array,
    junction: JunctionRegion,
) -> Array:
    """Run PCA independently on each annulus component of a junction."""
    directions: list[Array] = []
    for component in junction.arm_indices:
        indices = np.asarray(component, dtype=int)
        if len(indices) == 0:
            continue
        direction = _pca_direction(X[indices], junction.center)
        centroid = np.mean(X[indices], axis=0)
        outward = centroid - junction.center
        if float(direction @ outward) < 0.0:
            direction = -direction
        if np.linalg.norm(direction) <= 1e-12 and np.linalg.norm(outward) > 1e-12:
            direction = _unit(outward)
        directions.append(_unit(direction))
    if not directions:
        return np.empty((0, X.shape[1]), dtype=float)
    return np.asarray(directions, dtype=float)


def _attach_junction_directions(X: Array, junctions: Sequence[JunctionRegion]) -> dict[int, Array]:
    """Return branch tangent arrays keyed by region position."""
    return {
        index: _junction_branch_directions(X, junction)
        for index, junction in enumerate(junctions)
    }


def _tangent_inconsistency(left: Array, right: Array) -> float:
    """Compare unoriented regular-point tangents."""
    left = _unit(left)
    right = _unit(right)
    if np.linalg.norm(left) <= 1e-12 or np.linalg.norm(right) <= 1e-12:
        return 0.0
    return float(1.0 - abs(float(left @ right)))


def _departure_angle(direction: Array, edge_vector: Array) -> float:
    """Return the oriented angular cost ``1 - u.T d``."""
    direction = _unit(direction)
    edge_vector = _unit(edge_vector)
    if np.linalg.norm(direction) <= 1e-12 or np.linalg.norm(edge_vector) <= 1e-12:
        return np.inf
    return float(1.0 - np.clip(float(direction @ edge_vector), -1.0, 1.0))


__all__ = [
    "_attach_junction_directions",
    "_departure_angle",
    "_estimate_local_tangents",
    "_junction_branch_directions",
    "_pca_direction",
    "_tangent_inconsistency",
]
