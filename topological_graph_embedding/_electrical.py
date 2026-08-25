"""Electrical diagnostics for the dense routing graph."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ._topology import _WeightedKNNGraph

Array = np.ndarray


def _laplacian(graph: _WeightedKNNGraph) -> Array:
    """Construct the dense conductance Laplacian."""
    size = len(graph.points)
    laplacian = np.zeros((size, size), dtype=float)
    for (left, right), conductance in graph.conductances.items():
        value = float(conductance)
        laplacian[left, left] += value
        laplacian[right, right] += value
        laplacian[left, right] -= value
        laplacian[right, left] -= value
    return laplacian


def _effective_resistance(graph: _WeightedKNNGraph) -> tuple[Array, dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Return the Laplacian pseudoinverse, edge resistances, and leverage."""
    laplacian = _laplacian(graph)
    if len(laplacian) == 0:
        return laplacian, {}, {}
    pseudoinverse = np.linalg.pinv(laplacian, hermitian=True)
    resistance: dict[tuple[int, int], float] = {}
    leverage: dict[tuple[int, int], float] = {}
    for edge, conductance in graph.conductances.items():
        left, right = edge
        value = float(
            pseudoinverse[left, left]
            + pseudoinverse[right, right]
            - 2.0 * pseudoinverse[left, right]
        )
        value = max(0.0, value)
        resistance[edge] = value
        leverage[edge] = float(conductance * value)
    return pseudoinverse, resistance, leverage


def _electrical_flow(
    graph: _WeightedKNNGraph,
    pairs: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Aggregate normalized absolute current for source-target experiments."""
    if len(graph.points) == 0:
        return {}
    laplacian = _laplacian(graph)
    pseudoinverse = np.linalg.pinv(laplacian, hermitian=True)
    traffic = {edge: 0.0 for edge in graph.edges}
    for source, target in pairs:
        if source == target:
            continue
        injection = np.zeros(len(graph.points), dtype=float)
        injection[int(source)] = 1.0
        injection[int(target)] = -1.0
        potential = pseudoinverse @ injection
        for edge, conductance in graph.conductances.items():
            left, right = edge
            traffic[edge] += abs(float(conductance * (potential[left] - potential[right])))
    maximum = max(traffic.values(), default=0.0)
    if maximum > 1e-12:
        traffic = {edge: value / maximum for edge, value in traffic.items()}
    return traffic


def _kron_reduction(
    graph: _WeightedKNNGraph,
    landmark_vertices: Sequence[int],
) -> tuple[Array, Array]:
    """Return the Schur-complement Laplacian and retained vertex IDs."""
    retained = np.asarray(sorted(set(int(value) for value in landmark_vertices)), dtype=int)
    if len(retained) == 0:
        return np.empty((0, 0), dtype=float), retained
    laplacian = _laplacian(graph)
    interior = np.asarray([index for index in range(len(graph.points)) if index not in set(retained)], dtype=int)
    if len(interior) == 0:
        return laplacian[np.ix_(retained, retained)], retained
    boundary_block = laplacian[np.ix_(retained, retained)]
    cross_block = laplacian[np.ix_(retained, interior)]
    interior_block = laplacian[np.ix_(interior, interior)]
    try:
        solved = np.linalg.solve(interior_block, cross_block.T)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(interior_block, hermitian=True) @ cross_block.T
    reduced = boundary_block - cross_block @ solved
    reduced = 0.5 * (reduced + reduced.T)
    reduced[np.abs(reduced) < 1e-12] = 0.0
    return reduced, retained


__all__ = [
    "_electrical_flow",
    "_effective_resistance",
    "_kron_reduction",
    "_laplacian",
]
