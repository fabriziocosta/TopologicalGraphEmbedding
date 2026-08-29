"""Discrete selection helpers for the skeletal backbone."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _components(edge_pairs: Sequence[tuple[int, int]], n_nodes: int) -> int:
    parent = list(range(n_nodes))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edge_pairs:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    return len({find(node) for node in range(n_nodes)})


def select_backbone_mip(
    candidates: Sequence[Any],
    specifications: Sequence[dict[str, Any]],
    requested_cycles: int,
    *,
    use_mip: bool = True,
) -> tuple[dict[tuple[int, int], Any], str]:
    """Select candidate paths with a small mixed-integer model.

    The routing substrate remains responsible for producing feasible paths.
    This solver only selects among those paths, which keeps the optimization
    problem small and makes failures safely recoverable by the existing
    deterministic selector.
    """
    if not use_mip or not candidates:
        return {}, "disabled"
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError:
        return {}, "unavailable"

    unique: list[Any] = []
    pair_indices: dict[tuple[int, int], list[int]] = {}
    for candidate in candidates:
        pair = tuple(sorted((int(candidate.start_landmark), int(candidate.end_landmark))))
        index = len(unique)
        unique.append(candidate)
        pair_indices.setdefault(pair, []).append(index)
    n_edges = len(unique)
    n_nodes = len(specifications)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def degree_row(node: int) -> np.ndarray:
        row = np.zeros(n_edges, dtype=float)
        for index, candidate in enumerate(unique):
            if node in (candidate.start_landmark, candidate.end_landmark):
                row[index] = 1.0
        return row

    for node, specification in enumerate(specifications):
        kind = specification["kind"]
        if kind == "endpoint":
            target = 1.0
        elif kind == "junction":
            target = float(max(1, specification["region"].branch_count))
        elif kind == "cycle_anchor":
            target = 2.0
        else:
            continue
        rows.append(degree_row(node))
        lower.append(target)
        upper.append(target)

    # Never select two alternative paths with the same logical endpoints.
    for indices in pair_indices.values():
        if len(indices) > 1:
            row = np.zeros(n_edges, dtype=float)
            row[indices] = 1.0
            rows.append(row)
            lower.append(0.0)
            upper.append(1.0)

    candidate_pairs = [
        tuple(sorted((int(candidate.start_landmark), int(candidate.end_landmark))))
        for candidate in unique
    ]
    component_count = _components(candidate_pairs, n_nodes)
    target_edges = n_nodes - component_count + max(0, int(requested_cycles))
    rows.append(np.ones(n_edges, dtype=float))
    lower.append(float(target_edges))
    upper.append(float(target_edges))

    matrix = lil_matrix((len(rows), n_edges), dtype=float)
    for row_index, row in enumerate(rows):
        matrix[row_index] = row
    objective = np.asarray(
        [
            float(candidate.total_cost)
            - 0.05 * float(getattr(candidate, "electrical_support", 0.0))
            - 0.05 * float(getattr(candidate, "current_support", 0.0))
            for candidate in unique
        ],
        dtype=float,
    )
    result = milp(
        c=objective,
        integrality=np.ones(n_edges, dtype=int),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix.tocsr(), np.asarray(lower), np.asarray(upper)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        return {}, f"infeasible:{getattr(result, 'message', 'unknown')}"
    selected: dict[tuple[int, int], Any] = {}
    for index, value in enumerate(result.x):
        if value >= 0.5:
            candidate = unique[index]
            selected[tuple(sorted((candidate.start_landmark, candidate.end_landmark)))] = candidate
    return selected, "optimal"


__all__ = ["select_backbone_mip"]
