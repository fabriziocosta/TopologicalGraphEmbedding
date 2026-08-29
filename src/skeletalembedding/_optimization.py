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
    cycle_class_count: int = 0,
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
    # The first block contains binary path-selection variables.  Two
    # continuous directed-flow variables per candidate provide a compact
    # single-commodity connectivity certificate for the selected landmark
    # graph.
    flow_count = 2 * n_edges
    variable_count = n_edges + flow_count
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def degree_row(node: int) -> np.ndarray:
        row = np.zeros(variable_count, dtype=float)
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
            row = np.zeros(variable_count, dtype=float)
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
    row = np.zeros(variable_count, dtype=float)
    row[:n_edges] = 1.0
    rows.append(row)
    lower.append(float(target_edges))
    upper.append(float(target_edges))

    # Send one unit of flow from the first landmark to every other landmark.
    # A selected undirected path can carry flow in either direction, bounded
    # by the corresponding binary path variable.  This rules out disconnected
    # collections of otherwise degree- and cycle-feasible paths.
    if n_nodes > 1:
        root = 0
        for node in range(n_nodes):
            row = np.zeros(variable_count, dtype=float)
            for index, candidate in enumerate(unique):
                left, right = int(candidate.start_landmark), int(candidate.end_landmark)
                forward = n_edges + 2 * index
                backward = forward + 1
                if node == left:
                    row[forward] += 1.0
                    row[backward] -= 1.0
                elif node == right:
                    row[forward] -= 1.0
                    row[backward] += 1.0
            demand = float(n_nodes - 1) if node == root else -1.0
            rows.append(row)
            lower.append(demand)
            upper.append(demand)
        for index in range(n_edges):
            row = np.zeros(variable_count, dtype=float)
            row[n_edges + 2 * index] = 1.0
            row[n_edges + 2 * index + 1] = 1.0
            row[index] = -float(n_nodes - 1)
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0.0)

    for cycle_class in range(max(0, int(cycle_class_count))):
        row = np.zeros(variable_count, dtype=float)
        tagged = [
            index for index, candidate in enumerate(unique)
            if cycle_class in getattr(candidate, "persistent_cycle_classes", ())
        ]
        if not tagged:
            return {}, f"infeasible:missing-persistent-cycle-class:{cycle_class}"
        row[tagged] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(np.inf)

    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    for row_index, row in enumerate(rows):
        matrix[row_index] = row
    objective = np.asarray(
        [
            float(candidate.total_cost)
            - 0.05 * float(getattr(candidate, "electrical_support", 0.0))
            - 0.05 * float(getattr(candidate, "current_support", 0.0))
            - 0.05 * float(getattr(candidate, "stability_support", 1.0))
            for candidate in unique
        ],
        dtype=float,
    )
    objective = np.concatenate([objective, np.zeros(flow_count, dtype=float)])
    result = milp(
        c=objective,
        integrality=np.concatenate([
            np.ones(n_edges, dtype=int),
            np.zeros(flow_count, dtype=int),
        ]),
        bounds=Bounds(
            np.zeros(variable_count, dtype=float),
            np.concatenate([
                np.ones(n_edges, dtype=float),
                np.full(flow_count, max(0, n_nodes - 1), dtype=float),
            ]),
        ),
        constraints=LinearConstraint(matrix.tocsr(), np.asarray(lower), np.asarray(upper)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        return {}, f"infeasible:{getattr(result, 'message', 'unknown')}"
    selected: dict[tuple[int, int], Any] = {}
    for index, value in enumerate(result.x[:n_edges]):
        if value >= 0.5:
            candidate = unique[index]
            selected[tuple(sorted((candidate.start_landmark, candidate.end_landmark)))] = candidate
    return selected, "optimal"


__all__ = ["select_backbone_mip"]
