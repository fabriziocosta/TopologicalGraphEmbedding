"""Coverage-driven rib proposals for higher-dimensional structure."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._curves import _fit_curve
from ._local_geometry import _tangent_inconsistency

Array = np.ndarray


@dataclass
class RibCandidate:
    """A provisional coverage element and its local utility diagnostics."""

    points: Array
    seed_index: int
    source_element: int
    candidate_type: str
    coverage_gain: float
    support: float
    stability: float = 1.0
    utility: float = 0.0
    spline: Any = None
    subsample_support: float = float("nan")
    resolution_support: float = float("nan")
    length: float = 0.0
    seed_sources: tuple[str, ...] = ("residual",)
    descendant_original_indices: Array | None = None


def _orthogonal_direction(tangent: Array, residuals: Array) -> Array:
    tangent = np.asarray(tangent, dtype=float).copy()
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals - residuals @ tangent[:, None] * tangent[None, :]
    if len(residuals):
        _, _, vectors = np.linalg.svd(residuals, full_matrices=False)
        direction = vectors[0]
    else:
        direction = np.zeros_like(tangent)
        direction[0] = 1.0
    direction = direction - float(direction @ tangent) * tangent
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.zeros_like(tangent)
        direction[np.argmin(np.abs(tangent))] = 1.0
        direction -= float(direction @ tangent) * tangent
        norm = float(np.linalg.norm(direction))
    return direction / max(norm, 1e-12)


def _routing_edge_cost(model: Any, left: int, right: int) -> float:
    """Use the same length/tangent/density terms as backbone routing."""
    graph = model.routing_graph_
    edge = graph.key(left, right)
    reference = max(float(model.local_scale_), 1e-12)
    length = float(graph.edges[edge]) / reference
    tangent = _tangent_inconsistency(
        model.local_tangents_[left], model.local_tangents_[right]
    )
    density = 1.0 / max(float(graph.edge_density.get(edge, 0.0)), 1e-6)
    return max(
        model.routing_length_weight * length
        + model.routing_tangent_weight * tangent
        + model.routing_density_weight * density,
        1e-8,
    )


def _route_nodes(
    model: Any,
    start: int,
    target: int,
    *,
    alignment: Array | None = None,
) -> list[int]:
    """Route a rib between anchors using the original observation graph."""
    graph = getattr(model, "routing_graph_", None)
    if graph is None or start == target:
        return [int(start), int(target)] if start != target else [int(start)]
    queue: list[tuple[float, int, tuple[int, ...]]] = [(0.0, int(start), (int(start),))]
    best = {int(start): 0.0}
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == target:
            return list(path)
        if cost > best.get(node, np.inf) + 1e-12:
            continue
        for neighbour in sorted(graph.adjacency[node]):
            if neighbour in path:
                continue
            edge_cost = _routing_edge_cost(model, node, neighbour)
            if alignment is not None:
                displacement = graph.points[neighbour] - graph.points[node]
                norm = float(np.linalg.norm(displacement))
                if norm > 1e-12:
                    edge_cost += 0.75 * (1.0 - abs(float(
                        np.asarray(alignment) @ (displacement / norm)
                    )))
            candidate = cost + edge_cost
            if candidate < best.get(neighbour, np.inf):
                best[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour, path + (neighbour,)))
    return [int(start), int(target)]


def _nearest_graph_node(model: Any, point: Array) -> int:
    graph = getattr(model, "routing_graph_", None)
    if graph is None:
        return 0
    return int(np.argmin(np.sum((graph.points - point) ** 2, axis=1)))


def propose_ribs(
    model: Any,
    points: Array,
    result: Any,
    *,
    max_candidates: int,
    candidate_type: str = "transverse",
) -> list[RibCandidate]:
    """Generate sparse local parallel/transverse rib proposals.

    Candidate construction is deliberately local.  A proposal follows the
    observed manifold support in a residual direction instead of inventing a
    curve in empty ambient space.
    """
    errors = np.asarray(result.unexplained_residual, dtype=float) / model.scale_
    norms = np.linalg.norm(errors, axis=1)
    if not len(norms) or not np.any(norms > 1e-12):
        return []
    order = np.argsort(norms)[::-1]
    spacing = model.coverage_candidate_spacing
    if spacing is None:
        spacing = max(4.0 * model.local_scale_, 1e-8)
    chosen: list[int] = []
    for index in order:
        if all(np.linalg.norm(points[index] - points[other]) >= spacing for other in chosen):
            chosen.append(int(index))
        if len(chosen) >= max_candidates:
            break

    proposals: list[RibCandidate] = []
    for seed in chosen:
        source = int(result.route_id[seed])
        if source < 0:
            continue
        center = points[seed]
        distances = np.linalg.norm(points - center, axis=1)
        radius = max(6.0 * model.local_scale_, float(np.quantile(distances, 0.12)))
        local = np.flatnonzero(distances <= radius)
        if len(local) < 5:
            continue
        tangent = np.asarray(result.tangent[seed], dtype=float)
        direction = _orthogonal_direction(tangent, errors[local])
        signed = errors[local] @ direction
        threshold = max(float(np.quantile(np.abs(signed), 0.55)), 1e-8)
        support_indices = local[np.abs(signed) >= threshold]
        if len(support_indices) < 5:
            continue
        # Prefer the side with the stronger unresolved support for a parallel
        # highway; a transverse proposal is retained when both sides are
        # comparably populated.
        positive = support_indices[signed[np.searchsorted(local, support_indices)] >= 0]
        negative = support_indices[signed[np.searchsorted(local, support_indices)] < 0]
        balanced = bool(
            len(positive)
            and len(negative)
            and abs(len(positive) - len(negative)) <= max(2, len(support_indices) // 5)
        )
        inferred_type = "transverse" if balanced else "parallel"
        if candidate_type == "transverse":
            # Transverse is the conservative default: use the two most
            # separated residual sides even when finite sampling makes one
            # side smaller than the other.
            inferred_type = "transverse"
        elif candidate_type == "parallel":
            inferred_type = "parallel"
        if inferred_type == "transverse":
            positive_anchor = int(local[np.argmax(signed)])
            negative_anchor = int(local[np.argmin(signed)])
            start_node = _nearest_graph_node(model, points[positive_anchor])
            end_node = _nearest_graph_node(model, points[negative_anchor])
            nodes = _route_nodes(model, start_node, end_node, alignment=direction)
            side = support_indices
        else:
            side = positive if len(positive) >= len(negative) else negative
            if len(side) < 5:
                continue
            longitudinal = (points[side] - center) @ tangent
            ordered = side[np.argsort(longitudinal, kind="mergesort")]
            start_node = _nearest_graph_node(model, points[ordered[0]])
            end_node = _nearest_graph_node(model, points[ordered[-1]])
            nodes = _route_nodes(model, start_node, end_node, alignment=tangent)
        if len(nodes) < 2:
            continue
        graph = getattr(model, "routing_graph_", None)
        if graph is None:
            support = points[side]
            support = support[np.argsort((support - center) @ tangent, kind="mergesort")]
        else:
            support = graph.points[np.asarray(nodes, dtype=int)]
        support_scaled = np.asarray(support, dtype=float)
        spline = _fit_curve(
            support_scaled,
            closed=False,
            smoothing=model.spline_smoothing,
            sample_count=max(64, len(support_scaled) * 2),
        )
        _, _, distances2 = spline.project(points)
        local_error = np.sum(errors[local] ** 2, axis=1)
        local_gain = np.maximum(
            local_error - np.asarray(distances2[local], dtype=float), 0.0
        )
        gain = float(np.sum(local_gain))
        proposals.append(
            RibCandidate(
                points=support_scaled,
                seed_index=seed,
                source_element=source,
                candidate_type=inferred_type,
                coverage_gain=gain,
                support=float(len(side) / max(len(local), 1)),
                spline=spline,
            )
        )
    return proposals


def select_ribs(
    candidates: list[RibCandidate],
    *,
    max_ribs: int | None,
    min_gain: float,
    length_penalty: float,
    rib_penalty: float,
    junction_penalty: float,
    selection: str = "greedy",
    resolution_weight: float = 0.0,
) -> list[RibCandidate]:
    """Select non-overlapping useful ribs using the default greedy policy."""
    for candidate in candidates:
        length = float(np.sum(np.linalg.norm(np.diff(candidate.points, axis=0), axis=1)))
        candidate.utility = (
            candidate.coverage_gain
            + candidate.stability
            + (resolution_weight * candidate.resolution_support if np.isfinite(candidate.resolution_support) else 0.0)
            - length_penalty * length
            - rib_penalty
            - junction_penalty * max(0, len(candidate.points) - 2)
        )
    candidates.sort(key=lambda item: (-item.utility, item.seed_index))
    if selection == "mip" and candidates:
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
        except ImportError:
            selection = "greedy"
        else:
            limit = len(candidates) if max_ribs is None else max(0, int(max_ribs))
            if limit == 0:
                return []
            objective = -np.asarray([candidate.utility for candidate in candidates], dtype=float)
            result = milp(
                c=objective,
                integrality=np.ones(len(candidates), dtype=int),
                bounds=Bounds(0.0, 1.0),
                constraints=LinearConstraint(
                    np.ones((1, len(candidates))), -np.inf, float(limit)
                ),
            )
            if result.success and result.x is not None:
                return [
                    candidate for candidate, value in zip(candidates, result.x)
                    if value >= 0.5 and candidate.utility > min_gain
                ]
    selected: list[RibCandidate] = []
    for candidate in candidates:
        if candidate.utility <= min_gain:
            continue
        if any(np.linalg.norm(candidate.points[0] - other.points[0]) < 1e-8 for other in selected):
            continue
        selected.append(candidate)
        if max_ribs is not None and len(selected) >= max_ribs:
            break
    return selected


__all__ = ["RibCandidate", "propose_ribs", "select_ribs"]


def propose_hierarchy_ribs(model, points, result, *, max_candidates):
    """Propose unresolved fine representative paths, independently at each level."""
    from scipy.spatial import cKDTree

    from ._multiresolution import sparse_corridor_path
    from ._stability import match_paths_across_levels
    if len(model.levels_) < 2:
        return []
    errors2 = np.sum((result.unexplained_residual / model.scale_) ** 2, axis=1)
    threshold = max(float(np.quantile(errors2, 0.6)), 1e-16)
    by_level = []
    last = max(1, model.selected_backbone_level_)
    for level in model.levels_[:last + 1]:
        error = errors2[level.representative_indices]
        high = np.flatnonzero(error >= threshold)
        paths = []
        if len(high) >= 3:
            tree = cKDTree(level.points[high])
            spacing = tree.query(level.points[high], k=min(3, len(high)))[0][:, -1]
            covered = set()
            for seed in high[np.argsort(-error[high], kind="stable")]:
                if int(seed) in covered:
                    continue
                seed_position = int(np.searchsorted(high, seed))
                radius = max(4 * float(spacing[seed_position]), 1e-10)
                local = high[tree.query_ball_point(level.points[seed], radius)]
                if len(local) < 3:
                    continue
                local = np.sort(local)
                start = int(np.searchsorted(local, seed))
                end = int(np.argmax(np.linalg.norm(level.points[local] - level.points[seed], axis=1)))
                path = sparse_corridor_path(level.points[local], start, end,
                                             min(4, model.hierarchy_local_neighbors))
                if len(path) < 3:
                    continue
                ids = local[path]
                support = level.points[ids]
                steps = np.linalg.norm(np.diff(support, axis=0), axis=1)
                if np.any(steps > radius / 2):
                    continue
                covered.update(map(int, local))
                descendants = np.unique(np.concatenate([level.descendant_indices[i] for i in ids]))
                paths.append((support, int(level.representative_indices[seed]), descendants))
                if len(paths) >= max_candidates:
                    break
        by_level.append(paths)
    candidates = []
    for index, paths in enumerate(by_level):
        for support, seed, descendants in paths:
            tested = list(range(max(0, index - 1), min(len(by_level), index + 2)))
            hits = []
            for other in tested:
                tolerance = max(model.levels_[other].scale, model.levels_[index].scale,
                                float(np.median(np.linalg.norm(np.diff(support, axis=0), axis=1)))) * 3
                hits.append(bool(match_paths_across_levels([support], [p[0] for p in by_level[other]],
                    tolerance=tolerance, reference_descendants=[descendants],
                    candidate_descendants=[p[2] for p in by_level[other]])[0]))
            resolution = float(np.mean(hits)) if len(tested) > 1 else float("nan")
            if sum(hits) < 2:
                continue
            spline = _fit_curve(support, closed=False, smoothing=model.spline_smoothing,
                                sample_count=max(64, 2 * len(support)))
            _, _, distance2 = spline.project(points[descendants])
            gain = float(np.maximum(errors2[descendants] - distance2, 0).sum())
            if gain <= 0:
                continue
            candidate = RibCandidate(support, seed, int(result.route_id[seed]), "hierarchy", gain,
                                      len(descendants) / len(points), spline=spline)
            candidate.resolution_support = resolution
            candidate.seed_sources = ("hierarchy",)
            candidate.descendant_original_indices = descendants
            candidates.append(candidate)
    return candidates


def prepare_rib_candidates(model, points, result, residual_candidates):
    """Combine sources, score measured evidence, and deduplicate corridors."""
    from ._stability import corridor_recurs, path_resolution_support, route_distance
    if len(model.levels_) < 2:
        return list(residual_candidates) if model.rib_seed_source != "hierarchy" else []
    candidates = list(residual_candidates) if model.rib_seed_source != "hierarchy" else []
    if model.use_multiresolution and model.rib_seed_source in {"hierarchy", "both"}:
        candidates.extend(propose_hierarchy_ribs(model, points, result,
            max_candidates=model.coverage_max_candidates_per_iteration))
    for candidate in candidates:
        candidate.length = float(np.linalg.norm(np.diff(candidate.points, axis=0), axis=1).sum())
        if len(model.levels_) > 1 and not np.isfinite(candidate.resolution_support):
            candidate.resolution_support = path_resolution_support(model, candidate.points)
        if model._structural_subsamples_:
            hits = [corridor_recurs(entry, candidate.points) for entry in model._structural_subsamples_]
            candidate.subsample_support = float(np.mean(hits))
            candidate.stability = candidate.subsample_support
    unique = []
    for candidate in sorted(candidates, key=lambda c: -c.coverage_gain):
        match = next((other for other in unique if route_distance(candidate.points, other.points)
                      <= max(model.local_scale_, 1e-10)), None)
        if match is None:
            unique.append(candidate)
        else:
            match.seed_sources = tuple(sorted(set(match.seed_sources + candidate.seed_sources)))
    if len(model.levels_) > 1:
        unique = [candidate for candidate in unique if candidate.coverage_gain > model.coverage_min_gain
                  and (not np.isfinite(candidate.subsample_support) or candidate.subsample_support >= model.rib_min_support)]
    return unique[:model.coverage_max_candidates_per_iteration]
