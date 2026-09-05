"""MILK-inspired geometric compression, ancestry, and resolution evidence.

Compression is independent of topology. Coordinates use the estimator metric;
all representative and descendant identifiers refer to original input rows.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


@dataclass
class RepresentativeLevel:
    points: np.ndarray
    representative_indices: np.ndarray
    parent_indices: np.ndarray | None
    descendant_indices: list[np.ndarray]
    scale: float
    grouping_threshold: float = 0.0

    @property
    def descendant_original_indices(self):
        return self.descendant_indices


def validate_hierarchy_parameters(model):
    for name, minimum in (
        ("hierarchy_max_levels", 0),
        ("hierarchy_target_size", 3),
        ("hierarchy_local_neighbors", 2),
        ("backbone_max_representatives", 3),
        ("backbone_consensus_levels", 1),
    ):
        value = getattr(model, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in ("hierarchy_distance_quantile", "hierarchy_min_reduction"):
        value = getattr(model, name)
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    for name in ("route_resolution_weight", "rib_resolution_weight"):
        value = getattr(model, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if model.representative_method not in {"medoid", "approx_medoid"}:
        raise ValueError("representative_method must be 'medoid' or 'approx_medoid'")
    if model.rib_seed_source not in {"residual", "hierarchy", "both"}:
        raise ValueError("rib_seed_source must be 'residual', 'hierarchy', or 'both'")
    level = model.backbone_level
    if level != "auto" and (
        isinstance(level, bool) or not isinstance(level, (int, np.integer)) or level < 0
    ):
        raise ValueError("backbone_level must be 'auto' or a non-negative integer")


def medoid(points, members, original_indices, method="medoid", random_state=0):
    """Blocked exact distance objective; bounded candidates in approximate mode."""
    members = np.asarray(members, dtype=int)
    if np.all(points[members] == points[members[0]]):
        return int(members[np.argmin(original_indices[members])])
    candidates = members
    objective_points = points[members]
    objective_counts = np.ones(len(members))
    if len(members) > 64:
        ordered = members[np.argsort(original_indices[members], kind="stable")]
        objective_points, first, objective_counts = np.unique(
            points[ordered], axis=0, return_index=True, return_counts=True
        )
        candidates = ordered[first]
    if method == "approx_medoid" and len(candidates) > 64:
        candidates = np.random.default_rng(random_state).choice(
            candidates, 64, replace=False
        )
    candidates = candidates[np.argsort(original_indices[candidates], kind="stable")]
    best, best_cost = int(candidates[0]), np.inf
    for start in range(0, len(candidates), 128):
        block = candidates[start : start + 128]
        costs = np.zeros(len(block))
        for offset in range(0, len(objective_points), 1024):
            costs += (
                cdist(points[block], objective_points[offset : offset + 1024])
                @ objective_counts[offset : offset + 1024]
            )
        index = int(np.argmin(costs))
        if costs[index] < best_cost:
            best, best_cost = int(block[index]), float(costs[index])
    return best


def build_hierarchy(
    points,
    *,
    max_levels=8,
    target_size=1000,
    min_reduction=0.15,
    distance_quantile=0.1,
    local_neighbors=10,
    representative_method="medoid",
    random_state=0,
):
    points = np.asarray(points, dtype=float)
    levels = [
        RepresentativeLevel(
            points,
            np.arange(len(points)),
            None,
            [np.array([i], dtype=int) for i in range(len(points))],
            0.0,
        )
    ]
    for depth in range(max_levels):
        current = levels[-1]
        n = len(current.points)
        if n <= target_size:
            break
        k = min(local_neighbors, n - 1)
        distances, neighbors = cKDTree(current.points).query(current.points, k=k + 1)
        positive = distances[distances > 0]
        floor = (
            max(float(np.median(positive)) * 1e-12, np.finfo(float).tiny)
            if positive.size
            else 1.0
        )
        scales = np.maximum(distances[:, -1], floor)
        normalized = distances[:, 1:] / np.sqrt(
            scales[:, None] * scales[neighbors[:, 1:]]
        )
        threshold = float(np.quantile(normalized, distance_quantile))
        _, inverse, counts = np.unique(
            current.points, axis=0, return_inverse=True, return_counts=True
        )
        duplicates = {}
        if np.any(counts > 1):
            ordered = np.argsort(inverse, kind="stable")
            for group in np.split(ordered, np.cumsum(counts)[:-1]):
                if len(group) > 1:
                    duplicates[int(inverse[group[0]])] = group
        assigned = np.full(n, -1, dtype=int)
        groups = []
        # Sparse neighborhoods bound group size and avoid density-weighted flooding.
        for seed in np.argsort(current.representative_indices, kind="stable"):
            if assigned[seed] >= 0:
                continue
            row = neighbors[seed]
            admissible = (
                distances[seed] / np.sqrt(scales[seed] * scales[row]) <= threshold
            )
            group = np.unique(np.r_[seed, row[admissible & (assigned[row] < 0)]])
            if duplicates:
                extra = [
                    duplicates[int(label)]
                    for label in np.unique(inverse[group])
                    if int(label) in duplicates
                ]
                if extra:
                    group = np.unique(np.concatenate([group, *extra]))
                    group = group[assigned[group] < 0]
            assigned[group] = len(groups)
            groups.append(group)
        reduction = 1 - len(groups) / n
        if len(groups) == n or reduction < min_reduction:
            break
        selected = np.array(
            [
                medoid(
                    current.points,
                    group,
                    current.representative_indices,
                    representative_method,
                    random_state + depth,
                )
                for group in groups
            ]
        )
        descendants = [
            np.sort(np.concatenate([current.descendant_indices[i] for i in group]))
            for group in groups
        ]
        current.parent_indices = assigned
        levels.append(
            RepresentativeLevel(
                current.points[selected],
                current.representative_indices[selected],
                None,
                descendants,
                float(np.quantile(scales, distance_quantile)),
                grouping_threshold=threshold,
            )
        )
    return levels


def initialize_hierarchy(model, points):
    validate_hierarchy_parameters(model)
    model.hierarchy_levels_ = build_hierarchy(
        points,
        max_levels=model.hierarchy_max_levels if model.use_multiresolution else 0,
        target_size=model.hierarchy_target_size,
        min_reduction=model.hierarchy_min_reduction,
        distance_quantile=model.hierarchy_distance_quantile,
        local_neighbors=model.hierarchy_local_neighbors,
        representative_method=model.representative_method,
        random_state=model.random_state,
    )
    model.levels_ = model.hierarchy_levels_
    model.hierarchy_sizes_ = [len(level.points) for level in model.levels_]
    model.hierarchy_scales_ = [level.scale for level in model.levels_]
    model.hierarchy_descendants_ = [level.descendant_indices for level in model.levels_]
    model.topology_by_level_ = {}
    model.selected_backbone_level_ = 0
    model.selected_backbone_level_band_ = [0]
    model.route_refinement_failures_ = []
    model.route_descendant_support_ = []
    model.route_resolution_support_ = np.empty(0)
    model.rib_resolution_support_ = []
    model.cycle_resolution_support_ = np.empty(0)
    model.junction_resolution_support_ = np.empty(0)
    model._structural_subsamples_ = []
    model.mip_candidate_count_ = 0
    model.hierarchy_summary_ = {
        "n_levels": len(model.levels_),
        "level_sizes": model.hierarchy_sizes_,
        "selected_backbone_level": 0,
        "selection_reason": "single_level",
        "stable_cycle_count_by_level": [],
        "stable_junction_count_by_level": [],
    }
    if model.backbone_level != "auto" and model.backbone_level >= len(model.levels_):
        raise ValueError("backbone_level is outside the constructed hierarchy")


def evaluate_level(model, level, index):
    """Run existing topology primitives without fitting curves or selecting a MIP."""
    from copy import copy

    from ._topology import (
        PersistentCycle,
        _approximate_cycle_representatives,
        _estimate_persistence,
        _is_nearly_linear,
        _kmeans,
        _local_scale,
        _normalize_persistence_diagram,
    )

    points = level.points
    diagram, backend = _estimate_persistence(
        points, max_points=model.persistence_max_points, random_state=model.random_state
    )
    scale = _local_scale(points)
    normalized = _normalize_persistence_diagram(diagram, scale)
    threshold = (
        3.0 if model.persistence_threshold is None else model.persistence_threshold
    )
    bars = normalized[
        np.isfinite(normalized).all(axis=1)
        & (np.diff(normalized, axis=1).ravel() >= threshold)
    ]
    if not model.detect_cycles:
        bars = bars[:0]
    bars = bars[np.argsort(-(bars[:, 1] - bars[:, 0]), kind="stable")][
        : model.max_cycles
    ]
    prototypes = _kmeans(
        points, min(model.n_centroids, len(points)), model.random_state
    )
    probe = copy(model)
    probe.levels_ = [level]
    probe._evaluating_hierarchy_ = True
    probe.local_scale_ = scale
    probe.persistence_threshold_ = threshold
    probe.persistent_cycle_count_ = len(bars)
    probe.requested_cycle_count_ = len(bars)
    probe.persistent_cycles_ = [
        PersistentCycle(float(b), float(d), float(d - b)) for b, d in bars
    ]
    original_metric = points * model.scale_ + model.mean_
    probe.linear_structure_ = _is_nearly_linear(
        prototypes * model.scale_ + model.mean_, model.linear_structure_tolerance
    )
    probe.linear_center_ = None
    probe.linear_direction_ = None
    probe.hypercube_dimension_ = None
    probe.face_cycle_count_ = 0
    if probe.linear_structure_:
        center = original_metric.mean(axis=0)
        _, _, components = np.linalg.svd(original_metric - center, full_matrices=False)
        probe.linear_center_ = (center - model.mean_) / model.scale_
        direction = components[0] / model.scale_
        probe.linear_direction_ = direction / max(np.linalg.norm(direction), 1e-12)
        probe.requested_cycle_count_ = 0
    probe._topological_backbone(points, prototypes, topology_only=True)
    graph = probe.routing_graph_
    representatives = _approximate_cycle_representatives(graph, len(bars))
    for cycle, representative in zip(probe.persistent_cycles_, representatives):
        cycle.representative = representative
    return {
        "status": "tested",
        "cycle_count": probe.requested_cycle_count_,
        "persistent_cycles": probe.persistent_cycles_,
        "junctions": probe.junction_regions_,
        "endpoints": probe.endpoint_regions_,
        "graph": graph,
        "scale": scale,
        "persistence_diagram": diagram,
        "normalized_persistence_diagram": normalized,
        "persistence_backend": backend,
        "level": index,
        "n_points": len(points),
    }


def infer_hierarchy_topology(model):
    from ._stability import match_junctions_across_levels

    if len(model.levels_) < 2:
        return
    minimum = max(8, model.topology_neighbors_ + 1)
    # Include one finer guard level above the size budget, but never full-data
    # topology merely to validate a compressed hierarchy.
    eligible = [
        i for i, level in enumerate(model.levels_) if len(level.points) >= minimum
    ]
    bounded = [
        i
        for i in eligible
        if len(model.levels_[i].points) <= model.backbone_max_representatives
    ]
    chosen = set(bounded)
    if bounded:
        finer = [i for i in eligible if i < min(bounded)]
        if finer:
            chosen.add(max(finer))
    elif eligible:
        chosen.add(max(eligible))
    if model.backbone_level != "auto":
        chosen.add(model.backbone_level)
    for i, level in enumerate(model.levels_):
        if i not in chosen:
            model.topology_by_level_[i] = {
                "status": "skipped",
                "reason": "size_or_neighborhood_limit",
            }
            continue
        try:
            model.topology_by_level_[i] = evaluate_level(model, level, i)
        except (ValueError, np.linalg.LinAlgError) as error:
            model.topology_by_level_[i] = {"status": "failed", "reason": str(error)}
    tested = [
        i for i in sorted(chosen) if model.topology_by_level_[i]["status"] == "tested"
    ]
    if not tested:
        raise ValueError(
            "No hierarchy level has sufficient representatives for topology; increase hierarchy_target_size"
        )
    bands = []
    for i in tested:
        band = [i]
        for j in range(i - 1, max(-1, i - model.backbone_consensus_levels), -1):
            if j not in tested:
                break
            a, b = model.topology_by_level_[i], model.topology_by_level_[j]
            matches = match_junctions_across_levels(
                a["junctions"],
                b["junctions"],
                tolerance=4 * max(a["scale"], b["scale"]),
            )
            if (
                a["cycle_count"] != b["cycle_count"]
                or len(a["junctions"]) != len(b["junctions"])
                or not np.all(matches)
            ):
                break
            band.append(j)
        if len(band) >= min(2, model.backbone_consensus_levels):
            bands.append(band)
    # A collapsed coarsest plateau must not override richer finer evidence.
    richest = max(model.topology_by_level_[i]["cycle_count"] for i in tested)
    bands = [
        band
        for band in bands
        if model.topology_by_level_[band[0]]["cycle_count"] == richest
    ]
    guarded = []
    for band in bands:
        entry = model.topology_by_level_[band[0]]
        finer = [i for i in tested if i < min(band)]
        if finer:
            guard = model.topology_by_level_[max(finer)]
            matches = match_junctions_across_levels(
                guard["junctions"],
                entry["junctions"],
                tolerance=4 * max(guard["scale"], entry["scale"]),
            )
            if not np.all(matches):
                continue
        guarded.append(band)
    bands = guarded
    reason = "stable_band"
    within = [
        band
        for band in bands
        if len(model.levels_[band[0]].points) <= model.backbone_max_representatives
    ]
    if model.backbone_level != "auto":
        if model.backbone_level not in tested:
            raise ValueError("The explicit backbone_level could not be evaluated")
        selected, band, reason = (
            model.backbone_level,
            [model.backbone_level],
            "explicit",
        )
    elif within or bands:
        band = max(within or bands, key=lambda b: b[0])
        selected = band[0]
    else:
        selected, band, reason = min(tested), [min(tested)], "insufficient_consensus"
    model.selected_backbone_level_ = selected
    model.selected_backbone_level_band_ = sorted(band)
    if len(model.levels_[selected].points) > model.backbone_max_representatives:
        warnings.warn(
            "Preserving topology requires exceeding backbone_max_representatives.",
            RuntimeWarning,
            stacklevel=3,
        )
        reason += ":size_budget_exceeded"
    model.hierarchy_summary_.update(
        selected_backbone_level=selected,
        selection_reason=reason,
        stable_cycle_count_by_level=[
            model.topology_by_level_[i].get("cycle_count")
            for i in range(len(model.levels_))
        ],
        stable_junction_count_by_level=[
            len(model.topology_by_level_[i]["junctions"]) if i in tested else None
            for i in range(len(model.levels_))
        ],
    )


def sparse_corridor_path(
    points, start, end, neighbors=10, *, waypoints=None, model=None
):
    """Local kNN routing with ordered anchors and bounded-memory Dijkstra."""
    import heapq

    if start == end and waypoints is None:
        return np.array([start], dtype=int)
    distances, indices = cKDTree(points).query(
        points, k=min(neighbors + 1, len(points))
    )
    if distances.ndim == 1:
        return np.array([], dtype=int)
    rows = np.repeat(np.arange(len(points)), distances.shape[1] - 1)
    columns = indices[:, 1:].ravel()
    costs = np.maximum(distances[:, 1:].ravel(), 1e-12)
    if model is not None:
        scales = np.maximum(distances[:, -1], 1e-12)
        costs = (
            model.routing_length_weight * costs / max(float(np.median(scales)), 1e-12)
        )
        costs += (
            model.routing_density_weight
            * distances[:, 1:].ravel()
            / np.sqrt(scales[rows] * scales[columns])
        )
        if model.use_local_pca and model.routing_tangent_weight:
            tangents = []
            for ids in indices:
                local = points[ids] - points[ids].mean(axis=0)
                _, _, components = np.linalg.svd(local, full_matrices=False)
                tangents.append(components[0])
            tangents = np.asarray(tangents)
            costs += model.routing_tangent_weight * (
                1 - np.abs(np.sum(tangents[rows] * tangents[columns], axis=1))
            )
    graph = csr_matrix(
        (np.maximum(costs, 1e-12), (rows, columns)), shape=(len(points), len(points))
    )
    graph = graph.maximum(graph.T)
    anchors = [start, end] if waypoints is None else list(waypoints)
    path = [int(anchors[0])]
    for source, target in itertools.pairwise(anchors):
        if source == target:
            continue
        previous = {}
        best = {int(source): 0.0}
        queue = [(0.0, int(source))]
        while queue:
            cost, vertex = heapq.heappop(queue)
            if vertex == target:
                break
            if cost > best[vertex]:
                continue
            for offset in range(graph.indptr[vertex], graph.indptr[vertex + 1]):
                neighbor = int(graph.indices[offset])
                proposal = cost + graph.data[offset]
                if proposal < best.get(neighbor, np.inf):
                    best[neighbor] = proposal
                    previous[neighbor] = vertex
                    heapq.heappush(queue, (proposal, neighbor))
        if target not in best:
            return np.array([], dtype=int)
        segment = [int(target)]
        while segment[-1] != source:
            segment.append(previous[segment[-1]])
        path.extend(segment[-2::-1])
    return np.asarray(path, dtype=int)


def refine_backbone(model, graph, paths):
    """Expand local descendant corridors; retain the logical graph unchanged."""
    from ._stability import path_resolution_support

    level_index = model.selected_backbone_level_
    model.coarse_backbone_graph_ = graph.copy()
    model.coarse_backbone_paths_ = {
        key: model.routing_graph_.points[path.vertices].copy()
        for key, path in (paths or {}).items()
    }
    if not paths or level_index == 0:
        return
    for key, path in paths.items():
        vertices = np.asarray(path.vertices, dtype=int)
        support = model.levels_[level_index].points[vertices]
        descendant_ids = np.unique(
            np.concatenate(
                [model.levels_[level_index].descendant_indices[v] for v in vertices]
            )
        )
        path.descendant_original_indices = descendant_ids
        path.resolution_support = path_resolution_support(model, support)
        for depth in range(level_index, 0, -1):
            coarse, fine = model.levels_[depth], model.levels_[depth - 1]
            fine_ids = np.flatnonzero(np.isin(fine.parent_indices, vertices))
            # Shared endpoints are original observations retained at every level.
            anchors = coarse.representative_indices[vertices]
            positions = {
                int(original): i
                for i, original in enumerate(fine.representative_indices[fine_ids])
            }
            local_path = sparse_corridor_path(
                fine.points[fine_ids],
                positions[int(anchors[0])],
                positions[int(anchors[-1])],
                model.hierarchy_local_neighbors,
                waypoints=[positions[int(anchor)] for anchor in anchors],
                model=model,
            )
            if len(local_path) < 2:
                near = cKDTree(fine.points).query(
                    support, k=min(model.hierarchy_local_neighbors, len(fine.points))
                )[1]
                fine_ids = np.unique(np.r_[fine_ids, np.asarray(near).ravel()])
                positions = {
                    int(original): i
                    for i, original in enumerate(fine.representative_indices[fine_ids])
                }
                local_path = sparse_corridor_path(
                    fine.points[fine_ids],
                    positions[int(anchors[0])],
                    positions[int(anchors[-1])],
                    2 * model.hierarchy_local_neighbors,
                    waypoints=[positions[int(anchor)] for anchor in anchors],
                    model=model,
                )
            if len(local_path) < 2:
                model.route_refinement_failures_.append(
                    {"edge": key, "level": depth - 1}
                )
                break
            vertices = fine_ids[local_path]
            support = fine.points[vertices]
        path.support_points = support.copy()
        path.length = float(np.linalg.norm(np.diff(support, axis=0), axis=1).sum())
        graph.edges[key] = path.length


def finish_hierarchy_diagnostics(model):
    from ._stability import (
        match_cycles_across_levels,
        match_junctions_across_levels,
        path_resolution_support,
    )

    if not model.topology_by_level_:
        model.topology_by_level_[0] = {
            "status": "tested",
            "cycle_count": model.persistent_cycle_count_,
            "persistent_cycles": model.persistent_cycles_,
            "junctions": model.junction_regions_,
            "endpoints": model.endpoint_regions_,
            "scale": model.local_scale_,
            "n_points": model.backbone_input_size_,
            "graph": model.routing_graph_,
            "persistence_backend": model.persistence_backend_,
        }
        model.hierarchy_summary_.update(
            stable_cycle_count_by_level=[model.persistent_cycle_count_],
            stable_junction_count_by_level=[len(model.junction_regions_)],
        )
    tested = [
        entry
        for entry in model.topology_by_level_.values()
        if entry["status"] == "tested"
    ]
    model.topology_input_sizes_ = {
        i: entry["n_points"]
        for i, entry in model.topology_by_level_.items()
        if entry["status"] == "tested"
    }
    cycles = model.persistent_cycles_
    junctions = getattr(model, "junction_regions_", [])
    model.cycle_resolution_support_ = np.full(len(cycles), np.nan)
    model.junction_resolution_support_ = np.full(len(junctions), np.nan)
    if len(tested) > 1:
        model.cycle_resolution_support_ = np.mean(
            [
                match_cycles_across_levels(
                    cycles,
                    entry["persistent_cycles"],
                    tolerance=4 * max(model.local_scale_, entry["scale"]),
                )
                for entry in tested
            ],
            axis=0,
        )
        model.junction_resolution_support_ = np.mean(
            [
                match_junctions_across_levels(
                    junctions,
                    entry["junctions"],
                    tolerance=4 * max(model.local_scale_, entry["scale"]),
                )
                for entry in tested
            ],
            axis=0,
        )
    for cycle, support in zip(cycles, model.cycle_resolution_support_):
        cycle.resolution_support = float(support)
        cycle.subsample_support = (
            float(cycle.stability_support)
            if model.stability_selection
            else float("nan")
        )
    for junction, support in zip(junctions, model.junction_resolution_support_):
        junction.resolution_support = float(support)
        junction.branch_count_by_level = {}
        locations = []
        for entry in tested:
            matches = [
                region
                for region in entry["junctions"]
                if region.branch_count == junction.branch_count
                and np.linalg.norm(region.center - junction.center)
                <= 4 * max(model.local_scale_, entry["scale"])
            ]
            match = (
                min(
                    matches,
                    key=lambda region: np.linalg.norm(region.center - junction.center),
                )
                if matches
                else None
            )
            junction.branch_count_by_level[entry.get("level", 0)] = (
                match.branch_count if match is not None else None
            )
            if match is not None:
                locations.append(match.center)
        junction.location_dispersion = (
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (np.asarray(locations) - np.mean(locations, axis=0)) ** 2,
                            axis=1,
                        )
                    )
                )
            )
            if len(locations) > 1
            else float("nan")
        )
    model.cycle_subsample_support_ = (
        np.asarray(model.cycle_support_).copy()
        if model.stability_selection
        else np.full(len(cycles), np.nan)
    )
    count = getattr(model, "backbone_element_count_", len(model.routes_))
    model.route_resolution_support_ = np.array(
        [
            path_resolution_support(model, route.samples)
            for route in model.routes_[:count]
        ]
    )
    model.rib_resolution_support_ = np.asarray(
        model.rib_resolution_support_, dtype=float
    )
    consensus = list(getattr(model, "junction_consensus_", []))
    if len(consensus) != len(junctions):
        consensus = [
            {"center": junction.center.copy(), "branch_count": junction.branch_count}
            for junction in junctions
        ]
    for record, junction in zip(consensus, junctions):
        record.update(
            resolution_support=junction.resolution_support,
            branch_count_by_level=junction.branch_count_by_level,
            location_dispersion=junction.location_dispersion,
        )
    model.junction_consensus_ = consensus
    level = model.levels_[model.selected_backbone_level_]
    tree = cKDTree(level.points)
    model.route_descendant_support_ = []
    model._route_descendant_indices_ = []
    for route in model.routes_:
        ids = np.unique(tree.query(route.samples)[1])
        descendants = np.unique(
            np.concatenate([level.descendant_indices[i] for i in ids])
        )
        model._route_descendant_indices_.append(descendants)
        length = float(np.linalg.norm(np.diff(route.samples, axis=0), axis=1).sum())
        model.route_descendant_support_.append(
            {
                "n_original_descendants": len(descendants),
                "fraction_of_dataset": len(descendants) / len(model.levels_[0].points),
                "local_density": len(descendants) / max(length, 1e-12),
            }
        )
