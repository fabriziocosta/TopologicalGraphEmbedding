"""Consensus utilities for optional subsampling-based stability selection."""

from __future__ import annotations

import itertools

import numpy as np

Array = np.ndarray


def subsample_indices(n_samples: int, fraction: float, random_state: int) -> np.ndarray:
    """Return a without-replacement stability subsample."""
    size = max(3, int(np.ceil(float(fraction) * n_samples)))
    return np.random.default_rng(random_state).choice(n_samples, size=size, replace=False)


def jitter_points(
    points: Array,
    *,
    jitter: float,
    local_scale: float,
    rng: np.random.Generator,
) -> Array:
    """Perturb a stability sample in local-neighbourhood units."""
    points = np.asarray(points, dtype=float)
    if jitter <= 0.0:
        return points.copy()
    return points + rng.normal(
        0.0,
        max(float(local_scale), 1e-12) * float(jitter),
        size=points.shape,
    )


def _match_by_distance(
    reference: Array,
    candidates: Array,
    tolerance: float,
) -> list[int | None]:
    """Greedily match points one-to-one within a deterministic radius."""
    reference = np.asarray(reference, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    if reference.size == 0:
        return []
    if candidates.size == 0:
        return [None] * len(reference)
    distances = np.linalg.norm(reference[:, None, :] - candidates[None, :, :], axis=2)
    available = set(range(len(candidates)))
    matches: list[int | None] = []
    for row in range(len(reference)):
        choices = [
            (float(distances[row, column]), column)
            for column in sorted(available)
            if distances[row, column] <= tolerance
        ]
        if not choices:
            matches.append(None)
            continue
        _, column = min(choices)
        available.remove(column)
        matches.append(column)
    return matches


def match_cycles(
    reference: Array,
    candidate: Array,
    *,
    tolerance: float,
) -> np.ndarray:
    """Match persistence bars by birth, death, and lifetime."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    hits = np.zeros(len(reference), dtype=bool)
    if not len(reference) or not len(candidate):
        return hits
    scale = max(float(tolerance), 1e-12)
    distances = np.linalg.norm(reference[:, None, :] - candidate[None, :, :], axis=2)
    available = set(range(len(candidate)))
    for row in np.argsort(-reference[:, 1] + reference[:, 0], kind="mergesort"):
        choices = [
            (float(distances[row, column]), column)
            for column in sorted(available)
            if distances[row, column] <= scale
        ]
        if choices:
            _, column = min(choices)
            available.remove(column)
            hits[row] = True
    return hits


def match_regions(
    reference: list[object],
    candidate: list[object],
    *,
    tolerance: float,
    require_branch_count: bool = False,
) -> np.ndarray:
    """Match junction or endpoint regions by center and optional degree."""
    if not reference:
        return np.zeros(0, dtype=bool)
    reference_centers = np.asarray([region.center for region in reference], dtype=float)
    candidate_centers = np.asarray([region.center for region in candidate], dtype=float)
    matches = _match_by_distance(reference_centers, candidate_centers, tolerance)
    result = np.zeros(len(reference), dtype=bool)
    for index, match in enumerate(matches):
        if match is None:
            continue
        if require_branch_count and getattr(reference[index], "branch_count", None) != getattr(
            candidate[match], "branch_count", None
        ):
            continue
        result[index] = True
    return result


def route_distance(reference: Array, candidate: Array) -> float:
    """Return a symmetric sampled Hausdorff-like route distance."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if not len(reference) or not len(candidate):
        return float("inf")
    forward = np.min(
        np.linalg.norm(reference[:, None, :] - candidate[None, :, :], axis=2), axis=1
    )
    backward = np.min(
        np.linalg.norm(candidate[:, None, :] - reference[None, :, :], axis=2), axis=1
    )
    return float(max(np.quantile(forward, 0.9), np.quantile(backward, 0.9)))


def match_routes(
    reference: list[Array],
    candidate: list[Array],
    *,
    tolerance: float,
) -> np.ndarray:
    """Match route support curves by corridor overlap."""
    if not reference:
        return np.zeros(0, dtype=bool)
    if not candidate:
        return np.zeros(len(reference), dtype=bool)
    pairs = sorted(
        (route_distance(left, right), left_index, right_index)
        for left_index, left in enumerate(reference)
        for right_index, right in enumerate(candidate)
    )
    available = set(range(len(candidate)))
    result = np.zeros(len(reference), dtype=bool)
    for distance, left_index, right_index in pairs:
        if left_index >= len(result) or right_index not in available:
            continue
        if result[left_index] or distance > tolerance:
            continue
        result[left_index] = True
        available.remove(right_index)
    return result


def subspace_principal_angle(reference: Array, candidate: Array) -> float:
    """Return the largest principal angle between two orthonormal bases."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    rank = min(reference.shape[1], candidate.shape[1])
    if rank == 0:
        return 0.0
    singular_values = np.linalg.svd(reference[:, :rank].T @ candidate[:, :rank], compute_uv=False)
    return float(np.arccos(np.clip(np.min(singular_values), -1.0, 1.0)))


__all__ = [
    "jitter_points",
    "match_cycles",
    "match_regions",
    "match_routes",
    "route_distance",
    "subsample_indices",
    "subspace_principal_angle",
]


def match_junctions_across_levels(reference, candidate, *, tolerance):
    """Match compatible degrees before assigning spatially nearby regions."""
    hits = np.zeros(len(reference), dtype=bool)
    pairs = sorted((float(np.linalg.norm(left.center - right.center)), i, j)
                   for i, left in enumerate(reference) for j, right in enumerate(candidate)
                   if left.branch_count == right.branch_count)
    used = set()
    for distance, i, j in pairs:
        if distance <= tolerance and not hits[i] and j not in used:
            hits[i] = True
            used.add(j)
    return hits


def match_cycles_across_levels(reference, candidate, *, tolerance):
    """Match geometric cycles, checking relative filtration lifetime as well."""
    hits = np.zeros(len(reference), dtype=bool)
    pairs = []
    for i, left in enumerate(reference):
        for j, right in enumerate(candidate):
            if left.representative is None or right.representative is None:
                continue
            ratio = min(left.persistence, right.persistence) / max(left.persistence, right.persistence, 1e-12)
            if ratio < 0.1:
                continue
            distance = route_distance(left.representative, right.representative)
            if distance <= tolerance:
                pairs.append((distance, i, j))
    used = set()
    for _, i, j in sorted(pairs):
        if not hits[i] and j not in used:
            hits[i] = True
            used.add(j)
    return hits


def match_paths_across_levels(reference, candidate, *, tolerance,
                              reference_descendants=None, candidate_descendants=None):
    """One-to-one corridor matching with optional nested ancestry agreement."""
    if reference_descendants is None or candidate_descendants is None:
        return match_routes(reference, candidate, tolerance=tolerance)
    hits = np.zeros(len(reference), dtype=bool)
    pairs = []
    for i, left in enumerate(reference):
        for j, right in enumerate(candidate):
            overlap = np.intersect1d(reference_descendants[i], candidate_descendants[j]).size
            if not overlap:
                continue
            distance = route_distance(left, right)
            if distance <= tolerance:
                pairs.append((distance, -overlap, i, j))
    used = set()
    for _, _, i, j in sorted(pairs):
        if not hits[i] and j not in used:
            hits[i] = True
            used.add(j)
    return hits


def corridor_recurs(entry, support):
    """Test an ordered corridor against an independently constructed graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    support = np.asarray(support)
    if len(support) < 2:
        return False
    graph = entry["graph"]
    points = graph.points
    tolerance = 3 * entry["scale"]
    distances, nearest = cKDTree(points).query(support)
    if np.quantile(distances, 0.9) > tolerance:
        return False
    inside = cKDTree(support).query(points)[0] <= tolerance
    inside[nearest] = True
    rows, columns, lengths = [], [], []
    for (left, right), length in graph.edges.items():
        if inside[left] and inside[right]:
            rows.extend((left, right))
            columns.extend((right, left))
            lengths.extend((max(length, 1e-12),) * 2)
    adjacency = csr_matrix((lengths, (rows, columns)), shape=(len(points), len(points)))
    anchors = nearest[np.linspace(0, len(nearest) - 1, min(9, len(nearest)), dtype=int)]
    path = [int(anchors[0])]
    for start, end in itertools.pairwise(anchors):
        if start == end:
            continue
        _, previous = dijkstra(adjacency, indices=int(start), return_predecessors=True)
        segment = [int(end)]
        while segment[-1] != start:
            parent = int(previous[segment[-1]])
            if parent < 0:
                return False
            segment.append(parent)
        path.extend(segment[-2::-1])
    return len(set(path)) > 1 and route_distance(support, points[path]) <= tolerance


def path_resolution_support(model, support):
    """Measure independent graph-corridor recurrence at tested resolutions."""
    tested = [entry for entry in model.topology_by_level_.values() if entry["status"] == "tested"]
    if len(tested) < 2:
        return float("nan")
    return float(np.mean([corridor_recurs(entry, support) for entry in tested]))


def prepare_structural_subsamples(model, points):
    """Collect bounded topology probes before structural selection, without MIPs."""
    if not model.stability_selection or len(model.levels_) < 2:
        return
    from ._multiresolution import build_hierarchy, evaluate_level
    for run in range(model.stability_runs):
        seed = model.random_state + run + 1
        indices = subsample_indices(len(points), model.stability_fraction, seed)
        sample = jitter_points(points[indices], jitter=model.stability_jitter,
                               local_scale=model.local_scale_, rng=np.random.default_rng(seed))
        levels = build_hierarchy(sample, max_levels=model.hierarchy_max_levels,
                                 target_size=model.hierarchy_target_size,
                                 min_reduction=model.hierarchy_min_reduction,
                                 distance_quantile=model.hierarchy_distance_quantile,
                                 local_neighbors=model.hierarchy_local_neighbors,
                                 representative_method=model.representative_method, random_state=seed)
        # Match compression to the selected input size rather than growing a
        # topology/MIP problem with the original subsample size.
        target = model.backbone_input_size_
        level = min(levels, key=lambda level: abs(len(level.points) - target))
        entry = evaluate_level(model, level, -1)
        model._structural_subsamples_.append(entry)


def score_multiresolution_candidates(model, candidates, specifications):
    if len(model.levels_) < 2:
        return
    for candidate in candidates:
        support = model.routing_graph_.points[candidate.vertices]
        candidate.geometry_cost = candidate.total_cost
        for landmark in (candidate.start_landmark, candidate.end_landmark):
            region = specifications[landmark].get("region")
            confidence = getattr(region, "combined_confidence", float("nan"))
            if np.isfinite(confidence):
                candidate.total_cost += model.route_resolution_weight * (1 - confidence)
        for cycle_index in candidate.persistent_cycle_classes:
            confidence = getattr(model.persistent_cycles_[cycle_index], "combined_confidence", float("nan"))
            if np.isfinite(confidence):
                candidate.total_cost -= model.route_resolution_weight * confidence
        candidate.resolution_support = path_resolution_support(model, support)
        if np.isfinite(candidate.resolution_support):
            candidate.total_cost -= model.route_resolution_weight * candidate.resolution_support
        probes = model._structural_subsamples_
        if probes:
            hits = [corridor_recurs(entry, support) for entry in probes]
            candidate.stability_support = float(np.mean(hits))
            candidate.subsample_support = candidate.stability_support


def structural_feature_evidence(model, cycles=None, junctions=None):
    """Attach independent evidence before topology constraints are constructed."""
    tested = [entry for entry in model.topology_by_level_.values() if entry["status"] == "tested"]
    probes = model._structural_subsamples_
    if cycles is not None:
        matcher, field = match_cycles_across_levels, "persistent_cycles"
        features = cycles
    else:
        matcher, field = match_junctions_across_levels, "junctions"
        features = junctions
    for i, feature in enumerate(features):
        resolution = [matcher(features, entry[field], tolerance=4 * max(model.local_scale_, entry["scale"]))[i]
                      for entry in tested] if len(tested) > 1 else []
        sampling = [matcher(features, entry[field], tolerance=4 * max(model.local_scale_, entry["scale"]))[i]
                    for entry in probes]
        feature.resolution_support = float(np.mean(resolution)) if resolution else float("nan")
        feature.subsample_support = float(np.mean(sampling)) if sampling else float("nan")
        measured = [value for value in (feature.resolution_support, feature.subsample_support)
                    if np.isfinite(value)]
        feature.combined_confidence = float(np.prod(measured)) if measured else float("nan")
    # Geometric cycle representatives are approximate. Do not erase mandatory
    # topology constraints solely because that proxy did not match; confidence
    # instead changes competing candidate costs at the selection boundary.
    return features
