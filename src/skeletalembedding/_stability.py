"""Consensus utilities for optional subsampling-based stability selection."""

from __future__ import annotations

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
