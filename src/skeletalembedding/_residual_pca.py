"""Local residual-PCA fields attached to fitted spline routes."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._frames import _frame_from_grid, _normal_frame, _normal_frames
from .results import EmbeddingResult

Array = np.ndarray


def _projector_basis(
    projector: Array,
    tangent: Array,
    rank: int,
    fallback: Array,
) -> Array:
    """Return an orthonormal rank-r basis in the tangent normal space."""
    if rank == 0:
        return np.empty((len(tangent), 0), dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-12:
        tangent = np.zeros_like(tangent)
        tangent[0] = 1.0
    else:
        tangent = tangent / tangent_norm
    normal_projector = np.eye(len(tangent)) - np.outer(tangent, tangent)
    projected = normal_projector @ np.asarray(projector, dtype=float)
    # The eigenspace is computed in the normal coordinates, so this should
    # have rank ``rank``.  Numerical fallback keeps the fitted field valid for
    # sparse routes and degenerate local covariance matrices.
    basis, _ = np.linalg.qr(projected, mode="reduced")
    if basis.shape[1] < rank or not np.all(np.isfinite(basis)):
        projected = normal_projector @ np.asarray(fallback, dtype=float)
        basis, _ = np.linalg.qr(projected, mode="reduced")
    if basis.shape[1] < rank:
        normal = _normal_frame(tangent)
        basis = normal[:, :rank]
    return basis[:, :rank]


def _align_basis(previous: Array, current: Array) -> Array:
    """Resolve the arbitrary PCA orientation between neighboring bases."""
    if current.shape[1] == 0:
        return current
    left, _, right = np.linalg.svd(previous.T @ current, full_matrices=False)
    return current @ (right.T @ left.T)


def _route_distances(values: Array, center: float, closed: bool) -> Array:
    distances = np.abs(np.asarray(values, dtype=float) - center)
    if closed:
        distances = np.minimum(distances, 1.0 - distances)
    return distances


def _smooth_route_bases(
    bases: Array,
    tangents: Array,
    grid_t: Array,
    closed: bool,
    smoothness: float,
) -> Array:
    """Smooth route subspaces by repeated neighboring projector averaging."""
    if bases.shape[2] == 0:
        return bases
    unique_count = len(bases) - 1 if closed else len(bases)
    working = np.asarray(bases[:unique_count], dtype=float).copy()
    if unique_count == 0:
        return bases

    # First align the basis coordinates.  This leaves every projector intact
    # but makes interpolation and subsequent QR steps deterministic.
    for index in range(1, unique_count):
        working[index] = _align_basis(working[index - 1], working[index])
    if closed and unique_count > 1:
        working[-1] = _align_basis(working[0], working[-1])

    if smoothness > 0.0 and unique_count > 1:
        weight = smoothness / (1.0 + smoothness)
        for _ in range(5):
            projectors = np.einsum("ntr,nur->ntu", working, working)
            updated = np.empty_like(working)
            for index in range(unique_count):
                if closed:
                    neighbours = (projectors[(index - 1) % unique_count]
                                  + projectors[(index + 1) % unique_count]) / 2.0
                elif index == 0:
                    neighbours = projectors[1]
                elif index == unique_count - 1:
                    neighbours = projectors[-2]
                else:
                    neighbours = (projectors[index - 1] + projectors[index + 1]) / 2.0
                blended = (1.0 - weight) * projectors[index] + weight * neighbours
                values, vectors = np.linalg.eigh(blended)
                candidate = vectors[:, np.argsort(values)[::-1][: working.shape[2]]]
                updated[index] = _projector_basis(
                    candidate @ candidate.T,
                    tangents[index],
                    working.shape[2],
                    working[index],
                )
            for index in range(1, unique_count):
                updated[index] = _align_basis(updated[index - 1], updated[index])
            working = updated

    if closed:
        return np.concatenate((working, working[:1]), axis=0)
    return working


def fit_residual_pca(
    model: Any,
    points: Array,
    centerline_result: EmbeddingResult,
) -> None:
    """Fit fixed-dimensional, route-local residual PCA grids in place."""
    dimension = int(points.shape[1])
    rank = int(model.residual_dim_)
    model.residual_parameter_grids_ = []
    model.residual_pca_t_ = model.residual_parameter_grids_
    model.residual_bases_ = []
    model.residual_eigenvalues_ = []
    if rank == 0:
        for route in range(len(model.routes_)):
            grid_t = np.asarray(model.normal_frame_grids_[route]["t"])
            model.residual_parameter_grids_.append(grid_t.copy())
            model.residual_bases_.append(np.zeros((len(grid_t), dimension, 0), dtype=float))
            model.residual_eigenvalues_.append(np.zeros((len(grid_t), 0), dtype=float))
        return

    residual_scaled = np.asarray(centerline_result.residual, dtype=float) / model.scale_
    frames = _normal_frames(model, centerline_result)
    for route, spline in enumerate(model.routes_):
        grid_t = np.asarray(model.normal_frame_grids_[route]["t"], dtype=float).copy()
        tangents = np.asarray(spline.tangent(grid_t), dtype=float)
        bases = np.empty((len(grid_t), dimension, rank), dtype=float)
        eigenvalues = np.empty((len(grid_t), rank), dtype=float)
        members = np.flatnonzero(centerline_result.route_id == route)
        if len(members):
            local_coordinates = np.einsum(
                "ni,nij->nj", residual_scaled[members], frames[members]
            )
            local_t = centerline_result.position[members]
        else:
            local_coordinates = np.empty((0, max(0, dimension - 1)), dtype=float)
            local_t = np.empty(0, dtype=float)
        normal_grid = model.normal_frame_grids_[route]
        for index, center in enumerate(grid_t):
            if len(members):
                distances = _route_distances(local_t, center, spline.closed)
                scaled_distance = (distances - np.min(distances)) / model.residual_pca_bandwidth
                weights = np.exp(-0.5 * scaled_distance * scaled_distance)
                if not np.any(weights > 0.0):
                    weights[np.argmin(distances)] = 1.0
                covariance = (local_coordinates * weights[:, None]).T @ local_coordinates
                covariance /= max(float(np.sum(weights)), 1e-12)
            else:
                covariance = np.zeros((max(0, dimension - 1), max(0, dimension - 1)))
            values, vectors = np.linalg.eigh(covariance)
            order = np.argsort(values)[::-1]
            values = np.maximum(values[order], 0.0)
            vectors = vectors[:, order]
            normal = _frame_from_grid(normal_grid, float(center), tangents[index], spline.closed)
            bases[index] = normal @ vectors[:, :rank]
            eigenvalues[index] = values[:rank]
        bases = _smooth_route_bases(
            bases, tangents, grid_t, spline.closed, model.residual_subspace_smoothness
        )
        model.residual_parameter_grids_.append(grid_t)
        model.residual_bases_.append(bases)
        model.residual_eigenvalues_.append(eigenvalues)


def _basis_at(model: Any, route: int, t: float, tangent: Array) -> Array:
    spline = model.routes_[route]
    grid_t = model.residual_parameter_grids_[route]
    grid_basis = model.residual_bases_[route]
    if grid_basis.shape[2] == 0:
        return grid_basis[0]
    value = float(t % 1.0) if spline.closed else float(np.clip(t, 0.0, 1.0))
    index = int(np.searchsorted(grid_t, value, side="right") - 1)
    index = min(max(index, 0), len(grid_t) - 2)
    denominator = grid_t[index + 1] - grid_t[index]
    alpha = 0.0 if denominator <= 1e-12 else (value - grid_t[index]) / denominator
    base = (1.0 - alpha) * grid_basis[index] + alpha * grid_basis[index + 1]
    return _projector_basis(base @ base.T, tangent, grid_basis.shape[2], base)


def attach_residual_pca(
    model: Any,
    original: Array,
    points: Array,
    centerline_result: EmbeddingResult,
) -> EmbeddingResult:
    """Attach local PCA coordinates and the residual decomposition."""
    rank = int(model.residual_dim_)
    if rank == 0:
        return EmbeddingResult(
            route_id=centerline_result.route_id,
            position=centerline_result.position,
            projected=centerline_result.projected,
            residual=centerline_result.residual,
            residual_norm=centerline_result.residual_norm,
            tangent=centerline_result.tangent,
        )
    bases = np.empty((len(points), points.shape[1], rank), dtype=float)
    for index, (route, position) in enumerate(
        zip(centerline_result.route_id, centerline_result.position)
    ):
        bases[index] = _basis_at(
            model, int(route), float(position), centerline_result.tangent[index]
        )
    residual_scaled = np.asarray(centerline_result.residual, dtype=float) / model.scale_
    coordinates = np.einsum("ni,nir->nr", residual_scaled, bases)
    projection_scaled = (centerline_result.projected - model.mean_) / model.scale_
    reconstructed = (projection_scaled + np.einsum("nir,nr->ni", bases, coordinates))
    reconstructed_original = reconstructed * model.scale_ + model.mean_
    unexplained = np.asarray(original, dtype=float) - reconstructed_original
    return EmbeddingResult(
        route_id=centerline_result.route_id,
        position=centerline_result.position,
        projected=centerline_result.projected,
        residual=centerline_result.residual,
        residual_norm=centerline_result.residual_norm,
        tangent=centerline_result.tangent,
        residual_coordinates=coordinates,
        reconstructed=reconstructed_original,
        unexplained_residual=unexplained,
        unexplained_residual_norm=np.linalg.norm(unexplained, axis=1),
    )


__all__ = ["attach_residual_pca", "fit_residual_pca"]
