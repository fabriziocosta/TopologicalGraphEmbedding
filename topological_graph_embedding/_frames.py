"""Deterministic normal-frame construction for spline routes."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._curves import _SplineRoute

Array = np.ndarray

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


def _fit_normal_frame_grid(spline: _SplineRoute) -> dict[str, Any]:
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


def _normal_frames(model: Any, result: Any) -> Array:
    """Return one orthonormal normal frame for every projected observation.

    The returned array has shape ``(n_samples, n_features, n_features - 1)``.
    Coordinates in this frame describe only displacement perpendicular to the
    local spline tangent; the longitudinal direction is intentionally omitted.
    """
    highway_ids = np.asarray(result.route_id, dtype=int)
    t_values = np.asarray(result.position, dtype=float)
    n_samples = len(highway_ids)
    n_features = int(np.asarray(result.residual).shape[1])
    frames = np.zeros((n_samples, n_features, max(0, n_features - 1)), dtype=float)
    tangent_vectors = result.tangent
    if tangent_vectors is not None:
        tangent_vectors = np.asarray(tangent_vectors, dtype=float)
    frame_grids = getattr(model, "normal_frame_grids_", None)
    for route, spline in enumerate(model.routes_):
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


def _normal_coordinates(model: Any, result: Any) -> Array:
    """Project residuals into the local spline-normal hyperplane coordinates."""
    residual = np.asarray(result.residual, dtype=float)
    scale = np.asarray(getattr(model, "scale_", np.ones(residual.shape[1])), dtype=float)
    residual_scaled = residual / scale
    frames = _normal_frames(model, result)
    if frames.shape[2] == 0:
        return np.empty((len(residual), 0), dtype=float)
    return np.einsum("ni,nij->nj", residual_scaled, frames)


