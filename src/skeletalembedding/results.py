"""Typed outputs returned by the spline graph embedding estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Projection of observations onto a fitted spline route network.

    All arrays have one row per observation.  ``tangent`` is expressed in the
    standardized fitting coordinates used to construct the normal frame.
    """

    route_id: np.ndarray
    position: np.ndarray
    projected: np.ndarray
    residual: np.ndarray
    residual_norm: np.ndarray
    tangent: np.ndarray
    residual_coordinates: np.ndarray | None = None
    reconstructed: np.ndarray | None = None
    unexplained_residual: np.ndarray | None = None
    unexplained_residual_norm: np.ndarray | None = None

    def __post_init__(self) -> None:
        route_id = np.asarray(self.route_id, dtype=int)
        position = np.asarray(self.position, dtype=float)
        projected = np.asarray(self.projected, dtype=float)
        residual = np.asarray(self.residual, dtype=float)
        residual_norm = np.asarray(self.residual_norm, dtype=float)
        tangent = np.asarray(self.tangent, dtype=float)
        count = len(route_id)
        if self.residual_coordinates is None:
            residual_coordinates = np.empty((count, 0), dtype=float)
        else:
            residual_coordinates = np.asarray(self.residual_coordinates, dtype=float)
        if self.reconstructed is None:
            reconstructed = projected.copy()
        else:
            reconstructed = np.asarray(self.reconstructed, dtype=float)
        if self.unexplained_residual is None:
            unexplained_residual = residual.copy()
        else:
            unexplained_residual = np.asarray(self.unexplained_residual, dtype=float)
        if self.unexplained_residual_norm is None:
            unexplained_residual_norm = np.linalg.norm(unexplained_residual, axis=1)
        else:
            unexplained_residual_norm = np.asarray(self.unexplained_residual_norm, dtype=float)
        for name, value in (
            ("route_id", route_id),
            ("position", position),
            ("projected", projected),
            ("residual", residual),
            ("residual_norm", residual_norm),
            ("tangent", tangent),
            ("residual_coordinates", residual_coordinates),
            ("reconstructed", reconstructed),
            ("unexplained_residual", unexplained_residual),
            ("unexplained_residual_norm", unexplained_residual_norm),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.route_id.ndim != 1 or self.position.ndim != 1:
            raise ValueError("route_id and position must be one-dimensional")
        if self.residual_norm.ndim != 1 or self.unexplained_residual_norm.ndim != 1:
            raise ValueError("residual norms must be one-dimensional")
        if (
            self.projected.ndim != 2
            or self.residual.ndim != 2
            or self.tangent.ndim != 2
            or self.residual_coordinates.ndim != 2
            or self.reconstructed.ndim != 2
            or self.unexplained_residual.ndim != 2
        ):
            raise ValueError("result feature arrays must be two-dimensional")
        if any(len(array) != count for array in (
            self.position,
            self.projected,
            self.residual,
            self.residual_norm,
            self.tangent,
            self.residual_coordinates,
            self.reconstructed,
            self.unexplained_residual,
            self.unexplained_residual_norm,
        )):
            raise ValueError("all result arrays must contain the same number of observations")
        if self.projected.shape != self.residual.shape or self.projected.shape != self.tangent.shape:
            raise ValueError("projected, residual, and tangent must have matching shapes")
        if (
            self.reconstructed.shape != self.projected.shape
            or self.unexplained_residual.shape != self.projected.shape
        ):
            raise ValueError("reconstructed and unexplained_residual must match projected shape")

    @property
    def n_samples(self) -> int:
        """Number of observations represented by the result."""
        return len(self.route_id)

    @property
    def n_features(self) -> int:
        """Number of original features represented by the result."""
        return self.projected.shape[1]


__all__ = ["EmbeddingResult"]
